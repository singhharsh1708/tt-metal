# GLM-5.2 prefill → prefill loopback KV migration

How to migrate a GLM-5.2 KV cache from one prefill slot to another and prove the destination is
correct. This is **Gate 2** of [PREFILL_MIGRATION_TESTING.md](./PREFILL_MIGRATION_TESTING.md), i.e.
tt-llm-engine's `tests/docs/KV_MIGRATION_VALIDATION.md` §3, for the first *sparse* (two-cache) model.

GLM-5.2 owns **two** KV caches, and both must move and both must be checked:

| table config | cache | rows |
|---|---|---|
| 0 | MLA KVPE (bf16 ROW_MAJOR, head_dim 576) | one per layer (78) |
| 1 | lightning-indexer KEY (bfp8_b TILE, head_dim 128) | one per **`full` indexer rank** (21) — *not* per layer |

The engine handles the two together already: `dcn_sender_backend.cpp:222` (`migrate_slot`) iterates
every table config per layer and skips a config whose `num_layers` does not cover that layer, so a
single `migrate[0,78)` moves config 0 rows 0-77 and config 1 rows 0-20 with no `lookup_range` abort.
Config 1's rank-vs-global-layer compaction is harmless for a loopback because source and destination
use identical row indices.

## Validated result (2026-08-11, 8×4 Blackhole galaxy)

78 layers, 56320 tokens, slot 0 → slot 1. The copy itself took 3.4 min after 858/858 layer acks.

| check | where | result |
|---|---|---|
| **A** — `dst == src`, on-device, both caches | runner | `comparing 2 cache tensor(s), shapes [(78,1,56320,576), (21,1,56320,128)]` → **min_pcc 1.000000** |
| **B** — bit-exact, both configs | driver, over UMD | config 0: 274560 reads, config 1: 73920 → **174240 chunks byte-identical** |
| **C** — source KV vs golden | producer, over UMD | **min 0.860911 ≥ 0.85**, plus index PCC over 21/21 |

A and B answer "was the copy lossless"; C answers "was the thing we copied correct". C's 0.860911 is
identical to the standalone (Gate 0/1) figure, which is what lets you attribute a future failure to one
side or the other rather than guessing.

## 1. Prerequisites

**tt-llm-engine**, built against *this* tt-metal. Do **not** pass `--synthetic-only` — that produces a
`SimulatedDram` worker which never touches galaxy DRAM:

```bash
cd $ENGINE
TT_METAL_DIR=$TT_METAL TT_METAL_BUILD_DIR=$TT_METAL/build_Release \
Python_ROOT_DIR=$TT_METAL/python_env PATH=$TT_METAL/python_env/bin:$PATH \
  ./build_migration_layer.sh --build-type=RelWithDebInfo --jobs=$(nproc)
```
Artifacts land in `disaggregation/migration/build_RelWithDebInfo/` (`bin/migration_endpoint`,
`bin/migration_worker`, `python/_migration_client*.so`) — the directory
`launch_migration_endpoints.sh` defaults to. The worker never links `libtt_metal.so` (pure-UMD path,
`CMakeLists.txt:385-394`), so cross-tree ABI drift is not the hazard it looks like.

Optional build sanity checks. **The MPI vars are required for the second one** — on a box with real
NICs, OpenMPI otherwise picks one nondeterministically and the rendezvous times out after 60 s, which
looks like a hang (`run_single_host_tests.sh:64-75`):

```bash
MIG=$ENGINE/disaggregation/migration/build_RelWithDebInfo
$MIG/test/migration_tests                                  # 92 tests PASSED
OMPI_MCA_btl_tcp_if_include=127.0.0.1/8 OMPI_MCA_oob_tcp_if_include=127.0.0.1/8 \
OMPI_MCA_oob_tcp_disable_family=IPv6 \
  $MIG/test/test_worker_synthetic_routing \
    --gtest_filter='Ranks2/SyntheticRoutingTest.MigrateAllSlots/small_1L_1S__p2p'   # ~0.4 s, OK
```
(`--gtest_filter='*small_1L_1S*'` matches ~6 suites, not one case — budget minutes, not seconds.)

## 2. Configure

These are tt-llm-engine launch-harness configs — the same schema as the engine's own
`disagg_harness_prefill_loopback.yaml`, which you could equally edit in place. They exist separately
only so the ten GLM-specific values (and the reason for each) are recorded rather than re-derived: that
file's defaults are DeepSeek's (61 layers, `num_users: 2`, `roce`, `NCHUNKS: 22`, `PCC 0.93`), and two
of those are silent traps — a wrong `NCHUNKS` **hangs the runner**, and `0.93` **false-fails** GLM-5.2's
source check. Editing it in place also puts your run config in a file the engine repo tracks.

The GLM-5.2 configs live in the engine, next to the example they derive from:
`$ENGINE/disaggregation/launch_harness/disagg_harness_glm52_loopback_{1layer,78layer}.yaml`. Copy one
out and fill in the six `[EDIT]`-marked machine-specific values (everything else is portable).
The 1-layer one is a fast smoke; the 78-layer one is the real gate:

```bash
export ENGINE=/path/to/tt-llm-engine   TT_METAL=/path/to/tt-metal
export RUNS=$HOME/disagg_runs   SHARED=$HOME/tmp/pd   BLAZE=/path/to/tt-blaze
mkdir -p "$RUNS" "$SHARED"

sed -e "s|/path/to/tt-llm-engine|$ENGINE|g" -e "s|/path/to/tt-metal|$TT_METAL|g" \
    -e "s|/path/to/disagg_runs|$RUNS|g"     -e "s|/path/to/tmp/pd|$SHARED|g" \
    -e "s|/path/to/tt-blaze|$BLAZE|g"       -e "s|<hostname>|$(hostname)|g" \
    "$ENGINE/disaggregation/launch_harness/disagg_harness_glm52_loopback_78layer.yaml" \
    > /tmp/glm52_loopback.yaml
```

## 3. Run

```bash
cd "$ENGINE"
export TT_METAL_HOME="$TT_METAL"        # MANDATORY — see the gotcha below
python3 -m disaggregation.launch_harness pd_producer --config /tmp/glm52_loopback.yaml --dry-run
timeout --signal=KILL 5400 python3 -m disaggregation.launch_harness pd_producer --config /tmp/glm52_loopback.yaml
```

Five steps: `preclean_pd_producer → migration_endpoints → prefill_runner` (waits for
`[migration] WORKER_READY`) `→ prefill_producer → pd_producer_complete`. Harness exit 0 == all checks
passed. Always `--dry-run` after editing a config: it prints every step command without touching the
device. Budget ~20 min at 78 layers (~7 min of weight load, ~1 min compile, 11 chunks, a 3.4 min copy,
then two read-back passes).

## 4. Read the verdicts

```bash
D=$(ls -dt "$RUNS"/*/ | head -1)

# CHECK A -- the `shapes` line is the proof of two-cache coverage. A byte copy scores exactly 1.0 on
# every tensor, so the PASS line ALONE cannot tell you whether one cache or two were compared.
grep -aE "kv-migrate-validate" "$D/prefill_runner.log" | sed 's/^\[1,0\]<st[a-z]*>: //'

# CHECK B (bit-exact) and CHECK C (source vs golden)
grep -aiE "MIGRATE slot|verify bytes" "$D/prefill_producer.log"
grep -aiE "KV PCC over|index PCC over|PCC PASSED" "$D/prefill_producer.log"
```

## 5. Cleanup after a crash or Ctrl-C

Unnecessary after a clean exit. The H2D sockets are created `O_CREAT|O_EXCL`, so a leftover segment
fails the next run outright, and a stale DONE sentinel makes the validator pick up the wrong pairs.
The harness's own preclean will not kill a stray *current* runner — `launch_harness/cleanup.py:13-14`
still greps the retired `deepseek_v3_d_p` module paths.

```bash
pkill -f "disaggregation.launch_harness"; pkill -f launch_migration_endpoints
pkill -f migration_endpoint; pkill -f migration_worker; pkill -f prefill_runner; pkill -f ttrun.py; pkill prte
rm -f /dev/shm/mig_ep* /dev/shm/ep_1_* /dev/shm/tt_h2d_* /dev/shm/tt_d2h_* \
      /dev/shm/tt_prefill_layer_acks_* "$SHARED"/migration_done.sentinel*
```
On a wedged device: `tt-smi -glx_reset`.

## Gotchas

- **`export TT_METAL_HOME` to a real tt-metal tree.** The worker resolves SoC descriptors under it and
  `launch_migration_endpoints.sh:211` defaults it to the engine's *uninitialized* `tt-metal` submodule
  → the worker aborts with `YAML::BadFile … /tt_metal/soc_descriptors/blackhole_140_arch.yaml`, and the
  runner then blocks on `WORKER_READY` until the gate times out. No harness-config field covers this.
- **Scale depth with `num_layers`, never `PREFILL_MIGRATION_LAYERS`.** With a layer subset the bit-exact
  check silently drops to config 0 only (`migration_driver.py:613-619`) and the golden check refuses
  outright — because config 1's rows are full-indexer *ranks*, so global layer 3 would read index rank
  3, i.e. global layer 6.
- **`PREFILL_STANDALONE_CHUNKED_NCHUNKS` must equal `producer_chunks × producer_max_requests`.** The
  runner only validates after its request loop exits, which is after that many chunks; a mismatch hangs
  it.
- **Use the `glm-traces/vllm-glm52-indexer-kcache-55k` golden.** The adapter's own
  `prefill_trace_default` lacks `dsa/indexer_k_layer_*`, so `index_golden_present()` is false and every
  indexer-K check is skipped with only a warning.
- **`PREFILL_STANDALONE_CHUNKED_PCC=0.85`** for GLM-5.2. Its KVPE `nope` PCC bottoms at ~0.86 @ L75
  (`test_prefill_transformer_chunked.py:134-137`); the 0.93 default is above the model's documented
  minimum and false-fails.
- **Slots**: `num_users: N` migrates src `[0,N)` → dst `[N,2N)` and the harness sizes the runner's table
  to `2N`. Do not double it by hand. `migration_driver` also rejects a loopback mapping where a slot is
  both source and destination.
- **Cross-talk is not covered** by a single pair driven from one prompt: all sources are byte-identical,
  so a mis-routed copy is indistinguishable from a correct one. Use `PREFILL_PRODUCER_SLOT_TRACES` with
  distinct per-slot prompts once you migrate more than one pair.

## Related

- [PREFILL_MIGRATION_TESTING.md](./PREFILL_MIGRATION_TESTING.md) — the three gates, all config files,
  and the values that must agree across processes.
- `models/demos/deepseek_v3_d_p/tt/runners/kv_chunk_table.py:158` — where the merged two-config table
  is built; `utils/kv_cache_utils.py:457` for the block-cyclic address math.
- `models/demos/deepseek_v3_d_p/tests/test_kv_cache_table.py:688` — `test_glm52_kv_cache_table`, the
  cheapest device-side reference for the two-config table.
- tt-llm-engine `disaggregation/launch_harness/disagg_harness_prefill_loopback.yaml` — the upstream
  config these are adapted from. Note its sibling
  `tests/docs/PrefillLoopbackMigrationTestGuide.md` is stale (it targets the retired
  `deepseek_v3_d_p/.../prefill_runner.py`).
