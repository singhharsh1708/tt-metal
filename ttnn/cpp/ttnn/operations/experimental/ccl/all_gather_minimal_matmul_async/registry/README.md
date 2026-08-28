# AGMM configuration registry

This directory contains the native runtime boundary for a future certified
configuration registry for `all_gather_minimal_matmul_async`. It is deliberately
separate from the one-chip `matmul`/`linear`/`addmm` registry: it has its own
configuration switch (`agmm_registry_mode`), compact key and replay ABI,
compiled table, structural compatibility check, circuit breaker, and telemetry.

The offline-to-native boundary is specified in
[`GENERATED_TABLE_CONTRACT.md`](GENERATED_TABLE_CONTRACT.md). The generated
lock and exact recipes are typed C++ constants compiled into TT-metal; there is
no runtime predictor, wrapper, sidecar, or filesystem lookup. A constexpr
validator rejects ABI drift, missing provenance, device-domain mismatch, wrong
entry counts, and non-strict key ordering before a populated header can build.
Each populated lock is restricted to one exact Blackhole 32-chip 8x4 device
descriptor at one worker-grid size.

The checked-in table holds 40 exact BH32 entries drawn from the `20260822d`
silicon campaign, and the mode defaults to `Off`. `Off` bypasses lookup,
request construction, telemetry, and the selected-call guard. `Shadow` and `On`
still fall back whenever the live request cannot be built. An explicit program configuration or compute-kernel configuration is
ineligible and remains observable in `Shadow`; trace capture, including an
unknown trace state, is also ineligible. These preflight reasons intentionally
precede `empty_registry`.

## Exact native contract

The key is an immutable, bounded POD. It distinguishes:

- exact local logical and padded tensor shapes, dtype, layout, buffer type,
  memory layout, and 32x32 tile metadata for every input and optional tensor;
- effective logical and padded M/K/N plus batch;
- requested output dtype, layout, tile, buffer type, and memory layout;
- TP and FSDP effective topology, ring sizes, axes, links, workers, buffers,
  chunking, semaphore counts (never semaphore addresses),
  barrier/persistent-buffer presence, transpose, SwiGLU, activation, and exact
  IEEE-754 ternary-scalar bits;
- architecture and device/mesh/worker-grid dimensions.

`num_workers_per_link` is derived from the worker grid, is itself a key axis,
and the operation asserts the two agree. Emitted entries therefore carry the
measured `(grid, links, workers, buffers)` tuple consistently; an entry whose
fabric facts disagree with its replay grid is rejected before export.

The replay descriptor contains only `MinimalMatmulConfig` and the complete
`DeviceComputeKernelConfig`, including an explicit `throttle_level` whose six
supported values are bound one-for-one during materialization. Materialization
validates schema/ABI, the exact BH32 8x4 domain, default tile
geometry, exact workload consistency, grid bounds, block divisibility, and
destination-register constraints before the public operation is dispatched.
Any validation or materialization failure falls back and opens the per-registry
circuit breaker.

The evidence source schema used by the exporter may evolve independently; the
native compact key and replay schemas start at version 1.

## Why there is no attestation gate

An earlier revision gated every lookup behind a `production_attestation`
provider and three compatibility digests: a semantic-source digest over the AGMM
sources, a build-identity digest, and a runtime-capability/firmware digest. The
device key additionally bound a board capability class and ordered-mesh,
fabric-topology, and runtime-capability digests, and every tensor bound opaque
memory-config and tensor-topology digests.

None of those could ever hold. Build identity binds the exact flag set, so only
a bit-identical rebuild of the promoter's tree matches; the semantic digest
invalidates the whole table on a comment edit; the runtime-capability digest
invalidates it on a firmware bump; and the ordered-mesh, fabric, memory-config
and tensor-topology digests have no versioned canonical preimage any offline
promoter can reproduce. The net effect was total: `production_attestation`
returned `UnsupportedAttestation` unconditionally, so the registry could not
serve a single call.

The gate also did not guard correctness. A stale recipe is slower, not wrong:
`materialize_recipe` still proves candidate legality on whatever the registry
injects, and the operation's own validators still run.

What remains is the real safety surface:

- the constexpr and runtime table validators (schema/ABI, entry count, nonzero
  entry ids, exact BH32 8x4 domain, strict key order);
- an exact device-domain match between the request and `certified_device`; and
- `materialize_recipe`: exact tile geometry, workload/tensor shape consistency,
  the collective K equation, grid bounds, block and subblock divisibility, the
  SwiGLU even-N-block rule, K-block versus local K tiles, and the exact
  default-32x32 `get_dest_reg_count` destination-register capacity.

`content_sha256` stays because it is the lock's own integrity digest.
`semantic_source_sha256`, `evidence_manifest_sha256`, `predictor_sha256` and
`exporter_sha256` stay as inert provenance: emitted so a table's origin is
readable in a diff and in telemetry, required nonzero, never compared against a
running build, firmware, or source tree.

`build_registry_request` remains the device-free seam that turns resolved
compact facts into a request. It rejects trace capture and explicit
program/kernel overrides, and checks mesh, optional-tensor, operation, tile and
workload consistency. Wiring it to live TT objects is still open work: the
device operation currently resolves with no request, so `On` observes
`incomplete_request` and falls back.

## One-shot execution

`Shadow` can record an exact would-hit but never materializes or applies a
recipe. `On` resolves and materializes at most once. Once a selected public
launch begins there is no baseline retry path: an execution exception propagates
with existing operation semantics, opens the circuit breaker, and does not
increment `launch_completed_hits`. `selected_hits` and `launch_completed_hits`
are separate. The latter means only that the asynchronous public launch API
returned without throwing; it is not a device synchronization, execution
completion, PCC, or silicon-correctness signal.
