set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
cd "$TT_METAL_HOME" || { echo "no $TT_METAL_HOME"; exit 1; }
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
P=ttnn/cpp/ttnn/operations/pool
RUN=/home/user/fixpool_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
git checkout -q -- $P/upsample/device $P/grid_sample/device
python ~/patch_pool.py "$TT_METAL_HOME" || { echo "PATCH FAILED, nothing built"; git checkout -q -- $P/upsample/device $P/grid_sample/device; exit 1; }
git diff --stat -- $P | tail -5
echo "=== rebuild start $(date -u +%H:%M)"; t0=$(date +%s)
ninja -C build > "$RUN/build.log" 2>&1; RC=$?; echo "=== rebuild rc=$RC $(( $(date +%s) - t0 ))s"
if [ $RC -ne 0 ]; then grep -E "error:|FAILED" "$RUN/build.log" | head -12 | cut -c1-260; git checkout -q -- $P/upsample/device $P/grid_sample/device; echo "BUILD FAILED; pool patch reverted, nothing installed"; exit 1; fi
cmake --install build > "$RUN/install.log" 2>&1; echo "=== install rc=$?"
one () { command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3; timeout -k 20 900 "${@:2}" > "$RUN/$1.log" 2>&1; echo "===== [$1] rc=$?"; grep -E '^RESULT' "$RUN/$1.log" | cut -c1-220; grep -E "TT_FATAL|TT_THROW" "$RUN/$1.log" | tail -1 | cut -c1-220; }
for c in up_shard up_fp32 grid; do one ${c}_patched python ~/pool_probe.py $c; done
pt () { command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3; timeout -k 20 2400 pytest -q -p no:cacheprovider "${@:2}" > "$RUN/$1.log" 2>&1; echo "===== [$1] rc=$? | $(grep -E '(passed|failed|error).* in [0-9.]+s' "$RUN/$1.log" | tail -1 | cut -c1-120)"; grep -E "^(FAILED|ERROR) " "$RUN/$1.log" | head -6 | cut -c1-200; }
pt tt_test_upsample tests/ttnn/unit_tests/operations/pool/test_upsample.py
pt tt_test_grid_sample tests/ttnn/unit_tests/operations/pool/test_grid_sample.py
echo "source LEFT PATCHED, dirty: $(git status --porcelain | wc -l | tr -d ' ').  done $(date -u +%H:%M)  logs: $RUN"
