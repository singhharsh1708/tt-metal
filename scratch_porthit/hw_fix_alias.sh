set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
cd "$TT_METAL_HOME" || { echo "no $TT_METAL_HOME"; exit 1; }
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
F=ttnn/api/ttnn/mesh_device_operation_adapter.hpp
RUN=/home/user/fixalias_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
git checkout -q -- "$F"
python ~/patch_alias.py "$F" || { echo "PATCH FAILED, nothing built"; exit 1; }
echo "patch applied: +$(git diff --numstat -- "$F" | cut -f1) lines; tree dirty files: $(git status --porcelain | wc -l | tr -d ' ')"
echo "=== rebuild start $(date -u +%H:%M) (this header is included by every op, expect a long build)"
t0=$(date +%s)
if [ -f build/build.ninja ]; then ninja -C build > "$RUN/build.log" 2>&1; else cmake --build build -- -j"$(nproc)" > "$RUN/build.log" 2>&1; fi
RC=$?; echo "=== rebuild rc=$RC $(( $(date +%s) - t0 ))s"
if [ $RC -ne 0 ]; then grep -E "error:|FAILED" "$RUN/build.log" | head -14 | cut -c1-260; git checkout -q -- "$F"; echo "BUILD FAILED; adapter patch reverted (library on disk may be half built: rerun ninja before other tests)"; exit 1; fi
cmake --install build > "$RUN/install.log" 2>&1; echo "=== install rc=$?"
one () { command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3; timeout -k 20 900 "${@:2}" > "$RUN/$1.log" 2>&1; echo "===== [$1] rc=$?"; grep -E '^(RESULT|INFO)' "$RUN/$1.log" | cut -c1-220; }
one alias_sampling_patched python ~/alias_sampling.py
one alias_manual_seed_patched python ~/alias_manual_seed.py
one slice_ring_sanity python ~/slice_ring.py
pt () { command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3; timeout -k 20 1800 pytest -q -p no:cacheprovider "${@:2}" > "$RUN/$1.log" 2>&1; echo "===== [$1] rc=$? | $(grep -E '(passed|failed|error).* in [0-9.]+s' "$RUN/$1.log" | tail -1 | cut -c1-120)"; grep -E "^(FAILED|ERROR) " "$RUN/$1.log" | head -6 | cut -c1-200; }
pt tt_alias_test tests/ttnn/unit_tests/base_functionality/test_program_cache_tensor_aliasing.py
pt tt_sampling_tests tests/ttnn/unit_tests/operations/reduce/test_sampling.py
pt tt_batch_norm tests/ttnn/unit_tests/operations/fused/test_batch_norm.py
echo "source LEFT PATCHED (slice + adapter), dirty: $(git status --porcelain | wc -l | tr -d ' ').  done $(date -u +%H:%M)  logs: $RUN"
