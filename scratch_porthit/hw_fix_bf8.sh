set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
cd "$TT_METAL_HOME" || { echo "no $TT_METAL_HOME"; exit 1; }
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
F=ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp
RUN=/home/user/fixbf8_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
git checkout -q -- $F
python ~/patch_bf8.py "$TT_METAL_HOME" || { echo "PATCH FAILED, nothing built"; git checkout -q -- $F; exit 1; }
git diff --stat -- $F | tail -2
echo "=== rebuild start $(date -u +%H:%M)"; t0=$(date +%s)
ninja -C build > "$RUN/build.log" 2>&1; RC=$?; echo "=== rebuild rc=$RC $(( $(date +%s) - t0 ))s"
if [ $RC -ne 0 ]; then grep -E "error:|FAILED|undefined" "$RUN/build.log" | head -12 | cut -c1-260; git checkout -q -- $F; echo "BUILD FAILED; bf8 patch reverted, nothing installed"; exit 1; fi
cmake --install build > "$RUN/install.log" 2>&1; echo "=== install rc=$?"
command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
timeout -k 20 900 python ~/cand_probe.py bf8 > "$RUN/bf8.log" 2>&1; echo "===== [bf8_patched] rc=$?"; grep '^RESULT' "$RUN/bf8.log" | cut -c1-330
command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
timeout -k 20 3000 pytest -q -p no:cacheprovider tests/ttnn/unit_tests/operations/eltwise/test_unary.py -k "recip or rsqrt or log" > "$RUN/test_unary.log" 2>&1
echo "===== [TT test_unary.py -k recip/rsqrt/log] rc=$? | $(grep -E '(passed|failed|error).* in [0-9.]+s' "$RUN/test_unary.log" | tail -1 | cut -c1-120)"; grep -E "^(FAILED|ERROR) " "$RUN/test_unary.log" | head -6 | cut -c1-200
echo "source LEFT PATCHED, dirty: $(git status --porcelain | wc -l | tr -d ' ').  done $(date -u +%H:%M)  logs: $RUN"
