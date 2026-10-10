set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
cd "$TT_METAL_HOME" || { echo "no $TT_METAL_HOME"; exit 1; }
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
R=${R2:-$HOME/round2_repro}
H=ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_hwc
RUN=/home/user/fixhwc2_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
git diff -- $H > "$RUN/before.patch"
back () { git checkout -q -- $H && git apply "$RUN/before.patch" && echo "restored to the first patch"; }
python $R/patch_to_hwc2.py || { echo "PATCH FAILED"; back; exit 1; }
git diff --stat -- $H | tail -1
echo "=== rebuild start $(date -u +%H:%M)"; t0=$(date +%s)
ninja -C build > "$RUN/build.log" 2>&1; RC=$?; echo "=== rebuild rc=$RC $(( $(date +%s) - t0 ))s"
if [ $RC -ne 0 ]; then grep -E "error:|FAILED" "$RUN/build.log" | head -20 | cut -c1-300; back; echo "BUILD FAILED, nothing installed"; exit 1; fi
cmake --install build > "$RUN/install.log" 2>&1; echo "=== install rc=$?"
command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
timeout -k 20 600 python $R/repro_rowpitch_family.py to_hwc > "$RUN/to_hwc.log" 2>&1; echo "===== [to_hwc] rc=$?"
grep -E '^RESULT' "$RUN/to_hwc.log" | cut -c1-300
grep -E "error:|TT_FATAL|TT_THROW|Traceback" "$RUN/to_hwc.log" | tail -6 | cut -c1-300
command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
f=$(find tests -name "test_convert_to_hwc.py" | head -1)
timeout -k 20 1800 pytest -q -p no:cacheprovider "$f" > "$RUN/tt_test.log" 2>&1; echo "===== [tt_test_convert_to_hwc] rc=$? | $(grep -E '(passed|failed|error).* in [0-9.]+s' "$RUN/tt_test.log" | tail -1 | cut -c1-120)"
grep -E "^(FAILED|ERROR) " "$RUN/tt_test.log" | head -6 | cut -c1-200
echo "done $(date -u +%H:%M)  logs: $RUN"
