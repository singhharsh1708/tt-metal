# Find what hangs in the width-concat fix (tiny pieces), then try a plain copy loop in place of tt_memmove.
set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
cd "$TT_METAL_HOME" || exit 1
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
R=${R3:-$HOME/round3}
RUN=/home/user/concatfix2_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
K=ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/reader_concat_stick_layout_interleaved_start_id.cpp
tiny () {  # tag
  for c in "2 1 1" "2 1 0" "2 4 1" "2 8 1" "3 1 1" "47 1 0"; do
    command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 2
    timeout -k 10 90 python $R/concat_tiny.py $c > "$RUN/$1_${c// /_}.log" 2>&1; rc=$?
    echo "[$1] n,w,dim=$c rc=$rc $(grep -aE '^RESULT' "$RUN/$1_${c// /_}.log" | cut -c1-120) $(grep -aE 'TT_FATAL|TT_THROW|Error' "$RUN/$1_${c// /_}.log" | tail -1 | cut -c1-160)"
  done
}
grep -c "tt_memmove" $K
tiny memmove
cp $K "$RUN/kernel_before_v2.cpp"
python $R/patch_concat_v2.py || { echo "PATCH FAILED"; exit 1; }
ninja -C build > "$RUN/build.log" 2>&1 && cmake --install build > "$RUN/install.log" 2>&1; echo "=== build+install rc=$?"
tiny copyloop
pt () { command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3; timeout -k 20 "$2" pytest -q -p no:cacheprovider -p no:timeout "${@:3}" > "$RUN/$1.log" 2>&1; rc=$?; echo "===== [$1] rc=$rc | $(grep -aE '(passed|failed|error).* in [0-9.]+s' "$RUN/$1.log" | tail -1 | cut -c1-120)"; grep -aE "^(FAILED|ERROR) " "$RUN/$1.log" | head -5 | cut -c1-200; }
command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
timeout -k 20 900 python $R/repro_noc_align.py concat > "$RUN/probe.log" 2>&1; echo "===== [probe concat] rc=$?"; grep -aE '^RESULT' "$RUN/probe.log" | cut -c1-200
pt test_concat 2400 tests/ttnn/unit_tests/operations/data_movement/test_concat.py
pt test_concat_iterative 3000 tests/ttnn/unit_tests/operations/data_movement/test_concat_iterative.py
echo "kernel left with the copy loop (previous version saved at $RUN/kernel_before_v2.cpp).  done $(date -u +%H:%M)  logs: $RUN"
