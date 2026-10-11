# Is the test_concat_many_inputs hang ours? Run it on the patched tree with and without the concat edits.
set -uo pipefail
unset LD_LIBRARY_PATH CI
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
RUN=/home/user/concathang_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
T=tests/ttnn/unit_tests/operations/data_movement/test_concat.py
use () { cd "$1" || exit 1; export TT_METAL_HOME=$1 TT_METAL_RUNTIME_ROOT=$1 PYTHONPATH=$1:$1/ttnn; }
pt () {  # label, timeout, args
  command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
  timeout -k 20 "$2" pytest -q -p no:cacheprovider -p no:timeout "${@:3}" > "$RUN/$1.log" 2>&1; rc=$?
  echo "===== [$1] rc=$rc | $(grep -aE '(passed|failed|error|deselected).* in [0-9.]+s' "$RUN/$1.log" | tail -1 | cut -c1-120)"
  grep -aE "^(FAILED|ERROR) " "$RUN/$1.log" | head -5 | cut -c1-200
}
C=ttnn/cpp/ttnn/operations/data_movement/concat
use /home/user/ttm-main
echo "patched tree $(git log -1 --format=%h), dirty $(git status --porcelain | grep -vc '^??')"
pt A_patched_many47 240 "$T::test_concat_many_inputs" -k "47"
git diff -- $C > "$RUN/concat_fix.patch"; [ -s "$RUN/concat_fix.patch" ] || { echo "concat not patched here, stopping"; exit 1; }
git checkout -q -- $C
ninja -C build > "$RUN/build_stock.log" 2>&1 && cmake --install build > "$RUN/inst_stock.log" 2>&1; echo "=== build+install [concat stock] rc=$?"
pt B_concat_stock_many47 240 "$T::test_concat_many_inputs" -k "47"
git apply "$RUN/concat_fix.patch" || { echo "COULD NOT RE-APPLY concat fix, saved at $RUN/concat_fix.patch"; exit 1; }
ninja -C build > "$RUN/build_fix.log" 2>&1 && cmake --install build > "$RUN/inst_fix.log" 2>&1; echo "=== build+install [concat fix restored] rc=$?"
pt C_patched_rest_of_file 2400 "$T" -k "not many_inputs"
pt D_patched_many48_100 400 "$T::test_concat_many_inputs" -k "48 or 100"
if [ -d /home/user/ttm-latest/build/lib ]; then use /home/user/ttm-latest; pt E_clean_main_many47 240 "$T::test_concat_many_inputs" -k "47"; fi
echo "done $(date -u +%H:%M)  logs: $RUN"
