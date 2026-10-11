# Whose hang is it? test_concat_many_inputs[47] and test_concat_iterative on: patched tree, concat stock, adapter stock, clean main.
set -uo pipefail
unset LD_LIBRARY_PATH CI
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
RUN=/home/user/concathang_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
T=tests/ttnn/unit_tests/operations/data_movement/test_concat.py
TI=tests/ttnn/unit_tests/operations/data_movement/test_concat_iterative.py
use () { cd "$1" || exit 1; export TT_METAL_HOME=$1 TT_METAL_RUNTIME_ROOT=$1 PYTHONPATH=$1:$1/ttnn; }
pt () {  # label, timeout, args
  command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
  timeout -k 20 "$2" pytest -v -p no:cacheprovider -p no:timeout "${@:3}" > "$RUN/$1.log" 2>&1; rc=$?
  echo "===== [$1] rc=$rc | $(grep -aE '(passed|failed|error|deselected).* in [0-9.]+s' "$RUN/$1.log" | tail -1 | cut -c1-120)"
  grep -aE "^(FAILED|ERROR) " "$RUN/$1.log" | head -5 | cut -c1-200
  [ $rc -eq 124 ] && echo "   passed before the hang: $(grep -ac PASSED "$RUN/$1.log"); last test started: $(grep -aoE "test_concat[a-z_]*\.py::[^ ]+" "$RUN/$1.log" | tail -1)"
}
two () { pt "$1_many47" 240 "$T::test_concat_many_inputs" -k "47"; pt "$1_iter_small1" 600 "$TI" -k "small_tensors_1"; }
rebuild () { ninja -C build > "$RUN/build_$1.log" 2>&1 && cmake --install build > "$RUN/inst_$1.log" 2>&1; echo "=== build+install [$1] rc=$?"; }
C=ttnn/cpp/ttnn/operations/data_movement/concat
A=$(cd /home/user/ttm-main && git status --porcelain | grep -oE "[^ ]*mesh_device_operation_adapter.hpp" | head -1)
use /home/user/ttm-main
echo "patched tree $(git log -1 --format=%h), dirty $(git status --porcelain | grep -vc '^??'), adapter file: ${A:-not dirty}"
two A_patched
git diff -- $C > "$RUN/concat_fix.patch"; [ -s "$RUN/concat_fix.patch" ] || { echo "concat not patched here, stopping"; exit 1; }
git checkout -q -- $C; rebuild concat_stock
two B_concat_stock
if [ -n "$A" ]; then
  git diff -- "$A" > "$RUN/adapter_fix.patch"; git checkout -q -- "$A"; rebuild adapter_stock
  two C_concat_and_adapter_stock
  git apply "$RUN/adapter_fix.patch" || { echo "COULD NOT RE-APPLY adapter fix: $RUN/adapter_fix.patch"; exit 1; }
fi
git apply "$RUN/concat_fix.patch" || { echo "COULD NOT RE-APPLY concat fix: $RUN/concat_fix.patch"; exit 1; }
rebuild all_fixes_restored
pt D_patched_rest_of_test_concat 2400 "$T" -k "not many_inputs"
if [ -d /home/user/ttm-latest/build/lib ]; then use /home/user/ttm-latest; echo "clean main $(git log -1 --format=%h), dirty $(git status --porcelain | grep -vc '^??')"; two E_clean_main; fi
echo "done $(date -u +%H:%M)  logs: $RUN"
