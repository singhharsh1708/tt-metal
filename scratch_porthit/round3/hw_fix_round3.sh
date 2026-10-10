set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
cd "$TT_METAL_HOME" || { echo "no $TT_METAL_HOME"; exit 1; }
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
R=${R3:-$HOME/round3}
O=ttnn/cpp/ttnn/operations
DIRS="$O/data_movement/reshape_on_device $O/embedding $O/data_movement/concat"
RUN=/home/user/fixround3_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
git checkout -q -- $DIRS
python $R/patch_noc_align.py || { echo "PATCH FAILED, nothing built"; git checkout -q -- $DIRS; exit 1; }
git diff --stat -- $DIRS | tail -1
echo "=== rebuild start $(date -u +%H:%M)"; t0=$(date +%s)
ninja -C build > "$RUN/build.log" 2>&1; RC=$?; echo "=== rebuild rc=$RC $(( $(date +%s) - t0 ))s"
if [ $RC -ne 0 ]; then grep -E "error:|FAILED" "$RUN/build.log" | head -20 | cut -c1-300; echo "BUILD FAILED; source LEFT PATCHED, nothing installed. log: $RUN/build.log"; exit 1; fi
cmake --install build > "$RUN/install.log" 2>&1; echo "=== install rc=$?"
for c in reshape embedding concat; do
  command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
  timeout -k 20 900 python $R/repro_noc_align.py $c > "$RUN/$c.log" 2>&1; echo "===== [$c] rc=$?"
  grep -E '^RESULT' "$RUN/$c.log" | cut -c1-300
  grep -E "error:|TT_FATAL|TT_THROW|Traceback" "$RUN/$c.log" | tail -4 | cut -c1-300
done
pt () { command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3; timeout -k 20 2400 pytest -q -p no:cacheprovider "${@:2}" > "$RUN/$1.log" 2>&1; echo "===== [$1] rc=$? | $(grep -E '(passed|failed|error).* in [0-9.]+s' "$RUN/$1.log" | tail -1 | cut -c1-120)"; grep -E "^(FAILED|ERROR) " "$RUN/$1.log" | head -6 | cut -c1-200; }
i=0
for f in tests/ttnn/unit_tests/base_functionality/test_reshape.py tests/ttnn/unit_tests/operations/conv/data_movement/test_fold_op.py tests/tt_eager/python_api_testing/unit_testing/misc/test_reshape.py \
  tests/ttnn/unit_tests/operations/data_movement/test_embedding.py tests/tt_eager/python_api_testing/unit_testing/misc/test_embedding.py tests/ttnn/docs_examples/test_embedding_examples.py \
  tests/ttnn/unit_tests/operations/data_movement/test_concat.py tests/ttnn/unit_tests/operations/data_movement/test_concat_program_cache.py tests/ttnn/unit_tests/operations/data_movement/test_concat_cache_aliasing.py tests/ttnn/unit_tests/operations/data_movement/test_concat_iterative.py tests/ttnn/unit_tests/base_functionality/test_concat_issue.py tests/tt_eager/python_api_testing/unit_testing/misc/test_concat.py tests/ttnn/nightly/unit_tests/operations/data_movement/test_concat.py tests/ttnn/nightly/unit_tests/operations/data_movement/test_concat_memory_configs_and_layouts.py; do
  i=$((i+1)); if [ -f "$f" ]; then pt "tt${i}_$(basename $f .py)" "$f"; else echo "===== [$f] no such file"; fi
done
echo "source LEFT PATCHED, dirty: $(git status --porcelain | wc -l | tr -d ' ').  done $(date -u +%H:%M)  logs: $RUN"
