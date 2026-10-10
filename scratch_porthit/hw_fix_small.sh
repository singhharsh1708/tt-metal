set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
cd "$TT_METAL_HOME" || { echo "no $TT_METAL_HOME"; exit 1; }
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
FILES="ttnn/cpp/ttnn/operations/index_fill/index_fill.cpp ttnn/cpp/ttnn/operations/data_movement/gather/gather.cpp ttnn/cpp/ttnn/operations/embedding_backward/embedding_backward.cpp"
RUN=/home/user/fixsmall_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
git checkout -q -- $FILES
python ~/patch_small.py "$TT_METAL_HOME" || { echo "PATCH FAILED, nothing built"; git checkout -q -- $FILES; exit 1; }
git diff --stat -- $FILES | tail -4
echo "=== rebuild start $(date -u +%H:%M)"; t0=$(date +%s)
ninja -C build > "$RUN/build.log" 2>&1; RC=$?; echo "=== rebuild rc=$RC $(( $(date +%s) - t0 ))s"
if [ $RC -ne 0 ]; then grep -E "error:|FAILED" "$RUN/build.log" | head -12 | cut -c1-260; git checkout -q -- $FILES; echo "BUILD FAILED; small patch reverted, nothing installed"; exit 1; fi
cmake --install build > "$RUN/install.log" 2>&1; echo "=== install rc=$?"
one () { command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3; timeout -k 20 900 "${@:2}" > "$RUN/$1.log" 2>&1; echo "===== [$1] rc=$?"; grep -E "^(RESULT|GAP|BOTH) " "$RUN/$1.log" | grep -v -E "^BOTH .* n=(3|100|1000) " | cut -c1-260; grep -c "^OK " "$RUN/$1.log" | sed "s/^/   OK lines: /"; }
one index_fill_patched python ~/index_fill_probe.py
one gather_patched python ~/cand_probe.py gather
one embbw_patched python ~/cand_probe.py embbw
one rank1_gather_all_sizes python ~/rank1_probe.py
pt () { command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3; timeout -k 20 2400 pytest -q -p no:cacheprovider "${@:2}" > "$RUN/$1.log" 2>&1; echo "===== [$1] rc=$? | $(grep -E '(passed|failed|error).* in [0-9.]+s' "$RUN/$1.log" | tail -1 | cut -c1-120)"; grep -E "^(FAILED|ERROR) " "$RUN/$1.log" | head -6 | cut -c1-200; }
pt tt_index_fill tests/ttnn/nightly/unit_tests/operations/data_movement/test_index_fill.py
pt tt_gather_unit tests/ttnn/unit_tests/operations/data_movement/test_gather.py
pt tt_embedding_bw tests/ttnn/unit_tests/operations/data_movement/test_backward_embedding.py
echo "source LEFT PATCHED, dirty: $(git status --porcelain | wc -l | tr -d ' ').  done $(date -u +%H:%M)  logs: $RUN"
