set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
cd "$TT_METAL_HOME" || exit 1
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
F=ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl
T=tests/ttnn/nightly/unit_tests/operations/moreh
RUN=/home/user/fix53927_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
git checkout -q -- "$F"
echo "tree: $(git log -1 --format='%h %cd' --date=short)   dirty: $(git status --porcelain | wc -l | tr -d ' ')   artifacts: $RUN"
reset () { command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3; }
wpytest () {  # label timeout-s pytest-args...
  L="$RUN/$1.log"; reset; t0=$(date +%s)
  TT_METAL_WATCHER=5 timeout -k 20 "$2" pytest -v -p no:cacheprovider "${@:3}" > "$L" 2>&1; RC=$?
  echo "[$1] rc=$RC $(( $(date +%s) - t0 ))s | $(grep -E '^=+ .*(passed|failed|error).* =+$' "$L" | tail -1 | cut -c1-120)"
  grep -E "tripped an assert" "$L" | head -1 | cut -c1-330
  [ "$RC" -ne 0 ] && echo "  last test reached: $(grep -E '::test_' "$L" | tail -1 | cut -c1-200)"
  grep -E "^(FAILED|ERROR) " "$L" | head -8 | cut -c1-200
}
one () {  # label case watcher-level
  L="$RUN/$1.log"; reset
  if [ "$3" -gt 0 ]; then TT_METAL_WATCHER=$3 timeout -k 20 600 python ~/probe53927.py "$2" > "$L" 2>&1
  else timeout -k 20 600 python ~/probe53927.py "$2" > "$L" 2>&1; fi
  echo "[$1] rc=$? | $(grep -E '^RESULT' "$L" | cut -c1-300)"
  grep -E "tripped an assert" "$L" | head -1 | cut -c1-330
}
echo "=== BEFORE (unpatched), watcher on"
wpytest before_softmax_large 900 "$T/test_moreh_softmax.py" -k "large_algorithm_for_dim_hw and not backward"
echo "=== PATCH"
python - "$F" <<'PY' || exit 1
import sys
p = sys.argv[1]; s = open(p).read()
old = "input_dfb_id, Ht * chunk_size, total_input_tiles)));"
new = "input_dfb_id, Ht * ((Wt < chunk_size) ? Wt : chunk_size), total_input_tiles)));"
if s.count(old) != 1:
    sys.exit(f"expected one match in {p}, found {s.count(old)}")
open(p, "w").write(s.replace(old, new))
PY
git diff -- "$F" | grep -E "^[+-] "
echo "=== AFTER (patched), watcher on"
one after_h_large_watcher h_large 5
one after_h_small_watcher h_small 5
one after_w_large_watcher w_large 5
wpytest after_softmax 2400 "$T/test_moreh_softmax.py"
wpytest after_softmin 1800 "$T/test_moreh_softmin.py"
wpytest after_logsoftmax 1800 "$T/test_moreh_logsoftmax.py"
echo "=== AFTER (patched), watcher off"
one after_h_large_nowatcher h_large 0
git checkout -q -- "$F"; echo "patch reverted; dirty: $(git status --porcelain | wc -l | tr -d ' ')"
echo "done $(date -u +%H:%M)"
