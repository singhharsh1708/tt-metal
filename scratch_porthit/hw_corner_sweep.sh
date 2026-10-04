set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
cd "$TT_METAL_HOME" || exit 1
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
RUN=/home/user/corner_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
F=ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl
git checkout -q -- "$F"
echo "tree: $(git log -1 --format='%h %cd' --date=short)   dirty: $(git status --porcelain | wc -l | tr -d ' ')   artifacts: $RUN"
reset () { command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3; }

echo "=== gtest for #53927 (NormalizationSmoke.DISABLED_SoftmaxGeneralHLarge, watcher on)"
BIN=$(find build -type f -name unit_tests_ttnn -perm -u+x 2>/dev/null | head -1)
if [ -n "$BIN" ]; then
  gt () {
    reset; TT_METAL_WATCHER=5 timeout -k 20 600 "$BIN" --gtest_also_run_disabled_tests --gtest_filter='NormalizationSmoke.*SoftmaxGeneralHLarge' > "$RUN/$1.log" 2>&1
    echo "[$1] rc=$? | $(grep -E '^\[  (PASSED|FAILED) +\]|^\[ +(OK|FAILED) +\]' "$RUN/$1.log" | tail -2 | tr '\n' ' ' | cut -c1-200)"
    grep -E "tripped an assert" "$RUN/$1.log" | head -1 | cut -c1-260
  }
  gt gtest_unpatched
  python - "$F" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = "input_dfb_id, Ht * chunk_size, total_input_tiles)));"
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, "input_dfb_id, Ht * ((Wt < chunk_size) ? Wt : chunk_size), total_input_tiles)));"))
PY
  gt gtest_patched
  git checkout -q -- "$F"; echo "patch reverted; dirty: $(git status --porcelain | wc -l | tr -d ' ')"
else
  echo "no unit_tests_ttnn binary under build/, skipped"
fi

echo "=== padding invariance sweep"
for G in sm_bf16 sm_fp32 moreh norm reduce moreh_norm; do
  L="$RUN/$G.log"; reset; t0=$(date +%s)
  timeout -k 20 2400 python ~/corner_sweep.py "$G" > "$L" 2>&1; RC=$?
  c () { grep -c "^$1 " "$L"; }
  echo "[$G] rc=$RC $(( $(date +%s) - t0 ))s | cases $(c CASE) OK $(c OK) LEAK $(c LEAK) REF $(c REF) EXC $(c EXC) | $(grep -c '^GROUP .* finished' "$L") finished"
  [ "$(grep -c '^GROUP .* finished' "$L")" -eq 0 ] && echo "  stopped at: $(grep '^CASE ' "$L" | tail -1 | cut -c1-160)" && grep -E "TT_FATAL|TT_THROW|Aborted|Traceback" "$L" | tail -2 | cut -c1-220
done
echo "======== LEAK lines"; grep -h "^LEAK " "$RUN"/*.log | head -80 | cut -c1-330
echo "======== REF lines (informational)"; grep -h "^REF " "$RUN"/*.log | head -30 | cut -c1-200
echo "======== EXC lines (distinct messages)"; grep -h "^EXC " "$RUN"/*.log | sed -E 's/^EXC ([^|]*)\| /\1@@/' | awk -F'@@' '{ if (!seen[$2]++) print "EXC " $1 "| " $2 }' | head -25 | cut -c1-300
echo "done $(date -u +%H:%M)"
