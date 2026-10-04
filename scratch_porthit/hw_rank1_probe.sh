set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
cd "$TT_METAL_HOME" || exit 1
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
RUN=/home/user/rank1_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"; L="$RUN/probe.log"
echo "tree: $(git log -1 --format='%h %cd' --date=short)   dirty: $(git status --porcelain | wc -l | tr -d ' ')   log: $L"
command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
timeout -k 20 1500 python ~/rank1_probe.py > "$L" 2>&1; RC=$?
c () { grep -c "^$1 " "$L"; }
echo "rc=$RC | cases $(c CASE) OK $(c OK) GAP $(c GAP) BOTH $(c BOTH) | $(grep -c '^PROBE finished' "$L") finished"
[ "$(grep -c '^PROBE finished' "$L")" -eq 0 ] && echo "stopped at: $(grep '^CASE ' "$L" | tail -1)" && grep -E "TT_FATAL|TT_THROW|Aborted|Traceback" "$L" | tail -3 | cut -c1-220
echo "======== GAP lines (rank 2 works, rank 1 does not)"; grep "^GAP " "$L" | cut -c1-420
echo "======== BOTH lines, n=33 only (call wrong or op has no such form)"; grep "^BOTH .* n=33 " "$L" | cut -c1-330
echo "done $(date -u +%H:%M)"
