set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
cd "$TT_METAL_HOME" || { echo "no $TT_METAL_HOME: rebuild first"; exit 1; }
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
echo "tree: $(git log -1 --format='%h %cd' --date=short)   dirty: $(git status --porcelain | wc -l | tr -d ' ')"
RUN=/home/user/cand_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
for C in slice bf8 embbw gather sdpamask; do
  command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
  L="$RUN/$C.log"; t0=$(date +%s)
  timeout -k 20 900 python ~/cand_probe.py "$C" > "$L" 2>&1; RC=$?
  echo "===== [$C] rc=$RC $(( $(date +%s) - t0 ))s finished=$(grep -c '^CASE .* finished' "$L")"
  grep '^RESULT' "$L" | cut -c1-520
  [ "$RC" -ne 0 ] && grep -E "TT_FATAL|TT_THROW|Traceback|Error" "$L" | tail -3 | cut -c1-240
done
echo "done $(date -u +%H:%M)  logs: $RUN"
