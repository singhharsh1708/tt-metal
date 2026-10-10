set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
cd "$TT_METAL_HOME" || { echo "no $TT_METAL_HOME"; exit 1; }
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
R=${R3:-$HOME/round3}
RUN=/home/user/round3_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
for c in reshape embedding concat; do
  command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
  timeout -k 20 900 python $R/repro_noc_align.py $c > "$RUN/$c.log" 2>&1; echo "===== [$c] rc=$?"
  grep -E '^(RESULT|MODEL)' "$RUN/$c.log" | cut -c1-300
  grep -E "TT_FATAL|TT_THROW|Traceback" "$RUN/$c.log" | tail -2 | cut -c1-240
done
echo "done $(date -u +%H:%M)  logs: $RUN"
