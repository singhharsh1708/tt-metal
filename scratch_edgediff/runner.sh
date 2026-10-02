set -uo pipefail
exec 9>/home/user/census.lock
flock -n 9 || { echo "another device job is running; not starting"; exit 1; }
[ -d /home/user/ttm-main/build/lib ] || { echo "ttm-main is missing (pod was wiped): rebuild first"; exit 1; }
RUN=/home/user/edgediff_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
S_HOME=${TT_METAL_HOME:-/home/user/tt-metal}
echo "stable $S_HOME $(cd $S_HOME && git log -1 --format='%h %cd' --date=short)   main $(cd /home/user/ttm-main && git log -1 --format='%h %cd' --date=short)   artifacts $RUN"
sweep () {  # tag
  for i in $(seq 1 40); do
    tt-smi -r 0 >/dev/null 2>&1; touch "$RUN/$1.log"
    python ~/edge.py run "$RUN/$1.pt" >> "$RUN/$1.log" 2>&1 &
    pid=$!
    while kill -0 $pid 2>/dev/null; do
      sleep 15
      if [ $(( $(date +%s) - $(stat -c %Y "$RUN/$1.log") )) -gt 600 ]; then echo "[$1] watchdog: killing at: $(cat "$RUN/$1.pt.cur" 2>/dev/null)"; kill -9 $pid 2>/dev/null; fi
    done
    wait $pid; echo "[$1] pass $i rc=$? $(grep -cE '^[0-9]+/' "$RUN/$1.log") progress lines, last: $(grep -E '^[0-9]+/' "$RUN/$1.log" | tail -1 | cut -c1-80)"
    grep -q "SWEEP DONE" "$RUN/$1.log" && break
    grep -E "Error|Traceback" "$RUN/$1.log" | tail -2 | cut -c1-200
    [ -f "$RUN/$1.pt" ] || [ $i -lt 2 ] || { echo "no output file after 2 passes, stopping"; tail -5 "$RUN/$1.log" | cut -c1-220; return; }
  done
}
( cd "$S_HOME" && export TT_METAL_CACHE=/home/user/.cache/ttm-cache-stable && sweep stable )
( unset LD_LIBRARY_PATH CI; export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn TT_METAL_CACHE=/home/user/.cache/ttm-cache-main; cd /home/user/ttm-main && sweep main )
python ~/edge.py report "$RUN/stable.pt" "$RUN/main.pt" 2>&1 | grep -vE "Warning|warn" | cut -c1-420
echo "done: $RUN"
