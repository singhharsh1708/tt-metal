set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
cd "$TT_METAL_HOME" || { echo "no $TT_METAL_HOME: run hw_build_main_pinned.sh first"; exit 1; }
[ -d build ] || { echo "no build in $TT_METAL_HOME: run hw_build_main_pinned.sh first"; exit 1; }
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
RUN=/home/user/probe58082_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
export PROBE_OUT=$RUN PROBE_PY=/home/user/probe58082.py PROBE_REPS=${PROBE_REPS:-4}
LOOPS=${LOOPS:-200}; MAXHITS=${MAXHITS:-5}
T=tests/ttnn/unit_tests/operations/fused/test_softmax_probe58082.py
echo "tree: $(git log -1 --format='%h %cd' --date=short)   dirty: $(git status --porcelain | wc -l | tr -d ' ')   artifacts: $RUN"
echo "kmd: $(cat /sys/module/tenstorrent/version 2>/dev/null || echo unknown)   loops: $LOOPS   reps per loop: $PROBE_REPS"
python "$PROBE_PY" make || exit 1
command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1
START=$(date +%s); OTHER=0
for i in $(seq 1 "$LOOPS"); do
  [ -e /home/user/STOP58082 ] && { echo "stop file found"; break; }
  L="$RUN/loop_$i.log"; t0=$(date +%s)
  PROBE_LOOP=$i timeout -k 20 1200 pytest -q -m merge_gate "$T" -p no:cacheprovider > "$L" 2>&1; RC=$?
  HITS=$( [ -f "$RUN/hits.jsonl" ] && wc -l < "$RUN/hits.jsonl" || echo 0 ); HITS=${HITS// /}
  NOW=$(date +%s)
  echo "loop $i rc=$RC $((NOW - t0))s hits=$HITS eta=$(( (NOW - START) / i * (LOOPS - i) / 60 ))min | $(tail -1 "$L" | cut -c1-120)"
  if [ "$RC" -eq 0 ]; then rm -f "$L"; OTHER=0
  elif ! grep -q "corrupted softmax run" "$L"; then
    OTHER=$((OTHER + 1)); echo "  failure without a probe hit, log kept: $L"; grep -E "^(FAILED|ERROR)|TT_FATAL|TT_THROW" "$L" | head -5 | cut -c1-200
    command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1
    [ "$OTHER" -ge 3 ] && { echo "three such failures in a row, stopping"; break; }
  fi
  [ "$HITS" -ge "$MAXHITS" ] && { echo "$HITS hits collected, stopping"; break; }
done
rm -f "$T"
echo "======== REPORT ($RUN) ========"
python "$PROBE_PY" report "$RUN"
echo "done $(date -u +%H:%M)"
