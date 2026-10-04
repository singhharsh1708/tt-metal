set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
cd "$TT_METAL_HOME" || exit 1
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
echo "tree: $(git log -1 --format='%h %cd' --date=short)"
one () {  # label case watcher-level
  command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
  L=/home/user/p53927_$1.log
  if [ "$3" -gt 0 ]; then TT_METAL_WATCHER=$3 timeout -k 20 600 python ~/probe53927.py "$2" > "$L" 2>&1
  else timeout -k 20 600 python ~/probe53927.py "$2" > "$L" 2>&1; fi
  echo "[$1] rc=$? | $(grep -E '^RESULT' "$L" | cut -c1-420)"
  grep -E "tripped an assert|Aborting wait|watcher error" "$L" | head -3 | cut -c1-400
}
one h_large_nowatcher h_large 0
one h_large_watcher h_large 5
one h_small_watcher h_small 5
one w_large_watcher w_large 5
echo "--- watcher log tail"; tail -15 generated/watcher/watcher.log 2>/dev/null | cut -c1-220
echo "done $(date -u +%H:%M)"
