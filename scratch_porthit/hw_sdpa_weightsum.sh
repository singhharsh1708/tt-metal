set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
cd "$TT_METAL_HOME" || { echo "no $TT_METAL_HOME: pod was wiped, rebuild first"; exit 1; }
[ -d build ] || { echo "no build dir: rebuild first"; exit 1; }
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
echo "tree: $(git log -1 --format='%h %cd' --date=short)   dirty: $(git status --porcelain | wc -l | tr -d ' ')"
command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
L=/home/user/sdpa_weightsum.log
timeout -k 20 1500 python ~/sdpa_weightsum.py > "$L" 2>&1; echo "rc=$?  results: $(grep -c '^RESULT' "$L")  WRONG: $(grep -c 'WRONG$' "$L")"
grep '^RESULT' "$L" | cut -c1-420
echo "done $(date -u +%H:%M)"
