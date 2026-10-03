set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
cd "$TT_METAL_HOME" || exit 1
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
echo "tree: $(git log -1 --format='%h %cd' --date=short)   dirty: $(git status --porcelain | wc -l | tr -d ' ')"
command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1
timeout 3000 python ~/sdpa_race.py 2>&1 | grep -v -E "\| (DEBUG|TRACE) "
echo "rc=${PIPESTATUS[0]} done $(date -u +%H:%M)"
