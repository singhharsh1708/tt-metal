set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
cd "$TT_METAL_HOME" || { echo "no $TT_METAL_HOME: rebuild first"; exit 1; }
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
echo "tree: $(git log -1 --format='%h %cd' --date=short)"
command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
timeout -k 20 900 python ~/slice_ring.py > ~/slice_ring.log 2>&1; echo "rc=$?"; grep '^RESULT' ~/slice_ring.log | cut -c1-300
grep -E "TT_FATAL|TT_THROW|Traceback" ~/slice_ring.log | tail -3 | cut -c1-240
echo "done $(date -u +%H:%M)"
