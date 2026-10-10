set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
cd "$TT_METAL_HOME" || { echo "no $TT_METAL_HOME: rebuild first"; exit 1; }
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
echo "tree: $(git log -1 --format='%h %cd' --date=short)"
for S in slice_ring index_fill_probe alias_sampling alias_manual_seed; do
  command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
  timeout -k 20 600 python ~/$S.py > ~/new4_$S.log 2>&1; echo "===== [$S] rc=$?"
  grep -E '^(RESULT|INFO)' ~/new4_$S.log | cut -c1-320
  grep -E "TT_FATAL|TT_THROW|Traceback" ~/new4_$S.log | tail -2 | cut -c1-240
done
echo "done $(date -u +%H:%M)"
