set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
cd "$TT_METAL_HOME" || { echo "no $TT_METAL_HOME: rebuild first"; exit 1; }
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
echo "tree: $(git log -1 --format='%h %cd' --date=short)"
RUN=/home/user/allnew_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
one () {  # label, command...
  command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
  timeout -k 20 600 "${@:2}" > "$RUN/$1.log" 2>&1; echo "===== [$1] rc=$?"
  grep -E '^(RESULT|INFO)' "$RUN/$1.log" | cut -c1-330
  grep -E "TT_FATAL|TT_THROW|Traceback" "$RUN/$1.log" | tail -2 | cut -c1-240
}
one slice_ring python ~/slice_ring.py
one index_fill python ~/index_fill_probe.py
one alias_sampling python ~/alias_sampling.py
one alias_manual_seed python ~/alias_manual_seed.py
for c in up_shard up_fp32 grid up_shape; do one $c python ~/pool_probe.py $c; done
echo "done $(date -u +%H:%M)  logs: $RUN"
