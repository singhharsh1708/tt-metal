set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
cd "$TT_METAL_HOME" || exit 1
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
RUN=/home/user/sdpafix_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
K=ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp
T=tests/ttnn/unit_tests/operations/sdpa/test_sdpa_prefill.py
echo "tree: $(git log -1 --format='%h %cd' --date=short)   dirty (must be 0): $(git status --porcelain | wc -l | tr -d ' ')   artifacts: $RUN"
[ "$(git status --porcelain | wc -l | tr -d ' ')" = 0 ] || { echo "tree not clean, stopping"; exit 1; }
restore () { git checkout -q HEAD -- "$K"; rm -rf /home/user/.cache/tt-metal-cache; echo "[restored] dirty: $(git status --porcelain | wc -l | tr -d ' ')"; }
trap restore EXIT
for V in stock fixed; do
  if [ $V = fixed ]; then
    python - "$K" <<'PYEOF' || exit 1
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
old = "                        ((int32_t)k_tile_start >= min_window_start) &&\n"
assert s.count(old) == 1, s.count(old)
p.write_text(s.replace(old, "                        ((int32_t)k_tile_start >= max_window_start) &&\n"))
print("patched: fully-contained check uses the last row's window start")
PYEOF
    git diff --stat
  fi
  rm -rf /home/user/.cache/tt-metal-cache; command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1
  timeout 3000 python ~/sdpa_fix.py $V 2>&1 | grep "^\[$V\]" | tee "$RUN/sweep_$V.txt"
  timeout 3000 pytest "$T" -k "sliding_window" -q -p no:cacheprovider -rfE > "$RUN/tests_$V.log" 2>&1
  echo "[$V] TT test_sdpa_prefill -k sliding_window: $(tail -1 "$RUN/tests_$V.log")"
done
echo "done $(date -u +%H:%M)"
