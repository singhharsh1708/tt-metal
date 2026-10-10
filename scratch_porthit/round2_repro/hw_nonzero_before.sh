set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
cd "$TT_METAL_HOME" || { echo "no $TT_METAL_HOME"; exit 1; }
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
R=${R2:-$HOME/round2_repro}
N=ttnn/cpp/ttnn/operations/data_movement/non_zero_indices
RUN=/home/user/nonzero_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
git diff -- $N > "$RUN/fix.patch"
[ -s "$RUN/fix.patch" ] || { echo "nonzero is not patched in this tree, stopping"; exit 1; }
build () { ninja -C build > "$RUN/build_$1.log" 2>&1 && cmake --install build > "$RUN/install_$1.log" 2>&1; echo "=== build+install [$1] rc=$?"; }
probe () {
  command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
  timeout -k 20 600 python $R/repro_f_nonzero_bfloat8.py > "$RUN/$1.log" 2>&1; echo "===== [$1] rc=$?"
  grep -E '^RESULT' "$RUN/$1.log" | cut -c1-260
}
git checkout -q -- $N; build unpatched; probe UNPATCHED
git apply "$RUN/fix.patch" || { echo "COULD NOT RE-APPLY the fix, patch saved at $RUN/fix.patch"; exit 1; }
build patched; probe PATCHED
echo "fix restored, dirty: $(git status --porcelain | wc -l | tr -d ' ').  done $(date -u +%H:%M)  logs: $RUN"
