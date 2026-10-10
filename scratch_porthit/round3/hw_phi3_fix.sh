# Phi-3-mini-128k on the unpatched latest-main tree: host weight check, then the one-change fix with a fresh weight cache.
set -uo pipefail
unset LD_LIBRARY_PATH CI
D=/home/user/ttm-latest
R=${R3:-$HOME/round3}
cd "$D" || { echo "no $D"; exit 1; }
export TT_METAL_HOME=$D TT_METAL_RUNTIME_ROOT=$D PYTHONPATH=$D:$D/ttnn HF_HUB_OFFLINE=0
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
echo "tree: $(git log -1 --format='%h %cd' --date=short)  dirty files: $(git status --porcelain | grep -vc '^??')"
RUN=/home/user/phi3fix_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
export HF_MODEL=microsoft/Phi-3-mini-128k-instruct MESH_DEVICE=N150
echo "existing cache dirs:"; ls -d $D/model_cache/microsoft/Phi-3-mini-128k-instruct/* 2>/dev/null
python $R/phi3_weight_check.py > "$RUN/weight_check.log" 2>&1; echo "===== [host_weight_check] rc=$?"
grep -aE "^(VERSIONS|LOAD|WEIGHT)" "$RUN/weight_check.log" | cut -c1-240
grep -aE "Error" "$RUN/weight_check.log" | tail -3 | cut -c1-240
pt () {
  command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
  timeout -k 30 "$2" pytest -q -p no:cacheprovider "${@:3}" > "$RUN/$1.log" 2>&1; rc=$?
  echo "===== [$1] rc=$rc | $(grep -aE '(passed|failed|error|skipped|deselected).* in [0-9.]+s' "$RUN/$1.log" | tail -1 | cut -c1-140)"
  grep -aE "^(FAILED|ERROR) |^E +[A-Za-z]*Error|TT_THROW:|TT_FATAL:|Top1 Accuracy|Accuracy Check|Warm ttnn weight cache|skipping HF" "$RUN/$1.log" | grep -av "riscv-tt-elf" | awk '!s[$0]++' | tail -8 | cut -c1-230
  grep -aE "\[User 0\]" "$RUN/$1.log" | awk 'NR<=1; {l=$0} END{print l}' | cut -c1-260
}
TTT=models/tt_transformers
# BEFORE: stock main, brand-new empty weight cache (what a fresh install does)
export TT_CACHE_PATH=/home/user/phi3_cache_stock_$(date +%s); mkdir -p "$TT_CACHE_PATH"
pt before_perf_tokmatch 1800 --timeout=1740 $TTT/demo/simple_text_demo.py -k "performance and ci-token-matching"
# AFTER: load Phi-3-mini-128k with the library implementation, brand-new empty weight cache
python $R/patch_phi3_trc.py || { echo "PATCH FAILED"; exit 1; }
git diff --stat | tail -1
export TT_CACHE_PATH=/home/user/phi3_cache_fixed_$(date +%s); mkdir -p "$TT_CACHE_PATH"
pt after_perf_tokmatch 1800 --timeout=1740 $TTT/demo/simple_text_demo.py -k "performance and ci-token-matching"
pt after_acc_tokmatch 1800 --timeout=1740 $TTT/demo/simple_text_demo.py -k "accuracy and ci-token-matching"
pt after_ci1 900 --timeout=840 $TTT/demo/simple_text_demo.py -k "performance and ci-1"
pt after_long16k 1800 --timeout=1740 $TTT/demo/simple_text_demo.py -k "performance and long-context-16k and not ci-long"
git checkout -q -- models/tt_transformers/tt/model_config.py
echo "tree restored, dirty files: $(git status --porcelain | grep -vc '^??').  done $(date -u +%H:%M)  logs: $RUN"
