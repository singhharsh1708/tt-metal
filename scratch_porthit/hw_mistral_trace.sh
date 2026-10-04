set -uo pipefail
STABLE_LD=${LD_LIBRARY_PATH:-}; STABLE_HOME=/home/user/tt-metal
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
export MESH_DEVICE=N150 TOKENIZERS_PARALLELISM=false
cd "$TT_METAL_HOME" || exit 1
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
RUN=/home/user/mistral_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"; SUM="$RUN/summary.txt"
echo "main: $(git log -1 --format='%h %cd' --date=short)  stable: $(git -C $STABLE_HOME log -1 --format='%h %cd' --date=short 2>/dev/null)  artifacts: $RUN"
one () {  # label model extra-pytest-args timeout-s
  L="$RUN/$1.log"; command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 5
  t0=$(date +%s); HF_MODEL=$2 timeout -k 20 $4 pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1" $3 > "$L" 2>&1; RC=$?
  pkill -9 -f "simple_text_demo.py" 2>/dev/null
  SPD=$(grep -oE "Average speed: [0-9.]+ms @ [0-9.]+ tok/s/user" "$L" | tail -1)
  LAST=$(grep -E "\| (INFO|info) " "$L" | tail -1 | cut -c1-170)
  ERR=$(grep -E "TT_FATAL|TT_THROW|Out of Memory|^E +[A-Za-z]+Error" "$L" | tail -1 | cut -c1-150)
  echo "[$1] rc=$RC $(( $(date +%s) - t0 ))s | ${SPD:-no speed line} | err: ${ERR:-none} | last log: $LAST" | tee -a "$SUM"
}
M=mistralai/Mistral-7B-Instruct-v0.3
export TT_CACHE_PATH=$RUN/tt_cache_main
one mistral_untraced_1 $M "--disable_trace" 2700          # also builds the weight cache
one mistral_traced_1 $M "" 1500
one mistral_traced_2 $M "" 1500
export TT_METAL_WATCHER=10
one mistral_traced_watcher $M "" 1500
echo "--- watcher tail:"; tail -30 generated/watcher/watcher.log 2>/dev/null | cut -c1-200 | tee -a "$SUM"
unset TT_METAL_WATCHER
one llama8b_traced unsloth/Meta-Llama-3.1-8B-Instruct "" 2700
rm -rf "$RUN/tt_cache_main"
echo "--- same traced demo on the image's stable tree"
( export LD_LIBRARY_PATH=$STABLE_LD TT_METAL_HOME=$STABLE_HOME TT_METAL_RUNTIME_ROOT=$STABLE_HOME PYTHONPATH=$STABLE_HOME:$STABLE_HOME/ttnn TT_CACHE_PATH=$RUN/tt_cache_stable; cd $STABLE_HOME || exit 1
  L="$RUN/stable_mistral_traced.log"; tt-smi -r 0 >/dev/null 2>&1; sleep 5; t0=$(date +%s)
  HF_MODEL=$M timeout -k 20 3000 pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1" > "$L" 2>&1; RC=$?
  pkill -9 -f "simple_text_demo.py" 2>/dev/null
  echo "[stable_mistral_traced] rc=$RC $(( $(date +%s) - t0 ))s | $(grep -oE 'Average speed: [0-9.]+ms @ [0-9.]+ tok/s/user' "$L" | tail -1) | err: $(grep -E 'TT_FATAL|TT_THROW|Out of Memory|^E +[A-Za-z]+Error' "$L" | tail -1 | cut -c1-150) | last log: $(grep -E '\| (INFO|info) ' "$L" | tail -1 | cut -c1-170)" | tee -a "$SUM" )
rm -rf "$RUN/tt_cache_stable"
echo "======== SUMMARY ($RUN) ========"; cat "$SUM"
