set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
export MESH_DEVICE=N150 TOKENIZERS_PARALLELISM=false
cd "$TT_METAL_HOME" || exit 1
exec 9>/home/user/census.lock
flock -n 9 || { echo "another census is already running; not starting"; exit 1; }
RUN=/home/user/perfcensus_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
REFDIR=models/tt_transformers/tests/reference_outputs
SUM="$RUN/summary.txt"
echo "tree: $(git log -1 --format='%h %cd' --date=short)   dirty (must be 0): $(git status --porcelain | wc -l | tr -d ' ')   artifacts: $RUN"
[ "$(git status --porcelain | wc -l | tr -d ' ')" = 0 ] || { echo "tree not clean, stopping"; exit 1; }
restore () { git checkout -q -- models/tt_transformers; git clean -qf -- "$REFDIR"; echo "[restored] dirty: $(git status --porcelain | wc -l | tr -d ' ')"; }
trap restore EXIT
# Copy of TT's refgen without its Qwen-only YaRN override (old "type" key, KeyError on transformers 5.x;
# the device path does not apply YaRN), so each model is scored against its own unmodified HF config.
mkdir -p "$RUN/refgen"; cp models/tt_transformers/tests/tale-of-two-cities.txt.bz2 "$RUN/refgen/"
python - "$RUN/refgen/generate_reference_hf.py" <<'PYEOF'
import sys
src = open("models/tt_transformers/tests/generate_reference_hf.py").read()
block = '    if "Qwen" in model_name:\n        config.rope_scaling = {"factor": 4.0, "original_max_position_embeddings": 32768, "type": "yarn"}\n'
assert src.count(block) == 1, "Qwen override block not found"
open(sys.argv[1], "w").write(src.replace(block, ""))
print("refgen copy written without the Qwen YaRN override")
PYEOF
[ -f "$RUN/refgen/generate_reference_hf.py" ] || exit 1
[ $# -gt 0 ] || set -- unsloth/Llama-3.2-1B-Instruct Qwen/Qwen2.5-1.5B-Instruct
echo "models ($#): $*"
for M in "$@"; do
  FREE=$(df -BG /home/user | tail -1 | awk '{print $4}' | tr -d G)
  [ "$FREE" -lt 40 ] && { echo "[$M] STOP: only ${FREE}G free" | tee -a "$SUM"; break; }
  B=$(basename "$M"); REF="$REFDIR/$B.refpt"; t0=$(date +%s)
  timeout 3600 python "$RUN/refgen/generate_reference_hf.py" --model "$M" --total_length 1024 --output_file "$REF" > "$RUN/ref_$B.log" 2>&1
  if [ ! -f "$REF" ]; then
    echo "[$B] REFGEN FAILED: $(grep -iE 'error|exception' "$RUN/ref_$B.log" | grep -v DEBUG | tail -1 | cut -c1-160)" | tee -a "$SUM"
  else
    LINE="[$B]"
    for MODE in accuracy performance; do
      export HF_MODEL=$M TT_CACHE_PATH=$RUN/tt_cache_$MODE
      rm -rf "$TT_CACHE_PATH"
      command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1
      timeout 3600 pytest models/tt_transformers/demo/simple_text_demo.py -k "$MODE and ci-token-matching" > "$RUN/demo_${B}_$MODE.log" 2>&1
      RC=$?
      ACC=$(grep -oE "Top1 Accuracy: [0-9.]+%, Top5 Accuracy: [0-9.]+%" "$RUN/demo_${B}_$MODE.log" | head -1 | sed -E 's/Top1 Accuracy: ([0-9.]+)%, Top5 Accuracy: ([0-9.]+)%/\1\/\2/')
      NOTE=""; [ -z "$ACC" ] && NOTE="(rc=$RC $(grep -E 'TT_FATAL|TT_THROW|Out of Memory|^E +[A-Za-z]+Error' "$RUN/demo_${B}_$MODE.log" | grep -v DEBUG | tail -1 | cut -c1-110))"
      SPD=$(grep -oE "Average speed: [0-9.]+ms @ [0-9.]+ tok/s/user" "$RUN/demo_${B}_$MODE.log" | tail -1 | grep -oE "[0-9.]+ tok/s/user")
      LINE="$LINE $MODE=${ACC:--} ${NOTE}"
      rm -rf "$TT_CACHE_PATH"
    done
    echo "$LINE  ($(( ($(date +%s) - t0) / 60 )) min)" | tee -a "$SUM"
  fi
  git checkout -q -- "$REF" 2>/dev/null || rm -f "$REF"
  rm -rf "$HOME/.cache/huggingface/hub/models--${M//\//--}"
done
echo "======== SUMMARY ($RUN) ========"; cat "$SUM"
