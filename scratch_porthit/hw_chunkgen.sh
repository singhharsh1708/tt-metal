set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
export MESH_DEVICE=N150 TOKENIZERS_PARALLELISM=false
cd "$TT_METAL_HOME" || exit 1
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
RUN=/home/user/chunkgen_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"; SUM="$RUN/summary.txt"
echo "tree: $(git log -1 --format='%h %cd' --date=short)   dirty (must be 0): $(git status --porcelain | wc -l | tr -d ' ')   artifacts: $RUN"
[ "$(git status --porcelain | wc -l | tr -d ' ')" = 0 ] || { echo "tree not clean, stopping"; exit 1; }
T=models/tt_transformers/tests/test_chunked_generation.py
A=models/tt_transformers/tests/test_chunked_generation_all.py
trap 'rm -f "$A"; echo "[restored] dirty: $(git status --porcelain | wc -l | tr -d " ")"' EXIT
# Same test, but log every (last_token_idx, start_pos) case instead of stopping at the first failure.
python - "$T" "$A" <<'PYEOF'
import sys
s = open(sys.argv[1]).read()
old = "            assert passing\n"
assert s.count(old) == 1
s = s.replace(old, '            logger.info(f"CASE last_token_idx={last_token_idx} start_pos={start_pos} pass={passing} PCC: {pcc_message}")\n            fails.append((last_token_idx, start_pos)) if not passing else None\n')
s = s.replace('    logger.info("Running TT model")\n', '    logger.info("Running TT model")\n    fails = []\n')
s = s.rstrip("\n") + "\n    assert not fails, f\"failing cases: {fails}\"\n"
open(sys.argv[2], "w").write(s)
print("patched copy written")
PYEOF
[ $# -gt 0 ] || set -- unsloth/Llama-3.2-1B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen3-1.7B HuggingFaceTB/SmolLM2-1.7B-Instruct TinyLlama/TinyLlama-1.1B-Chat-v1.0 Qwen/Qwen2.5-3B-Instruct unsloth/Llama-3.2-3B-Instruct microsoft/Phi-3-mini-128k-instruct tiiuae/Falcon3-3B-Instruct mistralai/Mistral-7B-Instruct-v0.3
echo "models ($#): $*"
for M in "$@"; do
  FREE=$(df -BG /home/user | tail -1 | awk '{print $4}' | tr -d G)
  [ "$FREE" -lt 40 ] && { echo "[$M] STOP: only ${FREE}G free" | tee -a "$SUM"; break; }
  B=$(basename "$M"); L="$RUN/$B.log"; t0=$(date +%s)
  export HF_MODEL=$M TT_CACHE_PATH=$RUN/tt_cache; rm -rf "$TT_CACHE_PATH"
  command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1
  timeout 5400 pytest "$A" > "$L" 2>&1; RC=$?
  N=$(grep -c "CASE last_token_idx" "$L"); F=$(grep "CASE last_token_idx" "$L" | grep -c "pass=False")
  WORST=$(grep -oE "CASE last_token_idx=[0-9]+ start_pos=[0-9]+ pass=(True|False) PCC: [0-9.e-]+" "$L" | awk '{print $NF, $2, $3}' | sort -g | head -3 | tr '\n' ';')
  NOTE=""; [ "$N" = 0 ] && NOTE="(rc=$RC $(grep -E 'TT_FATAL|TT_THROW|Out of Memory|^E +[A-Za-z]+Error' "$L" | grep -v DEBUG | tail -1 | cut -c1-120))"
  echo "[$B] cases=$N fail=$F worst: $WORST rc=$RC $NOTE ($(( ($(date +%s) - t0) / 60 )) min)" | tee -a "$SUM"
  rm -rf "$TT_CACHE_PATH" "$HOME/.cache/huggingface/hub/models--${M//\//--}"
done
echo "======== SUMMARY ($RUN) ========"; cat "$SUM"
