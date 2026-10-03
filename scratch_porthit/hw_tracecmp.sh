set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
export MESH_DEVICE=N150 TOKENIZERS_PARALLELISM=false
cd "$TT_METAL_HOME" || exit 1
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
RUN=/home/user/tracecmp_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"; SUM="$RUN/summary.txt"
echo "tree: $(git log -1 --format='%h %cd' --date=short)   dirty: $(git status --porcelain | wc -l | tr -d ' ')   artifacts: $RUN"
cat > "$RUN/cmp.py" <<'PYEOF'
import re, sys
def outs(path):
    s = open(path, errors="ignore").read()
    res = {}
    for m in re.finditer(r"==REPEAT BATCH (\d+)\n==USER (\d+) - PROMPT\n.*?\n==USER \d+ - OUTPUT\n(.*?)(?=\n\d{4}-\d\d-\d\d |\n==REPEAT BATCH|\Z)", s, re.S):
        res[(int(m.group(1)), int(m.group(2)))] = m.group(3).strip()
    return res
a, b = outs(sys.argv[1]), outs(sys.argv[2])
if not a or not b:
    print(f"users traced={len(a)} untraced={len(b)} (no outputs parsed)"); sys.exit()
diff = []
for k in sorted(set(a) & set(b)):
    x, y = a[k], b[k]
    if x != y:
        i = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
        diff.append((i, k, x[max(0, i - 30):i + 40].replace("\n", " "), y[max(0, i - 30):i + 40].replace("\n", " ")))
print(f"users={len(set(a) & set(b))} differ={len(diff)}" + ("" if not diff else f" earliest char {min(d[0] for d in diff)}"))
for i, k, x, y in sorted(diff)[:3]:
    print(f"   user {k[1]} @char {i}\n     traced  : {x}\n     untraced: {y}")
PYEOF
[ $# -gt 0 ] || set -- unsloth/Llama-3.2-1B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen3-1.7B HuggingFaceTB/SmolLM2-1.7B-Instruct TinyLlama/TinyLlama-1.1B-Chat-v1.0 Qwen/Qwen2.5-3B-Instruct tiiuae/Falcon3-3B-Instruct mistralai/Mistral-7B-Instruct-v0.3
echo "models ($#): $*"
for M in "$@"; do
  FREE=$(df -BG /home/user | tail -1 | awk '{print $4}' | tr -d G)
  [ "$FREE" -lt 40 ] && { echo "[$M] STOP: only ${FREE}G free" | tee -a "$SUM"; break; }
  B=$(basename "$M"); export HF_MODEL=$M
  for LANE in batch-1 batch-32; do
    for TR in trace notrace; do
      export TT_CACHE_PATH=$RUN/tt_cache; command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1
      FLAG=""; [ $TR = notrace ] && FLAG="--disable_trace"
      timeout 3600 pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and $LANE" $FLAG > "$RUN/${B}_${LANE}_$TR.log" 2>&1
      echo "rc=$?" >> "$RUN/${B}_${LANE}_$TR.log"
    done
    R=$(python "$RUN/cmp.py" "$RUN/${B}_${LANE}_trace.log" "$RUN/${B}_${LANE}_notrace.log")
    echo "[$B] $LANE $R" | tee -a "$SUM"
  done
  rm -rf "$RUN/tt_cache" "$HOME/.cache/huggingface/hub/models--${M//\//--}"
done
echo "======== SUMMARY ($RUN) ========"; cat "$SUM"
