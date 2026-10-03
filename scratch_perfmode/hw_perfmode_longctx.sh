set -uo pipefail
unset LD_LIBRARY_PATH
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
export MESH_DEVICE=N150 TOKENIZERS_PARALLELISM=false
unset CI
cd "$TT_METAL_HOME" || exit 1
exec 9>/home/user/census.lock
flock -n 9 || { echo "another device job is running; not starting"; exit 1; }
RUN=/home/user/perfmode_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
D=models/tt_transformers/demo/simple_text_demo.py
REFDIR=models/tt_transformers/tests/reference_outputs
cp -p $D "$RUN/demo.orig"
restore () {
  cp -p "$RUN/demo.orig" $D
  for f in "$RUN"/*.refpt.bak; do [ -f "$f" ] || continue; b=$(basename "$f" .refpt.bak); mv -f "$f" "$REFDIR/$b.refpt"; done
  git ls-files --others --exclude-standard -- "$REFDIR" | xargs -r rm -f
  echo "[restored] dirty: $(git status --porcelain | wc -l | tr -d ' ')"
}
trap restore EXIT
echo "tree: $(git log -1 --format='%h %cd' --date=short)   dirty (must be 0): $(git status --porcelain | wc -l | tr -d ' ')   artifacts: $RUN"
[ "$(git status --porcelain | wc -l | tr -d ' ')" = 0 ] || { echo "tree not clean, stopping"; exit 1; }
command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1 && echo "[reset] ok"

cat > "$RUN/refgen.py" <<'PY'
import bz2, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
name, out, P, ANCHOR = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
SCORE = 512
tok = AutoTokenizer.from_pretrained(name)
m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
text = bz2.open("models/tt_transformers/tests/tale-of-two-cities.txt.bz2", "rt", encoding="utf-8").read()
all_ids = tok.encode(text, add_special_tokens=False)
assert ANCHOR >= P and len(all_ids) >= ANCHOR + P, (len(all_ids), ANCHOR, P)
ids = torch.tensor(all_ids[ANCHOR - P : ANCHOR + P]).unsqueeze(0)
used = P + SCORE  # only the prompt and the scored window matter; the unused tail gets placeholder top-5 rows
with torch.no_grad():
    logits = m(ids[:, :used]).logits[0].float()
top5 = torch.zeros(ids.shape[-1] - 1, 5, dtype=torch.long)
top5[: used - 1] = torch.topk(logits[:-1], k=5, dim=-1).indices
nxt = ids[0, 1:]
scored = slice(P - 1, P - 1 + SCORE)
print(f"REFGEN {name} anchor={ANCHOR} prefill={P} HF_self_top1={(top5[scored][:, 0] == nxt[scored]).float().mean():.4f} "
      f"first_scored_id={int(ids[0, P])} book_tokens={len(all_ids)}", flush=True)
torch.save({"reference_tokens": ids, "top5_tokens": top5}, out)
PY

patch_row () {  # max_seq_len  page_blocks
python - "$1" "$2" <<'PY' || exit 1
import pathlib, sys
msl, blocks = sys.argv[1], sys.argv[2]
p = pathlib.Path("models/tt_transformers/demo/simple_text_demo.py"); s = p.read_text()
a = s.index("(  # ci-token-matching run"); b = s.index("(  # ci-eval-1", a)
row = s[a:b]
for old, new in [("            1024,  # max_seq_len\n", f"            {msl},  # max_seq_len\n"),
                 ("            500,  # max_generated_tokens\n", "            512,  # max_generated_tokens\n"),
                 ('"page_max_num_blocks_per_dp": 1024}', f'"page_max_num_blocks_per_dp": {blocks}}}')]:
    if row.count(old) != 1: raise SystemExit(f"row anchor found {row.count(old)} times: {old!r}")
    row = row.replace(old, new)
p.write_text(s[:a] + row + s[b:])
PY
}

run_case () {  # model  prefill  anchor
  M=$1; P=$2; A=$3; BASE=$(basename "$M"); REF="$REFDIR/$BASE.refpt"; TAG="${BASE}_${P}_$A"
  [ -f "$REF" ] && [ ! -f "$RUN/$BASE.refpt.bak" ] && git ls-files --error-unmatch "$REF" >/dev/null 2>&1 && cp -p "$REF" "$RUN/$BASE.refpt.bak"
  timeout 7200 python "$RUN/refgen.py" "$M" "$REF" "$P" "$A" > "$RUN/ref_$TAG.log" 2>&1
  SELF=$(grep -oE "HF_self_top1=[0-9.]+" "$RUN/ref_$TAG.log" | cut -d= -f2)
  if [ -z "$SELF" ]; then echo "[$BASE prefill=$P] REFGEN FAILED: $(grep -iE 'error|assert' "$RUN/ref_$TAG.log" | tail -1 | cut -c1-160)"; return; fi
  MSL=8192; while [ $MSL -lt $(( P + 1024 )) ]; do MSL=$(( MSL * 2 )); done
  LINE="[$BASE anchor=$A prefill=$P] HF_self=$SELF"
  for MODE in accuracy performance; do
    patch_row $MSL $(( MSL / 32 ))
    export HF_MODEL=$M TT_CACHE_PATH=/home/user/tt_cache_perfmode_$MODE
    timeout 5400 pytest models/tt_transformers/demo/simple_text_demo.py -k "$MODE and ci-token-matching" > "$RUN/demo_${TAG}_$MODE.log" 2>&1
    RC=$?
    ACC=$(grep -oE "Top1 Accuracy: [0-9.]+%, Top5 Accuracy: [0-9.]+%" "$RUN/demo_${TAG}_$MODE.log" | head -1 | sed -E 's/Top1 Accuracy: ([0-9.]+)%, Top5 Accuracy: ([0-9.]+)%/\1\/\2/')
    NOTE=""; [ -z "$ACC" ] && NOTE="(rc=$RC $(grep -E 'TT_FATAL|TT_THROW|Out of Memory|Error' "$RUN/demo_${TAG}_$MODE.log" | grep -v DEBUG | tail -1 | cut -c1-110))"
    LINE="$LINE | $MODE ${ACC:--} $NOTE"
    cp -p "$RUN/demo.orig" $D
    command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1
  done
  echo "$LINE"
}

for M in unsloth/Llama-3.2-3B-Instruct unsloth/Meta-Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.3; do
  for P in 512 8192 16384 32768; do
    case "$M" in *8B*|*7B*) [ $P -gt 16384 ] && continue;; esac
    run_case "$M" "$P" 40000
  done
  rm -rf /home/user/tt_cache_perfmode_accuracy /home/user/tt_cache_perfmode_performance "$HOME/.cache/huggingface/hub/models--${M//\//--}"
done
echo "done: $RUN"
