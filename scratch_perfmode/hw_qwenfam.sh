set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
export MESH_DEVICE=N150 TOKENIZERS_PARALLELISM=false
cd "$TT_METAL_HOME" || exit 1
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
RUN=/home/user/qwenfam_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
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
python - <<'PYEOF' || exit 1
import pathlib
p = pathlib.Path("models/tt_transformers/tt/model_config.py"); s = p.read_text()
anchor = '        """Configuration optimized for performance\n        All models use bfp4 in FF1 and FF3 MLPs in this configuration\n        """\n'
assert s.count(anchor) == 1, s.count(anchor)
add = """        import os as _os
        _variant = _os.environ.get("PERF_VARIANT")
        if _variant:
            _attn16 = {TensorGroup.WQKV: PrecisionSetting.BF16, TensorGroup.KV_CACHE: PrecisionSetting.BF16, TensorGroup.WO: PrecisionSetting.BF16}
            _hifi4 = {g: MathFidelitySetting.HIFI4 for g in (OpGroup.LI_QKV_DECODE, OpGroup.LI_QKV_PREFILL, OpGroup.SDPA_DECODE, OpGroup.SDPA_PREFILL, OpGroup.LI_O_DECODE, OpGroup.LI_O_PREFILL)}
            _table = {
                "mlp8": ({}, {}),
                "plain": ({TensorGroup.FF1_FF3: PrecisionSetting.BFP4}, {OpGroup.LI_FF1_FF3: MathFidelitySetting.LOFI}),
                "mlp4_hifi2": ({TensorGroup.FF1_FF3: PrecisionSetting.BFP4}, {OpGroup.LI_FF1_FF3: MathFidelitySetting.HIFI2_FP16}),
                "kv16": ({TensorGroup.FF1_FF3: PrecisionSetting.BFP4, TensorGroup.KV_CACHE: PrecisionSetting.BF16}, {OpGroup.LI_FF1_FF3: MathFidelitySetting.LOFI}),
                "attn16": ({TensorGroup.FF1_FF3: PrecisionSetting.BFP4, **_attn16}, {OpGroup.LI_FF1_FF3: MathFidelitySetting.LOFI, **_hifi4}),
                "qwen7b": (dict(_attn16), dict(_hifi4)),
            }
            _tp, _of = _table[_variant]
            _inst = cls({"TensorPrecision": _tp, "OpFidelity": _of})
            _inst.__name__ = "performance"
            return _inst
"""
p.write_text(s.replace(anchor, anchor + add))
a = '        if self.base_model_name in ["Qwen2.5-7B", "Qwen2.5-VL-7B"] and self.num_devices not in [0, 2, 4]:\n'
s = p.read_text(); assert s.count(a) == 1, "N150 assertion anchor not found"
p.write_text(s.replace(a, a.replace("        if ", "        if False and ")))
print("model_config patched with PERF_VARIANT hook; Qwen2.5-7B N150 assertion disabled")
PYEOF
REFGEN="$RUN/refgen/generate_reference_hf.py"
num () { grep -oE "$2" "$1" | tail -1 | grep -oE "[0-9.]+" | tail -1; }
err () { grep -E 'TT_FATAL|TT_THROW|Out of Memory|out of memory|^E +[A-Za-z]+Error' "$1" | grep -v DEBUG | tail -1 | cut -c1-110; }
setv () { MODE=performance; case $1 in stock|tt) unset PERF_VARIANT;; accmode) unset PERF_VARIANT; MODE=accuracy;; *) export PERF_VARIANT=$1;; esac; }
run_demo () {  # model variant lane -> log path
  setv "$2"; export HF_MODEL=$1 TT_CACHE_PATH=$RUN/tt_cache_$2; rm -rf "$TT_CACHE_PATH"
  command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1
  L="$RUN/$(basename "$1")_$2_$3.log"
  timeout 3600 pytest models/tt_transformers/demo/simple_text_demo.py -k "$MODE and $3" > "$L" 2>&1
  echo "rc=$?" >> "$L"; rm -rf "$TT_CACHE_PATH"; unset PERF_VARIANT
}
acc_model () {  # model variants...
  M=$1; shift; B=$(basename "$M"); REF="$REFDIR/$B.refpt"; t0=$(date +%s)
  FREE=$(df -BG /home/user | tail -1 | awk '{print $4}' | tr -d G)
  [ "$FREE" -lt 45 ] && { echo "[$B] SKIP: only ${FREE}G free" | tee -a "$SUM"; return; }
  timeout 5400 python "$REFGEN" --model "$M" --total_length 1024 --output_file "$REF" > "$RUN/ref_$B.log" 2>&1
  [ -f "$REF" ] || { echo "[$B] REFGEN FAILED: $(err "$RUN/ref_$B.log")" | tee -a "$SUM"; return; }
  LINE="[$B] top1/top5"
  for V in "$@"; do
    run_demo "$M" "$V" ci-token-matching
    ACC=$(grep -oE "Top1 Accuracy: [0-9.]+%, Top5 Accuracy: [0-9.]+%" "$L" | head -1 | sed -E 's/Top1 Accuracy: ([0-9.]+)%, Top5 Accuracy: ([0-9.]+)%/\1\/\2/')
    LINE="$LINE | $V ${ACC:-FAIL($(tail -1 "$L") $(err "$L"))}"
  done
  echo "$LINE  ($(( ($(date +%s) - t0) / 60 )) min)" | tee -a "$SUM"
  git checkout -q -- "$REF" 2>/dev/null || rm -f "$REF"
}
spd_model () {  # model lane variants...
  M=$1; LANE=$2; shift 2; B=$(basename "$M"); LINE="[$B] $LANE decode tok/s/user (TTFT ms)"
  for V in "$@"; do
    run_demo "$M" "$V" "$LANE"
    SPD=$(num "$L" "Average speed: [0-9.]+ms @ [0-9.]+ tok/s/user"); TTFT=$(num "$L" "Average Time to First Token \(TTFT\): [0-9.]+ms")
    LINE="$LINE | $V ${SPD:-FAIL($(tail -1 "$L") $(err "$L"))} (${TTFT:-?})"
  done
  echo "$LINE" | tee -a "$SUM"
}
drop_hf () { rm -rf "$HOME/.cache/huggingface/hub/models--${1//\//--}"; }
echo "== part 1: Qwen2.5 family on N150, bf16 KV cache vs stock vs accuracy mode"
for M in Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-Coder-3B-Instruct deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B SakanaAI/TinySwallow-1.5B-Instruct Qwen/Qwen2.5-Math-1.5B-Instruct; do
  acc_model "$M" stock kv16 accmode; drop_hf "$M"
done
echo "== part 2: decode cost of bf16 KV at 16k context"
spd_model Qwen/Qwen2.5-1.5B-Instruct long-context-16k stock kv16; drop_hf Qwen/Qwen2.5-1.5B-Instruct
echo "== part 3: 7B on one N150 (tt = TT's own Qwen2.5-7B settings)"
acc_model Qwen/Qwen2.5-7B-Instruct tt plain kv16
spd_model Qwen/Qwen2.5-7B-Instruct batch-1 tt plain kv16; drop_hf Qwen/Qwen2.5-7B-Instruct
acc_model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B stock kv16
spd_model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B batch-1 stock kv16; drop_hf deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
echo "======== SUMMARY ($RUN) ========"; cat "$SUM"
