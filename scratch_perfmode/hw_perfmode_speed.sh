set -uo pipefail
unset LD_LIBRARY_PATH CI
export TT_METAL_HOME=/home/user/ttm-main TT_METAL_RUNTIME_ROOT=/home/user/ttm-main
export PYTHONPATH=/home/user/ttm-main:/home/user/ttm-main/ttnn
export MESH_DEVICE=N150 TOKENIZERS_PARALLELISM=false
cd "$TT_METAL_HOME" || exit 1
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
RUN=/home/user/perfspeed_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
REFDIR=models/tt_transformers/tests/reference_outputs
SUM="$RUN/summary.txt"
echo "tree: $(git log -1 --format='%h %cd' --date=short)   dirty (must be 0): $(git status --porcelain | wc -l | tr -d ' ')   artifacts: $RUN"
[ "$(git status --porcelain | wc -l | tr -d ' ')" = 0 ] || { echo "tree not clean, stopping"; exit 1; }
restore () { git checkout -q -- models/tt_transformers; echo "[restored] dirty: $(git status --porcelain | wc -l | tr -d ' ')"; }
trap restore EXIT
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
print("model_config patched with PERF_VARIANT hook")
PYEOF
[ $# -gt 0 ] || set -- Qwen/Qwen2.5-1.5B-Instruct openbmb/MiniCPM5-2B unsloth/Llama-3.2-1B-Instruct Qwen/Qwen3-1.7B
echo "models ($#): $*"
num () { grep -oE "$2" "$1" | tail -1 | grep -oE "[0-9.]+" | tail -1; }
for M in "$@"; do
  FREE=$(df -BG /home/user | tail -1 | awk '{print $4}' | tr -d G)
  [ "$FREE" -lt 40 ] && { echo "[$M] STOP: only ${FREE}G free" | tee -a "$SUM"; break; }
  B=$(basename "$M"); t0=$(date +%s); LINE="[$B] decode tok/s/user (TTFT ms)"
  for V in stock kv16 mlp8 qwen7b accmode stock; do
    MODE=performance
    case $V in stock) unset PERF_VARIANT;; accmode) unset PERF_VARIANT; MODE=accuracy;; *) export PERF_VARIANT=$V;; esac
    export HF_MODEL=$M TT_CACHE_PATH=$RUN/tt_cache_$V
    rm -rf "$TT_CACHE_PATH"
    command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1
    LOG="$RUN/demo_${B}_${V}_$(date +%s).log"
    timeout 3600 pytest models/tt_transformers/demo/simple_text_demo.py -k "$MODE and batch-1" > "$LOG" 2>&1
    RC=$?
    SPD=$(num "$LOG" "Average speed: [0-9.]+ms @ [0-9.]+ tok/s/user")
    TTFT=$(num "$LOG" "Average Time to First Token \(TTFT\): [0-9.]+ms")
    CFG=$(grep -c "requires specific TensorPrecision" "$LOG")
    NOTE=""; [ -z "$SPD" ] && NOTE="(rc=$RC $(grep -E 'TT_FATAL|TT_THROW|Out of Memory|^E +[A-Za-z]+Error' "$LOG" | grep -v DEBUG | tail -1 | cut -c1-90))"
    LINE="$LINE | $V ${SPD:-?} (${TTFT:-?}) cfg=$CFG $NOTE"
    rm -rf "$TT_CACHE_PATH"
  done
  unset PERF_VARIANT
  echo "$LINE  ($(( ($(date +%s) - t0) / 60 )) min)" | tee -a "$SUM"
  rm -rf "$HOME/.cache/huggingface/hub/models--${M//\//--}"
done
echo "======== SUMMARY ($RUN) ========"; cat "$SUM"
