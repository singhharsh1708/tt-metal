# Localise the Phi-3-mini-128k failure on the unpatched latest-main tree: host reference first, then TT unit tests.
set -uo pipefail
unset LD_LIBRARY_PATH CI
D=/home/user/ttm-latest
cd "$D" || { echo "no $D"; exit 1; }
export TT_METAL_HOME=$D TT_METAL_RUNTIME_ROOT=$D PYTHONPATH=$D:$D/ttnn HF_HUB_OFFLINE=0
exec 9>/home/user/census.lock
echo "waiting for the device lock..."; flock 9; echo "lock acquired $(date -u +%H:%M)"
echo "tree: $(git log -1 --format='%h %cd' --date=short)  dirty files: $(git status --porcelain | grep -vc '^??')"
RUN=/home/user/phi3loc_$(date -u +%Y%m%d_%H%M%S); mkdir -p "$RUN"
export HF_MODEL=microsoft/Phi-3-mini-128k-instruct MESH_DEVICE=N150
python - > "$RUN/host_ref.log" 2>&1 <<'PY'
import torch, transformers, tokenizers
print("VERSIONS torch", torch.__version__, "transformers", transformers.__version__, "tokenizers", tokenizers.__version__)
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
name = "microsoft/Phi-3-mini-128k-instruct"
cfg = AutoConfig.from_pretrained(name)
rs = getattr(cfg, "rope_scaling", None)
print("CONFIG max_pos", cfg.max_position_embeddings, "orig_max_pos", getattr(cfg, "original_max_position_embeddings", None),
      "rope_scaling keys", sorted(rs.keys()) if isinstance(rs, dict) else rs)
tok = AutoTokenizer.from_pretrained(name)
m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
print("STATE_DICT sample keys", [k for k in m.state_dict().keys() if "layers.0." in k][:8])
msgs = [{"role": "user", "content": "What is your favorite condiment? Answer in one sentence."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
if not torch.is_tensor(ids): ids = ids["input_ids"]
print("PROMPT ids", ids[0].tolist()[:12], "len", ids.shape[1])
out = m.generate(ids, max_new_tokens=24, do_sample=False)
print("HOST_REF_TEXT", repr(tok.decode(out[0, ids.shape[1]:])))
PY
echo "===== [host_ref] rc=$?"; grep -aE "^(VERSIONS|CONFIG|STATE_DICT|PROMPT|HOST_REF_TEXT)|Error" "$RUN/host_ref.log" | cut -c1-300
pt () {
  command -v tt-smi >/dev/null && tt-smi -r 0 >/dev/null 2>&1; sleep 3
  timeout -k 30 "$2" pytest -q -p no:cacheprovider "${@:3}" > "$RUN/$1.log" 2>&1; rc=$?
  echo "===== [$1] rc=$rc | $(grep -aE '(passed|failed|error|skipped|deselected).* in [0-9.]+s' "$RUN/$1.log" | tail -1 | cut -c1-140)"
  grep -aE "^(FAILED|ERROR) |^E +[A-Za-z]*Error|TT_THROW:|TT_FATAL:| PCC: |PCC value|pcc_message|Passed!|Failed!" "$RUN/$1.log" | grep -av "riscv-tt-elf" | awk '!s[$0]++' | head -6 | cut -c1-220
  grep -aE " PCC: |PCC value" "$RUN/$1.log" | awk '!s[$0]++' | tail -3 | cut -c1-220
}
T=models/tt_transformers/tests
pt embedding 600 $T/test_embedding.py
pt rms_norm 600 $T/test_rms_norm.py
pt mlp 900 $T/test_mlp.py
pt attention 900 $T/test_attention.py
pt attention_prefill 900 $T/test_attention_prefill.py
pt decoder 900 $T/test_decoder.py
pt decoder_prefill 900 $T/test_decoder_prefill.py
pt lm_head 900 $T/test_lm_head.py
pt model_full 1800 -p no:timeout $T/test_model.py -k "full"
pt model_prefill 1800 -p no:timeout $T/test_model_prefill.py
echo "done $(date -u +%H:%M)  logs: $RUN"
