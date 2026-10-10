# Host only: does from_pretrained return the checkpoint's weights for Phi-3-mini-128k, with and without remote code?
import json

import torch
import transformers
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM

R = "microsoft/Phi-3-mini-128k-instruct"
print("VERSIONS transformers", transformers.__version__, "torch", torch.__version__, flush=True)
idx = json.load(open(hf_hub_download(R, "model.safetensors.index.json")))["weight_map"]
keys = [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.qkv_proj.weight",
    "model.layers.0.mlp.gate_up_proj.weight",
    "model.layers.31.mlp.down_proj.weight",
    "model.norm.weight",
    "lm_head.weight",
]
ref = {}
for k in keys:
    with safe_open(hf_hub_download(R, idx[k]), "pt") as f:
        ref[k] = f.get_tensor(k).float()
for trc in (True, False):
    try:
        m = AutoModelForCausalLM.from_pretrained(R, torch_dtype="auto", trust_remote_code=trc)
    except Exception as e:
        print(f"LOAD trust_remote_code={trc}: EXC {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)
        continue
    sd = m.state_dict()
    print(f"LOAD trust_remote_code={trc}: class module {type(m).__module__}", flush=True)
    for k in keys:
        a, b = sd[k].float(), ref[k]
        corr = torch.corrcoef(torch.stack([a.flatten()[:2000000], b.flatten()[:2000000]]))[0, 1].item()
        print(f"WEIGHT trc={trc} {k}: equal={torch.equal(a, b)} std_loaded={a.std().item():.4f} std_ckpt={b.std().item():.4f} corr={corr:.4f}", flush=True)
    del m, sd
