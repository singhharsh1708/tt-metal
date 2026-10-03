import torch, ttnn

dev = ttnn.open_device(device_id=0, l1_small_size=32768)
grid = dev.compute_with_storage_grid_size()
SCALE = 128 ** -0.5


def mask_for(S, W):
    i = torch.arange(S)[:, None]; j = torch.arange(S)[None, :]
    m = torch.zeros(S, S, dtype=torch.float64)
    m[j > i] = -torch.inf
    if W is not None:
        m[j <= i - W] = -torch.inf
    return m


def ref64(q, k, v, W):
    q, k, v = q.double(), k.double(), v.double()
    r = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(r, 1), v.repeat_interleave(r, 1)
    s = (q @ k.transpose(-2, -1)) * SCALE + mask_for(q.shape[2], W)
    return torch.softmax(s, -1) @ v, s


def dev_sdpa(q, k, v, W, qc, kc, fp32, approx, l1acc=False, via_mask=False):
    comp = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=l1acc)
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=approx)
    t = [ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev) for x in (q, k, v)]
    if via_mask:
        m = ttnn.from_torch(mask_for(q.shape[2], W).float()[None, None].expand(q.shape[0], 1, -1, -1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        o = ttnn.transformer.scaled_dot_product_attention(*t, attn_mask=m, is_causal=False, scale=SCALE, program_config=pc, compute_kernel_config=comp)
    else:
        o = ttnn.transformer.scaled_dot_product_attention(*t, is_causal=True, sliding_window_size=W, scale=SCALE, program_config=pc, compute_kernel_config=comp)
    return ttnn.to_torch(o).double()


def stats(out, gold, W=None):
    err = (out - gold).abs().amax(-1); rel = err / gold.abs().amax(-1).clamp_min(1e-3)
    S = out.shape[2]
    rows = torch.arange(S)
    win = (rows >= W) if W is not None else torch.zeros(S, dtype=torch.bool)
    bad = rel > 0.10
    return int(bad.sum()), int(bad[..., win].sum()), int(bad[..., ~win].sum()), float(rel.max()), rel


def run(tag, q, k, v, W, qc, kc, variants):
    gold, scores = ref64(q, k, v, W)
    for name, kw in variants:
        try:
            out = dev_sdpa(q, k, v, W, qc, kc, **kw)
            n, nw, nn, worst, rel = stats(out, gold, W)
            print(f"{tag:<44} {name:<22} bad={n:<5} (rows>=W {nw}, rows<W {nn}) worst_rel={worst:.3f}", flush=True)
        except Exception as e:
            m = [l for l in str(e).splitlines() if "TT_THROW" in l or "TT_FATAL" in l]
            print(f"{tag:<44} {name:<22} EXC {(m[0] if m else str(e).splitlines()[0])[:110]}", flush=True)
    return gold, scores


STD = [("fp32 accurate", dict(fp32=True, approx=False)), ("fp32 approx", dict(fp32=True, approx=True)), ("bf16dest(stream)", dict(fp32=False, approx=False))]

print("== part 1: window alignment, tt_transformers chunks (q256 k256), via sliding_window_size and via attn_mask")
torch.manual_seed(0)
S = 2048
q = torch.randn(1, 8, S, 128).bfloat16(); k = torch.randn(1, 2, S, 128).bfloat16(); v = torch.randn(1, 2, S, 128).bfloat16()
for W in (1000, 1008, 1016, 1023, 1024, 1025, 1040, 2047):
    run(f"S={S} W={W} sliding_window_size", q, k, v, W, 256, 256, STD[:1] + STD[2:])
    run(f"S={S} W={W} attn_mask", q, k, v, W, 256, 256, [(n + " mask", dict(kw, via_mask=True)) for n, kw in (STD[0], STD[2])])

print("== part 2: plain causal (no window), sporadic rows: determinism, seeds, packer_l1_acc, worst-row anatomy")
for seed in range(4):
    torch.manual_seed(seed)
    q = torch.randn(1, 8, S, 128).bfloat16(); k = torch.randn(1, 2, S, 128).bfloat16(); v = torch.randn(1, 2, S, 128).bfloat16()
    gold, scores = ref64(q, k, v, None)
    for name, kw in STD + [("fp32 accurate l1acc", dict(fp32=True, approx=False, l1acc=True))]:
        outs = [dev_sdpa(q, k, v, None, 256, 256, **kw) for _ in range(2)]
        same = bool(torch.equal(outs[0], outs[1]))
        n, _, _, worst, rel = stats(outs[0], gold)
        line = f"seed{seed} S={S} causal q256 k256 {name:<22} bad={n:<4} worst_rel={worst:.3f} deterministic={same}"
        if n:
            b, h, i = [int(x) for x in (rel == rel.max()).nonzero()[0]]
            s = scores[b, h, i, : i + 1]
            top = torch.topk(s, min(3, i + 1)).values.tolist()
            o = outs[0][b, h, i]; kv = v.double().repeat_interleave(4, 1)[b, h, : i + 1]
            j = int((kv - o).abs().amax(-1).argmin())
            line += f" | worst h{h} r{i}: top scaled scores {[round(t, 2) for t in top]}, out nearest v[{j}] (dist {float((kv[j]-o).abs().max()):.3f}), gold-out max {float((gold[b,h,i]-o).abs().max()):.3f}"
        print(line, flush=True)
ttnn.close_device(dev)
