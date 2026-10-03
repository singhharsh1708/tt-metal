import torch, ttnn

dev = ttnn.open_device(device_id=0, l1_small_size=32768)
grid = dev.compute_with_storage_grid_size()


def window_mask(S, W):
    i = torch.arange(S)[:, None]; j = torch.arange(S)[None, :]
    m = torch.zeros(S, S); m[(j > i) | (j <= i - W)] = -torch.inf
    return m


def ref(q, k, v, W, scale):
    q, k, v = q.float(), k.float(), v.float()
    r = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(r, 1), v.repeat_interleave(r, 1)
    return torch.softmax((q @ k.transpose(-2, -1)) * scale + window_mask(q.shape[2], W), -1) @ v


def sdpa(q, k, v, W, scale, qc, kc, fp32, approx, dt):
    comp = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=False)
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=approx)
    t = [ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev) for x in (q, k, v)]
    o = ttnn.transformer.scaled_dot_product_attention(*t, is_causal=True, sliding_window_size=W, scale=scale, program_config=pc, compute_kernel_config=comp)
    return ttnn.to_torch(o).float()


def report(tag, out, gold, W, qc, kc):
    err = (out - gold).abs().amax(-1)              # [b, h, S]
    scale_row = gold.abs().amax(-1).clamp_min(1e-3)
    rel = err / scale_row
    bad = (rel > 0.10).nonzero().tolist()
    worst = float(rel.max())
    print(f"{tag:<62} rows_rel>10%={len(bad):<4} worst_rel={worst:.3f} max_abs={float(err.max()):.3f} median_rel={float(rel.median()):.4f}", flush=True)
    for b, h, i in sorted(bad, key=lambda x: -float(rel[x[0], x[1], x[2]]))[:6]:
        lo = max(0, i - W + 1)
        print(f"      head {h} row {i}: rel {float(rel[b,h,i]):.3f} abs {float(err[b,h,i]):.3f} |gold| {float(scale_row[b,h,i]):.3f}  window keys [{lo},{i}]  q-chunk {i//qc} (row {i%qc} in chunk)  first k-chunk {lo//kc} (key {lo%kc} in chunk)  k-chunks spanned {lo//kc}..{i//kc}", flush=True)


cfgs = [  # S, W, qc, kc ; the tt_transformers prefill config is q256/k256 at S>=2048, q64/k64 below
    (2048, 512, 256, 256), (2048, 1000, 128, 256), (2048, 512, 128, 512),
    (4096, 512, 256, 256), (8192, 512, 256, 256), (4096, 1024, 256, 256), (8192, 1024, 256, 256),
    (1024, 512, 64, 64), (2048, 4096, 256, 256),
]
for seed in (0, 1):
    for S, W, qc, kc in cfgs:
        torch.manual_seed(seed)
        q = torch.randn(1, 8, S, 128).bfloat16(); k = torch.randn(1, 2, S, 128).bfloat16(); v = torch.randn(1, 2, S, 128).bfloat16()
        gold = ref(q, k, v, W, 128 ** -0.5)
        for fp32, approx, dt in ((True, False, ttnn.bfloat16), (True, True, ttnn.bfloat16), (False, False, ttnn.bfloat16), (True, False, ttnn.bfloat8_b)):
            tag = f"seed{seed} S={S} W={W} q{qc} k{kc} fp32dest={int(fp32)} approx={int(approx)} {'bf8' if dt == ttnn.bfloat8_b else 'bf16'}"
            try:
                report(tag, sdpa(q, k, v, W, 128 ** -0.5, qc, kc, fp32, approx, dt), gold, W, qc, kc)
            except Exception as e:
                m = [l for l in str(e).splitlines() if "TT_THROW" in l or "TT_FATAL" in l]
                print(f"{tag:<62} EXC {(m[0] if m else str(e).splitlines()[0])[:120]}", flush=True)
ttnn.close_device(dev)
