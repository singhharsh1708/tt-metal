import sys, torch, ttnn

dev = ttnn.open_device(device_id=0, l1_small_size=32768)
grid = dev.compute_with_storage_grid_size()
SCALE = 128 ** -0.5
LABEL = sys.argv[1]


def mask_for(S, W, causal):
    i = torch.arange(S)[:, None]; j = torch.arange(S)[None, :]
    m = torch.zeros(S, S, dtype=torch.float64)
    if causal:
        m[j > i] = -torch.inf
        m[j <= i - W] = -torch.inf
    else:
        hw = W // 2
        m[(j < i - hw) | (j > i + hw)] = -torch.inf
    return m


def ref64(q, k, v, W, causal):
    q, k, v = q.double(), k.double(), v.double()
    r = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(r, 1), v.repeat_interleave(r, 1)
    return torch.softmax((q @ k.transpose(-2, -1)) * SCALE + mask_for(q.shape[2], W, causal), -1) @ v


def run(S, W, causal, qc, kc, fp32):
    torch.manual_seed(0)
    q = torch.randn(1, 8, S, 128).bfloat16(); k = torch.randn(1, 2, S, 128).bfloat16(); v = torch.randn(1, 2, S, 128).bfloat16()
    comp = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=False)
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=False)
    t = [ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev) for x in (q, k, v)]
    out = ttnn.to_torch(ttnn.transformer.scaled_dot_product_attention(*t, is_causal=causal, sliding_window_size=W, scale=SCALE, program_config=pc, compute_kernel_config=comp)).double()
    gold = ref64(q, k, v, W, causal)
    rel = (out - gold).abs().amax(-1) / gold.abs().amax(-1).clamp_min(1e-3)
    flat = torch.stack([out.flatten(), gold.flatten()])
    return int((rel > 0.10).sum()), float(rel.max()), float(torch.corrcoef(flat)[0, 1]), float(((out - gold) ** 2).mean().sqrt())


cases = [(2048, W, True, 256, 256) for W in (1000, 1008, 1016, 1023, 1024, 1025, 1040, 2047, 130, 512)]
cases += [(4096, 2047, True, 256, 256), (8192, 2047, True, 256, 256), (1024, 1000, True, 64, 64), (2048, 1000, True, 128, 256)]
for S, W, causal, qc, kc in cases:
    for fp32 in (True, False):
        try:
            n, worst, pcc, rmse = run(S, W, causal, qc, kc, fp32)
            print(f"[{LABEL}] S={S:<5} W={W:<5} {'causal' if causal else 'noncausal'} q{qc} k{kc} {'fp32dest' if fp32 else 'stream  '} rows_rel>10%={n:<5} worst_rel={worst:.3f} pcc={pcc:.6f} rmse={rmse:.5f}", flush=True)
        except Exception as e:
            m = [l for l in str(e).splitlines() if "TT_THROW" in l or "TT_FATAL" in l]
            print(f"[{LABEL}] S={S:<5} W={W:<5} {'causal' if causal else 'noncausal'} q{qc} k{kc} {'fp32dest' if fp32 else 'stream  '} EXC {(m[0] if m else str(e).splitlines()[0])[:110]}", flush=True)
ttnn.close_device(dev)
