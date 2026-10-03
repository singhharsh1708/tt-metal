import sys, torch, ttnn
sys.path.insert(0, "/home/user")
from sdpa_diag import diagnose, SCALE

dev = ttnn.open_device(device_id=0, l1_small_size=32768)
grid = dev.compute_with_storage_grid_size()


def ref64(q, k, v):
    q, k, v = q.double(), k.double(), v.double()
    r = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(r, 1), v.repeat_interleave(r, 1)
    S = q.shape[2]
    m = torch.triu(torch.full((S, S), -torch.inf, dtype=torch.float64), 1)
    return torch.softmax((q @ k.transpose(-2, -1)) * SCALE + m, -1) @ v


S = 2048
for seed in range(4):
    torch.manual_seed(seed)
    q = torch.randn(1, 8, S, 128).bfloat16(); k = torch.randn(1, 2, S, 128).bfloat16(); v = torch.randn(1, 2, S, 128).bfloat16()
    gold = ref64(q, k, v)
    for approx in (False, True):
        comp = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
        pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=256, k_chunk_size=256, exp_approx_mode=approx)
        t = [ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev) for x in (q, k, v)]
        out = ttnn.to_torch(ttnn.transformer.scaled_dot_product_attention(*t, is_causal=True, scale=SCALE, program_config=pc, compute_kernel_config=comp)).double()
        rel = (out - gold).abs().amax(-1) / gold.abs().amax(-1).clamp_min(1e-3)
        bad = (rel > 0.10).nonzero().tolist()
        typical = float((out - gold).abs().amax(-1).median())
        print(f"seed{seed} approx={int(approx)} bad rows {len(bad)} (typical row max-abs err {typical:.4f})", flush=True)
        for b, h, i in bad:
            top, maxes, am = diagnose(q, k, v, out, b, h, i)
            print(f"   h{h} r{i} (q-chunk {i // 256}, row {i % 256}, tile-row {i % 32}) rel {float(rel[b, h, i]):.3f} | chunk maxes {maxes} argmax key {am} (chunk {am // 256}, col {am % 32})", flush=True)
            print(f"      best explanations: " + "; ".join(f"{m}@chunk{c} err {e:.4f}" for e, m, c in top), flush=True)
ttnn.close_device(dev)
