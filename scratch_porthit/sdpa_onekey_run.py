import sys, torch, ttnn
sys.path.insert(0, "/home/user")
from sdpa_onekey import one_key_fit

dev = ttnn.open_device(device_id=0, l1_small_size=32768)
grid = dev.compute_with_storage_grid_size()
SCALE = 128 ** -0.5
S = 256
found = 0
for approx in (False, True):
    for seed in range(60):
        torch.manual_seed(1000 + seed)
        q = torch.randn(1, 8, S, 128).bfloat16(); k = torch.randn(1, 8, S, 128).bfloat16(); v = torch.randn(1, 8, S, 128).bfloat16()
        comp = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
        pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=256, k_chunk_size=256, exp_approx_mode=approx)
        t = [ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev) for x in (q, k, v)]
        out = ttnn.to_torch(ttnn.transformer.scaled_dot_product_attention(*t, is_causal=True, scale=SCALE, program_config=pc, compute_kernel_config=comp)).double()
        qd, kd, vd = q.double(), k.double(), v.double()
        sc = (qd @ kd.transpose(-2, -1)) * SCALE + torch.triu(torch.full((S, S), -torch.inf, dtype=torch.float64), 1)
        gold = torch.softmax(sc, -1) @ vd
        rel = (out - gold).abs().amax(-1) / gold.abs().amax(-1).clamp_min(1e-3)
        for b, h, i in (rel > 0.10).nonzero().tolist():
            found += 1
            s = sc[b, h, i, : i + 1]; m = float(s.max())
            e = (out[b, h, i] - gold[b, h, i]).abs()
            j, d, res = one_key_fit(s, vd[b, h, : i + 1], out[b, h, i])
            p = torch.softmax(s, 0)
            print(f"approx={int(approx)} seed{seed} h{h} r{i} (tile-row {i % 32}, face-row {i % 16}) rel {float(rel[b,h,i]):.3f} cols>0.05: {int((e > 0.05).sum())}/128 | "
                  f"one-key fit: key {j} (tile-col {j % 32}, face {(i % 32) // 16 * 2 + (j % 32) // 16}, k-tile {j // 32}) delta {d:+.3f} resid {res:.4f} "
                  f"| s_j-max {float(s[j]) - m:+.3f} p_j {float(p[j]):.4f} | row max key {int(s.argmax())} n_keys {i + 1}", flush=True)
print(f"total bad rows {found}")
ttnn.close_device(dev)
