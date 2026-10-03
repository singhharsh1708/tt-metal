import torch, ttnn

dev = ttnn.open_device(device_id=0, l1_small_size=32768)
FID = {"hifi4": ttnn.MathFidelity.HiFi4, "hifi3": ttnn.MathFidelity.HiFi3, "hifi2": ttnn.MathFidelity.HiFi2}


def p_like(rows, k, g):
    s = torch.randn(rows, k, generator=g, dtype=torch.float64) * 1.0
    return torch.exp(s - s.max(-1, keepdim=True).values).float()   # softmax numerators in (0, 1]


tot = {}
for seed in range(60):
    g = torch.Generator().manual_seed(seed)
    A = p_like(256, 256, g)[None, None]                       # P chunk: 256 q rows x 256 keys
    B = torch.randn(1, 1, 256, 128, generator=g).bfloat16()    # V chunk
    for adt_name, adt in (("A_fp32", ttnn.float32), ("A_bf16", ttnn.bfloat16)):
        a_host = A if adt == ttnn.float32 else A.bfloat16()
        ref = a_host.double() @ B.double()
        ta = ttnn.from_torch(a_host, dtype=adt, layout=ttnn.TILE_LAYOUT, device=dev)
        tb = ttnn.from_torch(B, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        for fname, fid in FID.items():
            cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
            out = ttnn.to_torch(ttnn.matmul(ta, tb, dtype=ttnn.float32, compute_kernel_config=cfg)).double()
            err = (out - ref).abs()
            typ = float(err.median()) + 1e-12
            big = (err > 0.02).nonzero().tolist()
            key = (adt_name, fname)
            n, mx = tot.get(key, (0, 0.0))
            tot[key] = (n + len(big), max(mx, float(err.max())))
            for _, _, i, c in big[:3]:
                print(f"seed{seed} {adt_name} {fname}: out[{i},{c}] dev {float(out[0,0,i,c]):+.5f} ref {float(ref[0,0,i,c]):+.5f} err {float(err[0,0,i,c]):.4f} (median err {typ:.2e})", flush=True)
print("== totals (elements with |err| > 0.02 over 60 seeds, max |err|)")
for k, (n, mx) in sorted(tot.items()):
    print(f"   {k[0]} {k[1]}: {n} outliers, max err {mx:.4g}")
ttnn.close_device(dev)
