import os, sys

import torch, ttnn

dev = ttnn.open_device(device_id=0)
g = torch.Generator().manual_seed(41)


def safe(name, fn):
    try:
        fn()
    except Exception as e:
        print(f"RESULT {name:70s} ERROR {' | '.join([l for l in str(e).splitlines() if l.strip()][:3])[:400]}", flush=True)


CFGS = {
    "default": None,
    "HiFi4_exact": ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                   fp32_dest_acc_en=False, packer_l1_acc=False),
    "HiFi2_approx_tt_decode": ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True,
                                                  fp32_dest_acc_en=False, packer_l1_acc=False),
    "HiFi2_exact": ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                                  fp32_dest_acc_en=False, packer_l1_acc=False),
}
POS = [0, 1, 2, 3, 7, 15, 31, 32, 33, 63, 100, 127, 128, 511, 1023]
for (b, nh, nkv, s, d) in [(4, 8, 1, 1024, 128), (4, 32, 8, 1024, 128)]:
    for cname, cfg in CFGS.items():
        for qk_scale in [0.0, 1.0, 4.0]:
            for c in [4.0]:
                def run(b=b, nh=nh, nkv=nkv, s=s, d=d, cfg=cfg, cname=cname, qk_scale=qk_scale, c=c):
                    res = {}
                    for k in range(0, len(POS), b):
                        pos = (POS[k:k + b] + POS)[:b]
                        Q = (torch.randn(1, b, nh, d, generator=g) * qk_scale / d ** 0.25).bfloat16().float()
                        K = (torch.randn(b, nkv, s, d, generator=g) * qk_scale / d ** 0.25).bfloat16().float()
                        V = torch.full((b, nkv, s, d), c)
                        tq = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
                        tk = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
                        tv = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
                        cp = ttnn.from_torch(torch.tensor(pos, dtype=torch.int32), dtype=ttnn.int32, device=dev)
                        kw = {"cur_pos_tensor": cp, "scale": d ** -0.5}
                        if cfg is not None:
                            kw["compute_kernel_config"] = cfg
                        o = ttnn.to_torch(ttnn.transformer.scaled_dot_product_attention_decode(tq, tk, tv, **kw)).float()[0, :b, :nh, :]
                        for i, p in enumerate(pos):
                            ratio = o[i] / c
                            res[p] = (float(ratio.min()), float(ratio.max()))
                    worst = max(abs(lo - 1) for lo, hi in res.values())
                    worst = max(worst, max(abs(hi - 1) for lo, hi in res.values()))
                    summary = {p: round((lo + hi) / 2, 5) for p, (lo, hi) in sorted(res.items())}
                    print(f"RESULT weightsum b={b} nh={nh} nkv={nkv} cfg={cname} qk_scale={qk_scale} c={c} worst_rel={worst:.3e} "
                          f"ratio_by_pos={summary} {'ok' if worst < 2e-2 else 'WRONG'}", flush=True)
                safe(f"weightsum b={b} nh={nh} nkv={nkv} cfg={cname} qk={qk_scale} c={c}", run)
ttnn.close_device(dev)
