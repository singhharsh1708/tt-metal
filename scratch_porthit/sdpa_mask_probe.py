import math, torch, ttnn

dev = ttnn.open_device(device_id=0, l1_small_size=32768)
grid = dev.compute_with_storage_grid_size()


def ref(q, k, v, mask, scale):
    q, k, v = q.float(), k.float(), v.float()
    if k.shape[1] != q.shape[1]:
        r = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(r, 1), v.repeat_interleave(r, 1)
    s = (q @ k.transpose(-2, -1)) * scale + mask
    return torch.softmax(s, -1) @ v


def window_mask(S, W):
    i = torch.arange(S)[:, None]; j = torch.arange(S)[None, :]
    m = torch.zeros(S, S)
    m[(j > i) | (j <= i - W)] = -torch.inf
    return m[None, None]


def run(tag, q, k, v, mask, scale, qc, kc, causal=False, W=None, use_mask=True):
    gold = ref(q, k, v, mask, scale)
    for fp32 in (True, False):
        for approx in (False, True):
            comp = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=False)
            pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=approx)
            t = [ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev) for x in (q, k, v)]
            kw = dict(is_causal=causal, scale=scale, program_config=pc, compute_kernel_config=comp)
            if use_mask:
                kw["attn_mask"] = ttnn.from_torch(mask.expand(q.shape[0], 1, -1, -1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            if W is not None:
                kw["sliding_window_size"] = W
            try:
                out = ttnn.to_torch(ttnn.transformer.scaled_dot_product_attention(*t, **kw)).float()
                nan = int(torch.isnan(out).sum()); err = (out - gold).abs().nan_to_num(1e9)
                rows_bad = int((err.amax(-1) > 0.1).sum())
                flat_o, flat_g = out.nan_to_num(0).flatten(), gold.flatten()
                pcc = float(torch.corrcoef(torch.stack([flat_o, flat_g]))[0, 1])
                print(f"{tag:<58} fp32dest={int(fp32)} approx={int(approx)} | nan={nan} bad_rows={rows_bad}/{out.shape[0]*out.shape[1]*out.shape[2]} max_err={float(err.max()):.3g} pcc={pcc:.5f}", flush=True)
            except Exception as e:
                m = [l for l in str(e).splitlines() if "TT_FATAL" in l or "TT_THROW" in l or l.strip()]
                print(f"{tag:<58} fp32dest={int(fp32)} approx={int(approx)} | EXC {m[0][:140] if m else repr(e)}", flush=True)


torch.manual_seed(0)
# A: explicit mask, first K chunk fully masked for half the query rows, later keys valid
for d in (64, 128):
    q = torch.randn(1, 4, 128, d).bfloat16(); k = torch.randn(1, 4, 1024, d).bfloat16(); v = torch.randn(1, 4, 1024, d).bfloat16()
    m = torch.zeros(1, 1, 128, 1024); m[:, :, :64, :512] = -torch.inf
    run(f"A mask first-chunk-masked rows 0-63 d={d} qc128 kc512", q, k, v, m, d ** -0.5, 128, 512)
    m2 = torch.zeros(1, 1, 128, 1024); m2[:, :, :, :512] = -torch.inf
    run(f"A mask first-chunk-masked all rows d={d} qc128 kc512", q, k, v, m2, d ** -0.5, 128, 512)
# B: causal sliding window via mask (how a model passes it) and via sliding_window_size
for S, W, qc, kc in ((1024, 128, 128, 128), (2048, 512, 128, 512), (2048, 512, 256, 256), (4096, 1024, 256, 512), (2048, 1000, 128, 256)):
    q = torch.randn(1, 8, S, 128).bfloat16(); k = torch.randn(1, 2, S, 128).bfloat16(); v = torch.randn(1, 2, S, 128).bfloat16()
    m = window_mask(S, W)
    run(f"B window via attn_mask S={S} W={W} qc{qc} kc{kc} gqa4", q, k, v, m, 128 ** -0.5, qc, kc)
    run(f"B window via sliding_window_size S={S} W={W} qc{qc} kc{kc} gqa4", q, k, v, m, 128 ** -0.5, qc, kc, causal=True, W=W, use_mask=False)
ttnn.close_device(dev)
