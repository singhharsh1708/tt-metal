import torch, ttnn

dev = ttnn.open_device(device_id=0, l1_small_size=32768)
TILE, DRAM = ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG
flagged = []


def rt(x, dt):
    """Value the device will hold after the host converts x to dt."""
    return ttnn.to_torch(ttnn.from_torch(x.float(), dtype=dt, layout=TILE)).double()


def cfg(fid, approx, fp32, l1):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=approx, fp32_dest_acc_en=fp32, packer_l1_acc=l1)


def corerange(n):
    nx = min(n, 8); ny = n // nx
    return ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nx - 1, ny - 1))


def report(tag, out, gold, thr=0.10):
    err = (out - gold).abs().amax(-1); rel = err / gold.abs().amax(-1).clamp_min(1e-3)
    n = int((rel > thr).sum()); tot = rel.numel()
    flat = torch.stack([out.flatten(), gold.flatten()])
    pcc = float(torch.corrcoef(flat)[0, 1]) if tot > 1 else float("nan")
    line = f"{tag:<86} bad={n}/{tot} worst_rel={float(rel.max()):.3f} median_rel={float(rel.median()):.4f} pcc={pcc:.5f}"
    print(line, flush=True)
    if n or float(rel.max()) > 0.25 or torch.isnan(out).any():
        flagged.append(line)
    return rel


def guard(tag, fn):
    try:
        fn()
    except Exception as e:
        m = [l for l in str(e).splitlines() if "TT_THROW" in l or "TT_FATAL" in l]
        line = f"{tag:<86} EXC {(m[0] if m else str(e).splitlines()[0])[:120]}"
        print(line, flush=True); flagged.append(line)


def paged(x, B, nkv, S, blk, hd):
    return x.reshape(B, nkv, S // blk, blk, hd).transpose(1, 2).reshape(B * S // blk, nkv, blk, hd)


def unpaged(x, B, nkv, S, blk, hd):
    return x.reshape(B, S // blk, nkv, blk, hd).transpose(1, 2).reshape(B, nkv, S, hd)


TT = dict(fid=ttnn.MathFidelity.HiFi2, approx=True, fp32=True, l1=True)   # tt_transformers SDPA_DECODE
VARIANTS = [("model(k=0,fp32)", 0, TT), ("k_chunk=32,fp32", 32, TT), ("k=0,bf16dest", 0, dict(TT, fp32=False))]

print("== gap 1: paged SDPA decode, dynamic chunk, tt_transformers config (HiFi2, approx, fp32 acc)")
S = 2048
for nh, nkv, hd, B, blk, kv_dt in ((32, 8, 64, 1, 32, ttnn.bfloat8_b), (32, 8, 64, 32, 32, ttnn.bfloat8_b), (24, 8, 128, 1, 32, ttnn.bfloat8_b),
                                   (14, 2, 64, 1, 32, ttnn.bfloat16), (12, 2, 128, 1, 32, ttnn.bfloat16), (16, 2, 128, 1, 32, ttnn.bfloat16),
                                   (32, 32, 96, 1, 32, ttnn.bfloat8_b), (32, 32, 64, 32, 32, ttnn.bfloat16), (32, 4, 64, 1, 64, ttnn.bfloat8_b),
                                   (12, 2, 128, 32, 32, ttnn.bfloat16)):
    torch.manual_seed(0)
    nb = S // blk
    perm = torch.randperm(B * nb); inv = torch.argsort(perm)
    Kp = rt(paged(torch.randn(B, nkv, S, hd), B, nkv, S, blk, hd)[perm], kv_dt)
    Vp = rt(paged(torch.randn(B, nkv, S, hd), B, nkv, S, blk, hd)[perm], kv_dt)
    Kc, Vc = unpaged(Kp[inv], B, nkv, S, blk, hd), unpaged(Vp[inv], B, nkv, S, blk, hd)
    tK = ttnn.as_tensor(Kp.float(), device=dev, dtype=kv_dt, layout=TILE, memory_config=DRAM)
    tV = ttnn.as_tensor(Vp.float(), device=dev, dtype=kv_dt, layout=TILE, memory_config=DRAM)
    tPT = ttnn.Tensor(inv.reshape(B, nb).to(torch.int32), ttnn.int32).to(dev)
    shard = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                              ttnn.ShardSpec(ttnn.CoreRangeSet({corerange(B)}), (32, hd), ttnn.ShardOrientation.ROW_MAJOR))
    r = nh // nkv; scale = hd ** -0.5
    for base in (0, 14, 31, 32, 33, 63, 64, 127, 128, 129, 255, 256, 1023, 1024, 1025, 2000):
        pos = [min(S - 1, base + 5 * u) for u in range(B)]
        Q = torch.randn(1, B, nh, hd).bfloat16()
        gold = torch.zeros(B, nh, hd, dtype=torch.float64); leak = torch.zeros_like(gold)
        for u in range(B):
            end = (pos[u] // 32 + 1) * 32   # keys the tile holds, including stale ones beyond cur_pos (#41215 model)
            for h in range(nh):
                k = Kc[u, h // r, : pos[u] + 1]; v = Vc[u, h // r, : pos[u] + 1]
                gold[u, h] = torch.softmax((k @ Q[0, u, h].double()) * scale, 0) @ v
                k2 = Kc[u, h // r, :end]; v2 = Vc[u, h // r, :end]
                leak[u, h] = torch.softmax((k2 @ Q[0, u, h].double()) * scale, 0) @ v2
        tPos = ttnn.Tensor(torch.tensor(pos, dtype=torch.int32), ttnn.int32).to(dev)
        for vname, kchunk, c in VARIANTS:
            tag = f"paged nh{nh} nkv{nkv} hd{hd} B{B} blk{blk} {'bfp8' if kv_dt == ttnn.bfloat8_b else 'bf16'} pos{base} {vname}"
            def go():
                tQ = ttnn.as_tensor(Q.float(), device=dev, dtype=ttnn.bfloat16, layout=TILE, memory_config=shard)
                pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), exp_approx_mode=False, q_chunk_size=0 if kchunk == 0 else 32, k_chunk_size=kchunk)
                o = ttnn.transformer.paged_scaled_dot_product_attention_decode(tQ, tK, tV, tPT, cur_pos_tensor=tPos, scale=scale, program_config=pc, compute_kernel_config=cfg(**c), memory_config=DRAM)
                out = ttnn.to_torch(o).double()[0, :, :nh]
                rel = report(tag, out, gold)
                if int((rel > 0.10).sum()):
                    report(tag + "  [vs stale-key leak model #41215]", out, leak)
            guard(tag, go)
    tK.deallocate(); tV.deallocate()

print("== gap 3: chunked prefill SDPA on the fp32 (legacy) kernel, Q bfp8, GQA, paged K/V block 32")
for nh, nkv, hd, chunk, qk in ((32, 8, 64, 1024, 64), (12, 2, 128, 1024, 64), (32, 8, 64, 2048, 256)):
    torch.manual_seed(1)
    Stot = 2 * chunk; blk = 32; nb = Stot // blk
    perm = torch.randperm(nb); inv = torch.argsort(perm)
    Kp = rt(paged(torch.randn(1, nkv, Stot, hd), 1, nkv, Stot, blk, hd)[perm], ttnn.bfloat8_b)
    Vp = rt(paged(torch.randn(1, nkv, Stot, hd), 1, nkv, Stot, blk, hd)[perm], ttnn.bfloat8_b)
    Kc, Vc = unpaged(Kp[inv], 1, nkv, Stot, blk, hd), unpaged(Vp[inv], 1, nkv, Stot, blk, hd)
    tK = ttnn.as_tensor(Kp.float(), device=dev, dtype=ttnn.bfloat8_b, layout=TILE, memory_config=DRAM)
    tV = ttnn.as_tensor(Vp.float(), device=dev, dtype=ttnn.bfloat8_b, layout=TILE, memory_config=DRAM)
    tPT = ttnn.Tensor(inv.reshape(1, nb).to(torch.int32), ttnn.int32).to(dev)
    r = nh // nkv; scale = hd ** -0.5
    for start in (0, chunk):
        Qc = rt(torch.randn(1, nh, chunk, hd), ttnn.bfloat8_b)
        Ke, Ve = Kc[0].repeat_interleave(r, 0)[:, : start + chunk], Vc[0].repeat_interleave(r, 0)[:, : start + chunk]
        sc = (Qc[0] @ Ke.transpose(-2, -1)) * scale
        i = torch.arange(chunk)[:, None] + start; j = torch.arange(start + chunk)[None, :]
        sc = sc.masked_fill(j > i, float("-inf"))
        gold = torch.softmax(sc, -1) @ Ve
        for fp32 in (True, False):
            tag = f"chunked nh{nh} nkv{nkv} hd{hd} chunk{chunk} start{start} q/k{qk} {'fp32dest(model)' if fp32 else 'bf16dest'}"
            def go():
                tQ = ttnn.as_tensor(Qc.float(), device=dev, dtype=ttnn.bfloat8_b, layout=TILE, memory_config=DRAM)
                pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), exp_approx_mode=False, q_chunk_size=qk, k_chunk_size=qk)
                o = ttnn.transformer.chunked_scaled_dot_product_attention(tQ, tK, tV, tPT, start, program_config=pc, compute_kernel_config=cfg(ttnn.MathFidelity.HiFi4, False, fp32, True))
                report(tag, ttnn.to_torch(o).double()[0], gold)
            guard(tag, go)
    tK.deallocate(); tV.deallocate()

print("== gap 6: batched prefill SDPA (batch > 1), fp32 kernel, Q bfp8")
for B, nh, nkv, hd, Sq in ((4, 32, 8, 64, 128), (4, 14, 2, 64, 128), (8, 32, 8, 64, 256)):
    torch.manual_seed(2)
    Q = rt(torch.randn(B, nh, Sq, hd), ttnn.bfloat8_b); K = rt(torch.randn(B, nkv, Sq, hd), ttnn.bfloat8_b); V = rt(torch.randn(B, nkv, Sq, hd), ttnn.bfloat8_b)
    r = nh // nkv; scale = hd ** -0.5
    sc = (Q @ K.repeat_interleave(r, 1).transpose(-2, -1)) * scale + torch.triu(torch.full((Sq, Sq), float("-inf"), dtype=torch.float64), 1)
    gold = torch.softmax(sc, -1) @ V.repeat_interleave(r, 1)
    for fp32 in (True, False):
        tag = f"batched prefill B{B} nh{nh} nkv{nkv} hd{hd} S{Sq} {'fp32dest(model)' if fp32 else 'bf16dest'}"
        def go():
            t = [ttnn.as_tensor(x.float(), device=dev, dtype=ttnn.bfloat8_b, layout=TILE, memory_config=DRAM) for x in (Q, K, V)]
            pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), exp_approx_mode=False, q_chunk_size=64, k_chunk_size=64)
            o = ttnn.transformer.scaled_dot_product_attention(*t, is_causal=True, scale=scale, program_config=pc, compute_kernel_config=cfg(ttnn.MathFidelity.HiFi4, False, fp32, True))
            report(tag, ttnn.to_torch(o).double(), gold)
        guard(tag, go)

print("======== FLAGGED ========")
print("\n".join(flagged) if flagged else "none")
ttnn.close_device(dev)
