import struct, torch, ttnn

dev = ttnn.open_device(device_id=0, l1_small_size=32768)
grid = dev.compute_with_storage_grid_size()
SCALE = 128 ** -0.5
S = 1024


def bf16_hex(x):
    return "%04x" % (struct.unpack("<I", struct.pack("<f", float(x)))[0] >> 16)


def f32_hex(x):
    return "%08x" % struct.unpack("<I", struct.pack("<f", float(x)))[0]


def run(q, k, v, fid, fp32, approx, vdt=ttnn.bfloat16):
    comp = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=False)
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=256, k_chunk_size=256, exp_approx_mode=approx)
    tq, tk = [ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev) for x in (q, k)]
    tv = ttnn.from_torch(v, dtype=vdt, layout=ttnn.TILE_LAYOUT, device=dev)
    return ttnn.to_torch(ttnn.transformer.scaled_dot_product_attention(tq, tk, tv, is_causal=True, scale=SCALE, program_config=pc, compute_kernel_config=comp)).double()


events = 0
for seed in range(24):
    torch.manual_seed(2000 + seed)
    q = torch.randn(1, 8, S, 128).bfloat16(); k = torch.randn(1, 8, S, 128).bfloat16(); v = torch.randn(1, 8, S, 128).bfloat16()
    qd, kd, vd = q.double(), k.double(), v.double()
    sc = (qd @ kd.transpose(-2, -1)) * SCALE + torch.triu(torch.full((S, S), -torch.inf, dtype=torch.float64), 1)
    gold = torch.softmax(sc, -1) @ vd
    outs = {name: run(q, k, v, *cfg) for name, cfg in {
        "hifi4": (ttnn.MathFidelity.HiFi4, True, False), "hifi4_approx": (ttnn.MathFidelity.HiFi4, True, True),
        "hifi2": (ttnn.MathFidelity.HiFi2, True, False), "stream": (ttnn.MathFidelity.HiFi4, False, False)}.items()}
    typical = float((outs["hifi4"] - gold).abs().median())
    for name in ("hifi4", "hifi4_approx", "hifi2", "stream"):
        err = (outs[name] - gold).abs()
        for b, h, i, c in (err > 0.05).nonzero().tolist():
            events += 1
            g, o = float(gold[b, h, i, c]), float(outs[name][b, h, i, c])
            others = " ".join(f"{n}={float(outs[n][b, h, i, c]):+.4f}" for n in outs if n != name)
            print(f"seed{seed} {name:<12} h{h} r{i} col{c} (tile-row {i % 32}, tile-col {c % 32}, face {(i % 32) // 16 * 2 + (c % 32) // 16}) "
                  f"gold {g:+.5f} [bf16 {bf16_hex(g)}] dev {o:+.5f} [bf16 {bf16_hex(o)}] diff {o - g:+.5f} ratio {o / g if g else float('nan'):+.3f} | {others}", flush=True)
    print(f"seed{seed} done (typical |err| {typical:.5f})", flush=True)
print(f"events {events}")
ttnn.close_device(dev)
