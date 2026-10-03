import sys, traceback, torch, ttnn

dev = ttnn.open_device(device_id=0, l1_small_size=32768)
torch.manual_seed(0)
TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
spacers = []
summary = []


def rnd(shape, dt, seed):
    g = torch.Generator().manual_seed(seed)
    if dt == torch.int32:
        return torch.randint(0, 1000, shape, generator=g, dtype=torch.int32)
    return (torch.randn(shape, generator=g) * 4).to(dt)


def diff(ref, out):
    ref = ref.to(torch.float32) if ref.is_floating_point() else ref.to(torch.int64)
    out = out.to(ref.dtype)
    if ref.shape != out.shape:
        return f"shape {tuple(out.shape)} vs {tuple(ref.shape)}"
    bad = (ref != out) & ~(torch.isnan(ref.float()) & torch.isnan(out.float()))
    n = int(bad.sum())
    if n == 0:
        return "ok"
    idx = bad.nonzero()[0].tolist()
    return f"{n}/{ref.numel()} differ, first at {idx}"


def case(name, run, ref):
    """run(seed) -> (torch_out); ref(seed) -> torch_ref. Run twice, the second a cache hit at new addresses."""
    try:
        r1 = diff(ref(1), run(1))
        e0 = dev.num_program_cache_entries()
        spacers.append(ttnn.from_torch(torch.zeros(1, 1, 32, 32 * 37), dtype=ttnn.bfloat16, layout=TILE, device=dev, memory_config=DRAM))
        spacers.append(ttnn.from_torch(torch.zeros(1, 1, 32, 32 * 5), dtype=ttnn.bfloat16, layout=TILE, device=dev, memory_config=L1))
        r2 = diff(ref(2), run(2))
        hit = dev.num_program_cache_entries() == e0
        line = f"{name:<70} run1 {r1:<34} run2(hit={hit}) {r2}"
    except Exception as e:
        msg = str(e).strip().splitlines()
        msg = next((l for l in msg if "TT_FATAL" in l or "TT_THROW" in l or "Error" in l), msg[0] if msg else repr(e))
        line = f"{name:<70} EXC {type(e).__name__}: {msg[:150]}"
    print(line, flush=True)
    if any(k in line for k in ("differ", "shape", "EXC", "hit=False")):
        summary.append(line)
    while len(spacers) > 40:
        spacers.pop(0).deallocate()


TDT = {torch.bfloat16: ttnn.bfloat16, torch.float32: ttnn.float32}

# tilize: block factory is picked when a row is > 32 tiles and the row-block split uses fewer cores than the wh split
for dt in (torch.bfloat16, torch.float32):
    for mc in (DRAM, L1):
        for shp in ([1, 1, 64, 4096], [1, 1, 32, 2080], [1, 1, 96, 3104], [2, 1, 64, 4160], [1, 1, 1024, 1056], [1, 3, 64, 2048], [1, 1, 160, 8192]):
            if mc is L1 and shp[-1] * shp[-2] * (4 if dt == torch.float32 else 2) * (shp[0] * shp[1]) > 600000:
                continue
            case(f"tilize {shp} {dt} {'L1' if mc is L1 else 'DRAM'}",
                 lambda s, shp=shp, dt=dt, mc=mc: ttnn.to_torch(ttnn.tilize(ttnn.from_torch(rnd(shp, dt, s), dtype=TDT[dt], layout=RM, device=dev, memory_config=mc), memory_config=mc, use_multicore=True)),
                 lambda s, shp=shp, dt=dt: rnd(shp, dt, s))

# tilize_with_val_padding, unaligned inputs padded up
for dt in (torch.bfloat16, torch.float32):
    for shp, out in (([1, 1, 50, 4100], [1, 1, 64, 4128]), ([1, 1, 33, 2050], [1, 1, 64, 2080]), ([2, 1, 70, 3000], [2, 1, 96, 3008]), ([1, 1, 1000, 1100], [1, 1, 1024, 1120])):
        for pv in (0.0, -7.5):
            def run(s, shp=shp, out=out, dt=dt, pv=pv):
                t = ttnn.tilize_with_val_padding(ttnn.from_torch(rnd(shp, dt, s), dtype=TDT[dt], layout=RM, device=dev, memory_config=DRAM), out, pv, memory_config=DRAM, use_multicore=True)
                return t.cpu().to_torch_with_padded_shape()
            def ref(s, shp=shp, out=out, dt=dt, pv=pv):
                r = torch.full(out, pv, dtype=dt)
                r[: shp[0], : shp[1], : shp[2], : shp[3]] = rnd(shp, dt, s)
                return r
            case(f"tilize_with_val_padding {shp}->{out} pad={pv} {dt}", run, ref)

# untilize / untilize_with_unpadding multicore (block factory for wide rows)
for dt in (torch.bfloat16, torch.float32):
    for shp in ([1, 1, 64, 4096], [1, 1, 32, 2080], [2, 1, 64, 4160], [1, 1, 1024, 1056], [1, 1, 160, 8192]):
        case(f"untilize {shp} {dt}",
             lambda s, shp=shp, dt=dt: ttnn.to_torch(ttnn.untilize(ttnn.from_torch(rnd(shp, dt, s), dtype=TDT[dt], layout=TILE, device=dev, memory_config=DRAM), memory_config=DRAM, use_multicore=True)),
             lambda s, shp=shp, dt=dt: rnd(shp, dt, s))
    for shp, end in (([1, 1, 64, 4128], [0, 0, 49, 4099]), ([2, 1, 96, 3008], [1, 0, 69, 2999]), ([1, 1, 1024, 1120], [0, 0, 999, 1099])):
        case(f"untilize_with_unpadding {shp} end={end} {dt}",
             lambda s, shp=shp, end=end, dt=dt: ttnn.to_torch(ttnn.untilize_with_unpadding(ttnn.from_torch(rnd(shp, dt, s), dtype=TDT[dt], layout=TILE, device=dev, memory_config=DRAM), output_tensor_end=end, memory_config=DRAM)),
             lambda s, shp=shp, end=end, dt=dt: rnd(shp, dt, s)[: end[0] + 1, : end[1] + 1, : end[2] + 1, : end[3] + 1])

# reshape on ROW_MAJOR (reshape_view RM path)
for dt in (torch.bfloat16, torch.float32):
    for mc in (DRAM, L1):
        for a, b in (([1, 1, 64, 4096], [1, 64, 64, 64]), ([2, 3, 40, 50], [6, 2000]), ([1, 128, 7, 13], [1, 1, 896, 13]), ([1, 1, 3, 1000], [1, 3, 10, 100]), ([4, 9, 33], [3, 4, 3, 33]), ([1, 1, 2, 30001], [2, 30001])):
            case(f"reshape RM {a}->{b} {dt} {'L1' if mc is L1 else 'DRAM'}",
                 lambda s, a=a, b=b, dt=dt, mc=mc: ttnn.to_torch(ttnn.reshape(ttnn.from_torch(rnd(a, dt, s), dtype=TDT[dt], layout=RM, device=dev, memory_config=mc), b)),
                 lambda s, a=a, b=b, dt=dt: rnd(a, dt, s).reshape(b))

# interleaved_to_sharded (height / width / block, even and uneven shards, RM and TILE)
grid = dev.compute_with_storage_grid_size()
def shard_cfg(layout_kind, shp, cores_x, cores_y, rm):
    H = shp[0] * shp[1] * shp[2]; W = shp[3]
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores_x - 1, cores_y - 1))})
    n = cores_x * cores_y
    al = 1 if rm else 32
    up = lambda v: ((v + al - 1) // al) * al
    if layout_kind == "height":
        ss, sl = [up(-(-H // n)), W], ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    elif layout_kind == "width":
        ss, sl = [H, up(-(-W // n))], ttnn.TensorMemoryLayout.WIDTH_SHARDED
    else:
        ss, sl = [up(-(-H // cores_y)), up(-(-W // cores_x))], ttnn.TensorMemoryLayout.BLOCK_SHARDED
    return ttnn.MemoryConfig(sl, ttnn.BufferType.L1, ttnn.ShardSpec(crs, ss, ttnn.ShardOrientation.ROW_MAJOR))
for dt in (torch.bfloat16, torch.float32):
    for rm in (False, True):
        for kind, shp, cx, cy in (("height", [1, 1, 320, 256], 8, 1), ("height", [1, 1, 288, 128], 5, 1), ("width", [1, 1, 64, 1024], 8, 1), ("width", [1, 1, 32, 960], 7, 1), ("block", [1, 1, 256, 512], 4, 4), ("block", [1, 1, 224, 480], 4, 3)):
            def run(s, kind=kind, shp=shp, cx=cx, cy=cy, rm=rm, dt=dt):
                x = ttnn.from_torch(rnd(shp, dt, s), dtype=TDT[dt], layout=RM if rm else TILE, device=dev, memory_config=DRAM)
                y = ttnn.interleaved_to_sharded(x, shard_cfg(kind, shp, cx, cy, rm))
                return ttnn.to_torch(ttnn.sharded_to_interleaved(y, DRAM))
            case(f"interleaved_to_sharded {kind} {shp} {cx}x{cy} {'RM' if rm else 'TILE'} {dt}", run, lambda s, shp=shp, dt=dt: rnd(shp, dt, s))

# argmax (single-core RM path and multicore), last dim
for shp in ([1, 1, 1, 32000], [1, 1, 32, 1000], [1, 1, 1, 151936], [1, 1, 8, 4096], [1, 1, 1, 50]):
    for mcore in (False, True):
        def run(s, shp=shp, mcore=mcore):
            x = ttnn.from_torch(rnd(shp, torch.bfloat16, s), dtype=ttnn.bfloat16, layout=RM, device=dev, memory_config=DRAM)
            i = ttnn.to_torch(ttnn.argmax(x, dim=-1, keepdim=True, use_multicore=mcore)).to(torch.int64).reshape(shp[:-1] + [1])
            return rnd(shp, torch.bfloat16, s).float().gather(-1, i)
        def ref(s, shp=shp):
            return rnd(shp, torch.bfloat16, s).float().max(-1, keepdim=True).values
        case(f"argmax {shp} multicore={mcore}", run, ref)

# embedding with fused tilize output
for vocab, dim, idx_shape in ((32000, 2048, [1, 128]), (32000, 2048, [1, 1000]), (151936, 1536, [1, 96]), (1000, 4096, [32, 1]), (128256, 2048, [1, 33]), (500, 64, [4, 70])):
    for lay in (TILE, RM):
        W = {}
        def run(s, vocab=vocab, dim=dim, idx_shape=idx_shape, lay=lay):
            w = rnd([vocab, dim], torch.bfloat16, 100 + s)
            i = torch.randint(0, vocab, idx_shape, generator=torch.Generator().manual_seed(s), dtype=torch.int32)
            tw = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=RM, device=dev, memory_config=DRAM)
            ti = ttnn.from_torch(i, dtype=ttnn.uint32, layout=RM, device=dev, memory_config=DRAM)
            out = ttnn.to_torch(ttnn.embedding(ti, tw, layout=lay, memory_config=DRAM))
            tw.deallocate(); return out.reshape(idx_shape + [dim])
        def ref(s, vocab=vocab, dim=dim, idx_shape=idx_shape):
            w = rnd([vocab, dim], torch.bfloat16, 100 + s)
            i = torch.randint(0, vocab, idx_shape, generator=torch.Generator().manual_seed(s), dtype=torch.int32)
            return w[i.long()]
        case(f"embedding vocab={vocab} dim={dim} idx={idx_shape} {'TILE' if lay is TILE else 'RM'}", run, ref)

print("======== FLAGGED ========")
print("\n".join(summary) if summary else "none")
ttnn.close_device(dev)
