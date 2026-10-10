# padded_slice, TILE float32 input -> ROW_MAJOR height-sharded output. Pure copy, expected bit-exact.
import torch, ttnn

device = ttnn.open_device(device_id=0)


def hs_cfg(shard_h, shard_w, n):
    grid = ttnn.num_cores_to_corerangeset(n, device.compute_with_storage_grid_size(), row_wise=True)
    spec = ttnn.ShardSpec(grid, (shard_h, shard_w), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)


def run(name, src, tt_dtype, layout, ends, n=2):
    shape = list(src.shape)
    tt_in = ttnn.from_torch(src, dtype=tt_dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out_h = ends[0] * ends[1] * ends[2]
    cfg = hs_cfg(out_h // n, ends[3], n)
    tt_out = ttnn.experimental.padded_slice(tt_in, [0, 0, 0, 0], list(ends), [1, 1, 1, 1], memory_config=cfg)
    out = ttnn.to_torch(tt_out).reshape(ends)
    ref = src[: ends[0], : ends[1], : ends[2], : ends[3]]
    if src.dtype == torch.bfloat16:
        out, ref = out.float(), ref.float()
    bad = out != ref
    line = f"{name:28s} out dtype {tt_out.dtype}  bad {int(bad.sum())}/{ref.numel()}"
    if ref.dtype == torch.float32:
        ob, rb = out.contiguous().view(torch.int32), ref.contiguous().view(torch.int32)
        line += f"  max abs err {float((out - ref).abs().max()):.3e}"
        line += f"  outputs with low16==0: {int(((ob & 0xFFFF) == 0).sum())}"
        line += f"  == bf16-truncated ref: {bool(torch.equal(ob, rb & ~0xFFFF))}"
    elif ref.dtype == torch.int32:
        line += f"  max abs err {int((out.long() - ref.long()).abs().max())}"
    print(line)


g = torch.Generator().manual_seed(0)
shape = (1, 1, 64, 32)
f32 = (torch.randn(shape, generator=g) * 3.0).float()  # full 23-bit mantissas
i32 = torch.randint(70000, 2**31 - 1, shape, generator=g, dtype=torch.int32)  # all above 65536

run("float32 TILE (bug)", f32, ttnn.float32, ttnn.TILE_LAYOUT, (1, 1, 64, 32))
run("bfloat16 TILE (control)", f32.to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT, (1, 1, 64, 32))
run("float32 ROW_MAJOR (control)", f32[:, :, :16], ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, (1, 1, 8, 32))
run("int32 TILE (probe)", i32, ttnn.int32, ttnn.TILE_LAYOUT, (1, 1, 64, 32))

ttnn.close_device(device)
