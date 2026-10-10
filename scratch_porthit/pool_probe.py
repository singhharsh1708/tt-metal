"""upsample / grid_sample candidates, one case per process: python pool_probe.py <up_shard|up_fp32|grid|up_shape>"""
import sys

import numpy as np
import torch
import torch.nn.functional as F
import ttnn


def nearest(x, s):
    return F.interpolate(x.float().permute(0, 3, 1, 2), scale_factor=s, mode="nearest").permute(0, 2, 3, 1)


def safe(tag, fn):
    try:
        fn()
    except Exception as e:
        print(f"RESULT {tag}: EXC {type(e).__name__}: {str(e).strip().splitlines()[0][:240]}", flush=True)


def up_shard(dev):
    def run(C, H=8, W=8, ncores=4, scale=2):
        torch.manual_seed(0)
        x = torch.randn(1, H, W, C, dtype=torch.bfloat16)
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(ncores - 1, 0))})
        mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                ttnn.ShardSpec(grid, [H * W // ncores, C], ttnn.ShardOrientation.ROW_MAJOR))
        t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mem)
        rt = torch.equal(ttnn.to_torch(t).float(), x.float())
        out = ttnn.to_torch(ttnn.upsample(t, [scale, scale], mode="nearest")).float()
        exp = nearest(x, scale)
        bad = int((out != exp).sum()) if out.shape == exp.shape else -1
        print(f"RESULT up_shard C={C} stick={2 * C}B: input_roundtrip_ok={rt} out_shape={tuple(out.shape)} bad={bad}/{exp.numel()}", flush=True)
    for C in (8, 16, 4, 12, 20):
        safe(f"up_shard C={C}", lambda: run(C))


def up_fp32(dev):
    def run(tag, dt, tdt, layout):
        torch.manual_seed(0)
        x = torch.rand(1, 32, 32, 32, dtype=torch.float32).to(tdt)
        t = ttnn.from_torch(x, dtype=dt, layout=layout, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.to_torch(ttnn.upsample(t, [2, 2], mode="nearest")).float()
        exp = nearest(x, 2)
        low0 = int(((out.contiguous().numpy().view(np.uint32) & 0xFFFF) == 0).sum())
        print(f"RESULT up_fp32 {tag}: bad={int((out != exp).sum())}/{exp.numel()} max_abs_err={float((out - exp).abs().max()):.3e} zero_low16={low0}", flush=True)
    safe("up_fp32 fp32_tile", lambda: run("fp32_tile", ttnn.float32, torch.float32, ttnn.TILE_LAYOUT))
    safe("up_fp32 ctrl_fp32_rm", lambda: run("ctrl_fp32_rm", ttnn.float32, torch.float32, ttnn.ROW_MAJOR_LAYOUT))
    safe("up_fp32 ctrl_bf16_tile", lambda: run("ctrl_bf16_tile", ttnn.bfloat16, torch.bfloat16, ttnn.TILE_LAYOUT))


def grid(dev):
    torch.manual_seed(42)
    x = torch.randn(1, 8, 16, 32, dtype=torch.bfloat16)
    g0 = torch.randint(-12, 13, (1, 4, 8, 2)).float() / 8
    exp = F.grid_sample(x.float().permute(0, 3, 1, 2), g0, mode="bilinear", padding_mode="zeros", align_corners=False).permute(0, 2, 3, 1)
    inp = ttnn.from_torch(x, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)

    def run(tag, packed, shard):
        g = ttnn.to_device(ttnn.reshape(ttnn.from_torch(g0, dtype=ttnn.bfloat16), packed), dev)
        mem = ttnn.create_sharded_memory_config(shard, ttnn.CoreGrid(y=1, x=2), ttnn.ShardStrategy.HEIGHT,
                                                ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        g = ttnn.to_memory_config(g, mem)
        ok = torch.equal(ttnn.to_torch(g).float().reshape(-1), g0.reshape(-1))
        out = ttnn.to_torch(ttnn.grid_sample(inp, g, mode="bilinear", align_corners=False)).float()
        bad = int((~torch.isclose(out, exp, atol=0.02, rtol=0.02)).sum()) if out.shape == exp.shape else -1
        print(f"RESULT grid {tag}: grid_roundtrip_ok={ok} out_shape={tuple(out.shape)} bad={bad}/{exp.numel()}", flush=True)
    safe("grid ctrl_packed_16B", lambda: run("ctrl_packed_16B", (1, 4, 2, 8), (4, 8)))
    safe("grid unpacked_4B", lambda: run("unpacked_4B", (1, 4, 8, 2), (16, 2)))


def up_shape(dev):
    for n, s in [(64, 1.5), (90, 1.3), (45, 2.6), (45, 1.4)]:
        def run():
            x = torch.randn(1, n, n, 32, dtype=torch.bfloat16)
            exp = nearest(x, s)
            y = ttnn.upsample(ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev), s, mode="nearest")
            print(f"RESULT up_shape in={n} scale={s}: ttnn={tuple(y.shape)} torch={tuple(exp.shape)} mismatch={tuple(y.shape) != tuple(exp.shape)}", flush=True)
        safe(f"up_shape {n} {s}", run)


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        {"up_shard": up_shard, "up_fp32": up_fp32, "grid": grid, "up_shape": up_shape}[sys.argv[1]](dev)
    finally:
        ttnn.close_device(dev)
