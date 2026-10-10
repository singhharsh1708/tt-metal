#!/usr/bin/env python3
"""Row-pitch family repro for Wormhole N150 (device 0).

A ROW_MAJOR sharded L1 buffer keeps each shard row at aligned_page_size() (16-byte pitch), while the
factories below step through the borrowed shard with the raw row size. Each case:
  1. writes the inputs, reads them back (ROUNDTRIP must be ok, so the host side is not at fault),
  2. runs the op and compares with torch (RESULT line: bad element count),
  3. has an aligned control that is expected to print bad=0.

Usage:  python repro_rowpitch_family.py            # all cases
        python repro_rowpitch_family.py clone pool  # a subset (substring match on the case name)
A case that raises is reported as RESULT ... EXC and the run continues. A device hang needs a reset; rerun the
remaining cases by name.
"""
import sys
import traceback

import torch
import ttnn

torch.manual_seed(0)
DEV = None
BF16, FP32 = ttnn.bfloat16, ttnn.float32


def grid(n):
    if n <= 8:
        return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n - 1, 0))])
    assert n % 8 == 0
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, n // 8 - 1))])


def shard_cfg(layout, ncores, h, w, buffer_type=ttnn.BufferType.L1):
    return ttnn.MemoryConfig(layout, buffer_type, ttnn.ShardSpec(grid(ncores), [h, w], ttnn.ShardOrientation.ROW_MAJOR))


def hs(ncores, h, w):
    return shard_cfg(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ncores, h, w)


def rnd(shape, dtype):
    x = torch.randn(shape)
    if dtype == BF16:
        x = x.to(torch.bfloat16).to(torch.float32)  # values exactly representable in bfloat16
    return x


def dev(x, dtype, mem):
    return ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=DEV, memory_config=mem)


def host(t):
    return ttnn.to_torch(t).to(torch.float32)


def poison(shape, dtype, mem):
    """Fill the L1 the output is likely to reuse with a sentinel, so unwritten rows cannot look right by luck."""
    p = dev(torch.full(shape, 123.0), dtype, mem)
    ttnn.deallocate(p)


def pitch(t):
    try:
        return f"row={t.buffer_page_size()}B pitch={t.buffer_aligned_page_size()}B"
    except Exception:
        return "row/pitch n/a"


def report(name, inputs, out, expected):
    rt = all(torch.equal(host(t).reshape(x.shape), x) for t, x in inputs)
    got = host(out)
    if got.numel() == expected.numel():
        got = got.reshape(expected.shape)
        bad = int((got != expected).sum())
    else:
        bad = f"shape {tuple(got.shape)} vs {tuple(expected.shape)}"
    lay = out.memory_config().memory_layout
    print(
        f"RESULT {name}: ROUNDTRIP={'ok' if rt else 'FAIL'} in[{pitch(inputs[0][0])}] out[{lay}, {pitch(out)}] "
        f"bad={bad}/{expected.numel()}",
        flush=True,
    )


# ---------------------------------------------------------------- transpose HC / permute (0,2,1,3)
def transpose_hc(name, shape, shard_h, dtype, use_permute=False):
    n, c, h, w = shape
    ncores = n * c * h // shard_h
    mem = hs(ncores, shard_h, w)
    x = rnd(shape, dtype)
    t = dev(x, dtype, mem)
    poison((n, h, c, w), dtype, mem)
    y = ttnn.permute(t, (0, 2, 1, 3)) if use_permute else ttnn.transpose(t, 1, 2)
    report(name, [(t, x)], y, x.transpose(1, 2))


# ---------------------------------------------------------------- concat, width, height-sharded RM
def concat_w(name, rows, shard_h, widths, dtype):
    ncores = rows // shard_h
    xs = [rnd((1, 1, rows, w), dtype) for w in widths]
    ts = [dev(x, dtype, hs(ncores, shard_h, w)) for x, w in zip(xs, widths)]
    out_mem = hs(ncores, shard_h, sum(widths))
    poison((1, 1, rows, sum(widths)), dtype, out_mem)
    y = ttnn.concat(ts, dim=-1, memory_config=out_mem)
    report(name, list(zip(ts, xs)), y, torch.cat(xs, dim=-1))


# ---------------------------------------------------------------- clone
def clone(name, shape, ncores, shard_h, dtype, layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED, shard_w=None):
    mem = shard_cfg(layout, ncores, shard_h, shard_w or shape[-1])
    x = rnd(shape, dtype)
    t = dev(x, dtype, mem)
    poison(shape, dtype, mem)
    y = ttnn.clone(t)
    report(name, [(t, x)], y, x)


# ---------------------------------------------------------------- max_pool2d on a pre-sharded RM input
def pool(name, H, W, C, ncores):
    rows = H * W
    mem = hs(ncores, rows // ncores, C)
    x = rnd((1, H, W, C), BF16)
    t = dev(x.reshape(1, 1, rows, C), BF16, mem)
    y = ttnn.max_pool2d(
        input_tensor=t,
        batch_size=1,
        input_h=H,
        input_w=W,
        channels=C,
        kernel_size=[2, 2],
        stride=[2, 2],
        padding=[0, 0],
        dilation=[1, 1],
    )
    exp = torch.nn.functional.max_pool2d(x.permute(0, 3, 1, 2), 2, 2).permute(0, 2, 3, 1)
    got = host(y)[..., :C]
    rt = torch.equal(host(t).reshape(x.shape), x)
    bad = int((got.reshape(exp.shape) != exp).sum()) if got.numel() == exp.numel() else f"shape {tuple(got.shape)}"
    print(f"RESULT {name}: ROUNDTRIP={'ok' if rt else 'FAIL'} in[{pitch(t)}] bad={bad}/{exp.numel()}", flush=True)


# ---------------------------------------------------------------- rotate nearest, height-sharded, angle 0
def rotate(name, shape, ncores, dtype):
    n, h, w, c = shape
    mem = hs(ncores, n * h * w // ncores, c)
    x = rnd(shape, dtype)
    t = dev(x, dtype, mem)
    poison(shape, dtype, mem)
    y = ttnn.rotate(t, angle=0.0, interpolation_mode="nearest")
    report(name, [(t, x)], y, x)


# ---------------------------------------------------------------- convert_to_hwc, width-sharded CHW input
def to_hwc(name, C, HW, in_cores, in_buffer):
    sw = HW // in_cores
    in_mem = shard_cfg(ttnn.TensorMemoryLayout.WIDTH_SHARDED, in_cores, C, sw, in_buffer)
    out_mem = hs(1, HW, 8)
    x = rnd((1, 1, C, HW), BF16)
    t = dev(x, BF16, in_mem)
    y = ttnn.experimental.convert_to_hwc(t, memory_config=out_mem, dtype=BF16)
    exp = x.transpose(2, 3).reshape(1, 1, HW, C)
    got = host(y)[..., :C]
    rt = torch.equal(host(t).reshape(x.shape), x)
    bad = int((got.reshape(exp.shape) != exp).sum())
    print(f"RESULT {name}: ROUNDTRIP={'ok' if rt else 'FAIL'} in[{pitch(t)}] bad={bad}/{exp.numel()}", flush=True)


# ---------------------------------------------------------------- indexed_fill, native height-sharded path
def indexed_fill(name, B, W, D, dtype):
    mem = hs(B, W, D)
    a = rnd((B, 1, W, D), dtype)
    b = rnd((1, 1, W, D), dtype)
    ids = torch.tensor([[[[1]]]])
    t_ids = ttnn.Tensor(ids, ttnn.uint32).to(DEV, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1))
    ta = dev(a, dtype, mem)
    tb = ttnn.from_torch(b, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=DEV)
    poison((B, 1, W, D), dtype, mem)
    y = ttnn.indexed_fill(t_ids, ta, tb, memory_config=mem)
    exp = a.clone()
    exp[1] = b[0]
    report(name, [(ta, a)], y, exp)


L = ttnn.TensorMemoryLayout
CASES = [
    # name, callable.   BUG = expected bad > 0, CTRL = expected bad == 0
    ("transpose_hc.BUG.bf16_W4_special", lambda n: transpose_hc(n, (1, 16, 16, 4), 256, BF16)),
    ("transpose_hc.BUG.bf16_W12_generic", lambda n: transpose_hc(n, (1, 3, 256, 12), 256, BF16)),
    ("transpose_hc.BUG.fp32_W2", lambda n: transpose_hc(n, (1, 16, 32, 2), 512, FP32)),
    ("transpose_hc.BUG.permute0213_bf16_W20", lambda n: transpose_hc(n, (1, 16, 64, 20), 256, BF16, use_permute=True)),
    ("transpose_hc.CTRL.bf16_W8_special", lambda n: transpose_hc(n, (1, 16, 16, 8), 128, BF16)),
    ("transpose_hc.CTRL.bf16_W16_generic", lambda n: transpose_hc(n, (1, 3, 256, 16), 128, BF16)),
    ("concat2.BUG.bf16_13+13", lambda n: concat_w(n, 64, 32, [13, 13], BF16)),
    ("concat2.BUG.bf16_16+12", lambda n: concat_w(n, 64, 32, [16, 12], BF16)),
    ("concat2.BUG.fp32_7+7", lambda n: concat_w(n, 64, 32, [7, 7], FP32)),
    ("concat2.CTRL.bf16_16+16", lambda n: concat_w(n, 64, 32, [16, 16], BF16)),
    ("concat3.BUG.bf16_12+12+12", lambda n: concat_w(n, 64, 32, [12, 12, 12], BF16)),
    ("concat3.BUG.bf16_20+20+24", lambda n: concat_w(n, 64, 32, [20, 20, 24], BF16)),
    ("concat3.CTRL.bf16_16+16+16", lambda n: concat_w(n, 64, 32, [16, 16, 16], BF16)),
    ("clone.BUG.bf16_height_W12", lambda n: clone(n, (1, 1, 64, 12), 2, 32, BF16)),
    ("clone.BUG.fp32_height_W3", lambda n: clone(n, (1, 1, 64, 3), 2, 32, FP32)),
    ("clone.BUG.bf16_width_shardW4", lambda n: clone(n, (1, 1, 64, 16), 4, 64, BF16, L.WIDTH_SHARDED, 4)),
    ("clone.CTRL.bf16_height_W8", lambda n: clone(n, (1, 1, 64, 8), 2, 32, BF16)),
    ("pool.BUG.maxpool_C12", lambda n: pool(n, 16, 16, 12, 2)),
    ("pool.CTRL.maxpool_C16", lambda n: pool(n, 16, 16, 16, 2)),
    ("rotate.BUG.bf16_C3", lambda n: rotate(n, (1, 16, 16, 3), 2, BF16)),
    ("rotate.BUG.fp32_C3", lambda n: rotate(n, (1, 16, 16, 3), 2, FP32)),
    ("rotate.CTRL.bf16_C8", lambda n: rotate(n, (1, 16, 16, 8), 2, BF16)),
    ("to_hwc.BUG.L1_in_shardW12", lambda n: to_hwc(n, 3, 192, 16, ttnn.BufferType.L1)),
    ("to_hwc.BUG.DRAM_in_shardW24", lambda n: to_hwc(n, 3, 192, 8, ttnn.BufferType.DRAM)),
    ("to_hwc.CTRL.L1_in_shardW32", lambda n: to_hwc(n, 3, 192, 6, ttnn.BufferType.L1)),
    ("to_hwc.CTRL.DRAM_in_shardW32", lambda n: to_hwc(n, 3, 192, 6, ttnn.BufferType.DRAM)),
    ("indexed_fill.BUG.bf16_D12", lambda n: indexed_fill(n, 4, 64, 12, BF16)),
    ("indexed_fill.CTRL.bf16_D16", lambda n: indexed_fill(n, 4, 64, 16, BF16)),
]


def main():
    global DEV
    want = sys.argv[1:]
    DEV = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        for name, fn in CASES:
            if want and not any(w in name for w in want):
                continue
            try:
                fn(name)
            except Exception as e:  # TT_FATAL etc.
                msg = str(e).strip().splitlines()
                print(f"RESULT {name}: EXC {type(e).__name__}: {msg[0][:200] if msg else ''}", flush=True)
                traceback.print_exc(limit=1)
    finally:
        ttnn.close_device(DEV)


if __name__ == "__main__":
    main()
