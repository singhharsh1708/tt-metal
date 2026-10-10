"""ttnn.slice RM, aligned begin: a core group with fewer rows reserves more DFB entries per batch than the ring holds."""
import torch, ttnn


def run(dev, shape, ends, dtype_t, dtype_tt):
    torch.manual_seed(0)
    x = (torch.rand(shape, dtype=torch.float32) + 1.0).to(dtype_t)
    a = ttnn.from_torch(x, dtype=dtype_tt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    got = ttnn.to_torch(ttnn.slice(a, [0] * len(shape), ends, [1] * len(shape))).to(dtype_t)
    want = x[tuple(slice(0, e) for e in ends)]
    bad = got != want
    return int(bad.sum()), int(bad.reshape(-1, want.shape[-1]).any(dim=1).sum()), want.numel()


dev = ttnn.open_device(device_id=0)
try:
    for name, shape, ends, tt_, dt in [
        ("f32_6160rows_uneven", (1, 80, 80, 80), [1, 80, 77, 80], torch.float32, ttnn.float32),
        ("ctrl_6080rows_even", (1, 80, 80, 80), [1, 80, 76, 80], torch.float32, ttnn.float32),
        ("ctrl_no_truncation", (1, 80, 80, 80), [1, 80, 80, 80], torch.float32, ttnn.float32),
        ("bf16_6160rows_uneven", (1, 80, 80, 160), [1, 80, 77, 160], torch.bfloat16, ttnn.bfloat16),
        ("f32_3d_1229x5", (1229, 6, 80), [1229, 5, 80], torch.float32, ttnn.float32),
        ("f32_2x40", (2, 40, 80, 80), [2, 40, 77, 80], torch.float32, ttnn.float32),
    ]:
        try:
            bad, rows, tot = run(dev, shape, ends, tt_, dt)
            print(f"RESULT slice_ring {name} {list(shape)}->{ends}: bad={bad}/{tot} bad_rows={rows}", flush=True)
        except Exception as e:
            print(f"RESULT slice_ring {name}: EXC {str(e).strip().splitlines()[0][:200]}", flush=True)
finally:
    ttnn.close_device(dev)
