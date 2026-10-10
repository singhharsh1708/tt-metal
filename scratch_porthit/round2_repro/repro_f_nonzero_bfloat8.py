# ttnn.nonzero: BFLOAT8_B / BFLOAT4_B pass the element-size check (1 byte) and the tile is scanned as flat bytes.
# ttnn.nonzero returns [count, indices]: count is [1,1,1,8] uint32 (value at [0,0,0,0]),
# indices is [1,1,1,4*volume] uint32 holding (b, n, h, c) per non-zero element.
import traceback

import torch
import ttnn


def first_line(ex):
    text = str(ex).strip() or repr(ex)
    return f"{type(ex).__name__}: {text.splitlines()[0][:200]}"


def onehot_col5():
    x = torch.zeros(1, 1, 1, 32)
    x[0, 0, 0, 5] = 1.0
    return x


def identity32():
    return torch.eye(32).reshape(1, 1, 32, 32)


def row6_ones():
    x = torch.zeros(1, 1, 8, 32)
    x[0, 0, 6, :] = 1.0
    return x


CASES = (
    ("onehot_col5_[1,1,1,32]", onehot_col5),
    ("identity_[1,1,32,32]", identity32),
    ("row6_ones_[1,1,8,32]", row6_ones),
)
DTYPES = (("bfloat16", "bfloat16"), ("bfloat8_b", "bfloat8_b"), ("bfloat4_b", "bfloat4_b"))


def run_case(device, x, dt):
    ref = {tuple(r) for r in torch.nonzero(x).tolist()}
    t = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.nonzero(t)
    cnt_t = ttnn.to_torch(ttnn.from_device(out[0]))
    idx_t = ttnn.to_torch(ttnn.from_device(out[1]))
    cnt = int(cnt_t.flatten()[0].item())
    flat = idx_t.flatten().to(torch.int64)
    n = min(cnt, flat.numel() // 4)
    got = {tuple(r) for r in flat[: 4 * n].reshape(-1, 4).tolist()}
    return len(ref), cnt, len(ref - got), len(got - ref), sorted(got)[:3]


def main():
    device = ttnn.open_device(device_id=0)
    try:
        for cname, make in CASES:
            for name, attr in DTYPES:
                tag = f"RESULT nonzero {cname} dtype={name}:"
                try:
                    x = make()
                    ref_n, cnt, missing, spurious, head = run_case(device, x, getattr(ttnn, attr))
                    print(
                        f"{tag} torch_count={ref_n} ttnn_count={cnt} missing={missing} spurious={spurious} first={head}",
                        flush=True,
                    )
                except Exception as ex:
                    print(f"{tag} REJECTED {first_line(ex)}", flush=True)
    finally:
        ttnn.close_device(device)


# model before the fix (bfloat8_b): onehot 1 vs 1 but at (0,0,0,0); identity 60 reported vs 32;
# row6 2 reported vs 32. bfloat16: all exact.
if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print(f"RESULT nonzero harness: CRASHED {first_line(ex)}", flush=True)
        traceback.print_exc()
        raise
