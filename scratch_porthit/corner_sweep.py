"""Padding-content invariance sweep for the softmax / normalisation family.

Each case runs three times with the implicit tile padding filled with 0, +37 and -37. The logical region of
the output must not depend on the padding, so any difference from the pad=0 run is a LEAK. TT's own tests
fill with -42, which exp() hides. REF (PCC against torch below 0.99) is informational.

  python corner_sweep.py <group>      groups: sm_bf16 sm_fp32 moreh norm reduce moreh_norm
"""
import sys
import traceback

import torch
import torch.nn.functional as F
import ttnn

PADS = (0.0, 37.0, -37.0)
EPS = 1e-5


def maxdiff(a, b):
    if a.shape != b.shape:
        return float("inf"), None
    same = (a == b) | (torch.isnan(a) & torch.isnan(b))
    d = torch.where(same, torch.zeros_like(a), torch.nan_to_num((a - b).abs(), nan=float("inf"), posinf=float("inf")))
    return float(d.max()) if d.numel() else 0.0, d


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    if a.shape != b.shape or not torch.isfinite(a).all():
        return float("nan")
    if a.std() == 0 or b.std() == 0:
        return 1.0 if torch.allclose(a, b, atol=1e-2) else 0.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def box(d):
    idx = (d > 0).nonzero()
    return int(idx.shape[0]), [f"{int(idx[:, k].min())}..{int(idx[:, k].max())}" for k in range(idx.shape[1])]


def back(t):
    if isinstance(t, (list, tuple)):
        t = t[0]
    return ttnn.to_torch(ttnn.from_device(t)).double()


def run_case(dev, label, shape, dt, call, ref, extra=None):
    """call(t, aux) -> device tensor; ref(x64, aux64) -> torch; extra(shape) -> dict of host-side aux tensors."""
    print(f"CASE {label} {list(shape)} {'fp32' if dt == ttnn.float32 else 'bf16'}", flush=True)
    try:
        torch.manual_seed(0)
        cast = (lambda v: v.float()) if dt == ttnn.float32 else (lambda v: v.to(torch.bfloat16).float())
        x = cast(torch.rand(shape) * 4 - 2)
        aux = {k: cast(v) for k, v in (extra(shape) if extra else {}).items()}
        outs = {}
        for p in PADS:
            t = ttnn.fill_implicit_tile_padding(ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev), p)
            aux_t = {
                k: ttnn.fill_implicit_tile_padding(ttnn.from_torch(v, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev), p)
                for k, v in aux.items()
            }
            outs[p] = back(call(t, aux_t))
        base = outs[0.0]
        r = pcc(base, ref(x.double(), {k: v.double() for k, v in aux.items()}))
        leaks, detail = [], ""
        for p in PADS[1:]:
            m, d = maxdiff(outs[p], base)
            leaks.append(m)
            if m > 0 and not detail:
                n, bb = box(d) if d is not None else (-1, ["shape mismatch"])
                detail = f" | pad {p:+.0f}: {n} elements differ, index ranges {bb}, out shape {list(base.shape)}"
        status = "LEAK" if max(leaks) > 0 else ("REF" if not r >= 0.99 else "OK")
        print(f"{status} {label} | leak(+37)={leaks[0]:.3g} leak(-37)={leaks[1]:.3g} pcc_vs_torch={r:.5f}{detail}", flush=True)
    except Exception as e:
        msg = str(e).strip().splitlines()
        print(f"EXC {label} | {type(e).__name__}: {(msg[0] if msg else '')[:260]}", flush=True)
        if "--trace" in sys.argv:
            traceback.print_exc()


SMALL = [(1, 1, 40, 70), (2, 3, 50, 100), (1, 2, 33, 33), (1, 1, 100, 31)]
BIG = [(1, 1, 1500, 40), (1, 1, 40, 1500), (1, 1, 4100, 40), (1, 1, 35, 16010)]


def softmax_group(dev, dt):
    def go(shape, dim, tag="", **kw):
        run_case(
            dev,
            f"ttnn.softmax dim={dim}{tag}",
            shape,
            dt,
            lambda t, a: ttnn.softmax(t, dim=dim, **kw),
            lambda x, a: torch.softmax(x, dim),
        )

    for shape, dims in [(s, (-1, -2, 1, 0)) for s in SMALL] + [(s, (-1, -2)) for s in BIG] + [
        ((3, 65, 47), (-1, -2, 0)),
        ((45, 70), (-1, -2)),
    ]:
        for dim in dims:
            go(shape, dim)
            if dim == -1:
                go(shape, dim, " numeric_stable=False", numeric_stable=False)
    for shape in SMALL[:3]:
        run_case(
            dev, "ttnn.softmax_in_place", shape, dt, lambda t, a: ttnn.softmax_in_place(t), lambda x, a: torch.softmax(x, -1)
        )


def moreh_group(dev):
    ops = {
        "softmax": (lambda: ttnn.operations.moreh.softmax, lambda x, d: torch.softmax(x, d)),
        "softmin": (lambda: ttnn.operations.moreh.softmin, lambda x, d: torch.softmax(-x, d)),
        "logsoftmax": (lambda: ttnn.operations.moreh.logsoftmax, lambda x, d: torch.log_softmax(x, d)),
    }
    for name, (get, ref) in ops.items():
        for shape, dims in [(s, (3, 2, 1, 0)) for s in SMALL[:3]] + [(s, (3, 2)) for s in BIG[:2]]:
            for dim in dims:
                for dt in (ttnn.bfloat16, ttnn.float32) if shape == SMALL[0] else (ttnn.bfloat16,):
                    run_case(
                        dev,
                        f"moreh.{name} dim={dim}",
                        shape,
                        dt,
                        lambda t, a, get=get, dim=dim: get()(t, dim),
                        lambda x, a, ref=ref, dim=dim: ref(x, dim),
                    )


def norm_group(dev):
    wb = lambda shape: {"w": torch.rand(shape[-1]) + 0.5, "b": torch.rand(shape[-1]) - 0.5}
    w_only = lambda shape: {"w": torch.rand(shape[-1]) + 0.5}
    rms = lambda x: x / torch.sqrt((x * x).mean(-1, keepdim=True) + EPS)
    for shape in [(1, 1, 40, 70), (2, 3, 50, 100), (1, 1, 33, 1500), (3, 65, 47)]:
        for dt in (ttnn.bfloat16, ttnn.float32):
            n = shape[-1]
            run_case(
                dev,
                "ttnn.layer_norm",
                shape,
                dt,
                lambda t, a: ttnn.layer_norm(t, epsilon=EPS),
                lambda x, a: F.layer_norm(x, (n,), eps=EPS),
            )
            run_case(
                dev,
                "ttnn.layer_norm weight+bias",
                shape,
                dt,
                lambda t, a: ttnn.layer_norm(t, epsilon=EPS, weight=a["w"], bias=a["b"]),
                lambda x, a: F.layer_norm(x, (n,), a["w"], a["b"], eps=EPS),
                wb,
            )
            run_case(dev, "ttnn.rms_norm", shape, dt, lambda t, a: ttnn.rms_norm(t, epsilon=EPS), lambda x, a: rms(x))
            run_case(
                dev,
                "ttnn.rms_norm weight",
                shape,
                dt,
                lambda t, a: ttnn.rms_norm(t, epsilon=EPS, weight=a["w"]),
                lambda x, a: rms(x) * a["w"],
                w_only,
            )
    for shape in [(2, 3, 40, 70), (1, 4, 33, 33), (2, 35, 50, 45)]:
        for dt in (ttnn.bfloat16, ttnn.float32):
            run_case(
                dev,
                "ttnn.batch_norm training",
                shape,
                dt,
                lambda t, a: ttnn.batch_norm(t, training=True, eps=EPS),
                lambda x, a: F.batch_norm(x, None, None, training=True, eps=EPS),
            )


def reduce_group(dev):
    refs = {
        "max": lambda x, d: x.amax(d, keepdim=True),
        "min": lambda x, d: x.amin(d, keepdim=True),
        "sum": lambda x, d: x.sum(d, keepdim=True),
        "mean": lambda x, d: x.mean(d, keepdim=True),
        "prod": lambda x, d: x.prod(d, keepdim=True),
        "std": lambda x, d: x.std(d, keepdim=True, correction=0),
        "var": lambda x, d: x.var(d, keepdim=True, correction=0),
    }
    for name, ref in refs.items():
        for shape in [(1, 1, 40, 70), (2, 3, 50, 100)]:
            for dim in (-1, -2):
                for dt in (ttnn.bfloat16, ttnn.float32):
                    run_case(
                        dev,
                        f"ttnn.{name} dim={dim}",
                        shape,
                        dt,
                        lambda t, a, name=name, dim=dim: getattr(ttnn, name)(t, dim=dim, keepdim=True),
                        lambda x, a, ref=ref, dim=dim: ref(x, dim),
                    )


def moreh_norm_group(dev):
    for shape in [(2, 3, 40, 70), (1, 4, 33, 100)]:
        for nd in (1, 2, 3):
            run_case(
                dev,
                f"moreh.layer_norm normalized_dims={nd}",
                shape,
                ttnn.bfloat16,
                lambda t, a, nd=nd: ttnn.operations.moreh.layer_norm(t, nd, EPS, None, None),
                lambda x, a, nd=nd, shape=shape: F.layer_norm(x, shape[-nd:], eps=EPS),
            )
    for shape, groups in [((2, 4, 40, 70), 2), ((1, 6, 33, 33), 3)]:
        run_case(
            dev,
            f"moreh.group_norm groups={groups}",
            shape,
            ttnn.bfloat16,
            lambda t, a, groups=groups: ttnn.operations.moreh.group_norm(t, groups, EPS),
            lambda x, a, groups=groups: F.group_norm(x, groups, eps=EPS),
        )


GROUPS = {
    "sm_bf16": lambda dev: softmax_group(dev, ttnn.bfloat16),
    "sm_fp32": lambda dev: softmax_group(dev, ttnn.float32),
    "moreh": moreh_group,
    "norm": norm_group,
    "reduce": reduce_group,
    "moreh_norm": moreh_norm_group,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in GROUPS:
        sys.exit(__doc__)
    device = ttnn.open_device(device_id=0)
    try:
        GROUPS[sys.argv[1]](device)
    finally:
        ttnn.close_device(device)
    print(f"GROUP {sys.argv[1]} finished", flush=True)
