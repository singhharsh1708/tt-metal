"""Rank-1 support probe. Each op runs on a 1D tensor [N] (dim 0) and, as a control, on the same data as
[1, N] (dim 1). GAP = the rank-2 form works and the rank-1 form crashes or is wrong. BOTH = both fail,
which usually means the call itself is wrong or the op has no such form.

Covers forge issues #34066 (sort), #45155 (gather), #43726 (repeat_interleave) plus a wider op list.
"""
import sys

import torch
import ttnn


def to_tt(x, dev, dt=None):
    if dt is None:
        dt = ttnn.float32 if x.dtype.is_floating_point else ttnn.uint32
    if not x.dtype.is_floating_point:
        x = x.to(torch.int32)
    return ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)


def back(t):
    return ttnn.to_torch(ttnn.from_device(t))


def same(got, ref):
    outs = list(got) if isinstance(got, (list, tuple)) else [got]
    refs = list(ref) if isinstance(ref, (list, tuple)) else [ref]
    if len(outs) != len(refs):
        return f"WRONG output count {len(outs)} vs {len(refs)}"
    for o, r in zip(outs, refs):
        o = back(o)
        if list(o.shape) != list(r.shape):
            return f"WRONG shape {list(o.shape)} vs {list(r.shape)}"
        if r.dtype.is_floating_point:
            if not torch.allclose(o.double(), r.double(), rtol=2e-2, atol=2e-2, equal_nan=True):
                return f"WRONG values, max abs diff {float((o.double() - r.double()).abs().max()):.4g}"
        elif not torch.equal(o.to(torch.int64), r.to(torch.int64)):
            return f"WRONG integers, {int((o.to(torch.int64) != r.to(torch.int64)).sum())} of {r.numel()} differ"
    return "OK"


def attempt(fn):
    try:
        return fn()
    except Exception as e:
        lines = [l.strip() for l in str(e).strip().splitlines() if l.strip()]
        info = next((l for l in lines if l.lower().startswith("info")), "")
        k = lines.index(info) + 1 if info and lines.index(info) + 1 < len(lines) else 0
        return f"EXC {type(e).__name__}: {(lines[0] if lines else '')[:150]} {lines[k][:150] if k else ''}"


def ops(n):
    """name -> (tt(t, d, aux), ref(x, d, aux)); aux holds host tensors built per shape."""
    k = min(5, n)
    return {
        "sort": (lambda t, d, a: ttnn.sort(t, dim=d, descending=True), lambda x, d, a: tuple(torch.sort(x, dim=d, descending=True))),
        "gather": (lambda t, d, a: ttnn.gather(t, dim=d, index=a["idx_tt"]), lambda x, d, a: torch.gather(x, d, a["idx"])),
        "repeat_interleave": (lambda t, d, a: ttnn.repeat_interleave(t, repeats=3, dim=d), lambda x, d, a: torch.repeat_interleave(x, 3, dim=d)),
        "topk": (lambda t, d, a: ttnn.topk(t, k=k, dim=d, largest=True, sorted=True), lambda x, d, a: tuple(torch.topk(x, k, dim=d))),
        "argmax": (lambda t, d, a: ttnn.argmax(t, dim=d), lambda x, d, a: torch.argmax(x, dim=d)),
        "max": (lambda t, d, a: ttnn.max(t, dim=d), lambda x, d, a: x.amax(d)),
        "min": (lambda t, d, a: ttnn.min(t, dim=d), lambda x, d, a: x.amin(d)),
        "sum": (lambda t, d, a: ttnn.sum(t, dim=d), lambda x, d, a: x.sum(d)),
        "mean": (lambda t, d, a: ttnn.mean(t, dim=d), lambda x, d, a: x.mean(d)),
        "std": (lambda t, d, a: ttnn.std(t, dim=d), lambda x, d, a: x.std(d, correction=0)),
        "cumsum": (lambda t, d, a: ttnn.cumsum(t, dim=d), lambda x, d, a: torch.cumsum(x, d)),
        "softmax": (lambda t, d, a: ttnn.softmax(t, dim=d), lambda x, d, a: torch.softmax(x, d)),
        "concat": (lambda t, d, a: ttnn.concat([t, t], dim=d), lambda x, d, a: torch.cat([x, x], d)),
        "stack": (lambda t, d, a: ttnn.stack([t, t], dim=0), lambda x, d, a: torch.stack([x, x], 0)),
        "repeat": (
            lambda t, d, a: ttnn.repeat(t, ttnn.Shape([2] if d == 0 else [1, 2])),
            lambda x, d, a: x.repeat(2) if d == 0 else x.repeat(1, 2),
        ),
        "roll": (lambda t, d, a: ttnn.roll(t, 3, d), lambda x, d, a: torch.roll(x, 3, d)),
        "flip": (lambda t, d, a: ttnn.flip(t, [d]), lambda x, d, a: torch.flip(x, [d])),
        "slice": (
            lambda t, d, a: ttnn.slice(t, [1], [n - 1]) if d == 0 else ttnn.slice(t, [0, 1], [1, n - 1]),
            lambda x, d, a: x[1 : n - 1] if d == 0 else x[:, 1 : n - 1],
        ),
        "pad": (
            lambda t, d, a: ttnn.pad(t, [(1, 2)] if d == 0 else [(0, 0), (1, 2)], 0.0),
            lambda x, d, a: torch.nn.functional.pad(x, (1, 2)),
        ),
        "unsqueeze": (lambda t, d, a: ttnn.unsqueeze(t, 0), lambda x, d, a: x.unsqueeze(0)),
        "reshape": (lambda t, d, a: ttnn.reshape(t, (n, 1)), lambda x, d, a: x.reshape(n, 1)),
        "where": (lambda t, d, a: ttnn.where(ttnn.gt(t, 0.0), t, ttnn.neg(t)), lambda x, d, a: x.abs()),
        "add": (lambda t, d, a: ttnn.add(t, t), lambda x, d, a: x + x),
        "mul_scalar": (lambda t, d, a: ttnn.multiply(t, 0.5), lambda x, d, a: x * 0.5),
        "relu": (lambda t, d, a: ttnn.relu(t), lambda x, d, a: torch.relu(x)),
        "typecast": (lambda t, d, a: ttnn.typecast(t, ttnn.int32), lambda x, d, a: x.to(torch.int32)),
        "embedding": (lambda t, d, a: ttnn.embedding(a["idx_tt"], a["emb_tt"]), lambda x, d, a: a["emb"][a["idx"]]),
        "scatter": (lambda t, d, a: ttnn.scatter(t, d, a["idx_tt"], a["src_tt"]), lambda x, d, a: torch.scatter(x, d, a["idx"], a["src"])),
    }


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        for n in (3, 33, 100, 1000):
            torch.manual_seed(n)
            x1 = (torch.randperm(n) - n // 2).float()
            m = max(1, n // 2)
            idx1 = torch.randperm(n)[:m]
            emb = torch.arange(n * 32).reshape(n, 32).float() / 8
            for name, (tt, ref) in ops(n).items():
                res = {}
                for rank in (1, 2):
                    x = x1 if rank == 1 else x1[None]
                    idx = idx1 if rank == 1 else idx1[None]
                    d = rank - 1
                    print(f"CASE {name} n={n} rank={rank}", flush=True)

                    def run():
                        aux = {"idx": idx, "idx_tt": to_tt(idx, dev), "emb": emb, "emb_tt": to_tt(emb, dev)}
                        aux["src"] = torch.full(idx.shape, 7.0)
                        aux["src_tt"] = to_tt(aux["src"], dev)
                        return same(tt(to_tt(x, dev), d, aux), ref(x, d, aux))

                    res[rank] = attempt(run)
                tag = "OK" if res[1] == "OK" else ("GAP" if res[2] == "OK" else "BOTH")
                print(f"{tag} {name} n={n} | rank1: {res[1]} | rank2: {res[2]}", flush=True)
    finally:
        ttnn.close_device(dev)
    print("PROBE finished", flush=True)


if __name__ == "__main__":
    sys.exit(main())
