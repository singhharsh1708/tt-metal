"""Candidate probes on the N150, one case per process: python cand_probe.py <case>
slice (#57382), bf8 (#51831), embbw (#45962), gather (#45155, #43869), sdpamask (#39946)."""
import sys

import torch
import ttnn


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    if a.shape != b.shape:
        return float("nan")
    if a.std() == 0 or b.std() == 0:
        return 1.0 if torch.allclose(a, b, atol=1e-2) else 0.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def safe(name, fn):
    try:
        fn()
    except Exception as e:
        lines = [l.strip() for l in str(e).splitlines() if l.strip()]
        print(f"RESULT {name}: EXC {type(e).__name__}: {' | '.join(lines[:3])[:300]}", flush=True)


def slice_case(dev):
    def one(shape, begin, end_mid, dt, layout, name):
        x = torch.arange(shape[0] * shape[1] * shape[2], dtype=torch.float32).reshape(shape)
        if dt == ttnn.bfloat16:
            x = (x % 251).to(torch.bfloat16).float() + torch.arange(shape[0] * shape[1]).reshape(shape[0], shape[1], 1) % 97
            x = x.to(torch.bfloat16).float()
        a = ttnn.from_torch(x, dtype=dt, layout=layout, device=dev)
        got = ttnn.to_torch(ttnn.slice(a, [0, 0, begin], [shape[0], end_mid, shape[2]], [1, 1, 1])).float()
        want = x[:, :end_mid, begin:]
        bad = (got != want).any(dim=-1)
        rows = bad.nonzero().tolist()
        rel = []
        for b, r in rows[:8]:
            src = "?"
            for bb in range(shape[0]):
                for rr in range(shape[1]):
                    if torch.equal(got[b, r], x[bb, rr, begin:]):
                        src = f"in[{bb},{rr}]"
            rel.append(f"out[{b},{r}]<-{src}")
        print(f"RESULT slice {name} shape={list(shape)} begin={begin} mid 0..{end_mid}: bad_rows={len(rows)} "
              f"bad_elems={int((got != want).sum())} {' '.join(rel)}", flush=True)

    f32, bf = ttnn.float32, ttnn.bfloat16
    RM, TL = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
    for shape, begin, em, dt, lay, nm in [
        ((5, 65, 1025), 1, 64, f32, RM, "issue_f32_rm"),
        ((5, 65, 1025), 1, 64, bf, RM, "issue_bf16_rm"),
        ((5, 65, 1025), 1, 64, f32, TL, "issue_f32_tile"),
        ((5, 65, 1025), 0, 64, f32, RM, "ctrl_begin0"),
        ((5, 65, 1025), 1, 65, f32, RM, "ctrl_mid_full"),
        ((5, 65, 1025), 2, 64, f32, RM, "begin2_row4092B"),
        ((5, 65, 1026), 1, 64, f32, RM, "row4100B"),
        ((5, 65, 1041), 1, 64, f32, RM, "row4160B"),
        ((5, 65, 513), 1, 64, f32, RM, "row2048B"),
        ((5, 65, 33), 1, 64, f32, RM, "row128B"),
        ((5, 65, 34), 1, 64, f32, RM, "row132B"),
        ((5, 65, 1025), 16, 64, f32, RM, "begin16_aligned64B"),
        ((3, 65, 1025), 1, 64, f32, RM, "3batches"),
        ((4, 65, 1025), 1, 64, f32, RM, "4batches"),
        ((5, 33, 1025), 1, 32, f32, RM, "mid33to32"),
        ((5, 65, 1025), 1, 60, f32, RM, "mid65to60"),
    ]:
        safe(f"slice {nm}", lambda: one(shape, begin, em, dt, lay, nm))


def bf8_case(dev):
    ops = {"reciprocal": torch.reciprocal, "rsqrt": torch.rsqrt, "log": torch.log, "sqrt": torch.sqrt, "exp": torch.exp}
    for dt, dn in ((ttnn.bfloat8_b, "bf8"), (ttnn.bfloat16, "bf16")):
        for shape in ([1, 64, 7, 7], [1, 2048, 7, 7], [1, 64, 32, 32], [7, 7], [1, 1, 7, 64]):
            torch.manual_seed(0)
            x = torch.randn(shape).abs() + 0.5
            a = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            ident = pcc(ttnn.to_torch(a).float(), x)
            res = []
            for n, f in ops.items():
                try:
                    o = ttnn.to_torch(getattr(ttnn, n)(a)).float()
                    res.append(f"{n} pcc={pcc(o, f(x)):.4f} max={float(o.max()):.3g}")
                except Exception as e:
                    res.append(f"{n} EXC {str(e).strip().splitlines()[0][:60]}")
            try:
                b = ttnn.from_torch(x + 1.0, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
                o = ttnn.to_torch(ttnn.div(a, b)).float()
                res.append(f"div pcc={pcc(o, x / (x + 1.0)):.4f}")
            except Exception as e:
                res.append(f"div EXC {str(e).strip().splitlines()[0][:60]}")
            print(f"RESULT bf8 {dn} {shape} identity={ident:.4f} | " + " | ".join(res), flush=True)


def embbw_case(dev):
    V, H, S = 1024, 64, 32
    g = ttnn.from_torch(torch.randn(1, 1, S, H, dtype=torch.bfloat16), device=dev, layout=ttnn.TILE_LAYOUT)
    ids = ttnn.from_torch(torch.randint(0, V, (1, S), dtype=torch.int32), dtype=ttnn.uint32, device=dev, layout=ttnn.ROW_MAJOR_LAYOUT)
    w = ttnn.from_torch(torch.randn(V, H, dtype=torch.bfloat16), device=dev, layout=ttnn.TILE_LAYOUT)
    out = ttnn.embedding_bw(ids, w, g)
    out = out[0] if isinstance(out, (list, tuple)) else out
    print(f"RESULT embbw grad shape={list(out.shape)} expected=[{V}, {H}] {'OK' if list(out.shape) == [V, H] else 'WRONG_SHAPE'}", flush=True)
    s = ttnn.sum(out, dim=[0, 1], keepdim=False)
    print(f"RESULT embbw sum(dim=[0,1]) shape={list(s.shape)} (a scalar is expected)", flush=True)


def gather_case(dev):
    torch.manual_seed(0)
    x = torch.randn(8, 64)
    idx = torch.randint(0, 64, (8, 32))
    ref = torch.gather(x, 1, idx)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    for name, dt, tdt in (("uint32", ttnn.uint32, torch.int32), ("uint16", ttnn.uint16, torch.int32), ("int32", ttnn.int32, torch.int32)):
        def go():
            ti = ttnn.from_torch(idx.to(tdt), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            o = ttnn.to_torch(ttnn.gather(tx, dim=1, index=ti)).float()
            print(f"RESULT gather rank2 index={name}: ran, pcc={pcc(o, ref.to(torch.bfloat16).float()):.4f}", flush=True)
        safe(f"gather rank2 index={name}", go)
    def r1():
        x1 = torch.randn(64); i1 = torch.randint(0, 64, (32,))
        o = ttnn.to_torch(ttnn.gather(ttnn.from_torch(x1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev), dim=0,
                                      index=ttnn.from_torch(i1.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=dev)))
        print(f"RESULT gather rank1: ran, shape={list(o.shape)}", flush=True)
    safe("gather rank1", r1)


def sdpamask_case(dev):
    torch.manual_seed(0)
    b, nh, s, d = 1, 32, 128, 64
    Q, K, V = torch.randn(1, b, nh, d), torch.randn(b, nh, s, d), torch.randn(b, nh, s, d)
    m1 = torch.zeros(b, 1, 1, s); m1[..., 40:] = -1e9
    put = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    ref = torch.softmax((Q.transpose(0, 1).transpose(1, 2) @ K.transpose(-1, -2)) * d ** -0.5 + m1, -1) @ V
    ref = ref.transpose(1, 2).transpose(0, 1)
    for name, m in (("explicit_heads", m1.expand(b, 1, nh, s).contiguous()), ("one_head_broadcast", m1)):
        def go():
            o = ttnn.to_torch(ttnn.transformer.scaled_dot_product_attention_decode(
                put(Q), put(K), put(V), is_causal=False, attn_mask=put(m), scale=d ** -0.5)).float()
            print(f"RESULT sdpamask {name} mask={list(m.shape)}: out={list(o.shape)} pcc={pcc(o[..., :nh, :], ref[..., :nh, :]):.4f}", flush=True)
        safe(f"sdpamask {name}", go)


CASES = {"slice": slice_case, "bf8": bf8_case, "embbw": embbw_case, "gather": gather_case, "sdpamask": sdpamask_case}
if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0)
    try:
        CASES[sys.argv[1]](dev)
    finally:
        ttnn.close_device(dev)
    print(f"CASE {sys.argv[1]} finished", flush=True)
