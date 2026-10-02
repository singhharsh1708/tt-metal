# Dense-domain differential sweep of eltwise and reduction ops: `run OUT.pt` on each build, then
# `report STABLE.pt MAIN.pt`. Goldens are torch float64 on the inputs read back from the device.
import os, sys, torch
import torch.nn.functional as F

MODE = sys.argv[1]
INF, NAN = float("inf"), float("nan")
sp = torch.special
b2f = lambda f: (lambda *a: f(*a).double())

UNARY = {
    "abs": torch.abs, "acos": torch.acos, "acosh": torch.acosh, "asin": torch.asin, "asinh": torch.asinh,
    "atan": torch.atan, "atanh": torch.atanh, "cbrt": lambda x: torch.sign(x) * x.abs().pow(1 / 3),
    "ceil": torch.ceil, "cos": torch.cos, "cosh": torch.cosh, "deg2rad": torch.deg2rad, "digamma": sp.digamma,
    "erf": torch.erf, "erfc": torch.erfc, "erfinv": torch.erfinv, "exp": torch.exp, "exp2": torch.exp2,
    "expm1": torch.expm1, "floor": torch.floor, "frac": torch.frac, "gelu": F.gelu, "hardsigmoid": F.hardsigmoid,
    "hardswish": F.hardswish, "hardtanh": F.hardtanh, "i0": sp.i0, "i1": sp.i1, "isfinite": b2f(torch.isfinite),
    "isinf": b2f(torch.isinf), "isnan": b2f(torch.isnan), "isneginf": b2f(torch.isneginf),
    "isposinf": b2f(torch.isposinf), "lgamma": torch.lgamma, "log": torch.log, "log10": torch.log10,
    "log1p": torch.log1p, "log2": torch.log2, "log_sigmoid": F.logsigmoid, "logical_not": lambda x: (x == 0).double(),
    "mish": F.mish, "neg": torch.neg, "rad2deg": torch.rad2deg, "reciprocal": torch.reciprocal, "relu": torch.relu,
    "relu6": F.relu6, "round": torch.round, "rsqrt": torch.rsqrt, "selu": F.selu, "sigmoid": torch.sigmoid,
    "sign": torch.sign, "signbit": b2f(torch.signbit), "silu": F.silu, "sin": torch.sin, "sinh": torch.sinh,
    "softplus": F.softplus, "softsign": F.softsign, "sqrt": torch.sqrt, "square": torch.square, "tan": torch.tan,
    "tanh": torch.tanh, "tanhshrink": F.tanhshrink, "trunc": torch.trunc, "nez": lambda x: (x != 0).double(),
    "eqz": lambda x: (x == 0).double(), "gtz": lambda x: (x > 0).double(), "ltz": lambda x: (x < 0).double(),
    "gez": lambda x: (x >= 0).double(), "lez": lambda x: (x <= 0).double(), "identity": lambda x: x,
    "multigammaln": lambda x: sp.multigammaln(x, 4), "logit": torch.logit,
    "elu": lambda x: F.elu(x, 1.0), "celu": lambda x: F.celu(x, 1.0), "leaky_relu": lambda x: F.leaky_relu(x, 0.01),
    "hardshrink": lambda x: F.hardshrink(x, 0.5), "softshrink": lambda x: F.softshrink(x, 0.5),
    "polygamma": lambda x: torch.polygamma(2, x), "clip": lambda x: torch.clamp(x, -3.0, 4.0),
    "pow_2.5": lambda x: torch.pow(x, 2.5), "pow_0.5": lambda x: torch.pow(x, 0.5), "pow_3": lambda x: torch.pow(x, 3.0),
    "pow_-1": lambda x: torch.pow(x, -1.0), "pow_2": lambda x: torch.pow(x, 2.0),
}
BINARY = {
    "add": torch.add, "subtract": torch.sub, "multiply": torch.mul, "divide": torch.div, "pow": torch.pow,
    "atan2": torch.atan2, "hypot": torch.hypot, "fmod": torch.fmod, "remainder": torch.remainder,
    "maximum": torch.maximum, "minimum": torch.minimum, "logaddexp": torch.logaddexp, "logaddexp2": torch.logaddexp2,
    "xlogy": torch.xlogy, "squared_difference": lambda a, b: (a - b) ** 2, "ldexp": lambda a, b: a * torch.pow(2.0, b),
    "floor_div": lambda a, b: torch.div(a, b, rounding_mode="floor"),
    "div_no_nan": lambda a, b: torch.where(b == 0, torch.zeros_like(a), a / b),
    "gt": b2f(torch.gt), "lt": b2f(torch.lt), "ge": b2f(torch.ge), "le": b2f(torch.le), "eq": b2f(torch.eq),
    "ne": b2f(torch.ne), "logical_and": lambda a, b: ((a != 0) & (b != 0)).double(),
    "logical_or": lambda a, b: ((a != 0) | (b != 0)).double(), "logical_xor": lambda a, b: ((a != 0) ^ (b != 0)).double(),
    "addalpha": lambda a, b: a + 1.5 * b, "subalpha": lambda a, b: a - 1.5 * b, "bias_gelu": lambda a, b: F.gelu(a + b),
}
TERNARY = {
    "addcmul": lambda a, b, c: a + 0.5 * b * c, "addcdiv": lambda a, b, c: a + 0.5 * b / c,
    "lerp": lambda a, b, c: a + c * (b - a), "mac": lambda a, b, c: a * b + c,
    "where": lambda a, b, c: torch.where(a != 0, b, c),
}
REDUCE = {
    "sum": lambda x: x.sum(-1, keepdim=True), "mean": lambda x: x.mean(-1, keepdim=True),
    "prod": lambda x: x.prod(-1, keepdim=True), "max": lambda x: x.amax(-1, keepdim=True),
    "min": lambda x: x.amin(-1, keepdim=True), "std": lambda x: x.std(-1, keepdim=True),
    "var": lambda x: x.var(-1, keepdim=True), "cumsum": lambda x: x.cumsum(-1), "cumprod": lambda x: x.cumprod(-1),
    "softmax": lambda x: torch.softmax(x, -1), "log_softmax": lambda x: torch.log_softmax(x, -1),
    "logsumexp": lambda x: torch.logsumexp(x, -1, keepdim=True),
    "layer_norm": lambda x: F.layer_norm(x, (x.shape[-1],), eps=1e-5),
    "rms_norm": lambda x: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5),
    "topk": lambda x: torch.topk(x, 4, dim=-1).values, "sort": lambda x: torch.sort(x, dim=-1).values,
}
GROUPS = {"u": UNARY, "b": BINARY, "t": TERNARY, "r": REDUCE}


def unary_grid():
    torch.manual_seed(1)
    logs = torch.pow(10.0, torch.linspace(-37, 38, 1200, dtype=torch.float64))
    ints = torch.arange(-128, 129, dtype=torch.float64)
    k = torch.arange(1, 24, dtype=torch.float64)
    parts = [torch.linspace(-12, 12, 3000, dtype=torch.float64), logs, -logs, ints, ints + 0.5,
             1 + torch.pow(2.0, -k), 1 - torch.pow(2.0, -k), torch.arange(-20, 21, dtype=torch.float64) * (torch.pi / 2),
             torch.tensor([0.0, -0.0, INF, -INF, NAN, 3.4028234e38, -3.4028234e38, 1.1755e-38, -1.1755e-38], dtype=torch.float64)]
    v = torch.cat(parts)
    v = torch.cat([v, torch.randn(8192 - v.numel(), dtype=torch.float64) * 3])
    return v.float().reshape(1, 1, 64, 128)


def nary_grid(n):
    torch.manual_seed(2)
    if n == 2:
        base = [0.0, 1, 0.5, 2, 3, 7.5, 0.1, 10, 100, 1e-3, 1e3, 1e-10, 1e10, 1e-30, 1e30, 3e38, INF, torch.pi, 2.718281828,
                1.5, 4, 0.25, 1e-5, 1e5, 20.5, 33, 1e-20, 1e20, 6, 0.9, 1.1, 255, 256, 12345.678, 77.7, 0.75, 9, 1e-38, 16, 5, 0.3, 1e15, 63, 1e-7]
        vals = torch.tensor(base + [-x for x in base] + [NAN], dtype=torch.float64)
    else:
        vals = torch.tensor([0.0, 1, -1, 0.5, -0.5, 2, -2, 3.5, -3.5, 10, -10, 1e-3, -1e3, 0.25, INF, -INF, NAN, 1e30, -1e30, 7], dtype=torch.float64)
    grids = torch.meshgrid(*([vals] * n), indexing="ij")
    cols = [g.reshape(-1) for g in grids]
    pad = 8192 - cols[0].numel()
    return [torch.cat([c[:8192], torch.randn(max(pad, 0), dtype=torch.float64) * 3]).float().reshape(1, 1, 64, 128) for c in cols]


def reduce_rows():
    torch.manual_seed(3)
    r = torch.randn(32, 64, dtype=torch.float64)
    cls = torch.zeros(32, dtype=torch.long)
    for i, s in enumerate([1, 1e-3, 1e3, 1e10, 1e-10, 100, 0.01, 30]):
        r[i] *= s
    r[8, 5] = INF; r[9, 40] = -INF; r[10, 3] = INF; r[10, 50] = -INF; r[11, 7] = NAN; r[19] = -INF; r[25, 63] = INF; r[26, 0] = INF
    cls[[8, 9, 10, 11, 19, 25, 26]] = 2
    r[12] = 3e38; r[13, 0] = 1e30; r[13, 1] = -1e30; r[18] *= 1e-30; r[24] *= 1e37
    cls[[12, 13, 18, 24]] = 1
    r[14] = 0; r[15] = 5.0; r[16] += 1000; r[17] -= 1000; r[20] = 0; r[20, 9] = 100
    r[21] = torch.arange(1, 65, dtype=torch.float64); r[22] = torch.tensor([1.0, -1.0] * 32, dtype=torch.float64)
    r[23] -= 5; r[27] = r[27].abs() + 0.5
    return r.float().reshape(1, 1, 32, 64), cls


def classes(group, xs):
    if group == "r":
        return None
    spec = torch.zeros_like(xs[0], dtype=torch.bool); ext = torch.zeros_like(spec)
    lim = 1e18 if group == "u" else 1e11
    for x in xs:
        a = x.double().abs()
        spec |= ~torch.isfinite(a)
        ext |= torch.isfinite(a) & (a != 0) & ((a > lim) | (a < 1 / lim))
    return torch.where(spec, 2, torch.where(ext, 1, 0))


def to_dtype(g, dt):
    g = g.double().float()
    return (g.to(torch.bfloat16) if dt == "bf16" else g).double()


def spacing(g, dt):
    a = g.abs()
    if dt == "fp32":
        a32 = a.float()
        s = (torch.nextafter(a32, torch.tensor(INF)) - a32).double()
    else:
        s = torch.pow(2.0, torch.floor(torch.log2(a.clamp(min=1e-45))) - 7)
    return torch.where(torch.isfinite(s), s, torch.full_like(s, 2.0**104)).clamp(min=1.1755e-38)


def run():
    import ttnn

    out_path = sys.argv[2]
    cur = out_path + ".cur"
    data = torch.load(out_path) if os.path.exists(out_path) else {"inputs": {}, "out": {}}
    if os.path.exists(cur):
        data["out"].setdefault(open(cur).read().strip(), "crash: process died or hung here")
        os.remove(cur)
    DT = {"fp32": ttnn.float32, "bf16": ttnn.bfloat16}
    state = {"dev": ttnn.open_device(device_id=0)}

    def put(x, dt):
        try:
            return ttnn.from_torch(x, dtype=DT[dt], layout=ttnn.TILE_LAYOUT, device=state["dev"], preserve_nan_values=True)
        except TypeError:
            return ttnn.from_torch(x, dtype=DT[dt], layout=ttnn.TILE_LAYOUT, device=state["dev"])

    UC = {
        "elu": lambda t: ttnn.elu(t, alpha=1.0), "celu": lambda t: ttnn.celu(t, alpha=1.0),
        "leaky_relu": lambda t: ttnn.leaky_relu(t, negative_slope=0.01), "hardshrink": lambda t: ttnn.hardshrink(t, lambd=0.5),
        "softshrink": lambda t: ttnn.softshrink(t, lambd=0.5), "polygamma": lambda t: ttnn.polygamma(t, 2),
        "clip": lambda t: ttnn.clip(t, -3.0, 4.0), "pow_2.5": lambda t: ttnn.pow(t, 2.5), "pow_0.5": lambda t: ttnn.pow(t, 0.5),
        "pow_3": lambda t: ttnn.pow(t, 3.0), "pow_-1": lambda t: ttnn.pow(t, -1.0), "pow_2": lambda t: ttnn.pow(t, 2.0),
        "addalpha": lambda a, b: ttnn.addalpha(a, b, 1.5), "subalpha": lambda a, b: ttnn.subalpha(a, b, 1.5),
        "addcmul": lambda a, b, c: ttnn.addcmul(a, b, c, value=0.5), "addcdiv": lambda a, b, c: ttnn.addcdiv(a, b, c, value=0.5),
        "cumsum": lambda t: ttnn.cumsum(t, dim=-1), "cumprod": lambda t: ttnn.cumprod(t, dim=-1),
        "softmax": lambda t: ttnn.softmax(t, dim=-1), "log_softmax": lambda t: ttnn.log_softmax(t, dim=-1),
        "layer_norm": lambda t: ttnn.layer_norm(t, epsilon=1e-5), "rms_norm": lambda t: ttnn.rms_norm(t, epsilon=1e-5),
        "topk": lambda t: ttnn.topk(t, k=4, dim=-1, largest=True, sorted=True)[0], "sort": lambda t: ttnn.sort(t, dim=-1)[0],
    }
    for n in ("sum", "mean", "prod", "max", "min", "std", "var", "logsumexp"):
        UC[n] = lambda t, n=n: getattr(ttnn, n)(t, dim=-1, keepdim=True)
    host = {"u": [unary_grid()], "b": nary_grid(2), "t": nary_grid(3), "r": [reduce_rows()[0]]}
    todo = [(g, n, dt) for g in "ubtr" for n in GROUPS[g] for dt in DT]
    for i, (g, name, dt) in enumerate(todo):
        key = f"{g}|{name}|{dt}"
        if key in data["out"]:
            continue
        open(cur, "w").write(key)
        try:
            ts = [put(x, dt) for x in host[g]]
            if f"{g}|{dt}" not in data["inputs"]:
                data["inputs"][f"{g}|{dt}"] = [ttnn.to_torch(t).float().clone() for t in ts]
            fn = UC.get(name) or getattr(ttnn, name)
            data["out"][key] = ttnn.to_torch(fn(*ts)).float().clone()
        except AttributeError as e:
            data["out"][key] = f"missing: {str(e)[:80]}"
        except Exception as e:
            msg = str(e).strip()
            data["out"][key] = "error: " + (msg.splitlines()[0][-170:] if msg else type(e).__name__)
            try:
                ttnn.close_device(state["dev"])
            except Exception:
                pass
            state["dev"] = ttnn.open_device(device_id=0)
        if i % 15 == 0 or i == len(todo) - 1:
            torch.save(data, out_path)
            print(f"{i}/{len(todo)} {key} {'ok' if torch.is_tensor(data['out'][key]) else data['out'][key][:60]}", flush=True)
    torch.save(data, out_path)
    if os.path.exists(cur):
        os.remove(cur)
    ttnn.close_device(state["dev"])
    print("SWEEP DONE", flush=True)


def metrics(out, gold, cls, dt, valid):
    o = out.double().reshape(gold.shape)
    g = to_dtype(gold, dt)
    cm = (torch.isnan(o) != torch.isnan(g)) | (torch.isinf(o) != torch.isinf(g)) | (torch.isinf(o) & torch.isinf(g) & (torch.sign(o) != torch.sign(g)))
    fin = torch.isfinite(o) & torch.isfinite(g)
    ulp = torch.where(fin, (o - g).abs() / spacing(g, dt), torch.zeros_like(g))
    res = {}
    for c in (0, 1, 2):
        m = (cls == c) & valid
        if int(m.sum()) == 0:
            res[c] = dict(n=0, cls=0, maxulp=0.0, nbad=0, arg=0)
            continue
        u = torch.where(m, ulp, torch.zeros_like(ulp))
        res[c] = dict(n=int(m.sum()), cls=int((cm & m).sum()), maxulp=float(u.max()), nbad=int((u > 8).sum()), arg=int((cm & m).reshape(-1).float().argmax()) if bool((cm & m).any()) else int(u.argmax()))
    score = torch.where(cm, torch.full_like(ulp, 1e30), ulp)
    return res, score, o, g


def fmt(v):
    return f"{float(v):.9g}"


def report():
    S, M = torch.load(sys.argv[2]), torch.load(sys.argv[3])
    CN = {0: "mid", 1: "extreme", 2: "special"}
    rcls = reduce_rows()[1]
    regs, worst, errs, improved = [], [], [], 0
    for key in sorted(set(S["out"]) | set(M["out"])):
        g, name, dt = key.split("|")
        so, mo = S["out"].get(key, "missing"), M["out"].get(key, "missing")
        if not torch.is_tensor(mo):
            errs.append((key, "RAN ON STABLE" if torch.is_tensor(so) else str(so)[:40], str(mo)[:150]))
            continue
        xin_m = [x.double() for x in M["inputs"][f"{g}|{dt}"]]
        try:
            gm = GROUPS[g][name](*xin_m)
        except Exception as e:
            errs.append((key, "golden failed", str(e)[:100])); continue
        if mo.numel() != gm.numel():
            errs.append((key, "shape", f"{tuple(mo.shape)} vs golden {tuple(gm.shape)}")); continue
        if g == "r":
            cls = rcls.view(1, 1, 32, 1).expand(gm.shape)
        else:
            cls = classes(g, xin_m)
        valid = torch.ones_like(gm, dtype=torch.bool)
        ms = None
        if torch.is_tensor(so) and so.numel() == gm.numel():
            xin_s = [x.double() for x in S["inputs"][f"{g}|{dt}"]]
            same = torch.ones_like(xin_m[0], dtype=torch.bool)
            for a, b in zip(xin_s, xin_m):
                same &= (a == b) | (torch.isnan(a) & torch.isnan(b))
            valid = same if g != "r" else same.all(-1, keepdim=True).expand(gm.shape)
            ms, s_score, s_o, _ = metrics(so, GROUPS[g][name](*xin_s), cls, dt, valid)
        mm, m_score, m_o, g_cast = metrics(mo, gm, cls, dt, valid)
        if mm[0]["maxulp"] > 2 or mm[0]["cls"] > 0:
            i = mm[0]["arg"]
            xs = "row" + str(i // gm.shape[-1]) if g == "r" else ",".join(fmt(t.reshape(-1)[i]) for t in xin_m)
            worst.append((mm[0]["maxulp"], mm[0]["cls"], mm[0]["nbad"], key, f"x=({xs}) gold={fmt(g_cast.reshape(-1)[i])} main={fmt(m_o.reshape(-1)[i])}"))
        if ms is None:
            errs.append((key, "no stable baseline", str(so)[:60])); continue
        for c in (0, 1, 2):
            a, b = ms[c], mm[c]
            if b["cls"] < a["cls"] or b["maxulp"] * 3 < a["maxulp"] and a["maxulp"] > 4:
                improved += 1
            if not (b["cls"] > a["cls"] or b["maxulp"] > max(4.0, 3 * a["maxulp"]) or b["nbad"] > 1.5 * a["nbad"] + 8):
                continue
            m = ((cls == c) & valid).reshape(-1)
            delta = torch.where(m, m_score.reshape(-1) - s_score.reshape(-1), torch.full_like(m_score.reshape(-1), -1.0))
            ex = []
            for idx in torch.topk(delta, 2).indices.tolist():
                if delta[idx] <= 0:
                    continue
                if g == "r":
                    x = f"row{idx // gm.shape[-1] if gm.shape[-1] > 1 else idx}"
                else:
                    x = ",".join(fmt(t.reshape(-1)[idx]) for t in xin_m)
                ex.append(f"x=({x}) gold={fmt(g_cast.reshape(-1)[idx])} stable={fmt(s_o.reshape(-1)[idx])} main={fmt(m_o.reshape(-1)[idx])}")
            regs.append((c, key, f"cls {a['cls']}->{b['cls']} maxulp {a['maxulp']:.3g}->{b['maxulp']:.3g} nbad {a['nbad']}->{b['nbad']} (n={b['n']})", ex))
    print(f"ops compared: {len(set(M['out']))}  regressions: {len(regs)}  improved entries: {improved}  main errors/missing: {len(errs)}")
    for c in (0, 1, 2):
        sel = [r for r in regs if r[0] == c]
        print(f"== REGRESSIONS stable -> main, {CN[c]} inputs: {len(sel)}")
        for _, key, line, ex in sel:
            print(f"  R {key} {line}")
            for e in ex:
                print(f"      {e}")
    print("== MAIN, mid-domain inputs, ops over 2 ULP or with nan/inf class mismatches, worst 60")
    for mu, c, nb, key, ex in sorted(worst, key=lambda w: (w[1] > 0, w[0]), reverse=True)[:60]:
        print(f"  W {key} maxulp={mu:.4g} class_mismatch={c} n>8ulp={nb} | {ex}")
    print("== main errors / missing")
    for key, a, b in errs:
        print(f"  E {key} | {a} | {b}")


run() if MODE == "run" else report()
