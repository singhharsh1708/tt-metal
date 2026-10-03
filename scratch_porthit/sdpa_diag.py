import math, torch

SCALE = 128 ** -0.5
KC = 256


def online(s, v, kc, mode=None, at=None):
    """Chunked online softmax for one row. mode breaks one step at chunk `at`."""
    m = -math.inf; l = 0.0; o = torch.zeros(v.shape[-1], dtype=torch.float64)
    n = s.shape[0]
    for c, st in enumerate(range(0, n, kc)):
        x = s[st:st + kc]; vv = v[st:st + kc]
        mc = max(m, float(x.max()))
        alpha = math.exp(m - mc) if m != -math.inf else 0.0
        if mode == "stale_max" and c == at and m != -math.inf:
            p = torch.exp(x - m); mc = m
        else:
            p = torch.exp(x - mc)
        if mode == "drop" and c == at:
            p = p * 0
        a_o = 1.0 if (mode in ("o_norescale", "both_norescale") and c == at) else alpha
        a_l = 1.0 if (mode in ("l_norescale", "both_norescale") and c == at) else alpha
        l = a_l * l + float(p.sum()); o = a_o * o + p @ vv; m = mc
    return o / l


def diagnose(q, k, v, out, b, h, i, kc=KC):
    r = q.shape[1] // k.shape[1]
    qq = q[b, h, i].double(); kk = k[b, h // r, : i + 1].double(); vv = v[b, h // r, : i + 1].double()
    s = (kk @ qq) * SCALE
    dev = out[b, h, i].double()
    nchunks = (i + kc) // kc
    cand = [("correct", None, None)]
    for c in range(nchunks):
        for mode in ("drop", "o_norescale", "l_norescale", "both_norescale", "stale_max"):
            cand.append((mode, c, None))
    correct = online(s, vv, kc)
    res = [(float((correct - dev).abs().max()), "correct", None)]
    for mode, c, _ in cand[1:]:
        pred = online(s, vv, kc, mode, c)
        if float((pred - correct).abs().max()) < 1e-9:
            continue  # this break is a no-op for this row
        res.append((float((pred - dev).abs().max()), mode, c))
    res.sort(key=lambda x: x[0])
    maxes = [round(float(s[st:st + kc].max()), 3) for st in range(0, i + 1, kc)]
    argmax = int(s.argmax())
    return res[:3], maxes, argmax
