import torch


def one_key_fit(s, v, dev):
    """s: scaled scores (n,), v: (n, d) float64, dev: device output (d,).
    Model: out = (sum_k p_k v_k + delta * v_j) / (1 + delta) with p = softmax(s). Return best j, delta, residual."""
    p = torch.softmax(s, 0); base = p @ v
    best = None
    for j in range(s.shape[0]):
        # out*(1+d) = base + d v_j  ->  d (out - v_j) = base - out  -> least squares for d
        a = dev - v[j]; b = base - dev
        den = float(a @ a)
        if den == 0:
            continue
        d = float(a @ b) / den
        if d <= -0.999:
            continue
        res = float(((base + d * v[j]) / (1 + d) - dev).abs().max())
        if best is None or res < best[2]:
            best = (j, d, res)
    return best
