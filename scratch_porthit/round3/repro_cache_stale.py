#!/usr/bin/env python3
"""Program-cache staleness probes for tt-metal (main 6024ed40f33, 2026-10-10).

Each case runs call A, then call B on the same device (B should hit A's cache entry),
then clears the program cache and runs B alone as the control. Output per case:

  RESULT <name>: second_call_bad=<n>/<total> control_bad=<n>/<total> cache_entries=<k>

second_call_bad counts elements of B-after-A that miss the golden; control_bad the same for
B on a cleared cache. cache_entries is how many entries A+B created (1 means B hit A's entry;
composite ops can add more). Where no torch golden is written (golden=fresh), B-after-A is
compared against B-fresh and control_bad is 0 by construction.

Usage: python repro_cache_stale.py [substring ...]   (no args = all cases)

Static reading found no new stale case; every case except the one tagged KNOWN is expected
to print second_call_bad=0. KNOWN_kv_update_cache_fp32 is issue #57364 item 4 and is here
only to show the harness can detect a stale hit.
"""
import sys
import traceback

import torch

import ttnn

torch.manual_seed(0)
CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn

    return deco


def ints(shape, lo=-8, hi=8):
    return torch.randint(lo, hi, tuple(shape)).float()


def dev(t, device, layout=None, dtype=None, mem=None):
    kw = dict(dtype=dtype or ttnn.bfloat16, layout=layout or ttnn.TILE_LAYOUT, device=device)
    if mem is not None:
        kw["memory_config"] = mem
    return ttnn.from_torch(t, **kw)


def host(t):
    return ttnn.to_torch(t).float()


def nbad(out, gold, rtol=0.03, atol=0.06):
    out = out.float().reshape(-1)
    gold = gold.float().reshape(-1)
    total = max(out.numel(), gold.numel())
    if out.numel() != gold.numel():
        return total, total
    bad = ~torch.isclose(out, gold, rtol=rtol, atol=atol, equal_nan=True)
    return int(bad.sum()), total


def clear_cache(device):
    if hasattr(device, "clear_program_cache"):
        device.clear_program_cache()
    else:
        device.disable_and_clear_program_cache()
        device.enable_program_cache()


def entries(device):
    try:
        return device.num_program_cache_entries()
    except Exception:
        return -1


def run_case(name, fn, device):
    clear_cache(device)
    n0 = entries(device)
    keep = []  # keep A's tensors alive so B lands at different addresses
    fn(device, "A", keep)
    out_hit, gold = fn(device, "B", keep)
    n2 = entries(device)
    keep.clear()
    clear_cache(device)
    out_ctl, gold_ctl = fn(device, "B", keep)
    if gold is None:
        bad, total = nbad(out_hit, out_ctl)
        cbad, ctotal = 0, total
        note = " golden=fresh"
    else:
        bad, total = nbad(out_hit, gold)
        cbad, ctotal = nbad(out_ctl, gold_ctl)
        note = ""
    print(f"RESULT {name}: second_call_bad={bad}/{total} control_bad={cbad}/{ctotal} cache_entries={n2 - n0}{note}")


# ---------------- shape-blind eltwise keys (unary / binary_ng / ternary) ----------------


def _relu(device, shape, keep):
    x = ints(shape)
    tx = dev(x, device)
    keep.append(tx)
    return host(ttnn.relu(tx)), torch.relu(x)


@case("unary_tile_small_then_large")
def _(device, v, keep):
    return _relu(device, [1, 1, 32, 32] if v == "A" else [1, 2, 256, 320], keep)


@case("unary_tile_large_then_small")
def _(device, v, keep):
    return _relu(device, [1, 2, 256, 320] if v == "A" else [1, 1, 32, 32], keep)


@case("unary_tile_rank2_then_rank4")
def _(device, v, keep):
    return _relu(device, [64, 64] if v == "A" else [2, 3, 64, 96], keep)


@case("binary_scalar_shape_and_value")
def _(device, v, keep):
    shape, s = ([1, 1, 32, 32], 1.5) if v == "A" else ([2, 3, 64, 96], 2.5)
    x = ints(shape)
    tx = dev(x, device)
    keep.append(tx)
    return host(ttnn.add(tx, s)), x + s


@case("binary_tt_outer_broadcast")
def _(device, v, keep):
    sa, sb = ([1, 1, 32, 32], [1, 1, 32, 32]) if v == "A" else ([4, 3, 32, 32], [1, 3, 32, 32])
    a, b = ints(sa), ints(sb)
    ta, tb = dev(a, device), dev(b, device)
    keep.extend([ta, tb])
    return host(ttnn.add(ta, tb)), a + b


@case("binary_row_major_width_change")
def _(device, v, keep):
    shape = [1, 1, 8, 64] if v == "A" else [1, 1, 8, 160]
    a, b = ints(shape), ints(shape)
    ta = dev(a, device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tb = dev(b, device, layout=ttnn.ROW_MAJOR_LAYOUT)
    keep.extend([ta, tb])
    return host(ttnn.add(ta, tb)), a + b


@case("binary_inplace_a_then_inplace_b")
def _(device, v, keep):
    a, b = ints([1, 1, 64, 64]), ints([1, 1, 64, 64])
    ta, tb = dev(a, device), dev(b, device)
    keep.extend([ta, tb])
    out = ttnn.add(ta, tb, output_tensor=ta if v == "A" else tb)
    return host(out), a + b


@case("where_ttt_batch_change")
def _(device, v, keep):
    shape = [1, 1, 32, 32] if v == "A" else [2, 4, 32, 32]
    p = (torch.rand(shape) > 0.5).float()
    t, f = ints(shape), ints(shape)
    tp, tt_, tf = dev(p, device), dev(t, device), dev(f, device)
    keep.extend([tp, tt_, tf])
    return host(ttnn.where(tp, tt_, tf)), torch.where(p.bool(), t, f)


@case("where_tts_scalar_value")
def _(device, v, keep):
    s = 1.0 if v == "A" else -7.0
    p = (torch.rand([1, 1, 64, 64]) > 0.5).float()
    t = ints([1, 1, 64, 64])
    tp, tt_ = dev(p, device), dev(t, device)
    keep.extend([tp, tt_])
    return host(ttnn.where(tp, tt_, s)), torch.where(p.bool(), t, torch.full_like(t, s))


@case("addcmul_value")
def _(device, v, keep):
    val = 0.5 if v == "A" else 2.0
    a, b, c = ints([1, 1, 64, 64]), ints([1, 1, 64, 64], -4, 4), ints([1, 1, 64, 64], -4, 4)
    ta, tb, tc = dev(a, device), dev(b, device), dev(c, device)
    keep.extend([ta, tb, tc])
    return host(ttnn.addcmul(ta, tb, tc, value=val)), a + val * b * c


# ---------------- positional tensor binding after an aliased first call ----------------


@case("matmul_alias_then_distinct")
def _(device, v, keep):
    a, b = ints([1, 1, 64, 64], -1, 2), ints([1, 1, 64, 64], -1, 2)
    if v == "A":
        b = a
    ta = dev(a, device)
    tb = ta if v == "A" else dev(b, device)
    keep.extend([ta, tb])
    return host(ttnn.matmul(ta, tb)), a @ b


@case("concat_alias_then_distinct")
def _(device, v, keep):
    a, b = ints([1, 1, 32, 64]), ints([1, 1, 32, 64])
    if v == "A":
        b = a
    ta = dev(a, device)
    tb = ta if v == "A" else dev(b, device)
    keep.extend([ta, tb])
    return host(ttnn.concat([ta, tb], dim=0)), torch.cat([a, b], dim=0)


@case("batch_norm_alias_mean_var_then_distinct")
def _(device, v, keep):
    x = torch.rand([2, 4, 32, 32]) * 4 - 2
    m = torch.rand([1, 4, 1, 1]) + 0.5
    var = m if v == "A" else torch.rand([1, 4, 1, 1]) * 3 + 1.0
    tx, tm = dev(x, device), dev(m, device)
    tv = tm if v == "A" else dev(var, device)
    keep.extend([tx, tm, tv])
    out = ttnn.batch_norm(tx, running_mean=tm, running_var=tv, training=False, eps=1e-5)
    return host(out), (x - m) / torch.sqrt(var + 1e-5)


@case("batch_norm_alias_weight_bias_then_distinct")
def _(device, v, keep):
    x = torch.rand([2, 4, 32, 32]) * 4 - 2
    m, var = torch.rand([1, 4, 1, 1]), torch.rand([1, 4, 1, 1]) + 1.0
    w = torch.rand([1, 4, 1, 1]) + 0.5
    b = w if v == "A" else torch.rand([1, 4, 1, 1]) * 3 - 1.5
    tx, tm, tv, tw = dev(x, device), dev(m, device), dev(var, device), dev(w, device)
    tb = tw if v == "A" else dev(b, device)
    keep.extend([tx, tm, tv, tw, tb])
    out = ttnn.batch_norm(tx, running_mean=tm, running_var=tv, training=False, eps=1e-5, weight=tw, bias=tb)
    return host(out), (x - m) / torch.sqrt(var + 1e-5) * w + b


@case("mul_bw_alias_then_distinct")
def _(device, v, keep):
    g, a, b = ints([1, 1, 64, 64], -2, 3), ints([1, 1, 64, 64], -4, 4), ints([1, 1, 64, 64], -4, 4)
    if v == "A":
        b = a
    tg, ta = dev(g, device), dev(a, device)
    tb = ta if v == "A" else dev(b, device)
    keep.extend([tg, ta, tb])
    outs = ttnn.mul_bw(tg, ta, tb)
    return torch.cat([host(outs[0]), host(outs[1])]), torch.cat([g * b, g * a])


# ---------------- hash-excluded scalars re-applied by an override ----------------


@case("rotary_embedding_token_idx")
def _(device, v, keep):
    idx = 0 if v == "A" else 37
    x = torch.rand([1, 1, 32, 64]) * 2 - 1
    cos, sin = torch.rand([1, 1, 64, 64]) * 2 - 1, torch.rand([1, 1, 64, 64]) * 2 - 1
    tx, tc, ts = dev(x, device), dev(cos, device), dev(sin, device)
    keep.extend([tx, tc, ts])
    out = ttnn.experimental.rotary_embedding(tx, tc, ts, idx)
    rot = torch.cat((-x[..., 32:], x[..., :32]), dim=-1)
    return host(out), x * cos[:, :, idx : idx + 1, :] + rot * sin[:, :, idx : idx + 1, :]


def _update_cache(device, idx, keep, cfg=None):
    cache = torch.zeros([32, 1, 64, 64])
    x = ints([1, 1, 32, 64])
    tc, tx = dev(cache, device), dev(x, device)
    keep.extend([tc, tx])
    kw = {} if cfg is None else {"compute_kernel_config": cfg}
    ttnn.update_cache(tc, tx, idx, **kw)
    gold = cache.clone()
    gold[:, 0, idx, :] = x[0, 0, :, :]
    return host(tc), gold


@case("kv_update_cache_update_idx")
def _(device, v, keep):
    return _update_cache(device, 3 if v == "A" else 41, keep)


@case("KNOWN_kv_update_cache_fp32_issue57364_item4")
def _(device, v, keep):
    cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=(v == "B"))
    return _update_cache(device, 5, keep, cfg)


@case("kv_fill_cache_batch_idx")
def _(device, v, keep):
    b = 0 if v == "A" else 1
    cache = torch.zeros([2, 1, 64, 64])
    x = ints([1, 1, 32, 64])
    tc, tx = dev(cache, device), dev(x, device)
    keep.extend([tc, tx])
    ttnn.fill_cache(tc, tx, b)
    gold = cache.clone()
    gold[b, 0, :32, :] = x[0, 0]
    return host(tc), gold


@case("slice_write_outer_start")
def _(device, v, keep):
    h0 = 0 if v == "A" else 24
    src = ints([1, 8, 16, 32])
    out = torch.zeros([1, 40, 16, 32])
    tin = ttnn.from_torch(src, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
    tout = ttnn.from_torch(out, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
    keep.extend([tin, tout])
    ttnn.experimental.slice_write(tin, tout, [0, h0, 0, 0], [1, h0 + 8, 16, 32], [1, 1, 1, 1])
    gold = out.clone()
    gold[:, h0 : h0 + 8] = src
    return host(tout), gold


@case("uniform_bounds")
def _(device, v, keep):
    lo, hi = (0.0, 1.0) if v == "A" else (5.0, 6.0)
    t = dev(torch.zeros([1, 1, 64, 64]), device, dtype=ttnn.float32)
    keep.append(t)
    ttnn.uniform(t, lo, hi, 1234)
    o = host(t)
    return torch.clamp(o, lo, hi) - o, torch.zeros_like(o)  # non-zero where out of [lo, hi]


@case("rand_bounds")
def _(device, v, keep):
    lo, hi = (0.0, 1.0) if v == "A" else (5.0, 6.0)
    t = ttnn.rand([1, 1, 64, 64], dtype=ttnn.float32, device=device, low=lo, high=hi, seed=1234)
    keep.append(t)
    o = host(t)
    return torch.clamp(o, lo, hi) - o, torch.zeros_like(o)


@case("dropout_seed")
def _(device, v, keep):
    seed = 11 if v == "A" else 4242
    t = dev(torch.ones([1, 1, 64, 64]), device)
    keep.append(t)
    return host(ttnn.experimental.dropout(t, probability=0.5, scale=2.0, seed=seed)), None


@case("bernoulli_seed")
def _(device, v, keep):
    seed = 11 if v == "A" else 4242
    t = dev(torch.full([1, 1, 64, 64], 0.5), device, dtype=ttnn.float32)
    keep.append(t)
    return host(ttnn.bernoulli(t, seed)), None


def main():
    wanted = sys.argv[1:]
    device = ttnn.open_device(device_id=0)
    try:
        if hasattr(device, "enable_program_cache"):
            device.enable_program_cache()
        for name, fn in CASES:
            if wanted and not any(w in name for w in wanted):
                continue
            try:
                run_case(name, fn, device)
            except Exception as e:  # keep going: one bad API guess must not hide the rest
                msg = str(e).strip().splitlines()[0][:200] if str(e).strip() else type(e).__name__
                print(f"RESULT {name}: ERROR {type(e).__name__}: {msg}")
                traceback.print_exc(limit=2)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
