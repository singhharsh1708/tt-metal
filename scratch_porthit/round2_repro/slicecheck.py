"""Exact port of the host-side sizing in slice_program_factory_rm.cpp (tt-metal 6024ed40) for WH."""
from math import prod
MAX_READ = 4096
ALIGN = {"L1": 16, "DRAM": 32}
ESZ = {"bf16": 2, "f32": 4, "u32": 4, "i32": 4, "u16": 2, "u8": 1}

def merge_num_sticks_to_read(n, stick, max_read):
    total = n * stick
    new = n
    cur = stick
    while cur <= max_read:
        if total % cur == 0:
            new = total // cur
        cur += stick
    return new

def rup(x, m): return (x + m - 1) // m * m

def split(R, cores):
    if R == 0: return 0, 0, 0, 0
    t = cores if R >= cores else R
    if R % t == 0: return t, 0, R // t, 0          # g1 cores, g2 cores, rows g1, rows g2
    return R % t, t - R % t, R // t + 1, R // t

def nrpb(rows, stride):
    p = rup(rows, 32)
    return p // merge_num_sticks_to_read(p, stride, MAX_READ)

def straddle(rows, k, entries):
    pos = 0; done = 0
    while done < rows:
        if pos % entries + k > entries: return True
        pos += k; done += k
    return False

def check(in_shape, begins, ends, dtype="bf16", src="DRAM", dst=None, cores=64):
    dst = dst or src
    out = [e - b for b, e in zip(begins, ends)]
    R = prod(out[:-1])
    es = ESZ[dtype]
    al = max(ALIGN[src], ALIGN[dst])
    row = out[-1] * es
    mis = (begins[-1] * es) % ALIGN[src]
    stride = rup(row, al)                              # stride_for_merge / stick_size_offset
    g1c, g2c, r1, r2 = split(R, cores)
    k1 = nrpb(r1, stride) if r1 else 0
    k2 = nrpb(r2, stride) if r2 else 0
    entries = 2 * k1                                   # DFB sized from the larger group
    mid = any(out[i] != in_shape[i] for i in range(1, len(out) - 1))
    strict = mid and g2c > 0 and (R // cores) % 32 == 0 and (R // cores) >= 96
    general = mid and g2c > 0 and k1 != k2
    ring = g2c > 0 and (straddle(r2, k2, entries) or straddle(r1, k1, entries))
    ii = mis != 0 and mid
    return dict(out=out, R=R, cores=cores, g1=(g1c, r1), g2=(g2c, r2), row_bytes=row, stride=stride,
                k1=k1, k2=k2, dfb=entries, mid_trunc=mid, mis=mis, A_i_strict=strict, A_i_general=general,
                ring_straddle=ring, A_ii_candidate=ii)

def show(tag, in_shape, begins, ends, **kw):
    for cores in (64, 56):
        r = check(in_shape, begins, ends, cores=cores, **kw)
        print(f"{tag} | in={in_shape} b={begins} e={ends} {kw} | cores={cores} R={r['R']} g1={r['g1']} g2={r['g2']} "
              f"stride={r['stride']} k1={r['k1']} k2={r['k2']} dfb={r['dfb']} mid={r['mid_trunc']} mis={r['mis']} "
              f"-> strict={r['A_i_strict']} general={r['A_i_general']} ring={r['ring_straddle']} ii={r['A_ii_candidate']}")

if __name__ == "__main__":
    # sanity: user's reference cases
    show("ref97", [1, 6208 // 1, 4, 32][:0] or [1, 100, 70, 32], [0, 0, 0, 0], [1, 97, 64, 32])
    show("ref129", [1, 200, 70, 32], [0, 0, 0, 0], [1, 129, 64, 32])
