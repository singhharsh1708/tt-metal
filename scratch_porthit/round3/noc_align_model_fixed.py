"""Address model of the host sizing and kernel addressing after patch_noc_align.py.

Ports the patched reshape_on_device RM, embedding (RM, tilized indices, BINARY/PADDED cache) and
concat RM interleaved paths, and checks for every NoC read:
  * source and destination agree modulo the source's read alignment,
  * the bytes land inside the CB/DFB entry (or scratch area) the host allocates,
  * a case that was clean before the patch issues the same reads as before.

    python noc_align_model_fixed.py
"""
import itertools
import sys

ARCH = {"wormhole": {"L1": 16, "DRAM": 32}, "blackhole": {"L1": 16, "DRAM": 64}}
WRITE_ALIGN = 16
NCORES = 64
CB_BASE = 0x1A000  # local CB bases are aligned to the DRAM alignment


def align(x, a):
    return (x + a - 1) // a * a


def split(units, ncores=NCORES):
    n = min(units, ncores)
    base, extra = divmod(units, n)
    return [base + 1] * extra + [base] * (n - extra)


def merge_num_sticks_to_read(num_sticks, stick, max_read):
    total = num_sticks * stick
    out = num_sticks
    cur = stick
    while cur <= max_read:
        if total % cur == 0:
            out = total // cur
        cur += stick
    return out


class Log:
    """Collects reads as (source alignment, dst address, size) and checks them."""

    def __init__(self):
        self.bad = 0
        self.reads = 0
        self.bounced = 0
        self.overrun = 0

    def read(self, src_align, dst, size, lo, hi, src=0):
        self.reads += 1
        if (src ^ dst) & (src_align - 1):
            self.bad += 1
        if dst < lo or dst + size > hi:
            self.overrun += 1

    def span(self, dst, size, lo, hi):
        if dst < lo or dst + size > hi:
            self.overrun += 1


# ---------------------------------------------------------------------------------------------
# 1. reshape_on_device RM
# ---------------------------------------------------------------------------------------------
def reshape_host(num_old, num_new, old, new, A_src):
    split_by_old = old > new
    per_core = split(num_old if split_by_old else num_new)
    ratio = max(old, new) // min(old, new)
    num_pages = max(per_core)
    cb_total = num_pages * max(old, new)
    cores = []
    max_rpb = 1
    for n in per_core:
        if split_by_old:
            n_old = n
            n_read = merge_num_sticks_to_read(n_old, old, 2048)
            rpb = n_old // n_read
            push = rpb * ratio
        else:
            n_new = n
            n_old = n_new * ratio
            n_new_read = merge_num_sticks_to_read(n_new, new, 2048)
            push = n_new // n_new_read
            rpb = push * ratio
            n_read = n_old // rpb
        max_rpb = max(max_rpb, rpb)
        cores.append((n_read, rpb, push))
    stride = align(old, A_src)
    return cb_total, max_rpb * stride, stride, cores


def reshape_fixed(num_old, num_new, old, new, buf, arch, patched=True):
    A = ARCH[arch][buf]
    cb_total, scratch_total, stride, cores = reshape_host(num_old, num_new, old, new, A)
    scratch_base = CB_BASE + align(cb_total, ARCH[arch]["DRAM"])
    log = Log()
    direct = []
    for n_read, rpb, push in cores:
        wr = CB_BASE
        for _ in range(n_read):
            for i in range(rpb):
                dst = wr + i * old
                if not patched or (dst & (A - 1)) == 0:
                    log.read(A, dst, old, CB_BASE, CB_BASE + cb_total)
                    direct.append((dst - CB_BASE, old))
                else:
                    slot = scratch_base + i * stride
                    log.read(A, slot, old, scratch_base, scratch_base + scratch_total)
                    log.span(dst, old, CB_BASE, CB_BASE + cb_total)
                    log.bounced += 1
            wr += push * new
        if wr > CB_BASE + cb_total:
            log.overrun += 1
    return log, direct


# ---------------------------------------------------------------------------------------------
# 2. embedding
# ---------------------------------------------------------------------------------------------
def embedding_rm_fixed(tokens, page, idx_buf, w_buf, arch, patched=True, sharded_out=False):
    Aw = ARCH[arch][w_buf]
    a = ARCH[arch][idx_buf]
    staging = max(a, Aw) if (patched and not sharded_out) else a
    chunk = align(page, staging)
    log = Log()
    reads = []
    for n in split(tokens):
        entries = 2 if n > 1 else 1
        total = entries * chunk
        for r in range(n):
            dst = CB_BASE + (r % entries) * chunk
            log.read(Aw, dst, chunk, CB_BASE, CB_BASE + total)  # reader reads the rounded size
            if dst % WRITE_ALIGN:
                log.bad += 1
            reads.append(((r % entries), chunk))
    return log, reads, chunk


def embedding_tilized_fixed(tokens, page, idx_buf, w_buf, arch, patched=True):
    Aw = ARCH[arch][w_buf]
    a = ARCH[arch][idx_buf]
    entry = align(page, max(a, Aw) if patched else a)
    log = Log()
    for r in range(tokens):
        dst = CB_BASE + (r % 2) * entry
        log.read(Aw, dst, page, CB_BASE, CB_BASE + 2 * entry)
    return log, entry


def embedding_cache_fixed(page, w_buf, arch, kind="BINARY", patched=True, staging_align=16):
    """Local weight cache: host entry size, kernel row addresses, fill reads and replay reads."""
    Aw = ARCH[arch][w_buf]
    dram = ARCH[arch]["DRAM"]
    entry = align(page, max(32, dram)) if patched else align(page, 32)
    n_entries = 2 if kind == "BINARY" else 1
    log = Log()
    zero = CB_BASE
    log.read(Aw, zero, page, CB_BASE, CB_BASE + n_entries * entry)
    if kind == "BINARY":
        one = zero + (align(page, dram) if patched else page)
        log.read(Aw, one, page, CB_BASE, CB_BASE + n_entries * entry)
        # replay: loopback L1 read from the cached row into a staging entry start
        stage = CB_BASE + 0x4000
        for row in (zero, one):
            for e in range(2):
                log.read(ARCH[arch]["L1"], stage + e * align(page, staging_align), page, 0, 1 << 32, src=row)
    return log


# ---------------------------------------------------------------------------------------------
# 3. concat RM interleaved
# ---------------------------------------------------------------------------------------------
def concat_host(widths, bufs, out_buf, elem, arch, width_concat, patched=True):
    A = ARCH[arch]
    pages = [w * elem for w in widths]
    aligns = [A[b] for b in bufs]
    if patched:
        common = max([A[out_buf]] + aligns)
    else:
        common = max(A[bufs[0]], A[out_buf])
    out_stick = sum(pages) if width_concat else pages[0]
    stick_area = align(out_stick, common)
    entry = stick_area
    if patched and width_concat:
        off = 0
        for p, al in zip(pages, aligns):
            if off % al:
                entry += align(p, common)
            off += p
    return pages, aligns, common, stick_area, entry


def concat_fixed(widths, bufs, out_buf, elem, arch, width_concat=True, rows=8, patched=True, entries=2):
    pages, aligns, common, scratch_off, entry = concat_host(widths, bufs, out_buf, elem, arch, width_concat, patched)
    log = Log()
    rel = []
    total = entries * entry
    for r in range(rows):
        base = CB_BASE + (r % entries) * entry
        if base % WRITE_ALIGN:
            log.bad += 1
        if width_concat:
            l1 = base
            scratch = base + scratch_off
            placed = []
            has_scratch = entry > scratch_off  # the host passes scratch offset 0 when it allocates none
            for j, (p, al) in enumerate(zip(pages, aligns)):
                if not patched or not has_scratch or (l1 & (al - 1)) == 0:
                    log.read(al, l1, p, base, base + entry)
                    rel.append((j, l1 - base, p))
                    placed.append((l1, p))
                else:
                    log.read(al, scratch, p, base + scratch_off, base + entry)
                    log.bounced += 1
                    scratch += align(p, common)
                l1 += p
            # second pass: memmove of the bounced pieces
            l1 = base
            scratch = base + scratch_off
            for p, al in zip(pages, aligns):
                if patched and has_scratch and (l1 & (al - 1)) != 0:
                    log.span(l1, p, base, base + scratch_off)
                    if scratch + p > base + entry or scratch < base + scratch_off:
                        log.overrun += 1
                    scratch += align(p, common)
                l1 += p
            if l1 - base > scratch_off:
                log.overrun += 1
        else:
            j = (r * 7 + r // 3) % len(bufs)  # any tensor may own any row
            log.read(aligns[j], base, pages[j], base, base + entry)
            rel.append((j, 0, pages[j]))
    if total != entries * entry:
        log.overrun += 1
    return log, rel, entry


# ---------------------------------------------------------------------------------------------
def check(name, log, expect_bounce=None):
    ok = log.bad == 0 and log.overrun == 0
    extra = "" if expect_bounce is None else f" bounced={log.bounced}"
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: reads={log.reads} misaligned={log.bad} overrun={log.overrun}{extra}")
    return ok


def main():
    ok = True
    arch = "wormhole"
    print("== confirmed cases and controls (wormhole), before -> after")

    def rs(name, i, o, elem, buf):
        nonlocal ok
        n_old, n_new = i[0] * i[1] * i[2], o[0] * o[1] * o[2]
        before, _ = reshape_fixed(n_old, n_new, i[3] * elem, o[3] * elem, buf, arch, patched=False)
        after, _ = reshape_fixed(n_old, n_new, i[3] * elem, o[3] * elem, buf, arch)
        print(f"  before {name}: misaligned reads {before.bad}/{before.reads}")
        ok &= check(name, after, True)

    rs("reshape bf16 DRAM [256,8]->[128,16]", (1, 1, 256, 8), (1, 1, 128, 16), 2, "DRAM")
    rs("reshape bf16 DRAM [512,24]->[1536,8]", (1, 1, 512, 24), (1, 1, 1536, 8), 2, "DRAM")
    rs("reshape bf16 DRAM [256,24]->[128,48]", (1, 1, 256, 24), (1, 1, 128, 48), 2, "DRAM")
    rs("reshape ctrl bf16 L1 [256,8]->[128,16]", (1, 1, 256, 8), (1, 1, 128, 16), 2, "L1")
    rs("reshape ctrl bf16 DRAM [256,16]->[128,32]", (1, 1, 256, 16), (1, 1, 128, 32), 2, "DRAM")
    rs("reshape ctrl f32 DRAM [256,8]->[128,16]", (1, 1, 256, 8), (1, 1, 128, 16), 4, "DRAM")

    def em(name, tokens, dim, ib, wb):
        nonlocal ok
        before, _, _ = embedding_rm_fixed(tokens, dim * 2, ib, wb, arch, patched=False)
        after, _, _ = embedding_rm_fixed(tokens, dim * 2, ib, wb, arch)
        print(f"  before {name}: misaligned reads {before.bad}/{before.reads}")
        ok &= check(name, after)

    em("embedding RM idx L1 w DRAM dim24", 512, 24, "L1", "DRAM")
    em("embedding RM idx L1 w DRAM dim8", 512, 8, "L1", "DRAM")
    em("embedding RM idx L1 w DRAM dim20", 512, 20, "L1", "DRAM")
    em("embedding RM ctrl idx DRAM w DRAM dim24", 512, 24, "DRAM", "DRAM")
    em("embedding RM ctrl idx L1 w DRAM dim16", 512, 16, "L1", "DRAM")
    em("embedding RM ctrl idx L1 w L1 dim24", 512, 24, "L1", "L1")
    for ib in ("L1", "DRAM"):
        before, _ = embedding_tilized_fixed(64, 48, ib, "DRAM", arch, patched=False)
        after, _ = embedding_tilized_fixed(64, 48, ib, "DRAM", arch)
        print(f"  before embedding TILE idx {ib} w DRAM dim24: misaligned reads {before.bad}/{before.reads}")
        ok &= check(f"embedding TILE idx {ib} w DRAM dim24", after)
    for dim, wb in ((8, "DRAM"), (24, "DRAM"), (4, "DRAM"), (16, "DRAM"), (8, "L1")):
        before = embedding_cache_fixed(dim * 2, wb, arch, patched=False)
        after = embedding_cache_fixed(dim * 2, wb, arch)
        print(f"  before embedding BINARY w {wb} dim{dim}: misaligned reads {before.bad}/{before.reads}")
        ok &= check(f"embedding BINARY w {wb} dim{dim}", after)

    def cc(name, widths, bufs, out, width_concat):
        nonlocal ok
        before, _, _ = concat_fixed(widths, bufs, out, 2, arch, width_concat, patched=False)
        after, _, _ = concat_fixed(widths, bufs, out, 2, arch, width_concat)
        print(f"  before {name}: misaligned reads {before.bad}/{before.reads}")
        ok &= check(name, after, True)

    cc("concat width [L1 8, DRAM 16] out DRAM", [8, 16], ["L1", "DRAM"], "DRAM", True)
    cc("concat width [DRAM 16, L1 8, DRAM 16] out DRAM", [16, 8, 16], ["DRAM", "L1", "DRAM"], "DRAM", True)
    cc("concat height [L1, DRAM] W=24 out L1", [24, 24], ["L1", "DRAM"], "L1", False)
    cc("concat height [L1, DRAM] W=5 out L1", [5, 5], ["L1", "DRAM"], "L1", False)
    cc("concat ctrl width [DRAM 16, DRAM 16] out DRAM", [16, 16], ["DRAM", "DRAM"], "DRAM", True)
    cc("concat ctrl height [DRAM, DRAM] W=24 out DRAM", [24, 24], ["DRAM", "DRAM"], "DRAM", False)
    cc("concat ctrl height [L1, DRAM] W=24 out DRAM", [24, 24], ["L1", "DRAM"], "DRAM", False)

    print("== sweeps (wormhole and blackhole)")
    for arch in ARCH:
        # reshape: widths 1..64, elem 1/2/4, every divisor pair, several stick counts
        n = bad = over = changed = fixed = 0
        for elem, buf in itertools.product((1, 2, 4), ("L1", "DRAM")):
            for win, wout in itertools.product(range(1, 65), repeat=2):
                if win == wout or max(win, wout) % min(win, wout):
                    continue
                for rows_in in (1, 2, 3, 4, 7, 64, 96, 256, 1000):
                    if (rows_in * win) % wout:
                        continue
                    num_new = rows_in * win // wout
                    b, db = reshape_fixed(rows_in, num_new, win * elem, wout * elem, buf, arch, patched=False)
                    a, da = reshape_fixed(rows_in, num_new, win * elem, wout * elem, buf, arch)
                    n += 1
                    bad += a.bad
                    over += a.overrun
                    fixed += b.bad > 0
                    if b.bad == 0 and (db != da or a.bounced):
                        changed += 1
        print(f"  {arch} reshape: {n} configs, {fixed} were misaligned, after: misaligned={bad} overrun={over} "
              f"clean-configs-with-changed-reads={changed}")
        ok &= bad == 0 and over == 0 and changed == 0

        n = bad = over = changed = fixed = 0
        for elem, ib, wb in itertools.product((1, 2, 4), ("L1", "DRAM"), ("L1", "DRAM")):
            for dim in range(1, 65):
                for tokens in (1, 2, 63, 64, 65, 128, 512):
                    b, rb, cb = embedding_rm_fixed(tokens, dim * elem, ib, wb, arch, patched=False)
                    a, ra, ca = embedding_rm_fixed(tokens, dim * elem, ib, wb, arch)
                    n += 1
                    bad += a.bad
                    over += a.overrun
                    fixed += b.bad > 0
                    if b.bad == 0 and (rb != ra) and tokens > NCORES:
                        changed += 1
                b, eb = embedding_tilized_fixed(96, dim * elem, ib, wb, arch, patched=False)
                a, ea = embedding_tilized_fixed(96, dim * elem, ib, wb, arch)
                n += 1
                bad += a.bad
                over += a.overrun
                fixed += b.bad > 0
                changed += b.bad == 0 and eb != ea
                for kind in ("BINARY", "PADDED"):
                    for st in (16, 32, 64):
                        a = embedding_cache_fixed(dim * elem, wb, arch, kind, staging_align=st)
                        n += 1
                        bad += a.bad
                        over += a.overrun
        print(f"  {arch} embedding (RM, tilized ids, BINARY/PADDED cache): {n} configs, {fixed} were misaligned, "
              f"after: misaligned={bad} overrun={over} clean-configs-with-changed-reads={changed}")
        ok &= bad == 0 and over == 0 and changed == 0

        n = bad = over = changed = fixed = bounced = 0
        mems = ("L1", "DRAM")
        for elem in (1, 2, 4):
            for bufs in itertools.product(mems, repeat=2):
                for out in mems:
                    for w0, w1 in itertools.product(range(1, 65), repeat=2):
                        for entries in (1, 2):
                            b, rb, eb = concat_fixed([w0, w1], bufs, out, elem, arch, True, patched=False, entries=entries)
                            a, ra, ea = concat_fixed([w0, w1], bufs, out, elem, arch, True, entries=entries)
                            n += 1
                            bad += a.bad
                            over += a.overrun
                            fixed += b.bad > 0
                            bounced += a.bounced > 0
                            if b.bad == 0 and entries == 2 and (rb != ra or eb != ea):
                                changed += 1
                    for w in range(1, 65):
                        b, rb, eb = concat_fixed([w, w], bufs, out, elem, arch, False, patched=False)
                        a, ra, ea = concat_fixed([w, w], bufs, out, elem, arch, False)
                        n += 1
                        bad += a.bad
                        over += a.overrun
                        fixed += b.bad > 0
                        if b.bad == 0 and (rb != ra or eb != ea):
                            changed += 1
            for bufs in itertools.product(mems, repeat=3):
                for out in mems:
                    for ws in itertools.product((1, 3, 5, 8, 12, 16, 20, 24, 32, 40, 64), repeat=3):
                        b, rb, eb = concat_fixed(list(ws), bufs, out, elem, arch, True, patched=False)
                        a, ra, ea = concat_fixed(list(ws), bufs, out, elem, arch, True)
                        n += 1
                        bad += a.bad
                        over += a.overrun
                        fixed += b.bad > 0
                        if b.bad == 0 and (rb != ra or eb != ea):
                            changed += 1
        print(f"  {arch} concat (width 2 and 3 inputs, height): {n} configs, {fixed} were misaligned, "
              f"after: misaligned={bad} overrun={over} clean-configs-with-changed-reads={changed}")
        ok &= bad == 0 and over == 0 and changed == 0

    print("ALL OK" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
