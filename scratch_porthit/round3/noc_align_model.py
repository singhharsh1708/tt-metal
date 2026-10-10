"""Address model for NoC read alignment on Wormhole (L1 src: 16, DRAM src: 32).

A read is flagged when (src_addr mod A) != (dst_addr mod A), the watcher rule in
internal/debug/sanitize.h. CB bases and DRAM page bases are 0 mod 32, L1 page bases 0 mod 16.
"""
import itertools

A = {"DRAM": 32, "L1": 16}
NCORES = 64


def align(x, a):
    return (x + a - 1) // a * a


def split(units, ncores=NCORES):
    n = min(units, ncores)
    base, extra = divmod(units, n)
    return [base + 1] * extra + [base] * (n - extra)


def bad_read(src_off, dst_off, buf):
    return (src_off % A[buf]) != (dst_off % A[buf])


# ---- A: reshape_on_device RM (reshape_rm_program_factory.cpp + reader_unary_reshape_stick...) ----
def reshape_on_device(in_shape, out_shape, elem=2, buf="DRAM", ncores=NCORES):
    old, new = in_shape[3] * elem, out_shape[3] * elem
    assert in_shape[3] % 8 == 0 and out_shape[3] % 8 == 0 and max(old, new) % min(old, new) == 0
    num_old = in_shape[0] * in_shape[1] * in_shape[2]
    num_new = out_shape[0] * out_shape[1] * out_shape[2]
    ratio = max(old, new) // min(old, new)
    per_core_old = split(num_old, ncores) if old > new else [n * ratio for n in split(num_new, ncores)]
    bad_sticks = 0
    for n in per_core_old:
        for k in range(n):  # k-th old stick lands at cb_base + k*old, source is a page start
            bad_sticks += bad_read(0, k * old, buf)
    return bad_sticks * in_shape[3], num_old * in_shape[3]


# ---- B1: embedding, weight staging page aligned to the INDEX buffer (embeddings_rm_program_factory.cpp) ----
def embedding_rm(tokens, dim, idx_buf, w_buf, ncores=NCORES):
    chunk = align(dim * 2, A[idx_buf])
    bad = 0
    for n in split(tokens, ncores):
        entries = 2 if n > 1 else 1  # buffering_size
        for r in range(n):
            bad += bad_read(0, (r % entries) * chunk, w_buf)
    return bad * dim, tokens * dim


def embedding_tilized_indices(tokens, dim, idx_buf, w_buf):
    chunk = align(dim * 2, A[idx_buf])  # num_entries = 2 always
    bad = sum(bad_read(0, (r % 2) * chunk, w_buf) for r in range(tokens))
    return bad * dim, tokens * dim  # per-core restart ignored; rows per core are multiples of 16


# ---- B2: embedding BINARY, row 1 cached at zero_local_addr + weight_stick_size (embeddings_common.hpp) ----
def embedding_binary(dim, w_buf):
    stick = dim * 2
    fill_bad = bad_read(0, stick, w_buf)  # weights page 1 -> cache_base + stick
    replay_bad = bad_read(stick, 0, "L1")  # cache_base + stick -> staging page start (L1 loopback)
    return fill_bad, replay_bad


# ---- C: concat RM interleaved (concat_program_factory.cpp + reader_concat_stick_layout...) ----
def concat_rm(widths, bufs, out_buf, rows_per_core, elem=2, width_concat=True, rows=None):
    common = max(A[bufs[0]], A[out_buf])
    if width_concat:
        if any((w * elem) % A[b] for w, b in zip(widths, bufs)):
            return None  # front end takes the transpose path
        page = align(sum(widths) * elem, common)
        bad = 0
        for r in range(rows_per_core):
            cum = 0
            for w, b in zip(widths, bufs):
                bad += w * bad_read(0, (r % 2) * page + cum, b)
                cum += w * elem
        return bad, rows_per_core * sum(widths)
    # concat on an outer dim: rows[j] sticks from tensor j per block, one stick per DFB entry
    w = widths[0]
    page = align(w * elem, common)
    order = [j for j, n in enumerate(rows) for _ in range(n)]
    bad = sum(w * bad_read(0, (r % 2) * page, bufs[order[r % len(order)]]) for r in range(rows_per_core))
    return bad, rows_per_core * w


if __name__ == "__main__":
    print("== A reshape_on_device RM: bad/total (ncores=64)")
    for elem, name in ((2, "bf16"), (4, "f32")):
        for buf in ("DRAM", "L1"):
            hits = []
            for win, wout in itertools.product(range(8, 73, 8), repeat=2):
                if win == wout or max(win, wout) % min(win, wout):
                    continue
                for rows_in in (2, 4, 64, 256):
                    if (rows_in * win) % wout:
                        continue
                    b, t = reshape_on_device((1, 1, rows_in, win), (1, 1, rows_in * win // wout, wout), elem, buf)
                    if b:
                        hits.append((rows_in, win, wout, b, t))
            print(f"  {name} {buf}: {len(hits)} triggering configs; first: {hits[:6]}")
    wins = sorted({w for w in range(8, 257, 8) if reshape_on_device((1, 1, 256, w), (1, 1, 128, 2 * w))[0]})
    print("  bf16 DRAM W_in that trigger (merge 2 rows):", wins[:12], "... i.e. W % 16 == 8:", all(w % 16 == 8 for w in wins))

    print("== B1 embedding RM factory (indices L1, weights DRAM), 128 tokens")
    dims = [d for d in range(1, 97) if embedding_rm(128, d, "L1", "DRAM")[0]]
    print("  dims that trigger:", dims)
    print("  control idx DRAM:", [d for d in range(1, 97) if embedding_rm(128, d, "DRAM", "DRAM")[0]])
    for t in (32, 64, 65, 128, 256):
        print(f"  tokens={t} dim=24:", embedding_rm(t, 24, "L1", "DRAM"))
    print("  tilized indices, 64 tokens dim=24:", embedding_tilized_indices(64, 24, "L1", "DRAM"))

    print("== B2 embedding BINARY (fill_bad, replay_bad)")
    for wb in ("DRAM", "L1"):
        print(f"  weights {wb}:", {d: embedding_binary(d, wb) for d in (4, 8, 12, 16, 20, 24, 32, 40, 48, 64)})

    print("== C concat RM interleaved")
    print("  width [L1 8, DRAM 16] out DRAM :", concat_rm([8, 16], ["L1", "DRAM"], "DRAM", 4))
    print("  width [L1 8, DRAM 16] out L1   :", concat_rm([8, 16], ["L1", "DRAM"], "L1", 4))
    print("  width [DRAM 16, DRAM 16] ctrl  :", concat_rm([16, 16], ["DRAM", "DRAM"], "DRAM", 4))
    print("  width [L1 8, L1 8] ctrl        :", concat_rm([8, 8], ["L1", "L1"], "L1", 4))
    print("  width [DRAM 16, L1 8, DRAM 16] :", concat_rm([16, 8, 16], ["DRAM", "L1", "DRAM"], "DRAM", 4))
    print("  height [L1, DRAM] W=24 out L1  :", concat_rm([24], ["L1", "DRAM"], "L1", 8, width_concat=False, rows=[4, 4]))
    print("  height [L1, DRAM] W=24 out DRAM:", concat_rm([24], ["L1", "DRAM"], "DRAM", 8, width_concat=False, rows=[4, 4]))
    print("  height [DRAM, L1] W=24 out L1  :", concat_rm([24], ["DRAM", "L1"], "L1", 8, width_concat=False, rows=[4, 4]))
    print("  height [L1, DRAM] W=5  out L1  :", concat_rm([5], ["L1", "DRAM"], "L1", 8, width_concat=False, rows=[4, 4]))

