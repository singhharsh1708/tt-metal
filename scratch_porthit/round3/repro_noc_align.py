"""NoC read alignment candidates (Wormhole: L1 source 16 B, DRAM source 32 B).

    python repro_noc_align.py                 # every case
    python repro_noc_align.py embedding       # cases whose name contains a substring
    python repro_noc_align.py reshape concat

One line per case:  RESULT <name>: bad=<n>/<total>
Cases named *_ctrl are the aligned control for the op and are expected to print bad=0.
The MODEL line is the count predicted by the address model for a 64-core grid.
"""
import sys

import torch
import ttnn

RM = ttnn.ROW_MAJOR_LAYOUT
DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG

CASES = []


def case(name, model=""):
    def wrap(fn):
        CASES.append((name, model, fn))
        return fn

    return wrap


def rand_bf16(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g).to(torch.bfloat16)


def to_dev(dev, t, dtype, mem, layout=RM):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=dev, memory_config=mem)


def count_bad(got, want):
    got = got.to(torch.float32).reshape(-1)
    want = want.to(torch.float32).reshape(-1)
    if got.numel() != want.numel():
        raise RuntimeError(f"size mismatch: got {got.numel()} want {want.numel()}")
    return int((got != want).sum()), want.numel()


def report(name, got, want):
    bad, total = count_bad(got, want)
    print(f"RESULT {name}: bad={bad}/{total}", flush=True)


# ---------------------------------------------------------------------------------------------
# reshape_on_device, ROW_MAJOR: old sticks are packed at k * old_stick_bytes in one CB
# ---------------------------------------------------------------------------------------------
def _reshape_on_device(dev, name, in_shape, out_shape, dtype, torch_dtype, mem):
    x = torch.randn(in_shape, generator=torch.Generator().manual_seed(1)).to(torch_dtype)
    a = to_dev(dev, x, dtype, mem)
    y = ttnn.reshape_on_device(a, *out_shape)
    report(name, ttnn.to_torch(y), x.reshape(out_shape))


@case("reshape_on_device_bf16_dram_w8_to_w16", "1024/2048 (second half of every output row)")
def _(dev):
    _reshape_on_device(dev, "reshape_on_device_bf16_dram_w8_to_w16", (1, 1, 256, 8), (1, 1, 128, 16), ttnn.bfloat16, torch.bfloat16, DRAM)


@case("reshape_on_device_bf16_dram_w24_to_w8", "6144/12288 (every second input row on a core)")
def _(dev):
    _reshape_on_device(dev, "reshape_on_device_bf16_dram_w24_to_w8", (1, 1, 512, 24), (1, 1, 1536, 8), ttnn.bfloat16, torch.bfloat16, DRAM)


@case("reshape_on_device_bf16_dram_w24_to_w48", "3072/6144")
def _(dev):
    _reshape_on_device(dev, "reshape_on_device_bf16_dram_w24_to_w48", (1, 1, 256, 24), (1, 1, 128, 48), ttnn.bfloat16, torch.bfloat16, DRAM)


@case("reshape_on_device_bf16_l1_w8_to_w16_ctrl", "0 (L1 source, 16 B sticks)")
def _(dev):
    _reshape_on_device(dev, "reshape_on_device_bf16_l1_w8_to_w16_ctrl", (1, 1, 256, 8), (1, 1, 128, 16), ttnn.bfloat16, torch.bfloat16, L1)


@case("reshape_on_device_bf16_dram_w16_to_w32_ctrl", "0 (32 B sticks)")
def _(dev):
    _reshape_on_device(dev, "reshape_on_device_bf16_dram_w16_to_w32_ctrl", (1, 1, 256, 16), (1, 1, 128, 32), ttnn.bfloat16, torch.bfloat16, DRAM)


@case("reshape_on_device_f32_dram_w8_to_w16_ctrl", "0 (32 B sticks)")
def _(dev):
    _reshape_on_device(dev, "reshape_on_device_f32_dram_w8_to_w16_ctrl", (1, 1, 256, 8), (1, 1, 128, 16), ttnn.float32, torch.float32, DRAM)


# ---------------------------------------------------------------------------------------------
# embedding: weight staging page is aligned to the index buffer, the read comes from the weights
# ---------------------------------------------------------------------------------------------
def _embedding(dev, name, tokens, dim, idx_mem, w_mem, idx_layout=RM, vocab=64):
    g = torch.Generator().manual_seed(2)
    ids = torch.randint(0, vocab, (1, tokens), generator=g)
    w = rand_bf16((vocab, dim), seed=3)
    t_ids = ttnn.to_device(ttnn.from_torch(ids, dtype=ttnn.uint32, layout=idx_layout), dev, memory_config=idx_mem)
    t_w = to_dev(dev, w, ttnn.bfloat16, w_mem)
    out = ttnn.embedding(t_ids, t_w, layout=RM, memory_config=DRAM)
    report(name, ttnn.to_torch(out), torch.nn.functional.embedding(ids, w.float()))


@case("embedding_idx_l1_w_dram_dim24", "6144/12288 (every second token on a core)")
def _(dev):
    _embedding(dev, "embedding_idx_l1_w_dram_dim24", 512, 24, L1, DRAM)


@case("embedding_idx_l1_w_dram_dim8", "2048/4096")
def _(dev):
    _embedding(dev, "embedding_idx_l1_w_dram_dim8", 512, 8, L1, DRAM)


@case("embedding_idx_l1_w_dram_dim20", "5120/10240")
def _(dev):
    _embedding(dev, "embedding_idx_l1_w_dram_dim20", 512, 20, L1, DRAM)


@case("embedding_idx_dram_w_dram_dim24_ctrl", "0 (page aligned to 32)")
def _(dev):
    _embedding(dev, "embedding_idx_dram_w_dram_dim24_ctrl", 512, 24, DRAM, DRAM)


@case("embedding_idx_l1_w_dram_dim16_ctrl", "0 (32 B rows)")
def _(dev):
    _embedding(dev, "embedding_idx_l1_w_dram_dim16_ctrl", 512, 16, L1, DRAM)


@case("embedding_idx_l1_w_l1_dim24_ctrl", "0 (L1 source, 16 B rule)")
def _(dev):
    _embedding(dev, "embedding_idx_l1_w_l1_dim24_ctrl", 512, 24, L1, L1)


@case("embedding_tile_idx_l1_w_dram_dim24", "768/1536 (two staging pages, every second token)")
def _(dev):
    _embedding(dev, "embedding_tile_idx_l1_w_dram_dim24", 64, 24, L1, DRAM, idx_layout=ttnn.TILE_LAYOUT)


@case("embedding_tile_idx_dram_w_dram_dim24_ctrl", "0")
def _(dev):
    _embedding(dev, "embedding_tile_idx_dram_w_dram_dim24_ctrl", 64, 24, DRAM, DRAM, idx_layout=ttnn.TILE_LAYOUT)


# BINARY: row 1 is cached at cache_base + weight_stick_size
def _embedding_binary(dev, name, dim, w_mem, tokens=96):
    g = torch.Generator().manual_seed(4)
    ids = torch.randint(0, 2, (1, tokens), generator=g)
    w = rand_bf16((2, dim), seed=5)
    t_ids = to_dev(dev, ids, ttnn.uint32, DRAM)
    t_w = to_dev(dev, w, ttnn.bfloat16, w_mem)
    out = ttnn.embedding(t_ids, t_w, layout=RM, embeddings_type=ttnn.EmbeddingsType.BINARY, memory_config=DRAM)
    got = ttnn.to_torch(out).to(torch.float32).reshape(tokens, dim)
    want = torch.nn.functional.embedding(ids, w.float()).reshape(tokens, dim)
    ones = ids.reshape(-1) == 1
    bad1 = int((got[ones] != want[ones]).sum())
    bad0 = int((got[~ones] != want[~ones]).sum())
    print(f"RESULT {name}: bad={bad0 + bad1}/{tokens * dim}", flush=True)
    print(f"  detail {name}: token0 bad={bad0}/{int((~ones).sum()) * dim} token1 bad={bad1}/{int(ones.sum()) * dim}", flush=True)


@case("embedding_binary_dram_dim8", "every token==1 row (row 1 cached at +16 B from a DRAM read)")
def _(dev):
    _embedding_binary(dev, "embedding_binary_dram_dim8", 8, DRAM)


@case("embedding_binary_dram_dim24", "every token==1 row (+48 B)")
def _(dev):
    _embedding_binary(dev, "embedding_binary_dram_dim24", 24, DRAM)


@case("embedding_binary_dram_dim4", "every token==1 row (+8 B, also misaligned on the L1 replay)")
def _(dev):
    _embedding_binary(dev, "embedding_binary_dram_dim4", 4, DRAM)


@case("embedding_binary_dram_dim16_ctrl", "0 (+32 B)")
def _(dev):
    _embedding_binary(dev, "embedding_binary_dram_dim16_ctrl", 16, DRAM)


@case("embedding_binary_l1_dim8_ctrl", "0 (L1 source, +16 B)")
def _(dev):
    _embedding_binary(dev, "embedding_binary_l1_dim8_ctrl", 8, L1)


# ---------------------------------------------------------------------------------------------
# concat, ROW_MAJOR interleaved, inputs in different memories.
# The two-input width case is described in open PR #52312 (native "returns shifted data").
# ---------------------------------------------------------------------------------------------
def _concat(dev, name, shapes, mems, dim, out_mem):
    xs = [rand_bf16(s, seed=10 + i) for i, s in enumerate(shapes)]
    ts = [to_dev(dev, x, ttnn.bfloat16, m) for x, m in zip(xs, mems)]
    out = ttnn.concat(ts, dim=dim, memory_config=out_mem)
    got = ttnn.to_torch(out).to(torch.float32)
    want = torch.cat([x.float() for x in xs], dim=dim)
    report(name, got, want)
    if got.shape == want.shape:
        off = 0
        parts = []
        for i, s in enumerate(shapes):
            sl = [slice(None)] * 4
            sl[dim] = slice(off, off + s[dim])
            parts.append(f"in{i}={int((got[tuple(sl)] != want[tuple(sl)]).sum())}/{want[tuple(sl)].numel()}")
            off += s[dim]
        print(f"  detail {name}: " + " ".join(parts), flush=True)


@case("concat_width_l1_dram_out_dram", "2048/3072 (all of input 1)")
def _(dev):
    _concat(dev, "concat_width_l1_dram_out_dram", [(1, 1, 128, 8), (1, 1, 128, 16)], [L1, DRAM], 3, DRAM)


@case("concat_width_dram_l1_dram_out_dram", "2048/5120 (all of input 2)")
def _(dev):
    _concat(dev, "concat_width_dram_l1_dram_out_dram", [(1, 1, 128, 16), (1, 1, 128, 8), (1, 1, 128, 16)], [DRAM, L1, DRAM], 3, DRAM)


@case("concat_height_l1_dram_out_l1_w24", "1536/6144 (every second row of input 1)")
def _(dev):
    _concat(dev, "concat_height_l1_dram_out_l1_w24", [(1, 1, 128, 24), (1, 1, 128, 24)], [L1, DRAM], 2, L1)


@case("concat_height_l1_dram_out_l1_w5", "320/1280 (every second row of input 1)")
def _(dev):
    _concat(dev, "concat_height_l1_dram_out_l1_w5", [(1, 1, 128, 5), (1, 1, 128, 5)], [L1, DRAM], 2, L1)


@case("concat_width_dram_dram_ctrl", "0")
def _(dev):
    _concat(dev, "concat_width_dram_dram_ctrl", [(1, 1, 128, 16), (1, 1, 128, 16)], [DRAM, DRAM], 3, DRAM)


@case("concat_height_dram_dram_out_dram_w24_ctrl", "0")
def _(dev):
    _concat(dev, "concat_height_dram_dram_out_dram_w24_ctrl", [(1, 1, 128, 24), (1, 1, 128, 24)], [DRAM, DRAM], 2, DRAM)


@case("concat_height_l1_dram_out_dram_w24_ctrl", "0 (page aligned to the DRAM output)")
def _(dev):
    _concat(dev, "concat_height_l1_dram_out_dram_w24_ctrl", [(1, 1, 128, 24), (1, 1, 128, 24)], [L1, DRAM], 2, DRAM)


def main():
    filters = sys.argv[1:]
    selected = [c for c in CASES if not filters or any(f in c[0] for f in filters)]
    if not selected:
        print("no case matches; available:", " ".join(c[0] for c in CASES))
        return
    dev = ttnn.open_device(device_id=0)
    try:
        grid = dev.compute_with_storage_grid_size()
        print(f"INFO compute grid {grid.x}x{grid.y}", flush=True)
        for name, model, fn in selected:
            try:
                fn(dev)
            except Exception as e:  # noqa: BLE001
                lines = [ln.strip() for ln in str(e).splitlines() if ln.strip()]
                print(f"RESULT {name}: bad=ERROR/0 {type(e).__name__}: {' | '.join(lines[:3])[:300]}", flush=True)
            if model:
                print(f"  MODEL {name}: {model}", flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
