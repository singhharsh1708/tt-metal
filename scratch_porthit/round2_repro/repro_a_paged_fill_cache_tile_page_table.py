# paged_fill_cache: TILE-layout page_table is accepted and read as a flat ROW_MAJOR row.
import torch, ttnn

device = ttnn.open_device(device_id=0)
torch.manual_seed(0)
M, block, hd = 32, 32, 32  # 32 virtual blocks of 32 tokens, head_dim 32
nblk = M + 1  # physical block 0 is left unmapped on purpose
perm = torch.randperm(M) + 1
page_table = perm.reshape(1, M).to(torch.int32)
x = torch.randn(1, 1, M * block, hd).bfloat16().float()
expected = torch.zeros(nblk, 1, block, hd)
for v in range(M):
    expected[perm[v]] = x[0, :, v * block : (v + 1) * block, :]


def run(layout):
    cache = ttnn.from_torch(torch.zeros(nblk, 1, block, hd), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    pt = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=layout, device=device)
    out = ttnn.experimental.paged_fill_cache(cache, xt, pt, batch_idx=0)
    got = ttnn.to_torch(out).float()
    bad = [b for b in range(nblk) if not torch.allclose(got[b], expected[b], atol=1e-3)]
    unwritten = [b for b in bad if b != 0 and float(got[b].abs().max()) == 0.0]
    return len(bad), len(unwritten), float(got[0].abs().max()) != 0.0


for name, layout in (("ROW_MAJOR", ttnn.ROW_MAJOR_LAYOUT), ("TILE", ttnn.TILE_LAYOUT)):
    try:
        bad, unwritten, blk0 = run(layout)
        print(f"RESULT paged_fill_cache page_table={name}: bad_blocks={bad}/{nblk} never_written={unwritten} unmapped_block0_clobbered={blk0}")
    except Exception as e:
        print(f"RESULT paged_fill_cache page_table={name}: REJECTED {str(e).splitlines()[0][:160]}")
# model: ROW_MAJOR 0/33; TILE 17/33 (16 never written, block 0 clobbered)
ttnn.close_device(device)
