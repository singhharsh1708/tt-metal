# moreh getitem: the reader's index decoding (ROW_MAJOR vs TILE) is chosen from index_tensors[0] only.
import torch, ttnn

device = ttnn.open_device(device_id=0)
torch.manual_seed(2)
shape, dims, N = (10, 5, 5, 64), [1, 2], 32
x = torch.randint(0, 1000, shape, dtype=torch.int32)
i0 = torch.randint(0, 5, (N,), dtype=torch.int32)
i1 = torch.randint(1, 5, (N,), dtype=torch.int32)  # never 0, so a padding read is always visible
ref = x[:, i0.long(), i1.long()]  # [10, N, 64]
dev_x = ttnn.from_torch(x, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)


def idx(t, tile):
    if tile:
        return ttnn.from_torch(t.reshape(1, N), dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    return ttnn.from_torch(t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


for name, l0, l1 in (("RM,RM", 0, 0), ("TILE,TILE", 1, 1), ("RM,TILE", 0, 1)):
    try:
        out = ttnn.operations.moreh.getitem(dev_x, [idx(i0, l0), idx(i1, l1)], dims)
        got = ttnn.to_torch(out).reshape(ref.shape)
        bad = [j for j in range(N) if not torch.equal(got[:, j], ref[:, j])]
        as_zero = [j for j in bad if torch.equal(got[:, j], x[:, i0[j].long(), 0])]
        print(f"RESULT moreh_getitem index layouts={name}: bad_index_positions={len(bad)}/{N} first_bad={bad[:1]} bad_matching_index0={len(as_zero)}")
    except Exception as ex:
        print(f"RESULT moreh_getitem index layouts={name}: REJECTED {str(ex).splitlines()[0][:160]}")
# model: RM,RM 0/32; TILE,TILE 0/32; RM,TILE 16/32 (positions 16..31 use index 0)
# (TILE,RM is also wrong but reads past the ROW_MAJOR buffer, so it is left out: not deterministic)
ttnn.close_device(device)
