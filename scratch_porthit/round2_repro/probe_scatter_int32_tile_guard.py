# scatter: the "int32 TILE row > 256" guard tests shape[dim], the to_layout it protects runs on the last dim.
import torch, ttnn

device = ttnn.open_device(device_id=0)
torch.manual_seed(0)
for shape, dim in (((2, 300), 0), ((300, 8), 0)):
    inp = torch.zeros(shape, dtype=torch.int32)
    index = torch.randint(0, shape[dim], shape, dtype=torch.int64)
    src = torch.randint(1, 100, shape, dtype=torch.int32)
    ref = torch.scatter(inp, dim, index, src)
    for name, layout in (("ROW_MAJOR", ttnn.ROW_MAJOR_LAYOUT), ("TILE", ttnn.TILE_LAYOUT)):
        try:
            a = ttnn.from_torch(inp, dtype=ttnn.int32, layout=layout, device=device)
            i = ttnn.from_torch(index.to(torch.int32), dtype=ttnn.int32, layout=layout, device=device)
            s_ = ttnn.from_torch(src, dtype=ttnn.int32, layout=layout, device=device)
            got = ttnn.to_torch(ttnn.scatter(a, dim, i, s_))
            # duplicate indices make the winner order-dependent; compare only positions written once
            once = torch.zeros(shape, dtype=torch.int32).scatter_add(dim, index, torch.ones(shape, dtype=torch.int32)) <= 1
            print(f"RESULT scatter shape={shape} dim={dim} layout={name}: bad={int((got[once] != ref[once]).sum())}/{int(once.sum())}")
        except Exception as ex:
            print(f"RESULT scatter shape={shape} dim={dim} layout={name}: REJECTED {str(ex).splitlines()[0][:140]}")
ttnn.close_device(device)
