"""ttnn.index_fill with a TILE-layout index: kernels scan physical_volume entries, so tile padding (0) is treated as an index."""
import torch, ttnn

dev = ttnn.open_device(device_id=0)


def make_index(idx, layout):
    t = torch.tensor(idx, dtype=torch.int32)
    try:
        return ttnn.from_torch(t, dtype=ttnn.uint32, layout=layout, device=dev)
    except Exception:
        rm = ttnn.from_torch(t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        return ttnn.to_layout(rm, layout)


def run(tag, shape, dim, idx, layout, tt_dtype=ttnn.bfloat16):
    try:
        torch.manual_seed(0)
        if tt_dtype == ttnn.int32:
            x = torch.randint(1, 100, shape, dtype=torch.int32); value = 777
        else:
            x = torch.randint(1, 100, shape).float(); value = 512.0
        ref = torch.index_fill(x, dim, torch.tensor(idx, dtype=torch.long), value)
        tx = ttnn.from_torch(x, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        ti = make_index(idx, layout)
        out = ttnn.to_torch(ttnn.index_fill(tx, dim, ti, value)).to(ref.dtype)
        diff = out != ref
        other = [d for d in range(len(shape)) if d != dim]
        pos = torch.nonzero(diff.sum(dim=other) if other else diff).flatten().tolist()
        print(f"RESULT index_fill {tag}: shape={list(shape)} dim={dim} K={len(idx)} index_layout={ti.layout} "
              f"bad={int(diff.sum())}/{ref.numel()} wrong_positions_along_dim={pos[:12]}", flush=True)
    except Exception as e:
        print(f"RESULT index_fill {tag}: EXC {str(e).strip().splitlines()[0][:220]}", flush=True)


T, R = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
run("tile_lastdim", (4, 8), 1, [3, 5], T)
run("tile_dim0", (6, 8), 0, [3, 5], T)
run("tile_4d_dim3", (2, 3, 40, 64), 3, [7, 9, 33], T)
run("tile_4d_dim2", (2, 3, 40, 64), 2, [1, 2, 39], T)
run("tile_int32_dim0", (6, 8), 0, [3, 5], T, ttnn.int32)
run("tile_K40", (4, 64), 1, list(range(1, 41)), T)
run("ctrl_rm_index", (4, 8), 1, [3, 5], R)
run("ctrl_rm_index_4d", (2, 3, 40, 64), 2, [1, 2, 39], R)
run("ctrl_tile_index_has_0", (6, 8), 0, [0, 3], T)
ttnn.close_device(dev)
