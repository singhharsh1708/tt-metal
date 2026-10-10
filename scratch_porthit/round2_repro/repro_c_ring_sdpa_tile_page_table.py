# ring_distributed_scaled_dot_product_attention (chunked + paged KV): page_table layout is not validated.
import torch, ttnn

device = ttnn.open_device(device_id=0)
torch.manual_seed(1234)
b, nh, nkv, d = 1, 8, 1, 128
ring_size, q_chunk, k_chunk, block, prefix = 4, 64, 64, 64, 64
s = prefix + 2048  # 33 KV blocks of 64 tokens
qs = s - prefix
Q = torch.randn(b, nh, s, d)
K = torch.randn(b, nkv, s, d)
V = torch.randn(b, nkv, s, d)
Qs = Q[:, :, prefix:, :]
nblk = s // block
perm = torch.randperm(nblk)
page_table = torch.argsort(perm).reshape(b, nblk).to(torch.int32)


def paged(c):
    return c.reshape(b, nkv, nblk, block, d).transpose(1, 2).reshape(nblk, nkv, block, d)[perm]


mask = torch.full((qs, s), torch.finfo(torch.float32).min)
for i in range(qs):
    mask[i, : prefix + i + 1] = 0
expect = torch.nn.functional.scaled_dot_product_attention(Qs, K.repeat(1, nh, 1, 1), V.repeat(1, nh, 1, 1), mask)

dt = ttnn.bfloat8_b
tt_Q = ttnn.from_torch(Qs, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
tt_K = ttnn.from_torch(paged(K), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
tt_V = ttnn.from_torch(paged(V), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=device.compute_with_storage_grid_size(), q_chunk_size=q_chunk, k_chunk_size=k_chunk, exp_approx_mode=True)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=False)


def reshuffle(outs):
    c = qs // (2 * ring_size)
    full = torch.zeros(b, nh, qs, d)
    for r, o in enumerate(outs):
        full[:, :, r * c : (r + 1) * c] = o[:, :, :c]
        j = 2 * ring_size - 1 - r
        full[:, :, j * c : (j + 1) * c] = o[:, :, c : 2 * c]
    return full


for name, layout in (("ROW_MAJOR", ttnn.ROW_MAJOR_LAYOUT), ("TILE", ttnn.TILE_LAYOUT)):
    try:
        pt = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=layout, device=device)
        outs = []
        for ring_id in range(ring_size):
            o = ttnn.transformer.ring_distributed_scaled_dot_product_attention(
                tt_Q, tt_K, tt_V, ring_size=ring_size, ring_id=ring_id, program_config=cfg,
                compute_kernel_config=ckc, page_table=pt, chunk_start_idx=prefix,
            )
            outs.append(ttnn.to_torch(o).float()[:, :, : qs // ring_size, :])
        got = reshuffle(outs)
        e = ((got - expect).norm(dim=-1) / expect.norm(dim=-1)).amax(dim=1)[0]  # per query position
        bad = (e > 0.3).nonzero().flatten()
        first = int(bad[0]) + prefix if len(bad) else -1
        print(f"RESULT ring_sdpa page_table={name}: bad_query_positions={len(bad)}/{qs} first_bad_global_pos={first} median_rel_err={float(e.median()):.3f}")
    except Exception as ex:
        print(f"RESULT ring_sdpa page_table={name}: REJECTED {str(ex).splitlines()[0][:160]}")
# model: ROW_MAJOR 0 bad; TILE: every query at global position >= 16*64 = 1024 reads KV block ids from tile padding
ttnn.close_device(device)
