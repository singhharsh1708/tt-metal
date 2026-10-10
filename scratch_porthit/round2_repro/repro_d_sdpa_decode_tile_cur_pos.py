# scaled_dot_product_attention_decode (non-paged): rank-1 TILE cur_pos_tensor is accepted;
# users 16.. read tile padding (0) as their position.
import torch, ttnn

device = ttnn.open_device(device_id=0)
torch.manual_seed(1234)
b, nh, nkv, s, d = 32, 8, 1, 1024, 128
grid = (8, 4)
pos = [200 + 7 * i for i in range(b)]
K = torch.randn(b, nkv, s, d)
V = torch.randn(b, nkv, s, d)
Q = torch.randn(1, b, nh, d)
scale = d**-0.5
mask = torch.zeros(b, nh, 1, s)
for i in range(b):
    mask[i, :, :, pos[i] + 1 :] = torch.finfo(torch.float32).min
expect = torch.nn.functional.scaled_dot_product_attention(
    Q.permute(1, 2, 0, 3), K.repeat(1, nh, 1, 1), V.repeat(1, nh, 1, 1), mask, scale=scale
).squeeze(2)  # [b, nh, d]
pos0 = V[:, 0, 0, :].unsqueeze(1).expand(b, nh, d)  # what a user gets when its position reads as 0

tt_K = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tt_V = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tt_Q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=32, k_chunk_size=128, exp_approx_mode=False)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False)


def rel(a, ref):
    return ((a - ref).norm(dim=-1) / ref.norm(dim=-1)).amax(dim=-1)  # per user


for name, layout in (("ROW_MAJOR", ttnn.ROW_MAJOR_LAYOUT), ("TILE", ttnn.TILE_LAYOUT)):
    try:
        cur = ttnn.from_torch(torch.tensor(pos, dtype=torch.int32), dtype=ttnn.int32, layout=layout, device=device)
        out = ttnn.transformer.scaled_dot_product_attention_decode(
            tt_Q, tt_K, tt_V, cur_pos_tensor=cur, scale=scale, program_config=cfg,
            compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        got = ttnn.to_torch(out).float()[0, :, :nh, :]
        e = rel(got, expect)
        bad = [i for i in range(b) if e[i] > 0.1]
        as_pos0 = [i for i in bad if rel(got, pos0)[i] < 0.05]
        print(f"RESULT sdpa_decode cur_pos={name} padded_shape={cur.padded_shape}: bad_users={len(bad)}/{b} first_bad={bad[:1]} bad_users_matching_cur_pos0={len(as_pos0)} max_rel_err={float(e.max()):.3f}")
    except Exception as ex:
        print(f"RESULT sdpa_decode cur_pos={name}: REJECTED {str(ex).splitlines()[0][:160]}")
# model: ROW_MAJOR 0/32; TILE 16/32 (users 16..31, all matching the cur_pos=0 output)
ttnn.close_device(device)
