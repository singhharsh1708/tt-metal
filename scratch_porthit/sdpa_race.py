import torch, ttnn

dev = ttnn.open_device(device_id=0, l1_small_size=32768)
SCALE = 128 ** -0.5
S = 1024
EVENTS = [  # seed, head, row, col, approx  (from elem run 19:32)
    (10, 7, 318, 9, True), (12, 5, 609, 54, True), (10, 4, 789, 89, True), (22, 0, 758, 14, False), (13, 7, 753, 68, False)]


def inputs(seed):
    torch.manual_seed(2000 + seed)
    q = torch.randn(1, 8, S, 128).bfloat16(); k = torch.randn(1, 8, S, 128).bfloat16(); v = torch.randn(1, 8, S, 128).bfloat16()
    qd, kd, vd = q.double(), k.double(), v.double()
    sc = (qd @ kd.transpose(-2, -1)) * SCALE + torch.triu(torch.full((S, S), -torch.inf, dtype=torch.float64), 1)
    return q, k, v, torch.softmax(sc, -1) @ vd


def run(q, k, v, approx, grid, qc=256, kc=256):
    comp = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=approx)
    t = [ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev) for x in (q, k, v)]
    return ttnn.to_torch(ttnn.transformer.scaled_dot_product_attention(*t, is_causal=True, scale=SCALE, program_config=pc, compute_kernel_config=comp)).double()


cache = {}
for seed, h, i, c, approx in EVENTS:
    if seed not in cache:
        cache[seed] = inputs(seed)
    q, k, v, gold = cache[seed]
    g = float(gold[0, h, i, c])
    line = f"seed{seed} h{h} r{i} c{c} approx={int(approx)} gold {g:+.4f} |"
    for name, grid, qc, kc in (("8x8 run1", (8, 8), 256, 256), ("8x8 run2", (8, 8), 256, 256), ("8x8 run3", (8, 8), 256, 256),
                               ("8x7", (8, 7), 256, 256), ("7x8", (7, 8), 256, 256), ("4x4", (4, 4), 256, 256), ("1x1", (1, 1), 256, 256),
                               ("8x8 q128", (8, 8), 128, 256), ("8x8 k128", (8, 8), 256, 128)):
        out = run(q, k, v, approx, grid, qc, kc)
        err = (out - gold).abs()
        n_bad = int((err[..., 4:, :] > 0.05).sum())  # skip the first rows (approx-exp precision on 2-4 keys)
        line += f" {name}: {float(out[0, h, i, c]):+.4f} (bad elems {n_bad})"
    print(line, flush=True)
ttnn.close_device(dev)
