# K heads of the QKV split go through a transpose compute kernel; Q and V are reader->writer only.
# Splitting and transposing is a permutation, expected bit-exact for float32.
import torch, ttnn

device = ttnn.open_device(device_id=0, l1_small_size=8192)


def report(name, got, ref):
    got, ref = got.float().contiguous(), ref.float().contiguous()
    gb, rb = got.view(torch.int32), ref.view(torch.int32)
    bad = got != ref
    print(
        f"  {name}: bad {int(bad.sum())}/{ref.numel()}  max abs err {float((got - ref).abs().max()):.3e}  "
        f"low16==0: {int(((gb & 0xFFFF) == 0).sum())}  low13==0: {int(((gb & 0x1FFF) == 0).sum())}  "
        f"== tf32-trunc ref: {bool(torch.equal(gb, rb & ~0x1FFF))}  == bf16-trunc ref: {bool(torch.equal(gb, rb & ~0xFFFF))}"
    )


def refs(A, B, S, H, D):
    q, k, v = torch.split(A, [H * D, H * D, H * D], dim=-1)
    q = q.reshape(B, S, H, D).transpose(-3, -2)
    k = k.reshape(B, S, H, D).transpose(-3, -2).transpose(-2, -1)
    v = v.reshape(B, S, H, D).transpose(-3, -2)
    return q, k, v


def interleaved(tt_dtype, torch_dtype):
    B, S, H, D = 1, 128, 2, 64
    torch.manual_seed(0)
    A = (torch.randn(B, 1, S, 3 * H * D) * 3.0).to(torch_dtype)
    t = ttnn.from_torch(A, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        t, None, num_heads=H, num_kv_heads=H, transpose_k_heads=True, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    print(f"nlp_create_qkv_heads interleaved, {tt_dtype}")
    for n, got, ref in zip("QKV", (q, k, v), refs(A, B, S, H, D)):
        report(n, ttnn.to_torch(got), ref)


def public_op(tt_dtype, torch_dtype):
    B, S, H, D = 1, 128, 2, 64
    torch.manual_seed(0)
    A = (torch.randn(B, S, 3 * H * D) * 3.0).to(torch_dtype)
    t = ttnn.from_torch(A, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(t, num_heads=H)
    print(f"ttnn.transformer.split_query_key_value_and_split_heads, {tt_dtype}")
    for n, got, ref in zip("QKV", (q, k, v), refs(A.unsqueeze(1), B, S, H, D)):
        report(n, ttnn.to_torch(got), ref)


def sharded(tt_dtype, torch_dtype):
    # same construction as tests/tt_eager/.../test_create_qkv_heads.py
    B, S, HQ, HKV, D, ch, cw = 7, 224, 8, 8, 64, 7, 8
    torch.manual_seed(1234)
    Q = torch.randn([B, 1, S, HKV, HQ // HKV * D]).to(torch_dtype)
    K = torch.randn([B, 1, S, HKV, D]).to(torch_dtype)
    V = torch.randn([B, 1, S, HKV, D]).to(torch_dtype)
    QKV = torch.concat([Q.flatten(-2, -1), K.flatten(-2, -1), V.flatten(-2, -1)], -1)
    QKV_i = torch.concat([Q, K, V], -1).flatten(-2, -1)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cw - 1, ch - 1))})
    spec = ttnn.ShardSpec(grid, [S, QKV.shape[-1] // cw], ttnn.ShardOrientation.ROW_MAJOR)
    in_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, spec)
    out_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
    t = ttnn.Tensor(QKV_i, tt_dtype).to(ttnn.TILE_LAYOUT).to(device, in_cfg)
    q, k, v = ttnn.experimental.create_qkv_heads(
        t, num_heads=HQ, num_kv_heads=HKV, transpose_k_heads=True, memory_config=out_cfg
    )
    rq, rk, rv = torch.split(QKV, [HQ * D, HKV * D, HKV * D], dim=-1)
    rq = rq.reshape(B, S, HQ, D).transpose(-3, -2)
    rk = rk.reshape(B, S, HKV, D).transpose(-3, -2).transpose(-2, -1)
    rv = rv.reshape(B, S, HKV, D).transpose(-3, -2)
    print(f"create_qkv_heads block-sharded, {tt_dtype}")
    for n, got, ref in zip("QKV", (q, k, v), (rq, rk, rv)):
        report(n, ttnn.to_torch(got), ref)


for fn in (interleaved, public_op, sharded):
    fn(ttnn.float32, torch.float32)  # expect Q, V exact; K lossy (low 13 bits zero)
    fn(ttnn.bfloat16, torch.bfloat16)  # control: all exact

ttnn.close_device(device)
