# ttnn.nonzero: BFLOAT8_B / BFLOAT4_B pass the element-size check (1 byte) and the tile is scanned as flat bytes.
import torch, ttnn

device = ttnn.open_device(device_id=0)
cases = {
    "onehot_col5_[1,1,1,32]": torch.eye(1, 32, 5).reshape(1, 1, 1, 32),
    "identity_[1,1,32,32]": torch.eye(32).reshape(1, 1, 32, 32),
    "row6_ones_[1,1,8,32]": torch.nn.functional.pad(torch.ones(1, 32), (0, 0, 6, 1)).reshape(1, 1, 8, 32),
}
for cname, x in cases.items():
    ref = {tuple(r) for r in torch.nonzero(x).tolist()}
    for name, dt in (("bfloat16", ttnn.bfloat16), ("bfloat8_b", ttnn.bfloat8_b)):
        try:
            t = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
            cnt_t, idx_t = ttnn.nonzero(t)
            cnt = int(ttnn.to_torch(cnt_t).flatten()[0])
            got = {tuple(r) for r in ttnn.to_torch(idx_t).flatten()[: 4 * cnt].reshape(-1, 4).tolist()}
            print(f"RESULT nonzero {cname} dtype={name}: torch_count={len(ref)} ttnn_count={cnt} missing={len(ref - got)} spurious={len(got - ref)}")
        except Exception as ex:
            print(f"RESULT nonzero {cname} dtype={name}: REJECTED {str(ex).splitlines()[0][:160]}")
# model (bfloat8_b): onehot 1 vs 1 but at (0,0,0,0); identity 60 reported vs 32; row6 2 reported vs 32. bfloat16: all exact.
ttnn.close_device(device)
