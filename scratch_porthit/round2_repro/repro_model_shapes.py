# N150 probe: slice shapes taken from model call sites. Prints one RESULT line per case.
# Run: python repro_model_shapes.py   (inside a tt-metal env, one Wormhole card)
import torch
import ttnn

DT = {"bf16": (ttnn.bfloat16, torch.bfloat16), "f32": (ttnn.float32, torch.float32)}
LAYOUT = {"RM": ttnn.ROW_MAJOR_LAYOUT, "TILE": ttnn.TILE_LAYOUT}
MEM = {"DRAM": ttnn.DRAM_MEMORY_CONFIG, "L1": ttnn.L1_MEMORY_CONFIG}

# (tag, input shape, begins, ends, dtype, layout, memory, expectation from the host-side port)
SLICE_CASES = [
    # cosyvoice2 tt/hifigan/istft.py:282, bucketing=False, mel=47 -> R=22560 (353/352 rows per core)
    ("cv2_istft_mel47_RM_L1", [1, 1, 22576, 1], [0, 0, 8, 0], [1, 1, 22568, 1], "f32", "RM", "L1", "strict+general"),
    ("cv2_istft_mel47_TILE_L1", [1, 1, 22576, 1], [0, 0, 8, 0], [1, 1, 22568, 1], "f32", "TILE", "L1", "strict+general"),
    ("cv2_istft_mel47_RM_DRAM", [1, 1, 22576, 1], [0, 0, 8, 0], [1, 1, 22568, 1], "f32", "RM", "DRAM", "all three"),
    ("cv2_istft_mel111_RM_L1", [1, 1, 53296, 1], [0, 0, 8, 0], [1, 1, 53288, 1], "f32", "RM", "L1", "all three"),
    # control: shipped bucket mel=256 -> R=122880, no second core group
    ("cv2_istft_mel256_RM_L1_control", [1, 1, 122896, 1], [0, 0, 8, 0], [1, 1, 122888, 1], "f32", "RM", "L1", "clean"),
    # cosyvoice2 tt/hifigan/source.py:252, bucketing=False, mel=228 -> R=2052 (33/32 rows per core)
    ("cv2_sinegen_down_mel228", [2052, 480, 1], [0, 239, 0], [2052, 240, 1], "f32", "RM", "L1", "general only"),
    ("cv2_sinegen_down_mel256_control", [2304, 480, 1], [0, 239, 0], [2304, 240, 1], "f32", "RM", "L1", "clean"),
    # cosyvoice2 tt/hifigan/source.py:296, mel=229 -> R=2052
    ("cv2_sinegen_up_mel229", [9, 229, 1], [0, 1, 0], [9, 229, 1], "f32", "RM", "L1", "general only"),
    # qwen3_tts tt/ttnn_qwen3_codec.py:431, decode(bucket=1), T=77, block1 -> R=12320 (193/192), k equal
    ("q3tts_transconv_T77_block1", [1, 12325, 384], [0, 0, 0], [1, 12320, 384], "bf16", "RM", "DRAM", "strict only"),
    ("q3tts_transconv_T193_block0", [1, 6184, 768], [0, 0, 0], [1, 6176, 768], "bf16", "RM", "DRAM", "strict only"),
    # control: pipeline default bucket=32, T=32, block1 -> R=5120
    ("q3tts_transconv_T32_block1_control", [1, 5125, 384], [0, 0, 0], [1, 5120, 384], "bf16", "RM", "DRAM", "clean"),
    # bge_m3 demo/generator_vllm.py:600, B=1, S=6150 -> R=6149 (97/96), k equal
    ("bge_m3_colbert_S6150", [1, 1, 6150, 1024], [0, 0, 1, 0], [1, 1, 6150, 1024], "bf16", "TILE", "DRAM", "strict only"),
]

# ttnn.split on a ROW_MAJOR tensor lowers to one ttnn.slice per chunk (split.cpp split_with_slice_impl)
SPLIT_CASES = [
    # rtdetr tt/decoder.py:210 at 640x416: levels 80x52, 40x26, 20x13 -> last chunk R=8*260=2080 (33/32)
    ("rtdetr_value_640x416", [8, 5460, 32], [4160, 1040, 260], 1, "bf16", "DRAM", "general only (chunk 2)"),
    # control: shipped 640x640
    ("rtdetr_value_640x640_control", [8, 8400, 32], [6400, 1600, 400], 1, "bf16", "DRAM", "clean"),
    # near-miss square sizes: stride-16 level 62x62 at 992x992
    ("rtdetr_value_992x992", [8, 20181, 32], [15376, 3844, 961], 1, "bf16", "DRAM", "all three (chunk 1)"),
]


def bad_count(got, ref):
    return int((got.float() != ref.float()).sum().item())


def main():
    device = ttnn.open_device(device_id=0)
    grid = device.compute_with_storage_grid_size()
    print(f"grid {grid.x}x{grid.y}")
    try:
        for tag, shape, b, e, dt, lay, mem, expect in SLICE_CASES:
            tt_dt, th_dt = DT[dt]
            x = torch.randn(shape, dtype=torch.float32).to(th_dt)
            tx = ttnn.from_torch(x, dtype=tt_dt, layout=LAYOUT[lay], device=device, memory_config=MEM[mem])
            out = ttnn.slice(tx, b, e)
            got = ttnn.to_torch(out)
            ref = x[tuple(slice(lo, hi) for lo, hi in zip(b, e))]
            n = bad_count(got.reshape(ref.shape), ref)
            print(f"RESULT {tag} bad={n}/{ref.numel()} out_layout={out.layout} predicted={expect}")
            ttnn.deallocate(out)
            ttnn.deallocate(tx)

        for tag, shape, sizes, dim, dt, mem, expect in SPLIT_CASES:
            tt_dt, th_dt = DT[dt]
            x = torch.randn(shape, dtype=torch.float32).to(th_dt)
            tx = ttnn.from_torch(x, dtype=tt_dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=MEM[mem])
            outs = ttnn.split(tx, sizes, dim=dim)
            refs = torch.split(x, sizes, dim=dim)
            for i, (o, r) in enumerate(zip(outs, refs)):
                n = bad_count(ttnn.to_torch(o).reshape(r.shape), r)
                print(f"RESULT {tag} chunk={i} bad={n}/{r.numel()} predicted={expect}")
            ttnn.deallocate(tx)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
