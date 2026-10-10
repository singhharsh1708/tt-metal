import sys
root = sys.argv[1].rstrip("/")
B = "ttnn/cpp/ttnn/operations/pool"


def edit(rel, pairs):
    p = f"{root}/{rel}"; s = open(p).read()
    for old, new, n in pairs:
        assert s.count(old) == n, (rel, old[:50], s.count(old))
        s = s.replace(old, new)
    open(p, "w").write(s)


# upsample, sharded: the kernel must walk shards at the aligned row pitch the buffers use
edit(f"{B}/upsample/device/upsample_program_factory_multicore_sharded.cpp",
     [('        {"stick_nbytes", input_stick_nbytes},\n', '        {"stick_nbytes", aligned_input_stick_nbytes},\n', 1)])
# upsample, TILE interleaved: 32-bit inputs need a 32-bit DEST and unpack-to-DEST in the untilize
anchor = ("        const std::uint32_t num_input_tiles_in_row =\n"
          "            input.padded_shape()[-1] / input.tensor_spec().tile().get_tile_shape()[1];\n")
cfg = ("        // 32-bit inputs need a 32-bit DEST and unpack-to-DEST, or the untilize truncates them to 16 bits.\n"
       "        const bool is_32_bit = input_cb_data_format == tt::DataFormat::Float32 ||\n"
       "                               input_cb_data_format == tt::DataFormat::Int32 ||\n"
       "                               input_cb_data_format == tt::DataFormat::UInt32;\n"
       "        metal2::ComputeHardwareConfig compute_cfg{.enable_32_bit_dest = is_32_bit};\n"
       "        if (is_32_bit) {\n"
       "            compute_cfg.unpack_modes.insert({SRC0, UnpackMode::UnpackToDest});\n"
       "        }\n")
edit(f"{B}/upsample/device/upsample_program_factory_multicore_interleaved.cpp",
     [(anchor, anchor + cfg, 1),
      ("                .hw_config = metal2::ComputeHardwareConfig{},\n", "                .hw_config = compute_cfg,\n", 2)])
# grid_sample: a sharded grid is stored at the aligned row pitch too
old = "is_sharded ? grid_shape[-1] * grid_tensor.element_size() : get_aligned_stick_size(grid_shape, grid_tensor);"
for f in ("grid_sample_bilinear_program_factory.cpp", "grid_sample_nearest_program_factory.cpp"):
    edit(f"{B}/grid_sample/device/{f}", [(old, "get_aligned_stick_size(grid_shape, grid_tensor);", 2)])
print("pool patch applied")
