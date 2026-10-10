#!/usr/bin/env python3
"""Row-pitch fixes for tt-metal at 4502c6d9c57. Run from the repo root.

A ROW_MAJOR sharded buffer keeps rows at Buffer::aligned_page_size(). These edits make every walk
through a borrowed shard use that pitch instead of the raw row size.

Each edit is an exact string replacement whose anchor must occur exactly once in the file.
"""
import sys

OPS = "ttnn/cpp/ttnn/operations/"

CLONE_F = OPS + "data_movement/clone/device/clone_program_factory.cpp"
CLONE_R = OPS + "data_movement/clone/device/kernels/read_kernel_rm_sharded.cpp"
CLONE_W = OPS + "data_movement/clone/device/kernels/write_kernel_rm_sharded.cpp"
TRANSPOSE_F = OPS + "data_movement/transpose/device/transpose_hc_sharded_program_factory.cpp"
ROTATE_F = OPS + "pool/rotate/device/rotate_nearest_program_factory.cpp"
POOL_F = OPS + "pool/generic/device/pool_multi_core_program_factory.cpp"
HWC_GATHER_H = OPS + "experimental/cnn/convert_to_hwc/device/gather.hpp"
HWC_GATHER_C = OPS + "experimental/cnn/convert_to_hwc/device/gather.cpp"
HWC_F = OPS + "experimental/cnn/convert_to_hwc/device/convert_to_hwc_program_factory.cpp"
FILL_F = OPS + "data_movement/indexed_fill/device/indexed_fill_program_factory.cpp"
FILL_R = OPS + "data_movement/indexed_fill/device/kernels/dataflow/indexed_fill_reader.cpp"

# (path, label, old, new)
EDITS = [
    # ------------------------------------------------------------------ 1. clone, RM sharded
    (
        CLONE_F,
        "clone: define strides",
        """    uint32_t aligned_output_unit_size = tt::align(output_unit_size, alignment);
""",
        """    uint32_t aligned_output_unit_size = tt::align(output_unit_size, alignment);
    // Rows of a row-major shard sit at the buffer's aligned page size.
    const uint32_t input_stick_stride = static_cast<uint32_t>(input.buffer()->aligned_page_size());
    const uint32_t output_stick_stride = static_cast<uint32_t>(output.buffer()->aligned_page_size());
""",
    ),
    (
        CLONE_F,
        "clone: rta schema",
        """        rta_names = is_sharded ? Group<std::string>{"stick_size", "num_sticks"}
""",
        """        rta_names = is_sharded ? Group<std::string>{"stick_size", "num_sticks", "stick_stride"}
""",
    ),
    (
        CLONE_F,
        "clone: reader args",
        """                    {{"stick_size", input_unit_size}, {"num_sticks", num_units_per_core}});
""",
        """                    {{"stick_size", input_unit_size},
                     {"num_sticks", num_units_per_core},
                     {"stick_stride", input_stick_stride}});
""",
    ),
    (
        CLONE_F,
        "clone: writer args",
        """                    {{"stick_size", output_unit_size}, {"num_sticks", num_units_per_core}});
""",
        """                    {{"stick_size", output_unit_size},
                     {"num_sticks", num_units_per_core},
                     {"stick_stride", output_stick_stride}});
""",
    ),
    (
        CLONE_R,
        "clone reader: read stride arg",
        """    auto num_sticks = get_arg(args::num_sticks);
""",
        """    auto num_sticks = get_arg(args::num_sticks);
    auto stick_stride = get_arg(args::stick_stride);
""",
    ),
    (
        CLONE_R,
        "clone reader: step by stride",
        """        local_l1_read_addr += stick_size;
""",
        """        local_l1_read_addr += stick_stride;
""",
    ),
    (
        CLONE_W,
        "clone writer: read stride arg",
        """    auto num_sticks = get_arg(args::num_sticks);
""",
        """    auto num_sticks = get_arg(args::num_sticks);
    auto stick_stride = get_arg(args::stick_stride);
""",
    ),
    (
        CLONE_W,
        "clone writer: step by stride",
        """        local_l1_write_addr += stick_size;
""",
        """        local_l1_write_addr += stick_stride;
""",
    ),
    # ------------------------------------------------------------------ 2. transpose HC, RM height sharded
    (
        TRANSPOSE_F,
        "transpose_hc: special-case runtime args",
        """    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1], N = input_shape[0];
    const uint32_t total_height = N * C * H;
    uint32_t stick_size_bytes = W * input_tensor.element_size();
""",
        """    uint32_t H = input_shape[2], C = input_shape[1], N = input_shape[0];
    const uint32_t total_height = N * C * H;
    // Rows of the shard sit at the buffer's aligned page size.
    uint32_t stick_size_bytes = static_cast<uint32_t>(input_tensor.buffer()->aligned_page_size());
""",
    ),
    (
        TRANSPOSE_F,
        "transpose_hc: descriptor",
        """    uint32_t W = input_tensor.logical_shape()[3], H = input_tensor.logical_shape()[2];
    uint32_t C = input_tensor.logical_shape()[1], N = input_tensor.logical_shape()[0];
    uint32_t stick_size_bytes = W * input_tensor.element_size();
""",
        """    uint32_t H = input_tensor.logical_shape()[2];
    uint32_t C = input_tensor.logical_shape()[1], N = input_tensor.logical_shape()[0];
    // Rows of the shard sit at the buffer's aligned page size.
    uint32_t stick_size_bytes = static_cast<uint32_t>(input_tensor.buffer()->aligned_page_size());
""",
    ),
    # ------------------------------------------------------------------ 3. rotate nearest, sharded
    (
        ROTATE_F,
        "rotate: aligned stick sizes",
        """    const uint32_t aligned_input_stick_nbytes = any_sharded ? effective_channels * input_tensor.element_size()
                                                            : pool::get_aligned_stick_size(input_shape, input_tensor);
    const uint32_t aligned_output_stick_nbytes = any_sharded ? effective_channels * output_tensor.element_size()
                                                             : pool::get_aligned_stick_size(input_shape, output_tensor);
""",
        """    // Rows of a sharded buffer sit at its aligned page size.
    const uint32_t aligned_input_stick_nbytes =
        any_sharded ? static_cast<uint32_t>(input_tensor.buffer()->aligned_page_size())
                    : pool::get_aligned_stick_size(input_shape, input_tensor);
    const uint32_t aligned_output_stick_nbytes =
        any_sharded ? static_cast<uint32_t>(output_tensor.buffer()->aligned_page_size())
                    : pool::get_aligned_stick_size(input_shape, output_tensor);
""",
    ),
    # ------------------------------------------------------------------ 4. pool2d reader stride
    (
        POOL_F,
        "pool: shard row pitch",
        """    const uint32_t shard_width_bytes = input_shape[3] / num_shards_c * params.nbytes;
""",
        """    // Rows of the haloed input shard sit at the buffer's aligned pitch.
    const uint32_t shard_width_bytes =
        tt::round_up(input_shape[3] / num_shards_c * params.nbytes, input.buffer()->alignment());
""",
    ),
    # ------------------------------------------------------------------ 5. convert_to_hwc gather
    (
        HWC_GATHER_H,
        "gather.hpp: lower_gather_transfers decl",
        """    uint32_t element_size_bytes,
    uint32_t output_shard_width);

// Forward declaration for API that returns both groups and count
""",
        """    uint32_t element_size_bytes,
    uint32_t output_shard_width,
    uint32_t input_row_pitch_bytes = 0);

// Forward declaration for API that returns both groups and count
""",
    ),
    (
        HWC_GATHER_H,
        "gather.hpp: group_transfers decl",
        """    uint32_t block_size,
    uint32_t output_shard_width);

/**
 * @brief Tensor-based interface for gather transfers precomputation
""",
        """    uint32_t block_size,
    uint32_t output_shard_width,
    uint32_t input_row_pitch_bytes = 0);

/**
 * @brief Tensor-based interface for gather transfers precomputation
""",
    ),
    (
        HWC_GATHER_C,
        "gather.cpp: lower_gather_transfers signature",
        """    uint32_t element_size_bytes,
    uint32_t output_shard_width) {
    std::vector<LowLevelGatherTransfer> low_level_transfers;
""",
        """    uint32_t element_size_bytes,
    uint32_t output_shard_width,
    uint32_t input_row_pitch_bytes) {
    std::vector<LowLevelGatherTransfer> low_level_transfers;
""",
    ),
    (
        HWC_GATHER_C,
        "gather.cpp: input row bytes",
        """    TT_FATAL(output_shard_width != 0, "Output shard width must be provided");
""",
        """    TT_FATAL(output_shard_width != 0, "Output shard width must be provided");
    // Source rows sit at the input buffer's aligned pitch when one is given (0 keeps the raw row size).
    const uint32_t input_row_bytes =
        input_row_pitch_bytes != 0 ? input_row_pitch_bytes : input_shard_width * element_size_bytes;
""",
    ),
    (
        HWC_GATHER_C,
        "gather.cpp: source byte offset",
        """        uint32_t src_offset_bytes = src_absolute_offset * element_size_bytes;
""",
        """        uint32_t src_offset_bytes = (src_row * input_row_bytes) + (t.src_offset * element_size_bytes);
""",
    ),
    (
        HWC_GATHER_C,
        "gather.cpp: group_transfers signature",
        """    uint32_t block_size,
    uint32_t output_shard_width) {
    // Dictionary to group transfers by (dst_shard_idx, column_block_idx)
""",
        """    uint32_t block_size,
    uint32_t output_shard_width,
    uint32_t input_row_pitch_bytes) {
    // Dictionary to group transfers by (dst_shard_idx, column_block_idx)
""",
    ),
    (
        HWC_GATHER_C,
        "gather.cpp: pass pitch to lower_gather_transfers",
        """        transfers, B, C, HW, input_cores, num_output_cores, element_size_bytes, output_shard_width);

    // Group transfers by which column blocks they write to, splitting transfers that cross boundaries
""",
        """        transfers,
        B,
        C,
        HW,
        input_cores,
        num_output_cores,
        element_size_bytes,
        output_shard_width,
        input_row_pitch_bytes);

    // Group transfers by which column blocks they write to, splitting transfers that cross boundaries
""",
    ),
    (
        HWC_GATHER_C,
        "gather.cpp: split transfer source bytes",
        """                (low_level.src_offset + src_offset_in_transfer) * element_size_bytes,  // Source offset in bytes
""",
        """                low_level.src_offset_bytes + (src_offset_in_transfer * element_size_bytes),  // Source offset in bytes
""",
    ),
    (
        HWC_F,
        "convert_to_hwc: group_and_coalesce signature",
        """    uint32_t effective_hw_for_gather,
    uint32_t block_size_width) {
    // Use the actual output shard width for transfer generation (determines which output core)
""",
        """    uint32_t effective_hw_for_gather,
    uint32_t block_size_width,
    uint32_t input_row_pitch_bytes) {
    // Use the actual output shard width for transfer generation (determines which output core)
""",
    ),
    (
        HWC_F,
        "convert_to_hwc: pass pitch to grouping",
        """        /*output_shard_width=*/config.gather_l1_output_shard_width);

    auto blocked_gather_transfers = blocked_result.blocked_transfers;
""",
        """        /*output_shard_width=*/config.gather_l1_output_shard_width,
        /*input_row_pitch_bytes=*/input_row_pitch_bytes);

    auto blocked_gather_transfers = blocked_result.blocked_transfers;
""",
    ),
    (
        HWC_F,
        "convert_to_hwc: call site",
        """    auto grouping = group_and_coalesce_transfers(config, in_cores, effective_hw_for_gather, block_width);
""",
        """    // Input rows sit at the input buffer's own aligned page size (L1 and DRAM differ).
    auto grouping = group_and_coalesce_transfers(
        config,
        in_cores,
        effective_hw_for_gather,
        block_width,
        static_cast<uint32_t>(a.buffer()->aligned_page_size()));
""",
    ),
    # ------------------------------------------------------------------ 6. indexed_fill native path
    (
        FILL_F,
        "indexed_fill: native DFB entry size",
        """    const uint32_t kernel_rounded_page_size = is_shard_local ? round_up_to_mul32(shard_page_size)
                                              : is_native     ? rounded_page_size
                                                              : generic_aligned_page_size;
""",
        """    // Native path: the data DFB is the output shard, whose rows sit at the buffer's aligned page size.
    const uint32_t native_page_stride =
        is_tile ? rounded_page_size : static_cast<uint32_t>(output_buffer->aligned_page_size());
    const uint32_t kernel_rounded_page_size = is_shard_local ? round_up_to_mul32(shard_page_size)
                                              : is_native     ? native_page_stride
                                                              : generic_aligned_page_size;
""",
    ),
    (
        FILL_F,
        "indexed_fill: page_stride compile-time arg",
        """        .compile_time_args = {{"page_size", kernel_page_size}, {"mode", kernel_mode}},
""",
        """        .compile_time_args =
            {{"page_size", kernel_page_size},
             {"mode", kernel_mode},
             {"page_stride", is_native ? native_page_stride : kernel_page_size}},
""",
    ),
    (
        FILL_R,
        "indexed_fill reader: bulk read covers the pitch",
        """                dfb_in0.reserve_back(batch_size_in_pages);
                noc.async_read(
                    s0, dfb_in0, page_size * batch_size_in_pages, {.page_id = start_id}, {.offset_bytes = 0});
""",
        """                // Rows sit page_stride apart in the shard, so the slab is page_stride bytes per row.
                constexpr uint32_t page_stride = get_arg(args::page_stride);
                dfb_in0.reserve_back(batch_size_in_pages);
                noc.async_read(
                    s0, dfb_in0, page_stride * batch_size_in_pages, {.page_id = start_id}, {.offset_bytes = 0});
""",
    ),
]


def main():
    by_path = {}
    for path, label, old, new in EDITS:
        by_path.setdefault(path, []).append((label, old, new))
    for path, edits in by_path.items():
        with open(path) as f:
            s = f.read()
        for label, old, new in edits:
            assert s.count(old) == 1, (path, label)
            s = s.replace(old, new)
        with open(path, "w") as f:
            f.write(s)
        print(f"patched {path} ({len(edits)} edits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
