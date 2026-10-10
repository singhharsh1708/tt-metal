#!/usr/bin/env python3
"""Source fixes for tt-metal at 4502c6d9c57. Run from the repo root: python3 patch_aux_misc.py"""
import sys
from pathlib import Path

OPS = "ttnn/cpp/ttnn/operations/"
EDITS = {}


def edit(path, label, old, new):
    EDITS.setdefault(OPS + path, []).append((label, old, new))


# A1 paged_fill_cache: TILE page_table
edit(
    "experimental/paged_cache/paged_cache.cpp",
    "A1 convert page_table at the public entry",
    """    (void)compute_kernel_config;

    return ttnn::prim::paged_fill_cache(
        cache_tensor,
        input_tensor,
        page_table,
""",
    """    (void)compute_kernel_config;

    // The writer reads the page table as flat row-major sticks.
    const Tensor page_table_rm =
        page_table.layout() == Layout::ROW_MAJOR ? page_table : ttnn::to_layout(page_table, Layout::ROW_MAJOR);

    return ttnn::prim::paged_fill_cache(
        cache_tensor,
        input_tensor,
        page_table_rm,
""",
)
edit(
    "experimental/paged_cache/device/fill_cache/paged_fill_cache_device_operation.cpp",
    "A1 reject non ROW_MAJOR page_table in the prim",
    """    TT_FATAL(page_table_tensor.dtype() == DataType::INT32, "Expect page_table_tensor to have datatype INT32");
""",
    """    TT_FATAL(page_table_tensor.dtype() == DataType::INT32, "Expect page_table_tensor to have datatype INT32");
    TT_FATAL(
        page_table_tensor.layout() == Layout::ROW_MAJOR, "Expect page table to have memory layout in ROW MAJOR");
""",
)
edit(
    "experimental/paged_cache/device/fill_cache/paged_fill_cache_device_operation.cpp",
    "A1 bound checks on the logical page_table shape",
    """    auto page_table_shape = page_table_tensor.padded_shape();
""",
    """    auto page_table_shape = page_table_tensor.logical_shape();
""",
)

# A2 sdpa_decode: TILE cur_pos_tensor on the non-paged path
edit(
    "transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp",
    "A2 reject non ROW_MAJOR cur_pos on every causal path",
    """    if (operation_attributes.paged_attention) {
        // Paged attention verification
""",
    """    if (operation_attributes.is_causal && tensor_args.cur_pos_tensor.has_value()) {
        TT_FATAL(
            tensor_args.cur_pos_tensor->layout() == Layout::ROW_MAJOR,
            "Expect cur_pos to be ROW_MAJOR, got {}",
            tensor_args.cur_pos_tensor->layout());
    }

    if (operation_attributes.paged_attention) {
        // Paged attention verification
""",
)

# A3 ring distributed sdpa: TILE page_table
edit(
    "transformer/sdpa/device/ring_distributed_sdpa_device_operation.cpp",
    "A3 reject non ROW_MAJOR page_table",
    """            "page_table tensor must have INT32 dtype. Got {}",
            page_table_tensor.dtype());
""",
    """            "page_table tensor must have INT32 dtype. Got {}",
            page_table_tensor.dtype());
        TT_FATAL(page_table_tensor.layout() == Layout::ROW_MAJOR, "Page table must be row major");
""",
)

# A4 moreh_getitem: index tensors with mixed layouts
edit(
    "moreh/moreh_getitem/device/moreh_getitem_device_operation.cpp",
    "A4 reject mixed index layouts",
    """        TT_FATAL(index_size == index_shape[-1], "The shapes of all index tensors must be identical!");
""",
    """        TT_FATAL(index_size == index_shape[-1], "The shapes of all index tensors must be identical!");
        TT_FATAL(
            index_layout == index_tensors[0].layout(), "The layouts of all index tensors must be identical!");
""",
)

# B1 padded_slice TILE float32: 32-bit dest and unpack-to-dest
edit(
    "experimental/padded_slice/device/padded_slice_tile_program_factory.cpp",
    "B1 fp32 dest and unpack-to-dest for float32",
    """    untilize_compute_kernel.config = ComputeConfigDescriptor{.fp32_dest_acc_en = false, .dst_full_sync_en = false};
""",
    """    // Float32 goes through a 16-bit SrcA/dest unless both of these are set.
    const bool fp32_untilize = a.dtype() == DataType::FLOAT32;
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_untilize) {
        unpack_to_dest_mode[cb_input_index] = UnpackToDestMode::UnpackToDestFp32;
    }
    untilize_compute_kernel.config = ComputeConfigDescriptor{
        .fp32_dest_acc_en = fp32_untilize,
        .dst_full_sync_en = false,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
    };
""",
)

# B2 slice_write RM interleaved: per-core batch tied to the CB ring
SW = "experimental/slice_write/device/slice_write_rm_interleaved_program_factory.cpp"
edit(
    SW,
    "B2 runtime-args signature",
    """    uint32_t num_sticks_per_core_group_2,
    uint32_t max_read_size) {
""",
    """    uint32_t num_sticks_per_core_group_2,
    uint32_t ring_read_per_barrier) {
""",
)
edit(
    SW,
    "B2 per-core batch",
    """        if (num_sticks_per_core != 0) {
            auto num_sticks_per_core_pad32 = num_sticks_per_core + ((32 - num_sticks_per_core % 32) % 32);
            num_sticks_per_core_read =
                tt::tt_metal::merge_num_sticks_to_read(num_sticks_per_core_pad32, input_row_size_bytes, max_read_size);
            num_read_per_barrier = num_sticks_per_core_pad32 / num_sticks_per_core_read;
        }
""",
    """        if (num_sticks_per_core != 0) {
            // Every core batches by the count the CB ring was sized for.
            num_read_per_barrier = ring_read_per_barrier;
            num_sticks_per_core_read = (num_sticks_per_core + num_read_per_barrier - 1) / num_read_per_barrier;
        }
""",
)
edit(
    SW,
    "B2 call site",
    """        num_sticks_per_core_group_2,
        max_read_size);
""",
    """        num_sticks_per_core_group_2,
        num_read_per_barrier);
""",
)

# B3 nonzero: block-float input
NZ = "data_movement/non_zero_indices/"
edit(
    NZ + "non_zero_indices.cpp",
    "B3 typecast include",
    """#include "ttnn/operations/data_movement/non_zero_indices/device/non_zero_indices_device_operation.hpp"
""",
    """#include "ttnn/operations/data_movement/non_zero_indices/device/non_zero_indices_device_operation.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
""",
)
edit(
    NZ + "non_zero_indices.cpp",
    "B3 typecast block-float input at the public entry",
    """    auto [output_0, output_1] = ttnn::prim::nonzero(input_tensor, memory_config.value_or(default_mc));
""",
    """    // Block-float tiles share exponents, so the per-element scan needs a plain dtype.
    const bool is_block_float = input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT8_B ||
                                input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT4_B;
    const Tensor scan_input =
        is_block_float ? ttnn::typecast(input_tensor, tt::tt_metal::DataType::BFLOAT16) : input_tensor;
    auto [output_0, output_1] = ttnn::prim::nonzero(scan_input, memory_config.value_or(default_mc));
""",
)
edit(
    NZ + "device/non_zero_indices_device_operation.cpp",
    "B3 reject block-float dtypes in the prim",
    """        "non_zero_indices: unsupported element size {} bytes; only 1, 2, or 4-byte dtypes are supported",
        elem_size);
""",
    """        "non_zero_indices: unsupported element size {} bytes; only 1, 2, or 4-byte dtypes are supported",
        elem_size);
    TT_FATAL(
        input_tensor.dtype() != DataType::BFLOAT8_B && input_tensor.dtype() != DataType::BFLOAT4_B,
        "non_zero_indices: block-float dtype {} is not supported; typecast to BFLOAT16 first",
        input_tensor.dtype());
""",
)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    # check every anchor before writing anything
    patched = {}
    for path, edits in EDITS.items():
        s = (root / path).read_text()
        # main re-derives these args in override_runtime_arguments; B2 needs a port there
        assert path != OPS + SW or "override_runtime_arguments" not in s, (path, "B2 is for 4502c6d9c57 only")
        for label, old, new in edits:
            assert s.count(old) == 1, (path, label)
            s = s.replace(old, new)
        patched[path] = s
    for path, s in patched.items():
        (root / path).write_text(s)
        print(f"patched {path}: {len(EDITS[path])} edit(s): " + "; ".join(label for label, _, _ in EDITS[path]))


if __name__ == "__main__":
    main()
