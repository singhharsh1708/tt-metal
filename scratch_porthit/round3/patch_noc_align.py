#!/usr/bin/env python3
"""Keep NoC reads aligned in reshape_on_device (RM), embedding and concat (RM interleaved).

tt-metal at 4502c6d9c57. Run from the repo root.

A direct NoC read is only delivered correctly when its source and its local destination agree modulo
the read alignment of the source (Wormhole: L1 16 B, DRAM 32 B; Blackhole DRAM 64 B). Three ops place
reads at offsets that break this:

  reshape_on_device RM  packs old sticks at k * old_stick_size in the staging CB
  embedding             sizes the staging entry from the index buffer, and caches BINARY row 1 at
                        cache_base + weight_stick_size
  concat RM             sizes the entry from input 0 and the output only, and packs width pieces at
                        the running byte offset

Fixes: embedding and concat on an outer dim only widen the host-side entry so every read lands
aligned. reshape and width concat keep the direct read when the destination is aligned and otherwise
read into an aligned scratch slot and move the bytes with tt_memmove after the barrier.

Each edit is an exact string replacement whose anchor must occur exactly once in the file. All
anchors are checked before anything is written.
"""
import sys

OPS = "ttnn/cpp/ttnn/operations/"
RS_K = (
    OPS
    + "data_movement/reshape_on_device/device/kernels/dataflow/"
    + "reader_unary_reshape_stick_layout_interleaved_multi_core.cpp"
)
RS_F = OPS + "data_movement/reshape_on_device/device/reshape_rm_program_factory.cpp"
EM_RM = OPS + "embedding/device/embeddings_rm_program_factory.cpp"
EM_TI = OPS + "embedding/device/embeddings_tilized_indices_program_factory.cpp"
EM_FU = OPS + "embedding/device/embeddings_fused_program_factory.cpp"
EM_H2 = OPS + "embedding/device/kernels/dataflow/embeddings_common_metal2.hpp"
EM_H1 = OPS + "embedding/device/kernels/dataflow/embeddings_common.hpp"
CC_K = OPS + "data_movement/concat/device/kernels/dataflow/reader_concat_stick_layout_interleaved_start_id.cpp"
CC_F = OPS + "data_movement/concat/device/concat_program_factory.cpp"

HAL_INCLUDE_OLD = """#include <tt-metalium/host_api.hpp>
"""
HAL_INCLUDE_NEW = """#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
"""

CACHE_OLD = """        uint32_t cache_page_size = round_up_to_mul32(weight_page_size);
"""
CACHE_NEW = """        // Each cached row gets a slot that keeps the DRAM read alignment.
        const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
        const uint32_t cache_alignment = dram_alignment > 32 ? dram_alignment : 32;
        uint32_t cache_page_size = ((weight_page_size + cache_alignment - 1) / cache_alignment) * cache_alignment;
"""

ROW1_OLD = """    one_local_addr = zero_local_addr + weight_stick_size;
"""
ROW1_NEW = """    // Row 1 starts one read-aligned stride after row 0 so that its NoC read is aligned as well.
    one_local_addr = zero_local_addr + ((weight_stick_size + NOC_DRAM_READ_ALIGNMENT_BYTES - 1) /
                                        NOC_DRAM_READ_ALIGNMENT_BYTES) * NOC_DRAM_READ_ALIGNMENT_BYTES;
"""

NOC_PARAMS_OLD = """#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
"""
NOC_PARAMS_NEW = """#include "api/dataflow/dataflow_api.h"
#include "noc_parameters.h"
#include "api/dataflow/noc.h"
"""

# (path, label, old, new)
EDITS = [
    # ------------------------------------------------------------------ reshape_on_device RM
    (
        RS_K,
        "reshape kernel: includes",
        """#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
""",
        """#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
""",
    ),
    (
        RS_K,
        "reshape kernel: compile args",
        """    constexpr uint32_t old_stick_size = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();

    constexpr auto dfb_in0 = tt::CBIndex::c_0;
""",
        """    constexpr uint32_t old_stick_size = get_compile_time_arg_val(0);
    constexpr uint32_t src_alignment = get_compile_time_arg_val(1);
    constexpr uint32_t scratch_stride = ((old_stick_size + src_alignment - 1) / src_alignment) * src_alignment;
    constexpr auto src_args = TensorAccessorArgs<2>();

    constexpr auto dfb_in0 = tt::CBIndex::c_0;
    constexpr auto dfb_scratch0 = tt::CBIndex::c_1;
""",
    ),
    (
        RS_K,
        "reshape kernel: scratch address",
        """    DataflowBuffer dfb_input(dfb_in0);
""",
        """    DataflowBuffer dfb_input(dfb_in0);
    DataflowBuffer dfb_scratch(dfb_scratch0);
    const uint32_t scratch_addr = dfb_scratch.get_write_ptr();
""",
    ),
    (
        RS_K,
        "reshape kernel: aligned or bounced read",
        """        dfb_input.reserve_back(num_sticks_per_cb_push);
        uint32_t cb_write_offset = 0;

        for (uint32_t i = 0; i < num_read_per_barrier; ++i) {
            noc.async_read(
                s,
                dfb_input,
                old_stick_size,
                {.page_id = i_stick, .offset_bytes = 0},
                {.offset_bytes = cb_write_offset});
            cb_write_offset += old_stick_size;
            i_stick++;
        }
        noc.async_read_barrier();
        dfb_input.push_back(num_sticks_per_cb_push);
""",
        """        dfb_input.reserve_back(num_sticks_per_cb_push);
        const uint32_t cb_write_addr = dfb_input.get_write_ptr();
        uint32_t cb_write_offset = 0;

        for (uint32_t i = 0; i < num_read_per_barrier; ++i) {
            if (((cb_write_addr + cb_write_offset) & (src_alignment - 1)) == 0) {
                noc.async_read(
                    s,
                    dfb_input,
                    old_stick_size,
                    {.page_id = i_stick, .offset_bytes = 0},
                    {.offset_bytes = cb_write_offset});
            } else {
                // A NoC read needs source and destination equal modulo the read alignment, so bounce through scratch.
                noc.async_read(
                    s,
                    CoreLocalMem<uint32_t>(scratch_addr + i * scratch_stride),
                    old_stick_size,
                    {.page_id = i_stick, .offset_bytes = 0},
                    {});
            }
            cb_write_offset += old_stick_size;
            i_stick++;
        }
        noc.async_read_barrier();
        for (uint32_t i = 0; i < num_read_per_barrier; ++i) {
            const uint32_t dst_addr = cb_write_addr + i * old_stick_size;
            if ((dst_addr & (src_alignment - 1)) != 0) {
                tt::data_movement::common::tt_memmove<false, false, false, 0>(
                    noc, dst_addr, scratch_addr + i * scratch_stride, old_stick_size);
            }
        }
        dfb_input.push_back(num_sticks_per_cb_push);
""",
    ),
    (
        RS_F,
        "reshape factory: reader compile args",
        """    std::vector<uint32_t> reader_ct_args = {old_stick_size};
""",
        """    const uint32_t src_alignment = src0_buffer->alignment();
    std::vector<uint32_t> reader_ct_args = {old_stick_size, src_alignment};
""",
    ),
    (
        RS_F,
        "reshape factory: batch size tracker",
        """    constexpr uint32_t max_read_size = 2048;
    uint32_t curr_sticks_read = 0;
""",
        """    constexpr uint32_t max_read_size = 2048;
    uint32_t max_old_sticks_read_per_barrier = 1;
    uint32_t curr_sticks_read = 0;
""",
    ),
    (
        RS_F,
        "reshape factory: track batch size",
        """        // Per-core runtime args. Slot 0 holds the buffer address; idle cores (no sticks
""",
        """        max_old_sticks_read_per_barrier = std::max(max_old_sticks_read_per_barrier, num_old_sticks_read_per_barrier);

        // Per-core runtime args. Slot 0 holds the buffer address; idle cores (no sticks
""",
    ),
    (
        RS_F,
        "reshape factory: scratch CB",
        """    desc.kernels.push_back(std::move(reader_desc));
""",
        """    // Scratch for the sticks of one read batch whose packed offset is off the source read alignment.
    constexpr uint32_t scratch_cb_index = 1;
    const uint32_t scratch_stride = ((old_stick_size + src_alignment - 1) / src_alignment) * src_alignment;
    const uint32_t scratch_cb_size = max_old_sticks_read_per_barrier * scratch_stride;
    desc.cbs.push_back(CBDescriptor{
        .total_size = scratch_cb_size,
        .core_ranges = total_core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(scratch_cb_index),
            .data_format = cb_data_format,
            .page_size = scratch_cb_size,
        }}},
    });

    desc.kernels.push_back(std::move(reader_desc));
""",
    ),
    # ------------------------------------------------------------------ embedding
    (
        EM_RM,
        "embedding rm factory: staging alignment",
        """    uint32_t rounded_weight_page_size = tt::align(weight_page_size, alignment);
""",
        """    // An interleaved staging entry also has to keep the weight buffer's read alignment.
    const uint32_t weights_alignment = weights.buffer()->alignment();
    const uint32_t staging_alignment =
        (!output_sharded && weights_alignment > alignment) ? weights_alignment : alignment;
    uint32_t rounded_weight_page_size = tt::align(weight_page_size, staging_alignment);
""",
    ),
    (EM_RM, "embedding rm factory: hal include", HAL_INCLUDE_OLD, HAL_INCLUDE_NEW),
    (EM_RM, "embedding rm factory: cache slot", CACHE_OLD, CACHE_NEW),
    (
        EM_TI,
        "embedding tilized-indices factory: staging alignment",
        """    uint32_t rounded_weight_page_size = tt::align(weight_page_size, alignment);
""",
        """    // The staging entry also has to keep the weight buffer's read alignment.
    const uint32_t weights_alignment = weights.buffer()->alignment();
    const uint32_t staging_alignment = weights_alignment > alignment ? weights_alignment : alignment;
    uint32_t rounded_weight_page_size = tt::align(weight_page_size, staging_alignment);
""",
    ),
    (EM_TI, "embedding tilized-indices factory: hal include", HAL_INCLUDE_OLD, HAL_INCLUDE_NEW),
    (EM_TI, "embedding tilized-indices factory: cache slot", CACHE_OLD, CACHE_NEW),
    (EM_FU, "embedding fused factory: hal include", HAL_INCLUDE_OLD, HAL_INCLUDE_NEW),
    (EM_FU, "embedding fused factory: cache slot", CACHE_OLD, CACHE_NEW),
    (EM_H2, "embedding metal2 header: noc_parameters include", NOC_PARAMS_OLD, NOC_PARAMS_NEW),
    (EM_H2, "embedding metal2 header: BINARY row 1 address", ROW1_OLD, ROW1_NEW),
    (EM_H1, "embedding legacy header: noc_parameters include", NOC_PARAMS_OLD, NOC_PARAMS_NEW),
    (EM_H1, "embedding legacy header: BINARY row 1 address", ROW1_OLD, ROW1_NEW),
    # ------------------------------------------------------------------ concat RM interleaved
    (
        CC_K,
        "concat kernel: includes",
        """#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
""",
        """#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
""",
    ),
    (
        CC_K,
        "concat kernel: aligned or bounced width pieces",
        """        for (uint32_t j = 0; j < num_tensors; ++j) {
            // The per-tensor page sizes are compile-time varargs: baked into the program, but
            // selected here by a value that advances at run time.
            auto page_size = get_compile_time_vararg(curr_tensor);
            noc.async_read(
                abstract_tensor_accessor_wrappers[curr_tensor],
                CoreLocalMem<uint8_t>(l1_write_addr),
                page_size,
                {.page_id = page_id_per_tensor[curr_tensor]},
                {});
            l1_write_addr += page_size;
            page_id_per_tensor[curr_tensor]++;
            curr_tensor++;
        }
        curr_tensor = 0;
""",
        """        // After the page sizes the varargs hold each tensor's read alignment, the scratch offset
        // inside the entry (0: no scratch) and the alignment of a scratch slot.
        const uint32_t scratch_offset = get_compile_time_vararg(2 * num_tensors);
        const uint32_t scratch_alignment = get_compile_time_vararg(2 * num_tensors + 1);
        const uint32_t stick_addr = l1_write_addr;
        uint32_t scratch_addr = stick_addr + scratch_offset;
        for (uint32_t j = 0; j < num_tensors; ++j) {
            // The per-tensor page sizes are compile-time varargs: baked into the program, but
            // selected here by a value that advances at run time.
            auto page_size = get_compile_time_vararg(curr_tensor);
            const uint32_t read_alignment = get_compile_time_vararg(num_tensors + curr_tensor);
            // A NoC read needs source and destination equal modulo the read alignment, else it goes to scratch.
            const bool direct = scratch_offset == 0 || (l1_write_addr & (read_alignment - 1)) == 0;
            noc.async_read(
                abstract_tensor_accessor_wrappers[curr_tensor],
                CoreLocalMem<uint8_t>(direct ? l1_write_addr : scratch_addr),
                page_size,
                {.page_id = page_id_per_tensor[curr_tensor]},
                {});
            if (!direct) {
                scratch_addr += (page_size + scratch_alignment - 1) & ~(scratch_alignment - 1);
            }
            l1_write_addr += page_size;
            page_id_per_tensor[curr_tensor]++;
            curr_tensor++;
        }
        curr_tensor = 0;
        if (scratch_offset != 0) {
            noc.async_read_barrier();
            l1_write_addr = stick_addr;
            scratch_addr = stick_addr + scratch_offset;
            for (uint32_t j = 0; j < num_tensors; ++j) {
                const uint32_t page_size = get_compile_time_vararg(j);
                const uint32_t read_alignment = get_compile_time_vararg(num_tensors + j);
                if ((l1_write_addr & (read_alignment - 1)) != 0) {
                    tt::data_movement::common::tt_memmove<false, false, false, 0>(
                        noc, l1_write_addr, scratch_addr, page_size);
                    scratch_addr += (page_size + scratch_alignment - 1) & ~(scratch_alignment - 1);
                }
                l1_write_addr += page_size;
            }
        }
""",
    ),
    (
        CC_F,
        "concat factory: entry alignment and width scratch",
        """    const uint32_t common_align_len = std::max(
        inputs[0].get().mesh_buffer().get_reference_buffer()->alignment(),
        output.mesh_buffer().get_reference_buffer()->alignment());
    if (rm_layout) {
        num_output_pages = output.physical_volume() / output.padded_shape()[-1];
        single_page_size = tt::align(output.element_size() * output.padded_shape()[-1], common_align_len);
    } else {
""",
        """    // Every entry has to keep the read alignment of each input buffer, not only of the first one.
    uint32_t common_align_len = output.mesh_buffer().get_reference_buffer()->alignment();
    std::vector<uint32_t> read_align_per_tensor(num_input_tensors);
    for (uint32_t i = 0; i < num_input_tensors; ++i) {
        read_align_per_tensor[i] = inputs[i].get().mesh_buffer().get_reference_buffer()->alignment();
        common_align_len = std::max(common_align_len, read_align_per_tensor[i]);
    }
    uint32_t width_scratch_offset = 0;
    if (rm_layout) {
        num_output_pages = output.physical_volume() / output.padded_shape()[-1];
        single_page_size = tt::align(output.element_size() * output.padded_shape()[-1], common_align_len);
        if (dim == output.padded_shape().rank() - 1) {
            // A width piece that lands off its source's read alignment is read into scratch behind the stick.
            uint32_t scratch_size = 0;
            uint32_t piece_offset = 0;
            for (uint32_t i = 0; i < num_input_tensors; ++i) {
                const uint32_t piece_size = static_cast<uint32_t>(inputs[i].get().mesh_buffer().page_size());
                if (piece_offset % read_align_per_tensor[i] != 0) {
                    scratch_size += tt::align(piece_size, common_align_len);
                }
                piece_offset += piece_size;
            }
            if (scratch_size != 0) {
                width_scratch_offset = single_page_size;
                single_page_size += scratch_size;
            }
        }
    } else {
""",
    ),
    (
        CC_F,
        "concat factory: reader compile-time varargs",
        """        reader_advanced_options.compile_time_varargs = page_size_per_tensor;
""",
        """        // Then each tensor's read alignment, the scratch offset in the entry and the scratch slot alignment.
        std::vector<uint32_t> reader_ct_varargs = page_size_per_tensor;
        reader_ct_varargs.insert(reader_ct_varargs.end(), read_align_per_tensor.cbegin(), read_align_per_tensor.cend());
        reader_ct_varargs.push_back(width_scratch_offset);
        reader_ct_varargs.push_back(common_align_len);
        reader_advanced_options.compile_time_varargs = std::move(reader_ct_varargs);
""",
    ),
]


def main():
    by_path = {}
    for path, label, old, new in EDITS:
        by_path.setdefault(path, []).append((label, old, new))
    patched = {}
    for path, edits in by_path.items():
        with open(path) as f:
            s = f.read()
        for label, old, new in edits:
            assert s.count(old) == 1, (path, label)
            s = s.replace(old, new)
        patched[path] = s
    for path, s in patched.items():
        with open(path, "w") as f:
            f.write(s)
        print(f"patched {path} ({len(by_path[path])} edits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
