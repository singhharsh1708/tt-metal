#!/usr/bin/env python3
"""convert_to_hwc: bounce NoC reads whose source and destination disagree modulo the read alignment.

Applies on top of patch_rowpitch.py, tt-metal at 4502c6d9c57. Run from the repo root.

After the row-pitch patch every source offset is a multiple of the buffer alignment plus a column
offset, while the destination in the staging CB is channel * block + column. With a shard row that is
not a multiple of the NoC read alignment (L1 16 B, DRAM 32 B on Wormhole, 64 B on Blackhole) the two
differ modulo that alignment for part of the input cores, and such a NoC read is not delivered
correctly. Those transfers now go through a small scratch CB at a congruent address and are copied
into place on the core.

Each edit is an exact string replacement whose anchor must occur exactly once in the file. All
anchors are checked before anything is written.
"""
import sys

HWC = "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_hwc/device/"
HWC_FH = HWC + "convert_to_hwc_program_factory.hpp"
HWC_F = HWC + "convert_to_hwc_program_factory.cpp"
HWC_K = HWC + "kernels/writer_convert_to_hwc.cpp"

# (path, label, old, new)
EDITS = [
    (
        HWC_FH,
        "factory.hpp: scratch CB index",
        """constexpr uint32_t CB_OUT = tt::CBIndex::c_5;
""",
        """constexpr uint32_t CB_OUT = tt::CBIndex::c_5;
constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_6;
""",
    ),
    (
        HWC_F,
        "factory.cpp: scratch sizes",
        """struct GroupingResult {
    std::vector<std::vector<BlockedTransferGroup>> per_core_groups;
    uint32_t num_blocks;
};
""",
        """struct GroupingResult {
    std::vector<std::vector<BlockedTransferGroup>> per_core_groups;
    uint32_t num_blocks;
};

// Scratch CB for reads that cannot land in the staging CB directly: one chunk plus the largest NoC read alignment.
constexpr uint32_t SCRATCH_CHUNK_BYTES = 2048;
constexpr uint32_t SCRATCH_LEAD_BYTES = 64;
""",
    ),
    (
        HWC_F,
        "factory.cpp: writer compile args",
        """        tiling.output_addr_stride,
        tiling.block_size_bytes};
}
""",
        """        tiling.output_addr_stride,
        tiling.block_size_bytes,
        CBIndex::CB_SCRATCH,
        SCRATCH_CHUNK_BYTES};
}
""",
    ),
    (
        HWC_F,
        "factory.cpp: create scratch CB",
        """        program, core_grid, CBIndex::CB_IN_BATCH, cb_in_batch_total_size, cb_in_batch_page_size, config.input_format);
""",
        """        program, core_grid, CBIndex::CB_IN_BATCH, cb_in_batch_total_size, cb_in_batch_page_size, config.input_format);

    const uint32_t cb_scratch_size = SCRATCH_CHUNK_BYTES + SCRATCH_LEAD_BYTES;
    create_circular_buffer(
        program, core_grid, CBIndex::CB_SCRATCH, cb_scratch_size, cb_scratch_size, config.input_format);
""",
    ),
    (
        HWC_K,
        "kernel: includes",
        """#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
""",
        """#include "api/dataflow/dataflow_api.h"
#include "noc_parameters.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
""",
    ),
    (
        HWC_K,
        "kernel: scratch compile args",
        """    constexpr uint32_t block_size_bytes = get_compile_time_arg_val(13);
""",
        """    constexpr uint32_t block_size_bytes = get_compile_time_arg_val(13);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(14);
    constexpr uint32_t scratch_chunk_bytes = get_compile_time_arg_val(15);

    constexpr uint32_t noc_read_alignment =
        is_input_in_dram ? NOC_DRAM_READ_ALIGNMENT_BYTES : NOC_L1_READ_ALIGNMENT_BYTES;
    static_assert(scratch_chunk_bytes % noc_read_alignment == 0, "scratch chunk must keep the read alignment");
""",
    ),
    (
        HWC_K,
        "kernel: scratch CB object",
        """    experimental::CB cb_out_obj(cb_out);
""",
        """    experimental::CB cb_out_obj(cb_out);
    experimental::CB cb_scratch_obj(cb_scratch);
""",
    ),
    (
        HWC_K,
        "kernel: aligned or bounced read",
        """                // dst_offset_bytes is already relative to block buffer start (includes channel * block_size + column)
                if constexpr (is_input_in_dram) {
                    // DRAM bank-id read via the AllocatorBank<DRAM> endpoint. Folding src_offset_bytes into the
                    // bank address offset is equivalent to the legacy bank-id addr-gen result plus
                    // src_offset_bytes: both land the offset in the NOC address's low bits below
                    // NOC_ADDR_COORD_SHIFT (see the DRAM bank-id addr-gen in dataflow_api_addrgen.h).
                    AllocatorBank<AllocatorBankType::DRAM> dram_bank;
                    noc.async_read(
                        dram_bank,
                        cb_in_batch_obj,
                        transfer_size_bytes,
                        {.bank_id = bank_id, .addr = dram_base_read_addr + src_offset_bytes},
                        {.offset_bytes = dst_offset_bytes});
                } else {
                    UnicastEndpoint src_ep;
                    noc.async_read(
                        src_ep,
                        cb_in_batch_obj,
                        transfer_size_bytes,
                        {.noc_x = src_x, .noc_y = src_y, .addr = cb_in_obj.get_read_ptr() + src_offset_bytes},
                        {.offset_bytes = dst_offset_bytes});
                }
            }
""",
        """                const uint32_t src_base_addr = is_input_in_dram ? dram_base_read_addr : cb_in_obj.get_read_ptr();
                auto read_input = [&](
                                      const experimental::CB& dst_cb,
                                      uint32_t from_addr,
                                      uint32_t dst_offset,
                                      uint32_t size) {
                    if constexpr (is_input_in_dram) {
                        // DRAM bank-id read via the AllocatorBank<DRAM> endpoint. Folding the source offset into
                        // the bank address offset is equivalent to the legacy bank-id addr-gen result plus the
                        // offset: both land the offset in the NOC address's low bits below
                        // NOC_ADDR_COORD_SHIFT (see the DRAM bank-id addr-gen in dataflow_api_addrgen.h).
                        AllocatorBank<AllocatorBankType::DRAM> dram_bank;
                        noc.async_read(
                            dram_bank,
                            dst_cb,
                            size,
                            {.bank_id = bank_id, .addr = from_addr},
                            {.offset_bytes = dst_offset});
                    } else {
                        UnicastEndpoint src_ep;
                        noc.async_read(
                            src_ep,
                            dst_cb,
                            size,
                            {.noc_x = src_x, .noc_y = src_y, .addr = from_addr},
                            {.offset_bytes = dst_offset});
                    }
                };

                // dst_offset_bytes is already relative to block buffer start (includes channel * block_size + column)
                const uint32_t src_addr = src_base_addr + src_offset_bytes;
                const uint32_t dst_addr = cb_in_batch_obj.get_write_ptr() + dst_offset_bytes;
                if (((src_addr ^ dst_addr) & (noc_read_alignment - 1)) == 0) {
                    read_input(cb_in_batch_obj, src_addr, dst_offset_bytes, transfer_size_bytes);
                    continue;
                }

                // A NoC read needs source and destination equal modulo the read alignment, so bounce through scratch.
                const uint32_t scratch_addr = cb_scratch_obj.get_write_ptr();
                const uint32_t lead = (src_addr - scratch_addr) & (noc_read_alignment - 1);
                for (uint32_t done = 0; done < transfer_size_bytes; done += scratch_chunk_bytes) {
                    const uint32_t remaining = transfer_size_bytes - done;
                    const uint32_t chunk = remaining < scratch_chunk_bytes ? remaining : scratch_chunk_bytes;
                    read_input(cb_scratch_obj, src_addr + done, lead, chunk);
                    noc.async_read_barrier();
                    tt::data_movement::common::tt_memmove<false, false, false, 0>(
                        noc, dst_addr + done, scratch_addr + lead, chunk);
                }
            }
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
