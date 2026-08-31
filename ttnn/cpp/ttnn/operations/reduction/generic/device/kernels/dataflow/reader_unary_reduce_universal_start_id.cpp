// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(0);
    constexpr uint32_t tiles_per_batch = get_compile_time_arg_val(1);
    constexpr auto tensor_args = TensorAccessorArgs<2>();

    constexpr uint32_t dfb_id_in2 = 2;
    float scaler_f = __builtin_bit_cast(float, scaler_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<dfb_id_in2, REDUCE_OP, REDUCE_DIM>(scaler_f);

    constexpr uint32_t dfb_id_in0 = 0;

    auto tensor_accessor = TensorAccessor(tensor_args, src_addr);

    Noc noc;
    DataflowBuffer dfb_in0(dfb_id_in0);

    uint32_t tile_bytes = dfb_in0.get_tile_size();

    // Issue a whole batch of reads before the barrier so a core keeps that many tiles in flight;
    // barriering per tile exposes the full read latency on every one of them. The host sizes the
    // input CB at two batches, so a reservation is always contiguous: every full batch leaves the
    // write pointer batch-aligned, and a short batch can only be the last one.
    const uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id;) {
        const uint32_t batch = std::min(tiles_per_batch, end_id - i);
        dfb_in0.reserve_back(batch);
        for (uint32_t k = 0; k < batch; ++k) {
            noc.async_read(tensor_accessor, dfb_in0, tile_bytes, {.page_id = i + k}, {.offset_bytes = k * tile_bytes});
        }
        noc.async_read_barrier();
        dfb_in0.push_back(batch);
        i += batch;
    }
}
