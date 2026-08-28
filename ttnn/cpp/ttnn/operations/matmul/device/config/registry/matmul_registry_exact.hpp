// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <type_traits>

#include "ttnn/operations/matmul/device/config/registry/matmul_registry_descriptor.hpp"

namespace ttnn::operations::matmul::registry::compact {

// The checked runtime registry contains complete native recipes only. Model
// training and prediction happen offline before an exact entry is promoted.
struct ProgramConfigDescriptor {
    ProgramFamily family{};
    std::uint16_t compute_grid_x{};
    std::uint16_t compute_grid_y{};
    std::uint32_t in0_block_w{};
    std::uint32_t out_subblock_h{};
    std::uint32_t out_subblock_w{};
    std::uint32_t per_core_m{};
    std::uint32_t per_core_n{};
    std::uint32_t out_block_h{};
    std::uint32_t out_block_w{};
    std::uint32_t num_global_cb_receivers{};
    bool allowed_worker_cores_present{};
    bool fuse_batch{};
    bool mcast_in0{};
    bool transpose_mcast{};
    bool fused_activation_present{};
    bool gather_in0{};
    bool hop_cores_present{};
    bool untilize_out{};
    bool stream_in1{};

    auto operator<=>(const ProgramConfigDescriptor&) const = default;
};

struct ProgramConfigCandidate {
    ProgramConfigDescriptor program_config{};
    ComputeKernelDescriptor compute_kernel_config{};
};

struct ProgramConfigExactEntry {
    RegistryEntryId entry_id{};
    KeyDescriptor key{};
    ProgramConfigDescriptor program_config{};
    ComputeKernelDescriptor compute_kernel_config{};

    auto operator<=>(const ProgramConfigExactEntry&) const = default;
};

static_assert(std::is_trivially_copyable_v<ProgramConfigExactEntry>);
static_assert(std::is_standard_layout_v<ProgramConfigExactEntry>);

inline constexpr const ProgramConfigExactEntry* lookup_program_config_exact(
    const KeyDescriptor& key, const std::span<const ProgramConfigExactEntry> entries) noexcept {
    const auto candidate = std::lower_bound(
        entries.begin(),
        entries.end(),
        key,
        [](const ProgramConfigExactEntry& entry, const KeyDescriptor& requested_key) {
            return entry.key < requested_key;
        });
    return candidate != entries.end() && candidate->key == key ? &*candidate : nullptr;
}

// A table entry's compute_kernel_config is the configuration the entry was
// measured under, and KeyDescriptor::compute_kernel is the configuration a call
// must be asking for to be allowed to reuse that measurement. The two are the
// same fact, so they must be spelled identically; an entry that disagrees would
// answer a lookup by silently substituting knobs the caller never asked for.
// This is checked at compile time over the emitted table and again at runtime
// before any entry is served, so neither a bad emitter nor a stale artifact can
// reintroduce the substitution.
constexpr bool entry_binds_key_compute_kernel(const ProgramConfigExactEntry& entry) noexcept {
    return entry.compute_kernel_config == entry.key.compute_kernel;
}

constexpr bool entries_bind_key_compute_kernel(const std::span<const ProgramConfigExactEntry> entries) noexcept {
    for (const auto& entry : entries) {
        if (!entry_binds_key_compute_kernel(entry)) {
            return false;
        }
    }
    return true;
}

// The runtime key normalizes math_approx_mode to false rather than keying it
// (see normalize_key_compute_kernel), which is sound only because no admitted
// call carries a fused activation and so no admitted kernel contains an SFPU op
// for the knob to configure. Every shipped entry must therefore be
// activation-free and carry the normalized spelling; an entry that is not would
// be reachable from a key that no longer describes it. This turns the runtime
// precondition into a build failure at the moment an activation-carrying entry
// is first emitted.
constexpr bool entry_permits_math_approx_normalization(const ProgramConfigExactEntry& entry) noexcept {
    return !entry.key.has_activation && !entry.program_config.fused_activation_present &&
           !entry.key.compute_kernel.math_approx_mode;
}

constexpr bool entries_permit_math_approx_normalization(
    const std::span<const ProgramConfigExactEntry> entries) noexcept {
    for (const auto& entry : entries) {
        if (!entry_permits_math_approx_normalization(entry)) {
            return false;
        }
    }
    return true;
}

// Bank evidence is portable across board identities, but not harvested worker
// grids: distinct 11x10, 12x10, and 13x10 winners remain distinct exact keys.
constexpr KeyDescriptor direct_bank_key(KeyDescriptor key) noexcept {
    key.board_capability_class = 0;
    key.topology_sha256 = {};
    return key;
}

constexpr bool legal_program_config_candidate(
    const KeyDescriptor& key, const ProgramConfigCandidate& candidate) noexcept {
    const auto& program = candidate.program_config;
    if (program.fused_activation_present || program.gather_in0 || program.hop_cores_present || program.untilize_out ||
        program.stream_in1) {
        return false;
    }
    if (program.compute_grid_x == 0 || program.compute_grid_y == 0 || program.compute_grid_x > key.compute_grid_x ||
        program.compute_grid_y > key.compute_grid_y || program.in0_block_w == 0 || program.out_subblock_h == 0 ||
        program.out_subblock_w == 0 || program.per_core_m == 0 || program.per_core_n == 0 ||
        program.allowed_worker_cores_present || key.input_a.tile_height == 0 || key.input_a.tile_width == 0 ||
        key.input_b.tile_height == 0 || key.input_b.tile_width == 0 || key.padded_m % key.input_a.tile_height != 0 ||
        key.padded_k % key.input_a.tile_width != 0 || key.padded_k % key.input_b.tile_height != 0 ||
        key.padded_n % key.input_b.tile_width != 0) {
        return false;
    }
    const auto m_tiles = key.padded_m / key.input_a.tile_height;
    const auto a_k_tiles = key.padded_k / key.input_a.tile_width;
    const auto b_k_tiles = key.padded_k / key.input_b.tile_height;
    const auto n_tiles = key.padded_n / key.input_b.tile_width;
    if (a_k_tiles != b_k_tiles || a_k_tiles % program.in0_block_w != 0 ||
        program.per_core_m % program.out_subblock_h != 0 || program.per_core_n % program.out_subblock_w != 0 ||
        static_cast<std::uint64_t>(program.out_subblock_h) * program.out_subblock_w >
            (candidate.compute_kernel_config.fp32_dest_acc_en ? 4U : 8U)) {
        return false;
    }
    switch (program.family) {
        case ProgramFamily::MultiCoreReuse:
            return program.out_block_h == 0 && program.out_block_w == 0 && program.num_global_cb_receivers == 0 &&
                   !program.fuse_batch && !program.mcast_in0 && !program.transpose_mcast &&
                   m_tiles % program.per_core_m == 0 && n_tiles == program.per_core_n;
        case ProgramFamily::MultiCast1D: {
            if (!program.fuse_batch || program.transpose_mcast || program.per_core_n > 64 ||
                program.out_block_h != program.per_core_m || program.out_block_w != program.per_core_n ||
                program.num_global_cb_receivers != 1) {
                return false;
            }
            const auto m_blocks = m_tiles / program.per_core_m + (m_tiles % program.per_core_m != 0);
            const auto n_blocks = n_tiles / program.per_core_n + (n_tiles % program.per_core_n != 0);
            const auto core_count = static_cast<std::uint64_t>(program.compute_grid_x) * program.compute_grid_y;
            const bool complete_axis =
                program.mcast_in0 ? program.per_core_m == m_tiles : program.per_core_n == n_tiles;
            return complete_axis && n_blocks != 0 && m_blocks <= core_count / n_blocks;
        }
        case ProgramFamily::MultiCast2D: {
            if (!program.fuse_batch || program.mcast_in0 || program.out_block_h != program.per_core_m ||
                program.out_block_w != program.per_core_n || program.num_global_cb_receivers != 0) {
                return false;
            }
            const auto m_blocks = m_tiles / program.per_core_m + (m_tiles % program.per_core_m != 0);
            const auto n_blocks = n_tiles / program.per_core_n + (n_tiles % program.per_core_n != 0);
            return program.transpose_mcast ? m_blocks <= program.compute_grid_x && n_blocks <= program.compute_grid_y
                                           : m_blocks <= program.compute_grid_y && n_blocks <= program.compute_grid_x;
        }
    }
    return false;
}

}  // namespace ttnn::operations::matmul::registry::compact
