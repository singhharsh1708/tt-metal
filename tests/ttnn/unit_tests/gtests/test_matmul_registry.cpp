// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <variant>

#include <umd/device/types/arch.hpp>

#include "ttnn/operations/matmul/device/config/matmul_config_registry.hpp"

namespace ttnn::operations::matmul::registry {
namespace {

using tt::tt_metal::BufferType;
using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::TensorMemoryLayout;

compact::Sha256 digest(const std::uint8_t byte) {
    compact::Sha256 result{};
    result.fill(byte);
    return result;
}

TensorRequest tensor_request(const DataType dtype = DataType::BFLOAT16) {
    return TensorRequest{
        .dtype = dtype,
        .layout = Layout::TILE,
        .memory_layout = TensorMemoryLayout::INTERLEAVED,
        .buffer_type = BufferType::DRAM,
        .tile_height = 32,
        .tile_width = 32};
}

CallSemantics semantics(const OperationDomain domain) {
    return domain == OperationDomain::Addmm ? addmm_call_semantics(1.0F, 0.0F) : CallSemantics{.domain = domain};
}

MatmulRegistryRequest request(const OperationDomain domain = OperationDomain::DenseMatmul) {
    return MatmulRegistryRequest{
        .schema_version = compact::kKeySchemaVersion,
        .call = semantics(domain),
        .workload = {.logical_m = 64, .logical_k = 64, .logical_n = 64, .padded_m = 64, .padded_k = 64, .padded_n = 64},
        .input_a = tensor_request(),
        .input_b = tensor_request(),
        .output = tensor_request(),
        .device = {
            .architecture = static_cast<std::uint32_t>(tt::ARCH::BLACKHOLE),
            .device_count = 1,
            .mesh_rows = 1,
            .mesh_cols = 1,
            .compute_grid_x = 13,
            .compute_grid_y = 10}};
}

MatmulRegistryRequest checked_in_request(const OperationDomain domain = OperationDomain::DenseMatmul) {
    auto result = request(domain);
    result.workload = {
        .logical_m = 32, .logical_k = 1280, .logical_n = 2304, .padded_m = 32, .padded_k = 1280, .padded_n = 2304};
    result.device.compute_grid_x = 12;
    result.device.compute_grid_y = 10;
    return result;
}

// The checked-in table carries this same M/K/N at 12x10 with a bfp4 weight, so
// only the input_b precision separates it from checked_in_request().
MatmulRegistryRequest checked_in_bfloat4b_request(const OperationDomain domain = OperationDomain::DenseMatmul) {
    auto result = checked_in_request(domain);
    result.input_b = tensor_request(DataType::BFLOAT4_B);
    return result;
}

Eligibility eligibility(const OperationDomain domain = OperationDomain::DenseMatmul) {
    return Eligibility{.call = semantics(domain)};
}

compact::ComputeKernelDescriptor kernel(const compact::ThrottleLevel throttle = compact::ThrottleLevel::NoThrottle) {
    return compact::ComputeKernelDescriptor{
        .math_fidelity = compact::MathFidelity::HiFi2,
        .throttle_level = throttle,
        .math_approx_mode = false,
        .fp32_dest_acc_en = false,
        .packer_l1_acc = true,
        .dst_full_sync_en = false};
}

// A call that names its own compute kernel configuration. Those knobs are a
// key axis, so the request and the eligibility have to agree that they are set.
MatmulRegistryRequest request_with_kernel(MatmulRegistryRequest runtime_request, compact::ComputeKernelDescriptor ckc) {
    runtime_request.user_compute_kernel_config = ckc;
    return runtime_request;
}

Eligibility eligibility_with_kernel(const OperationDomain domain = OperationDomain::DenseMatmul) {
    auto result = eligibility(domain);
    result.has_compute_kernel_config = true;
    return result;
}

compact::ProgramConfigDescriptor reuse_program(const std::uint16_t grid_x = 2) {
    return compact::ProgramConfigDescriptor{
        .family = compact::ProgramFamily::MultiCoreReuse,
        .compute_grid_x = grid_x,
        .compute_grid_y = 2,
        .in0_block_w = 1,
        .out_subblock_h = 1,
        .out_subblock_w = 1,
        .per_core_m = 1,
        .per_core_n = 2};
}

Resolution invalid_materialization_resolution(
    const MatmulRegistryRequest& runtime_request, const Eligibility&) noexcept {
    return Resolution{
        .reason = ResolutionReason::CertifiedMatch,
        .program_config = reuse_program(14),
        .compute_kernel_config = kernel(),
        .key = compact_registry_key(runtime_request)};
}

struct RuntimeStateReset {
    MatmulRegistryMode original_mode = ttnn::CONFIG.get<"matmul_registry_mode">();

    RuntimeStateReset() {
        reset_stats_for_testing();
        reset_circuit_breakers_for_testing();
        reset_startup_mode_for_testing();
    }

    ~RuntimeStateReset() {
        ttnn::CONFIG.set<"matmul_registry_mode">(original_mode);
        reset_startup_mode_for_testing();
        reset_circuit_breakers_for_testing();
        reset_stats_for_testing();
    }
};

compact::ProgramConfigDescriptor multicast_1d_program() {
    return compact::ProgramConfigDescriptor{
        .family = compact::ProgramFamily::MultiCast1D,
        .compute_grid_x = 2,
        .compute_grid_y = 2,
        .in0_block_w = 1,
        .out_subblock_h = 1,
        .out_subblock_w = 1,
        .per_core_m = 1,
        .per_core_n = 2,
        .out_block_h = 1,
        .out_block_w = 2,
        .num_global_cb_receivers = 1,
        .fuse_batch = true,
        .mcast_in0 = false};
}

compact::ProgramConfigDescriptor multicast_2d_program() {
    return compact::ProgramConfigDescriptor{
        .family = compact::ProgramFamily::MultiCast2D,
        .compute_grid_x = 2,
        .compute_grid_y = 2,
        .in0_block_w = 1,
        .out_subblock_h = 1,
        .out_subblock_w = 1,
        .per_core_m = 1,
        .per_core_n = 1,
        .out_block_h = 1,
        .out_block_w = 1,
        .fuse_batch = true,
        .mcast_in0 = false,
        .transpose_mcast = false};
}

compact::TableMetadata metadata(const bool exact = true) {
    return compact::TableMetadata{
        .lock_schema_version = 2,
        .key_schema_version = compact::kKeySchemaVersion,
        .exact_recipe_evidence_schema_version = exact ? std::uint16_t{2} : std::uint16_t{0},
        .matmul_kernel_equivalence_schema_version = exact ? std::uint16_t{1} : std::uint16_t{0},
        .content_sha256 = digest(1),
        .semantic_source_sha256 = digest(2)};
}

compact::ProgramConfigExactEntry exact_entry(
    const MatmulRegistryRequest& runtime_request,
    const compact::ProgramConfigDescriptor& program = reuse_program(),
    const compact::ComputeKernelDescriptor& ckc = kernel(),
    const std::optional<std::uint16_t> grid_x = std::nullopt) {
    auto key = *compact_registry_key(runtime_request);
    key.compute_grid_x = grid_x.value_or(key.compute_grid_x);
    // The key binds the configuration the recipe was measured under, so an
    // emitted entry always agrees with itself. compact::entry_binds_key_compute_kernel
    // is the shipped table's compile-time form of this same assignment.
    key.compute_kernel = ckc;
    return compact::ProgramConfigExactEntry{
        .entry_id = digest(11), .key = key, .program_config = program, .compute_kernel_config = ckc};
}

TEST(MatmulConfigRegistry, BlackholeKeyUsesNativeArchitectureAndPortablePhysicalFields) {
    const auto key = compact_registry_key(request());
    ASSERT_TRUE(key.has_value());
    EXPECT_EQ(key->architecture, static_cast<std::uint32_t>(tt::ARCH::BLACKHOLE));
    EXPECT_EQ(key->board_capability_class, 0U);
    EXPECT_EQ(key->topology_sha256, compact::Sha256{});
    EXPECT_EQ(key->compute_grid_x, 13);
}

TEST(MatmulConfigRegistry, ExactMatchPreservesHarvestedGridCohorts) {
    auto live = request();
    const std::array entries{
        exact_entry(live, reuse_program(1), kernel(), 11),
        exact_entry(live, reuse_program(2), kernel(), 12),
        exact_entry(live, reuse_program(3), kernel(), 13)};
    for (const std::uint32_t grid_x : {11U, 12U, 13U}) {
        live.device.compute_grid_x = grid_x;
        const auto result = resolve_with_compact_table_for_testing(live, eligibility(), metadata(), entries);
        EXPECT_EQ(result.reason, ResolutionReason::CertifiedMatch);
        ASSERT_TRUE(result.program_config.has_value());
        EXPECT_EQ(result.program_config->compute_grid_x, grid_x - 10);
    }
    live.device.compute_grid_x = 10;
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(live, eligibility(), metadata(), entries).reason,
        ResolutionReason::EmptyRegistry);
}

TEST(MatmulConfigRegistry, ExactArchitectureNeverCrossMatches) {
    const auto dense = request();
    const auto entry = exact_entry(dense);
    auto wrong_arch = dense;
    wrong_arch.device.architecture = static_cast<std::uint32_t>(tt::ARCH::WORMHOLE_B0);
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(wrong_arch, eligibility(), metadata(), {&entry, 1}).reason,
        ResolutionReason::EmptyRegistry);
}

TEST(MatmulConfigRegistry, KernelEquivalentPublicWrappersReuseDenseMeasurements) {
    const auto dense = request();
    const auto entry = exact_entry(dense, reuse_program(3), kernel());
    for (const auto domain : {OperationDomain::Linear, OperationDomain::Addmm}) {
        const auto result = resolve_with_compact_table_for_testing(
            request(domain), eligibility(domain), metadata(), std::span{&entry, std::size_t{1}});
        EXPECT_EQ(result.reason, ResolutionReason::CertifiedMatch);
        ASSERT_TRUE(result.program_config.has_value());
        EXPECT_EQ(result.program_config->compute_grid_x, 3);
    }
}

TEST(MatmulConfigRegistry, KernelEquivalentWrapperFallbackRequiresBoundEvidence) {
    const auto dense = request();
    const auto entry = exact_entry(dense);
    auto unproven = metadata();
    unproven.matmul_kernel_equivalence_schema_version = 0;
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(
            request(OperationDomain::Linear), eligibility(OperationDomain::Linear), unproven, {&entry, 1})
            .reason,
        ResolutionReason::EmptyRegistry);
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(dense, eligibility(), unproven, {&entry, 1}).reason,
        ResolutionReason::CertifiedMatch);
}

TEST(MatmulConfigRegistry, KernelEquivalentWrapperFallbackStillFailsClosedBeforeLookup) {
    const auto dense = request();
    const auto entry = exact_entry(dense);

    auto linear = request(OperationDomain::Linear);
    auto linear_eligibility = eligibility(OperationDomain::Linear);
    linear_eligibility.has_bias = true;
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(linear, linear_eligibility, metadata(), {&entry, 1}).reason,
        ResolutionReason::UnsupportedSemantics);
    linear_eligibility = eligibility(OperationDomain::Linear);
    linear_eligibility.has_activation = true;
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(linear, linear_eligibility, metadata(), {&entry, 1}).reason,
        ResolutionReason::UnsupportedSemantics);
    linear_eligibility = eligibility(OperationDomain::Linear);
    linear_eligibility.transpose_b = true;
    linear.transpose_b = true;
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(linear, linear_eligibility, metadata(), {&entry, 1}).reason,
        ResolutionReason::UnsupportedSemantics);

    auto addmm = request(OperationDomain::Addmm);
    auto addmm_eligibility = eligibility(OperationDomain::Addmm);
    addmm.call = addmm_call_semantics(2.0F, 0.0F);
    addmm_eligibility.call = addmm.call;
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(addmm, addmm_eligibility, metadata(), {&entry, 1}).reason,
        ResolutionReason::MalformedOperationSemantics);
    addmm.call = addmm_call_semantics(1.0F, 1.0F);
    addmm_eligibility.call = addmm.call;
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(addmm, addmm_eligibility, metadata(), {&entry, 1}).reason,
        ResolutionReason::UnsupportedSemantics);
}

TEST(MatmulConfigRegistry, OperationSpecificRecipePrecedesDenseWrapperFallback) {
    const auto dense = request();
    const auto linear = request(OperationDomain::Linear);
    const std::array entries{
        exact_entry(dense, reuse_program(1), kernel()), exact_entry(linear, reuse_program(3), kernel())};
    const auto result =
        resolve_with_compact_table_for_testing(linear, eligibility(OperationDomain::Linear), metadata(), entries);
    EXPECT_EQ(result.reason, ResolutionReason::CertifiedMatch);
    ASSERT_TRUE(result.program_config.has_value());
    EXPECT_EQ(result.program_config->compute_grid_x, 3);

    auto unproven = metadata();
    unproven.matmul_kernel_equivalence_schema_version = 0;
    const auto direct_result =
        resolve_with_compact_table_for_testing(linear, eligibility(OperationDomain::Linear), unproven, entries);
    EXPECT_EQ(direct_result.reason, ResolutionReason::CertifiedMatch);
    ASSERT_TRUE(direct_result.program_config.has_value());
    EXPECT_EQ(direct_result.program_config->compute_grid_x, 3);
}

TEST(MatmulConfigRegistry, ExactKeyBindsBothInputAndOutputDtypes) {
    const auto dense = request();
    const auto entry = exact_entry(dense);
    for (const auto dtype : {DataType::BFLOAT8_B, DataType::FLOAT32}) {
        auto changed = dense;
        changed.input_a.dtype = dtype;
        EXPECT_EQ(
            resolve_with_compact_table_for_testing(changed, eligibility(), metadata(), {&entry, 1}).reason,
            ResolutionReason::EmptyRegistry);
        changed = dense;
        changed.input_b.dtype = dtype;
        EXPECT_EQ(
            resolve_with_compact_table_for_testing(changed, eligibility(), metadata(), {&entry, 1}).reason,
            ResolutionReason::EmptyRegistry);
        changed = dense;
        changed.output.dtype = dtype;
        EXPECT_EQ(
            resolve_with_compact_table_for_testing(changed, eligibility(), metadata(), {&entry, 1}).reason,
            ResolutionReason::EmptyRegistry);
    }
}

TEST(MatmulConfigRegistry, ExactCarriesPairedRecipe) {
    const auto req = request_with_kernel(request(), kernel(compact::ThrottleLevel::Throttle3));
    const auto entry = exact_entry(req, reuse_program(2), kernel(compact::ThrottleLevel::Throttle3));
    const auto result = resolve_with_compact_table_for_testing(req, eligibility_with_kernel(), metadata(), {&entry, 1});
    EXPECT_EQ(result.reason, ResolutionReason::CertifiedMatch);
    ASSERT_TRUE(result.compute_kernel_config.has_value());
    EXPECT_EQ(result.compute_kernel_config->throttle_level, compact::ThrottleLevel::Throttle3);
}

TEST(MatmulConfigRegistry, EmptyAndMalformedArtifactsFallBack) {
    const auto req = request();
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(req, eligibility(), metadata(false)).reason,
        ResolutionReason::EmptyRegistry);
    auto bad_metadata = metadata();
    bad_metadata.lock_schema_version = 99;
    const auto entry = exact_entry(req);
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(req, eligibility(), bad_metadata, {&entry, 1}).reason,
        ResolutionReason::UnsupportedArtifact);
    bad_metadata = metadata();
    bad_metadata.content_sha256 = {};
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(req, eligibility(), bad_metadata, {&entry, 1}).reason,
        ResolutionReason::UnsupportedArtifact);
}

TEST(MatmulConfigRegistry, EveryExplicitTuningAxisBypassesBeforeLookup) {
    RuntimeStateReset reset;
    const auto req = request();
    ttnn::prim::MatmulParams legacy;
    for (const auto axis : {0, 1}) {
        auto eligible = eligibility();
        eligible.has_program_config = axis == 0;
        eligible.has_user_core_grid = axis == 1;
        EXPECT_EQ(preflight_v1_eligibility(eligible), ResolutionReason::ExplicitOverride);
        const auto dispatched = resolve_for_dispatch(Mode::On, req, eligible, legacy);
        EXPECT_EQ(dispatched.resolution.reason, ResolutionReason::ExplicitOverride);
        EXPECT_FALSE(dispatched.materialized_parameters.has_value());
    }
    // A caller-supplied compute kernel config is not an override of anything
    // the registry answers: it is bound into the key instead, so the lookup
    // proceeds and the dispatch leaves that field to its owner.
    EXPECT_EQ(preflight_v1_eligibility(eligibility_with_kernel()), ResolutionReason::CertifiedMatch);
}

TEST(MatmulConfigRegistry, StartupModeDefaultsOffAndFreezesOnFirstUse) {
    RuntimeStateReset reset;
    ttnn::CONFIG.set<"matmul_registry_mode">(Mode::Off);
    EXPECT_EQ(current_mode(), Mode::Off);
    EXPECT_TRUE(stats_snapshot().mode_is_frozen);
    EXPECT_EQ(stats_snapshot().frozen_mode, Mode::Off);

    reset_startup_mode_for_testing();
    ttnn::CONFIG.set<"matmul_registry_mode">(Mode::Shadow);
    EXPECT_EQ(current_mode(), Mode::Shadow);
    ttnn::CONFIG.set<"matmul_registry_mode">(Mode::On);
    EXPECT_EQ(current_mode(), Mode::Shadow);
}

TEST(MatmulConfigRegistry, OffShadowAndOnHaveDistinctMutationAndTelemetryContracts) {
    RuntimeStateReset reset;
    const auto req = checked_in_request();
    const auto eligible = eligibility();
    const ttnn::prim::MatmulParams legacy;

    auto off = resolve_for_dispatch(Mode::Off, req, eligible, legacy);
    EXPECT_EQ(off.resolution.reason, ResolutionReason::Disabled);
    EXPECT_EQ(off.action, ExecutionAction::Fallback);
    EXPECT_FALSE(off.materialized_parameters.has_value());
    EXPECT_EQ(stats_snapshot().domains[0].resolution_attempts, 0U);

    auto shadow = resolve_for_dispatch(Mode::Shadow, req, eligible, legacy);
    EXPECT_EQ(shadow.resolution.reason, ResolutionReason::CertifiedMatch);
    EXPECT_EQ(shadow.action, ExecutionAction::ObserveOnly);
    EXPECT_FALSE(shadow.materialized_parameters.has_value());
    auto snapshot = stats_snapshot().domains[0];
    EXPECT_EQ(snapshot.resolution_attempts, 1U);
    EXPECT_EQ(snapshot.certified_hits, 1U);
    EXPECT_EQ(snapshot.shadow_would_hits, 1U);
    EXPECT_EQ(snapshot.selected_hits, 0U);
    EXPECT_EQ(snapshot.fallbacks, 0U);

    auto on = resolve_for_dispatch(Mode::On, req, eligible, legacy);
    EXPECT_EQ(on.resolution.reason, ResolutionReason::CertifiedMatch);
    EXPECT_EQ(on.action, ExecutionAction::ApplyRecipe);
    ASSERT_TRUE(on.materialized_parameters.has_value());
    EXPECT_TRUE(on.materialized_parameters->program_config.has_value());
    EXPECT_TRUE(on.materialized_parameters->compute_kernel_config.has_value());
    snapshot = stats_snapshot().domains[0];
    EXPECT_EQ(snapshot.resolution_attempts, 2U);
    EXPECT_EQ(snapshot.certified_hits, 2U);
    EXPECT_EQ(snapshot.selected_hits, 1U);
    EXPECT_EQ(snapshot.reasons[static_cast<std::size_t>(ResolutionReason::CertifiedMatch)], 2U);
}

TEST(MatmulConfigRegistry, CompactDataTypeEnumeratorsAreAppendOnly) {
    // The emitter's sort key and the selector ABI both assume these values.
    // Renumbering silently reorders every checked-in key.
    EXPECT_EQ(static_cast<std::uint8_t>(compact::DataType::BFloat16), 0U);
    EXPECT_EQ(static_cast<std::uint8_t>(compact::DataType::BFloat8B), 1U);
    EXPECT_EQ(static_cast<std::uint8_t>(compact::DataType::Float32), 2U);
    EXPECT_EQ(static_cast<std::uint8_t>(compact::DataType::BFloat4B), 3U);
}

TEST(MatmulConfigRegistry, Bfloat4BWeightsResolveAgainstTheCheckedInTable) {
    RuntimeStateReset reset;
    const auto req = checked_in_bfloat4b_request();
    const auto eligible = eligibility();
    const ttnn::prim::MatmulParams legacy;

    // Without the BFLOAT4_B -> compact::BFloat4B mapping the key is nullopt and
    // every bfp4 entry in the table is silently unreachable.
    const auto key = compact_registry_key(req);
    ASSERT_TRUE(key.has_value());
    EXPECT_EQ(key->input_b.dtype, compact::DataType::BFloat4B);

    const auto on = resolve_for_dispatch(Mode::On, req, eligible, legacy);
    EXPECT_EQ(on.resolution.reason, ResolutionReason::CertifiedMatch);
    EXPECT_EQ(on.action, ExecutionAction::ApplyRecipe);
    ASSERT_TRUE(on.materialized_parameters.has_value());
    EXPECT_TRUE(on.materialized_parameters->program_config.has_value());
    EXPECT_TRUE(on.materialized_parameters->compute_kernel_config.has_value());
}

TEST(MatmulConfigRegistry, MaterializationFailureBreaksOnlyAffectedDomain) {
    RuntimeStateReset reset;
    const auto req = checked_in_request();
    const auto eligible = eligibility();
    const ttnn::prim::MatmulParams legacy;

    const auto failed = resolve_for_dispatch(Mode::On, req, eligible, legacy, invalid_materialization_resolution);
    EXPECT_EQ(failed.resolution.reason, ResolutionReason::MaterializationRejected);
    EXPECT_EQ(failed.action, ExecutionAction::Fallback);
    EXPECT_FALSE(failed.materialized_parameters.has_value());
    EXPECT_TRUE(is_domain_circuit_broken(OperationDomain::DenseMatmul));
    EXPECT_FALSE(is_domain_circuit_broken(OperationDomain::Linear));

    const auto linear = resolve_for_dispatch(
        Mode::On, checked_in_request(OperationDomain::Linear), eligibility(OperationDomain::Linear), legacy);
    EXPECT_EQ(linear.resolution.reason, ResolutionReason::CertifiedMatch);
    EXPECT_EQ(linear.action, ExecutionAction::ApplyRecipe);

    const auto broken = resolve_for_dispatch(Mode::On, req, eligible, legacy);
    EXPECT_EQ(broken.resolution.reason, ResolutionReason::CircuitBroken);
    const auto snapshot = stats_snapshot().domains[0];
    EXPECT_EQ(snapshot.circuit_breaker_activations, 1U);
    EXPECT_EQ(snapshot.fallbacks, 2U);
    EXPECT_EQ(snapshot.reasons[static_cast<std::size_t>(ResolutionReason::CircuitBroken)], 1U);
}

TEST(MatmulConfigRegistry, SelectedExecutionGuardCompletesOrCircuitBreaks) {
    RuntimeStateReset reset;
    {
        SelectedExecutionGuard guard(OperationDomain::DenseMatmul, true);
    }
    EXPECT_EQ(stats_snapshot().domains[0].completed_hits, 1U);
    EXPECT_FALSE(is_domain_circuit_broken(OperationDomain::DenseMatmul));

    EXPECT_THROW(
        {
            SelectedExecutionGuard guard(OperationDomain::Linear, true);
            throw std::runtime_error("selected execution failed");
        },
        std::runtime_error);
    EXPECT_TRUE(is_domain_circuit_broken(OperationDomain::Linear));
    EXPECT_EQ(stats_snapshot().domains[1].circuit_breaker_activations, 1U);
}

TEST(MatmulConfigRegistry, MaterializationSupportsEveryNativeFamily) {
    const auto key = *compact_registry_key(request());
    const auto ckc = kernel();
    const auto reuse = materialize_registry_program_config(key, reuse_program(), ckc);
    const auto one_d = materialize_registry_program_config(key, multicast_1d_program(), ckc);
    const auto two_d = materialize_registry_program_config(key, multicast_2d_program(), ckc);
    ASSERT_TRUE(reuse.has_value());
    ASSERT_TRUE(one_d.has_value());
    ASSERT_TRUE(two_d.has_value());
    EXPECT_TRUE(std::holds_alternative<MatmulMultiCoreReuseProgramConfig>(*reuse));
    EXPECT_TRUE(std::holds_alternative<MatmulMultiCoreReuseMultiCast1DProgramConfig>(*one_d));
    EXPECT_TRUE(std::holds_alternative<MatmulMultiCoreReuseMultiCastProgramConfig>(*two_d));
}

TEST(MatmulConfigRegistry, MaterializationRejectsCandidateOutsideLiveGrid) {
    auto small = request();
    small.device.compute_grid_x = 2;
    const auto key = *compact_registry_key(small);
    EXPECT_FALSE(materialize_registry_program_config(key, reuse_program(3), kernel()).has_value());
}

TEST(MatmulConfigRegistry, ComputeKernelMaterializationCoversThrottleZeroThroughFive) {
    const std::array compact_levels{
        compact::ThrottleLevel::NoThrottle,
        compact::ThrottleLevel::Throttle1,
        compact::ThrottleLevel::Throttle2,
        compact::ThrottleLevel::Throttle3,
        compact::ThrottleLevel::Throttle4,
        compact::ThrottleLevel::Throttle5};
    const std::array native_levels{
        compute_throttle_utils::ThrottleLevel::NO_THROTTLE,
        compute_throttle_utils::ThrottleLevel::LEVEL_1,
        compute_throttle_utils::ThrottleLevel::LEVEL_2,
        compute_throttle_utils::ThrottleLevel::LEVEL_3,
        compute_throttle_utils::ThrottleLevel::LEVEL_4,
        compute_throttle_utils::ThrottleLevel::LEVEL_5};
    for (std::size_t index = 0; index < compact_levels.size(); ++index) {
        const auto result = materialize_registry_compute_kernel_config(kernel(compact_levels[index]));
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result->throttle_level, native_levels[index]);
    }
}

TEST(MatmulConfigRegistry, PairedMaterializationPreservesAllCallerOwnedState) {
    const auto req = request_with_kernel(request(), kernel(compact::ThrottleLevel::Throttle4));
    const auto entry = exact_entry(req, multicast_2d_program(), kernel(compact::ThrottleLevel::Throttle4));
    const auto selected =
        resolve_with_compact_table_for_testing(req, eligibility_with_kernel(), metadata(), {&entry, 1});
    ttnn::prim::MatmulParams legacy;
    legacy.output_dtype = DataType::FLOAT32;
    legacy.user_run_batched = false;
    const auto materialized = materialize_parameters_for_execution(selected, legacy);
    ASSERT_TRUE(materialized.has_value());
    ASSERT_TRUE(materialized->program_config.has_value());
    ASSERT_TRUE(materialized->compute_kernel_config.has_value());
    EXPECT_EQ(materialized->output_dtype, legacy.output_dtype);
    EXPECT_EQ(materialized->output_mem_config, legacy.output_mem_config);
    EXPECT_EQ(materialized->compute_kernel_config->throttle_level, compute_throttle_utils::ThrottleLevel::LEVEL_4);
}

TEST(MatmulConfigRegistry, EligibilityFailsClosedOnUnsupportedAndInconsistentCalls) {
    auto eligible = eligibility();
    eligible.has_bias = true;
    EXPECT_EQ(preflight_v1_eligibility(eligible), ResolutionReason::UnsupportedSemantics);
    eligible = eligibility();
    eligible.trace_capture_active = true;
    EXPECT_EQ(preflight_v1_eligibility(eligible), ResolutionReason::TraceCaptureUnsupported);
    auto req = request();
    req.run_batched = true;
    EXPECT_EQ(validate_v1_request_envelope(req, eligibility()), ResolutionReason::InconsistentRequest);
}

TEST(MatmulConfigRegistry, AddmmRequiresExactSafeScalarSemantics) {
    EXPECT_EQ(preflight_v1_eligibility(eligibility(OperationDomain::Addmm)), ResolutionReason::CertifiedMatch);
    auto bad = eligibility(OperationDomain::Addmm);
    bad.call = addmm_call_semantics(2.0F, 0.0F);
    EXPECT_EQ(preflight_v1_eligibility(bad), ResolutionReason::MalformedOperationSemantics);
    bad.call = addmm_call_semantics(1.0F, 1.0F);
    EXPECT_EQ(preflight_v1_eligibility(bad), ResolutionReason::UnsupportedSemantics);
}

TEST(MatmulConfigRegistry, CheckedTableReportsSemanticSourceAsInertProvenance) {
    const auto snapshot = stats_snapshot();
    EXPECT_GT(snapshot.exact_entry_count, 0U);
    // Provenance is reported, but it gates nothing: the shipped table resolves
    // without any attestation precondition.
    EXPECT_NE(snapshot.semantic_source_sha256, compact::Sha256{});
    EXPECT_EQ(preflight_v1_eligibility(eligibility()), ResolutionReason::CertifiedMatch);
}

TEST(MatmulConfigRegistry, DefaultComputeKernelKnobsReproduceCreateMatmulAttributes) {
    const auto blackhole = static_cast<std::uint32_t>(tt::ARCH::BLACKHOLE);
    const auto wormhole = static_cast<std::uint32_t>(tt::ARCH::WORMHOLE_B0);
    using compact::DataType;
    using compact::MathFidelity;

    // Neither input low precision, so create_matmul_attributes() raises the
    // fidelity: HiFi2, packer L1 accumulation on, fp32 dest accumulation off.
    const auto bf16 =
        default_compute_kernel_descriptor(blackhole, DataType::BFloat16, DataType::BFloat16, DataType::BFloat16);
    EXPECT_EQ(bf16.math_fidelity, MathFidelity::HiFi2);
    EXPECT_FALSE(bf16.math_approx_mode);
    EXPECT_FALSE(bf16.fp32_dest_acc_en);
    EXPECT_TRUE(bf16.packer_l1_acc);
    EXPECT_FALSE(bf16.dst_full_sync_en);
    EXPECT_EQ(bf16.throttle_level, compact::ThrottleLevel::NoThrottle);

    // Both inputs low precision drops it back to LoFi; one of the two does not.
    EXPECT_EQ(
        default_compute_kernel_descriptor(blackhole, DataType::BFloat8B, DataType::BFloat4B, DataType::BFloat16)
            .math_fidelity,
        MathFidelity::LoFi);
    EXPECT_EQ(
        default_compute_kernel_descriptor(blackhole, DataType::BFloat16, DataType::BFloat8B, DataType::BFloat16)
            .math_fidelity,
        MathFidelity::HiFi2);

    // FLOAT32 inputs override both, and Wormhole takes HiFi3 for hardware bug
    // #38306 where every other architecture takes HiFi4.
    EXPECT_EQ(
        default_compute_kernel_descriptor(blackhole, DataType::Float32, DataType::Float32, DataType::BFloat16)
            .math_fidelity,
        MathFidelity::HiFi4);
    EXPECT_EQ(
        default_compute_kernel_descriptor(wormhole, DataType::Float32, DataType::Float32, DataType::BFloat16)
            .math_fidelity,
        MathFidelity::HiFi3);

    // A FLOAT32 output swaps the two accumulation knobs.
    const auto f32_out =
        default_compute_kernel_descriptor(blackhole, DataType::Float32, DataType::Float32, DataType::Float32);
    EXPECT_TRUE(f32_out.fp32_dest_acc_en);
    EXPECT_FALSE(f32_out.packer_l1_acc);
}

TEST(MatmulConfigRegistry, DefaultKnobsAreTheKeyWhenTheCallerNamesNone) {
    const auto key = compact_registry_key(request());
    ASSERT_TRUE(key.has_value());
    EXPECT_EQ(
        key->compute_kernel,
        default_compute_kernel_descriptor(
            static_cast<std::uint32_t>(tt::ARCH::BLACKHOLE),
            compact::DataType::BFloat16,
            compact::DataType::BFloat16,
            compact::DataType::BFloat16));
}

TEST(MatmulConfigRegistry, ComputeKernelConfigRoundTripsThroughTheCompactSpelling) {
    const auto compact_config = kernel(compact::ThrottleLevel::Throttle5);
    const auto native = materialize_registry_compute_kernel_config(compact_config);
    ASSERT_TRUE(native.has_value());
    const auto round_tripped = compact_compute_kernel_config(*native);
    ASSERT_TRUE(round_tripped.has_value());
    EXPECT_EQ(*round_tripped, compact_config);
}

TEST(MatmulConfigRegistry, RecipeMeasuredAtOtherKnobsIsUnreachable) {
    // A recipe harvested at LoFi is not an answer for a call that asks for --
    // or defaults to -- HiFi2. Before the knobs were keyed, this lookup hit and
    // silently rewrote the caller's fidelity.
    auto lofi = kernel();
    lofi.math_fidelity = compact::MathFidelity::LoFi;
    const auto lofi_request = request_with_kernel(request(), lofi);
    const auto entry = exact_entry(lofi_request, reuse_program(2), lofi);

    EXPECT_EQ(
        resolve_with_compact_table_for_testing(request(), eligibility(), metadata(), {&entry, 1}).reason,
        ResolutionReason::EmptyRegistry);

    const auto matched =
        resolve_with_compact_table_for_testing(lofi_request, eligibility_with_kernel(), metadata(), {&entry, 1});
    EXPECT_EQ(matched.reason, ResolutionReason::CertifiedMatch);
    ASSERT_TRUE(matched.compute_kernel_config.has_value());
    EXPECT_EQ(*matched.compute_kernel_config, lofi);
}

TEST(MatmulConfigRegistry, MathApproxModeIsNormalizedOutOfTheKey) {
    // math_approx_mode only configures the SFPU, and no admitted call has an
    // SFPU op: preflight rejects has_activation. Keying it would halve the
    // reach of every measurement for no numerical difference, so both
    // spellings resolve to one key and one entry.
    auto approximate = kernel();
    approximate.math_approx_mode = true;
    const auto approximate_key = compact_registry_key(request_with_kernel(request(), approximate));
    const auto default_key = compact_registry_key(request());
    ASSERT_TRUE(approximate_key.has_value());
    ASSERT_TRUE(default_key.has_value());
    EXPECT_FALSE(approximate_key->compute_kernel.math_approx_mode);
    EXPECT_EQ(approximate_key->compute_kernel, default_key->compute_kernel);

    // The entry is measured precise, the way the emitter normalizes it. A
    // caller who asked for the approximate spelling still resolves, and keeps
    // their own spelling because the dispatch never writes that field back.
    const auto entry = exact_entry(request(), reuse_program(2), kernel());
    const auto matched = resolve_with_compact_table_for_testing(
        request_with_kernel(request(), approximate), eligibility_with_kernel(), metadata(), {&entry, 1});
    EXPECT_EQ(matched.reason, ResolutionReason::CertifiedMatch);
    EXPECT_TRUE(compact::entry_permits_math_approx_normalization(entry));
}

TEST(MatmulConfigRegistry, NormalizationTouchesNothingButMathApproxMode) {
    // The other five knobs select arithmetic and stay strict; fidelity is the
    // one that made a LoFi measurement answer a HiFi2 call.
    auto knobs = kernel(compact::ThrottleLevel::Throttle2);
    knobs.math_fidelity = compact::MathFidelity::HiFi4;
    knobs.math_approx_mode = true;
    knobs.fp32_dest_acc_en = true;
    knobs.packer_l1_acc = false;
    knobs.dst_full_sync_en = true;
    auto expected = knobs;
    expected.math_approx_mode = false;
    EXPECT_EQ(normalize_key_compute_kernel(knobs), expected);
}

TEST(MatmulConfigRegistry, EntryWhoseRecipeDisagreesWithItsKeyIsRefused) {
    // The compile-time compact::entries_bind_key_compute_kernel check covers the
    // shipped table; a table that reaches the runtime without it still may not
    // substitute knobs the caller never asked for.
    const auto req = request();
    auto entry = exact_entry(req);
    entry.compute_kernel_config.math_fidelity = compact::MathFidelity::HiFi4;
    EXPECT_FALSE(compact::entry_binds_key_compute_kernel(entry));
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(req, eligibility(), metadata(), {&entry, 1}).reason,
        ResolutionReason::MaterializationRejected);
}

TEST(MatmulConfigRegistry, RequestAndEligibilityMustAgreeOnCallerSuppliedKnobs) {
    EXPECT_EQ(
        validate_v1_request_envelope(request_with_kernel(request(), kernel()), eligibility()),
        ResolutionReason::InconsistentRequest);
    EXPECT_EQ(
        validate_v1_request_envelope(request(), eligibility_with_kernel()), ResolutionReason::InconsistentRequest);
}

}  // namespace
}  // namespace ttnn::operations::matmul::registry
