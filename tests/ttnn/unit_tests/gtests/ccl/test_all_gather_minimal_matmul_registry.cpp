// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <atomic>
#include <cstdint>
#include <stdexcept>

#include <gtest/gtest.h>

#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/registry/agmm_config_registry.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/registry/agmm_registry_data.hpp"

namespace {

namespace registry = ttnn::experimental::all_gather_minimal_matmul_registry;
namespace compact = registry::compact;
using Mode = registry::Mode;

compact::Sha256 digest(const std::uint8_t value) {
    compact::Sha256 result{};
    result.fill(value);
    return result;
}

compact::TensorDescriptor tensor_descriptor(
    const std::array<std::uint64_t, 4>& logical, const std::array<std::uint64_t, 4>& padded) {
    compact::TensorDescriptor tensor{
        .rank = 4,
        .logical_shape = {logical[0], logical[1], logical[2], logical[3]},
        .padded_shape = {padded[0], padded[1], padded[2], padded[3]},
        .dtype = 1,
        .layout = 1,
        .memory_layout = 0,
        .buffer_type = 1,
        .tile_height = 32,
        .tile_width = 32};
    return tensor;
}

compact::KeyDescriptor valid_key() {
    auto key = compact::KeyDescriptor{};
    key.device = compact::DeviceDescriptor{
        .architecture = compact::kBlackholeArchitecture,
        .device_count = compact::kBh32DeviceCount,
        .mesh_rows = compact::kBh32MeshRows,
        .mesh_cols = compact::kBh32MeshCols,
        .compute_grid_x = 13,
        .compute_grid_y = 10};
    key.workload = compact::WorkloadDescriptor{
        .logical_m = 64,
        .logical_k = 256,
        .logical_n = 128,
        .padded_m = 64,
        .padded_k = 256,
        .padded_n = 128,
        .batch = 1};
    key.operation.topology = 2;
    key.operation.fsdp_topology = 2;
    key.operation.num_links = 1;
    key.operation.ring_size = 8;
    key.operation.fsdp_ring_size = 1;
    key.operation.chunks = 1;
    key.operation.dim = -1;
    key.operation.output_dtype_present = true;
    key.operation.output_dtype = 1;
    key.operation.output_memory_config_present = true;
    key.operation.output_layout = 1;
    key.operation.output_tile_height = 32;
    key.operation.output_tile_width = 32;
    key.input = tensor_descriptor({1, 1, 64, 32}, {1, 1, 64, 32});
    key.weight = tensor_descriptor({1, 1, 256, 128}, {1, 1, 256, 128});
    return key;
}

compact::EntryDescriptor valid_entry() {
    auto entry = compact::EntryDescriptor{};
    entry.entry_id = digest(7);
    entry.key = valid_key();
    entry.replay.config = compact::MinimalMatmulConfigDescriptor{
        .m_block_size = 2,
        .k_block_size = 1,
        .n_block_size = 2,
        .subblock_h = 1,
        .subblock_w = 2,
        .compute_grid_x = 2,
        .compute_grid_y = 2};
    entry.replay.compute_kernel_config = compact::ComputeKernelDescriptor{
        .math_fidelity = static_cast<std::uint32_t>(tt::tt_metal::MathFidelity::HiFi2),
        .math_approx_mode = false,
        .fp32_dest_acc_en = true,
        .packer_l1_acc = true,
        .dst_full_sync_en = false,
        .throttle_level =
            static_cast<std::uint32_t>(ttnn::operations::compute_throttle_utils::ThrottleLevel::NO_THROTTLE)};
    return entry;
}

compact::TableLock valid_lock(const std::size_t entry_count = 1) {
    return compact::TableLock{
        .schema_version = compact::kTableLockSchemaVersion,
        .codegen_recipe_abi = compact::kCodegenRecipeAbi,
        .entry_count = entry_count,
        .metadata = compact::TableMetadata{
            .key_schema_version = compact::kKeySchemaVersion,
            .replay_schema_version = compact::kReplaySchemaVersion,
            .content_sha256 = digest(12), .semantic_source_sha256 = digest(1)},
        .certified_device = valid_key().device,
        .evidence_manifest_sha256 = digest(13),
        .predictor_sha256 = digest(14),
        .exporter_sha256 = digest(15)};
}

registry::RegistryRequestFacts valid_request_facts() {
    const auto key = valid_key();
    return registry::RegistryRequestFacts{
        .device = key.device,
        .workload = key.workload,
        .operation = key.operation,
        .input = key.input,
        .weight = key.weight,
        .bias = key.bias,
        .ternary_input_a = key.ternary_input_a,
        .ternary_input_b = key.ternary_input_b,
        .persistent_output = key.persistent_output,
        .persistent_weight = key.persistent_weight};
}

std::atomic<std::uint32_t> resolver_calls{0};
std::atomic<std::uint32_t> materializer_calls{0};
compact::EntryDescriptor selected_descriptor = valid_entry();

registry::Resolution selected_resolver(
    Mode, const std::optional<registry::RegistryRequest>&, const registry::Eligibility&) noexcept {
    resolver_calls.fetch_add(1, std::memory_order_relaxed);
    return {.reason = registry::ResolutionReason::CertifiedMatch, .descriptor = &selected_descriptor};
}

registry::MaterializationResult counting_materializer(const compact::EntryDescriptor& descriptor) {
    materializer_calls.fetch_add(1, std::memory_order_relaxed);
    return registry::materialize_recipe(descriptor);
}

registry::MaterializationResult throwing_materializer(const compact::EntryDescriptor&) {
    materializer_calls.fetch_add(1, std::memory_order_relaxed);
    throw std::runtime_error("injected materialization failure");
}

void throw_during_selected_execution(std::atomic<std::uint32_t>& launches) {
    const bool selected = true;
    const registry::SelectedExecutionGuard guard(&selected);
    registry::execute_selected_call_once(guard, [&]() -> int {
        launches.fetch_add(1, std::memory_order_relaxed);
        throw std::runtime_error("injected public execution failure");
    });
}

class AgmmRegistryTest : public ::testing::Test {
protected:
    void SetUp() override {
        registry::reset_circuit_breaker_for_testing();
        registry::reset_stats_for_testing();
        resolver_calls.store(0, std::memory_order_relaxed);
        materializer_calls.store(0, std::memory_order_relaxed);
    }
};

TEST_F(AgmmRegistryTest, ProductionTablePreservesOffShadowAndOnWithoutARequest) {
    const auto eligibility = registry::Eligibility{};
    auto off = registry::resolve_for_dispatch(
        Mode::Off,
        std::nullopt,
        eligibility,
        &selected_resolver,
        &counting_materializer);
    EXPECT_EQ(off.resolution.reason, registry::ResolutionReason::Disabled);
    EXPECT_EQ(off.action, registry::ExecutionAction::Fallback);
    EXPECT_EQ(resolver_calls.load(), 0U);
    EXPECT_EQ(materializer_calls.load(), 0U);

    // The checked-in table is populated, so the cheap empty-table exit no longer
    // fires. Live request construction is still unwired, so both observing modes
    // report an incomplete request and preserve the legacy launch.
    const auto shadow = registry::resolve(Mode::Shadow, std::nullopt, eligibility);
    const auto on = registry::resolve(Mode::On, std::nullopt, eligibility);
    EXPECT_EQ(shadow.reason, registry::ResolutionReason::IncompleteRequest);
    EXPECT_EQ(on.reason, registry::ResolutionReason::IncompleteRequest);
    EXPECT_EQ(registry::resolution_reason_name(on.reason), "incomplete_request");
}

TEST_F(AgmmRegistryTest, ModeIsFrozenAtFirstRead) {
    const auto original = ttnn::CONFIG.get<"agmm_registry_mode">();
    ttnn::CONFIG.set<"agmm_registry_mode">(Mode::Shadow);
    registry::reset_startup_mode_for_testing();
    EXPECT_EQ(registry::current_mode(), Mode::Shadow);
    ttnn::CONFIG.set<"agmm_registry_mode">(Mode::On);
    EXPECT_EQ(registry::current_mode(), Mode::Shadow);
    ttnn::CONFIG.set<"agmm_registry_mode">(original);
    registry::reset_startup_mode_for_testing();
}

TEST_F(AgmmRegistryTest, UnknownTraceStateFailsClosedInShadowAndOnOnly) {
    EXPECT_FALSE(registry::fail_closed_trace_capture_active(Mode::Off, std::nullopt));
    EXPECT_TRUE(registry::fail_closed_trace_capture_active(Mode::Shadow, std::nullopt));
    EXPECT_TRUE(registry::fail_closed_trace_capture_active(Mode::On, std::nullopt));
    EXPECT_FALSE(registry::fail_closed_trace_capture_active(Mode::Shadow, false));
}

TEST_F(AgmmRegistryTest, PreflightReasonsPrecedeEmptyTableAndAreNamed) {
    const auto trace = registry::resolve(
        Mode::Shadow,
        std::nullopt,
        {.trace_capture_active = true, .has_explicit_program_config = true});
    EXPECT_EQ(trace.reason, registry::ResolutionReason::TraceCaptureUnsupported);
    EXPECT_EQ(registry::resolution_reason_name(trace.reason), "trace_capture_unsupported");

    const auto explicit_config = registry::resolve(
        Mode::Shadow,
        std::nullopt,
        {.has_explicit_program_config = true});
    EXPECT_EQ(explicit_config.reason, registry::ResolutionReason::ExplicitProgramConfig);

    const auto explicit_kernel = registry::resolve(
        Mode::Shadow,
        std::nullopt,
        {.has_explicit_compute_kernel_config = true});
    EXPECT_EQ(explicit_kernel.reason, registry::ResolutionReason::ExplicitComputeKernelConfig);
}

TEST_F(AgmmRegistryTest, ExactLookupRequiresFullKeyAndCompatibility) {
    const auto entry = valid_entry();
    const std::array entries{entry};
    const auto request = registry::RegistryRequest{.key = entry.key};
    auto result = registry::resolve_with_table_for_testing(
        Mode::On, request, {}, valid_lock(), entries);
    ASSERT_EQ(result.reason, registry::ResolutionReason::CertifiedMatch);
    EXPECT_EQ(result.descriptor, &entries.front());

    auto miss = request;
    miss.key.operation.scalar_present = true;
    miss.key.operation.scalar_f32_bits = 0x80000000U;
    result = registry::resolve_with_table_for_testing(
        Mode::On, miss, {}, valid_lock(), entries);
    EXPECT_EQ(result.reason, registry::ResolutionReason::ExactMiss);

    // No build, firmware, or source digest can veto a lookup any more. An
    // absent request still fails closed and legality is proven downstream.
    result = registry::resolve_with_table_for_testing(Mode::On, std::nullopt, {}, valid_lock(), entries);
    EXPECT_EQ(result.reason, registry::ResolutionReason::IncompleteRequest);
}

TEST_F(AgmmRegistryTest, GeneratedProductionLockIsCanonicalAndRoundTripsEveryKey) {
    namespace generated = registry::generated;
    const auto entries = generated::entries();
    const auto& lock = generated::lock();
    ASSERT_FALSE(entries.empty());
    EXPECT_EQ(lock.entry_count, entries.size());
    EXPECT_EQ(compact::validate_table_lock(lock, entries), compact::TableValidationStatus::Valid);
    EXPECT_EQ(
        registry::validate_compatibility(lock, entries, lock.certified_device),
        registry::CompatibilityStatus::Compatible);

    for (const auto& entry : entries) {
        EXPECT_EQ(entry.key.device, lock.certified_device);
        // Every emitted key must find its own entry and materialize.
        const auto* hit = compact::lookup_exact(entry.key, entries);
        ASSERT_TRUE(hit != nullptr);
        EXPECT_EQ(hit->entry_id, entry.entry_id);
        EXPECT_EQ(hit->replay, entry.replay);
        EXPECT_EQ(registry::materialize_recipe(entry).status, registry::MaterializationStatus::Success);
        // num_workers_per_link is derived from the grid and is also a key axis.
        EXPECT_EQ(entry.key.operation.num_workers_per_link, 6U);
        EXPECT_EQ(entry.key.operation.num_links, 2U);
        EXPECT_EQ(entry.key.operation.num_buffers_per_channel, 24U);
        EXPECT_TRUE(entry.replay.config.compute_grid_x <= entry.key.device.compute_grid_x);
        EXPECT_TRUE(entry.replay.config.compute_grid_y <= entry.key.device.compute_grid_y);
    }

    // A request from a different worker-grid cohort is a device mismatch, not a
    // silent hit on the certified cohort.
    auto foreign = entries.front().key;
    foreign.device.compute_grid_x += 1;
    EXPECT_EQ(
        registry::validate_compatibility(lock, entries, foreign.device),
        registry::CompatibilityStatus::DeviceMismatch);
    EXPECT_EQ(compact::lookup_exact(foreign, entries), nullptr);
}

TEST_F(AgmmRegistryTest, SyntheticGeneratedLockRequiresTypedProvenanceAndStrictKeyOrder) {
    auto first = valid_entry();
    auto second = valid_entry();
    second.entry_id = digest(6);
    second.key.workload.logical_n += 32;
    const std::array sorted_entries{first, second};
    auto lock = valid_lock(sorted_entries.size());
    EXPECT_EQ(
        compact::validate_table_lock(lock, sorted_entries), compact::TableValidationStatus::Valid);

    auto missing_provenance = lock;
    missing_provenance.predictor_sha256 = {};
    EXPECT_EQ(
        compact::validate_table_lock(missing_provenance, sorted_entries),
        compact::TableValidationStatus::MissingLockDigest);

    auto wrong_count = lock;
    wrong_count.entry_count -= 1;
    EXPECT_EQ(
        compact::validate_table_lock(wrong_count, sorted_entries),
        compact::TableValidationStatus::EntryCountMismatch);

    auto wrong_abi = lock;
    wrong_abi.codegen_recipe_abi += 1;
    EXPECT_EQ(
        compact::validate_table_lock(wrong_abi, sorted_entries),
        compact::TableValidationStatus::LockSchemaMismatch);

    auto zero_entry_id = sorted_entries;
    zero_entry_id.front().entry_id = {};
    EXPECT_EQ(
        compact::validate_table_lock(lock, zero_entry_id), compact::TableValidationStatus::MissingEntryId);

    auto wrong_grid = sorted_entries;
    wrong_grid.back().key.device.compute_grid_y -= 1;
    EXPECT_EQ(
        compact::validate_table_lock(lock, wrong_grid),
        compact::TableValidationStatus::CertifiedDeviceMismatch);

    const std::array reversed_entries{second, first};
    EXPECT_EQ(
        compact::validate_table_lock(lock, reversed_entries),
        compact::TableValidationStatus::EntriesNotStrictlySorted);
    const std::array duplicate_entries{first, first};
    EXPECT_EQ(
        compact::validate_table_lock(lock, duplicate_entries),
        compact::TableValidationStatus::EntriesNotStrictlySorted);
}

TEST_F(AgmmRegistryTest, SyntheticGeneratedLockIsRestrictedToExactBh32EightByFourDomain) {
    const auto entry = valid_entry();
    const std::array entries{entry};
    auto lock = valid_lock();

    auto wrong_architecture = lock;
    wrong_architecture.certified_device.architecture = 2;
    EXPECT_EQ(
        compact::validate_table_lock(wrong_architecture, entries),
        compact::TableValidationStatus::UnsupportedDeviceDomain);

    auto eight_device_lock = lock;
    eight_device_lock.certified_device.device_count = 8;
    eight_device_lock.certified_device.mesh_rows = 2;
    EXPECT_EQ(
        compact::validate_table_lock(eight_device_lock, entries),
        compact::TableValidationStatus::UnsupportedDeviceDomain);

    auto transposed_mesh = lock;
    transposed_mesh.certified_device.mesh_rows = 4;
    transposed_mesh.certified_device.mesh_cols = 8;
    EXPECT_EQ(
        compact::validate_table_lock(transposed_mesh, entries),
        compact::TableValidationStatus::UnsupportedDeviceDomain);

    auto missing_grid = lock;
    missing_grid.certified_device.compute_grid_x = 0;
    EXPECT_EQ(
        compact::validate_table_lock(missing_grid, entries),
        compact::TableValidationStatus::UnsupportedDeviceDomain);

    auto mismatched_grid = entries;
    mismatched_grid.front().key.device.compute_grid_x -= 1;
    EXPECT_EQ(
        compact::validate_table_lock(lock, mismatched_grid),
        compact::TableValidationStatus::CertifiedDeviceMismatch);

    auto mismatched_mesh = entries;
    mismatched_mesh.front().key.device.mesh_cols = compact::kBh32MeshRows;
    EXPECT_EQ(
        compact::validate_table_lock(lock, mismatched_mesh),
        compact::TableValidationStatus::CertifiedDeviceMismatch);

    // Inert provenance must still be present so a table's origin is readable,
    // but it is never compared against a running build, firmware, or source.
    auto missing_provenance = lock;
    missing_provenance.metadata.semantic_source_sha256 = {};
    EXPECT_EQ(
        compact::validate_table_lock(missing_provenance, entries),
        compact::TableValidationStatus::MissingLockDigest);
}

TEST_F(AgmmRegistryTest, ExactBh32LockRejectsARequestFromAnotherDeviceDomain) {
    const auto entry = valid_entry();
    const std::array entries{entry};
    auto request = registry::RegistryRequest{.key = entry.key};
    request.key.device.compute_grid_x -= 1;
    auto result = registry::resolve_with_table_for_testing(
        Mode::On,
        request,
        {},
        valid_lock(),
        entries);
    EXPECT_EQ(result.reason, registry::ResolutionReason::CompatibilityMismatch);
    EXPECT_EQ(result.descriptor, nullptr);

    request = registry::RegistryRequest{.key = entry.key};
    request.key.device.mesh_rows = compact::kBh32MeshCols;
    request.key.device.mesh_cols = compact::kBh32MeshRows;
    result = registry::resolve_with_table_for_testing(Mode::On, request, {}, valid_lock(), entries);
    EXPECT_EQ(result.reason, registry::ResolutionReason::CompatibilityMismatch);
    EXPECT_EQ(result.descriptor, nullptr);
}

TEST_F(AgmmRegistryTest, MalformedSyntheticLockCannotSelectARecipe) {
    const auto entry = valid_entry();
    const std::array entries{entry};
    auto lock = valid_lock();
    lock.evidence_manifest_sha256 = {};
    const auto result = registry::resolve_with_table_for_testing(
        Mode::On,
        registry::RegistryRequest{.key = entry.key},
        {},
        lock,
        entries);
    EXPECT_EQ(result.reason, registry::ResolutionReason::CompatibilityMismatch);
    EXPECT_EQ(result.descriptor, nullptr);
}

TEST_F(AgmmRegistryTest, PureRequestBuilderBindsEveryResolvedDescriptor) {
    const auto facts = valid_request_facts();
    const auto built = registry::build_registry_request(facts);
    ASSERT_EQ(built.status, registry::RequestBuildStatus::Success);
    ASSERT_TRUE(built.request.has_value());
    EXPECT_EQ(built.request->key, valid_key());
}

TEST_F(AgmmRegistryTest, PureRequestBuilderFailsClosedWithoutExactDeviceFacts) {
    auto facts = valid_request_facts();
    facts.device = {};
    auto built = registry::build_registry_request(facts);
    EXPECT_EQ(built.status, registry::RequestBuildStatus::IncompleteDescriptor);
    EXPECT_FALSE(built.request.has_value());

    facts = valid_request_facts();
    facts.device.compute_grid_x = 0;
    built = registry::build_registry_request(facts);
    EXPECT_EQ(built.status, registry::RequestBuildStatus::IncompleteDescriptor);
    EXPECT_FALSE(built.request.has_value());
}

TEST_F(AgmmRegistryTest, PureRequestBuilderRejectsExplicitOverridesBeforeDeviceFacts) {
    auto facts = valid_request_facts();
    facts.eligibility.has_explicit_program_config = true;
    facts.device = {};
    auto built = registry::build_registry_request(facts);
    EXPECT_EQ(built.status, registry::RequestBuildStatus::ExplicitProgramConfig);

    facts.eligibility = {.has_explicit_compute_kernel_config = true};
    built = registry::build_registry_request(facts);
    EXPECT_EQ(built.status, registry::RequestBuildStatus::ExplicitComputeKernelConfig);
}

TEST_F(AgmmRegistryTest, PureRequestBuilderRejectsMissingAndInconsistentFacts) {
    auto facts = valid_request_facts();
    facts.input.rank = 0;
    auto built = registry::build_registry_request(facts);
    EXPECT_EQ(built.status, registry::RequestBuildStatus::IncompleteDescriptor);
    EXPECT_FALSE(built.request.has_value());

    facts = valid_request_facts();
    facts.device.device_count += 1;
    built = registry::build_registry_request(facts);
    EXPECT_EQ(built.status, registry::RequestBuildStatus::IncompleteDescriptor);

    facts = valid_request_facts();
    facts.device.device_count = 8;
    facts.device.mesh_rows = 2;
    built = registry::build_registry_request(facts);
    EXPECT_EQ(built.status, registry::RequestBuildStatus::IncompleteDescriptor);

    facts = valid_request_facts();
    facts.workload.logical_k += 32;
    built = registry::build_registry_request(facts);
    EXPECT_EQ(built.status, registry::RequestBuildStatus::InconsistentDescriptor);

    facts = valid_request_facts();
    facts.bias.tensor = tensor_descriptor({1, 1, 1, 128}, {1, 1, 32, 128});
    built = registry::build_registry_request(facts);
    EXPECT_EQ(built.status, registry::RequestBuildStatus::InconsistentDescriptor);
}

TEST_F(AgmmRegistryTest, ShadowObservesAndOnMaterializesAtMostOnce) {
    const auto request = registry::RegistryRequest{.key = selected_descriptor.key};
    auto dispatch = registry::resolve_for_dispatch(
        Mode::Shadow, request, {}, &selected_resolver, &counting_materializer);
    EXPECT_EQ(dispatch.action, registry::ExecutionAction::ObserveOnly);
    EXPECT_EQ(resolver_calls.load(), 1U);
    EXPECT_EQ(materializer_calls.load(), 0U);

    dispatch = registry::resolve_for_dispatch(
        Mode::On, request, {}, &selected_resolver, &counting_materializer);
    EXPECT_EQ(dispatch.action, registry::ExecutionAction::ApplyRecipe);
    EXPECT_TRUE(dispatch.recipe.has_value());
    EXPECT_EQ(resolver_calls.load(), 2U);
    EXPECT_EQ(materializer_calls.load(), 1U);
}

TEST_F(AgmmRegistryTest, MaterializationExceptionFallsBackAndCircuitBreaks) {
    const auto dispatch = registry::resolve_for_dispatch(
        Mode::On,
        registry::RegistryRequest{.key = selected_descriptor.key},
        {},
        &selected_resolver,
        &throwing_materializer);
    EXPECT_EQ(dispatch.action, registry::ExecutionAction::Fallback);
    EXPECT_EQ(dispatch.resolution.reason, registry::ResolutionReason::MaterializationRejected);
    EXPECT_EQ(materializer_calls.load(), 1U);
    EXPECT_TRUE(registry::is_circuit_broken());
}

TEST_F(AgmmRegistryTest, SelectedExecutionExceptionIsNotRetriedAndCircuitBreaks) {
    std::atomic<std::uint32_t> launches{0};
    EXPECT_THROW(throw_during_selected_execution(launches), std::runtime_error);
    EXPECT_EQ(launches.load(), 1U);
    EXPECT_TRUE(registry::is_circuit_broken());
    EXPECT_EQ(registry::stats_snapshot().launch_completed_hits, 0U);
}

TEST_F(AgmmRegistryTest, MaterializerRejectsShapeTileGridAndBlockTampering) {
    auto entry = valid_entry();
    EXPECT_EQ(registry::materialize_recipe(entry).status, registry::MaterializationStatus::Success);

    entry.key.device.device_count = 8;
    entry.key.device.mesh_rows = 2;
    EXPECT_EQ(registry::materialize_recipe(entry).status, registry::MaterializationStatus::InvalidProgramConfig);
    entry = valid_entry();
    entry.key.input.tile_width = 16;
    EXPECT_EQ(registry::materialize_recipe(entry).status, registry::MaterializationStatus::InvalidProgramConfig);
    entry = valid_entry();
    entry.key.workload.logical_k += 32;
    EXPECT_EQ(registry::materialize_recipe(entry).status, registry::MaterializationStatus::InvalidProgramConfig);
    entry = valid_entry();
    entry.replay.config.compute_grid_x = 1;
    EXPECT_EQ(registry::materialize_recipe(entry).status, registry::MaterializationStatus::InvalidProgramConfig);
    entry = valid_entry();
    entry.replay.config.k_block_size = 2;
    EXPECT_EQ(registry::materialize_recipe(entry).status, registry::MaterializationStatus::InvalidProgramConfig);
}

TEST_F(AgmmRegistryTest, MaterializerRejectsOddSwiGluNBlockBeforeLaunch) {
    auto entry = valid_entry();
    entry.key.operation.fuse_swiglu = true;
    entry.replay.config.n_block_size = 3;
    entry.replay.config.subblock_w = 1;

    EXPECT_EQ(registry::materialize_recipe(entry).status, registry::MaterializationStatus::InvalidProgramConfig);
}

TEST_F(AgmmRegistryTest, MaterializerUsesFullDestinationRegisterCapacity) {
    auto entry = valid_entry();
    entry.replay.config.m_block_size = 2;
    entry.replay.config.n_block_size = 4;
    entry.replay.config.subblock_h = 2;
    entry.replay.config.subblock_w = 4;

    // fp32 accumulation has four 32x32 destination tiles in half-sync mode.
    entry.replay.compute_kernel_config.dst_full_sync_en = false;
    EXPECT_EQ(registry::materialize_recipe(entry).status, registry::MaterializationStatus::InvalidProgramConfig);

    // Full-sync uses the entire destination register and therefore admits all eight tiles.
    entry.replay.compute_kernel_config.dst_full_sync_en = true;
    EXPECT_EQ(registry::materialize_recipe(entry).status, registry::MaterializationStatus::Success);

    // Non-fp32 full-sync exposes sixteen tiles through the shared hardware contract.
    entry.replay.config.m_block_size = 4;
    entry.replay.config.subblock_h = 4;
    entry.replay.compute_kernel_config.fp32_dest_acc_en = false;
    EXPECT_EQ(registry::materialize_recipe(entry).status, registry::MaterializationStatus::Success);
    entry.replay.compute_kernel_config.dst_full_sync_en = false;
    EXPECT_EQ(registry::materialize_recipe(entry).status, registry::MaterializationStatus::InvalidProgramConfig);
}

TEST_F(AgmmRegistryTest, ReplayBindsEverySupportedThrottleLevelExactly) {
    using ThrottleLevel = ttnn::operations::compute_throttle_utils::ThrottleLevel;
    constexpr std::array levels{
        ThrottleLevel::NO_THROTTLE,
        ThrottleLevel::LEVEL_1,
        ThrottleLevel::LEVEL_2,
        ThrottleLevel::LEVEL_3,
        ThrottleLevel::LEVEL_4,
        ThrottleLevel::LEVEL_5};
    for (const auto level : levels) {
        auto entry = valid_entry();
        entry.replay.compute_kernel_config.throttle_level = static_cast<std::uint32_t>(level);
        const auto materialized = registry::materialize_recipe(entry);
        ASSERT_EQ(materialized.status, registry::MaterializationStatus::Success);
        ASSERT_TRUE(materialized.recipe.has_value());
        EXPECT_EQ(materialized.recipe->compute_kernel_config.throttle_level, level);
    }

    auto entry = valid_entry();
    entry.replay.compute_kernel_config.throttle_level = 0xFFFFFFFFU;
    EXPECT_EQ(
        registry::materialize_recipe(entry).status,
        registry::MaterializationStatus::InvalidComputeKernelConfig);
}

TEST_F(AgmmRegistryTest, TelemetrySeparatesSelectedFromLaunchCompletedAndUsesNamedReasons) {
    const auto resolution =
        registry::Resolution{.reason = registry::ResolutionReason::CertifiedMatch, .descriptor = &selected_descriptor};
    registry::record_resolution(Mode::On, resolution, registry::ExecutionAction::ApplyRecipe);
    auto snapshot = registry::stats_snapshot();
    EXPECT_EQ(snapshot.selected_hits, 1U);
    EXPECT_EQ(snapshot.launch_completed_hits, 0U);
    EXPECT_EQ(snapshot.reasons[static_cast<std::size_t>(registry::ResolutionReason::CertifiedMatch)], 1U);
    EXPECT_EQ(registry::resolution_reason_name(registry::ResolutionReason::CertifiedMatch), "certified_match");

    const bool selected = true;
    {
        const registry::SelectedExecutionGuard guard(&selected);
    }
    snapshot = registry::stats_snapshot();
    EXPECT_EQ(snapshot.launch_completed_hits, 1U);
}

}  // namespace
