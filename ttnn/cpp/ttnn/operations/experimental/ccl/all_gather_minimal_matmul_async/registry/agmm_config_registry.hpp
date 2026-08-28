// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <span>
#include <string_view>
#include <utility>

#include "agmm_registry_descriptor.hpp"
#include "ttnn/config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp"

namespace ttnn::experimental::all_gather_minimal_matmul_registry {

using Mode = ttnn::MatmulRegistryMode;

enum class ResolutionReason : std::uint8_t {
    Disabled,
    TraceCaptureUnsupported,
    ExplicitProgramConfig,
    ExplicitComputeKernelConfig,
    IncompleteRequest,
    CompatibilityMismatch,
    CircuitBroken,
    EmptyRegistry,
    ExactMiss,
    UnsupportedReplay,
    MaterializationRejected,
    CertifiedMatch,
    Count,
};

std::string_view resolution_reason_name(ResolutionReason reason) noexcept;

inline constexpr std::size_t kResolutionReasonCount = static_cast<std::size_t>(ResolutionReason::Count);

struct Eligibility {
    bool trace_capture_active = false;
    bool has_explicit_program_config = false;
    bool has_explicit_compute_kernel_config = false;
};

struct RegistryRequest {
    compact::KeyDescriptor key{};

    bool operator==(const RegistryRequest&) const = default;
};

// Request construction stays separate from querying live TT objects. The
// caller resolves every descriptor before calling this pure, allocation-free
// seam; this function never invents a missing tensor, mesh, or fabric fact.
// The device facts are portable observations (architecture, mesh shape, worker
// grid), not an attestation: nothing here needs a digest the runtime cannot
// produce.
struct RegistryRequestFacts {
    Eligibility eligibility{};
    compact::DeviceDescriptor device{};
    compact::WorkloadDescriptor workload{};
    compact::OperationDescriptor operation{};
    compact::TensorDescriptor input{};
    compact::TensorDescriptor weight{};
    compact::OptionalTensorDescriptor bias{};
    compact::OptionalTensorDescriptor ternary_input_a{};
    compact::OptionalTensorDescriptor ternary_input_b{};
    compact::OptionalTensorDescriptor persistent_output{};
    compact::OptionalTensorDescriptor persistent_weight{};
};

enum class RequestBuildStatus : std::uint8_t {
    Success,
    TraceCaptureUnsupported,
    ExplicitProgramConfig,
    ExplicitComputeKernelConfig,
    IncompleteDescriptor,
    InconsistentDescriptor,
};

struct RequestBuildResult {
    RequestBuildStatus status = RequestBuildStatus::IncompleteDescriptor;
    std::optional<RegistryRequest> request = std::nullopt;
};

RequestBuildResult build_registry_request(const RegistryRequestFacts& facts) noexcept;

// Structural table health plus the one device fact that actually matters: a
// table certified for the BH32 8x4 domain and a given worker grid must not
// serve a call on a different one. There is deliberately no build, firmware, or
// source digest gate -- those could never match a running build, and a stale
// recipe is slower, not wrong: materialize_recipe still proves legality.
enum class CompatibilityStatus : std::uint8_t {
    Compatible,
    EmptyRegistry,
    SchemaMismatch,
    MalformedTable,
    DeviceMismatch,
};

CompatibilityStatus validate_compatibility(
    const compact::TableLock& lock,
    std::span<const compact::EntryDescriptor> entries,
    const compact::DeviceDescriptor& device) noexcept;

struct Recipe {
    ttnn::experimental::prim::MinimalMatmulConfig config{};
    DeviceComputeKernelConfig compute_kernel_config{};
};

enum class MaterializationStatus : std::uint8_t {
    Success,
    UnsupportedSchema,
    InvalidProgramConfig,
    InvalidComputeKernelConfig,
};

struct MaterializationResult {
    MaterializationStatus status = MaterializationStatus::UnsupportedSchema;
    std::optional<Recipe> recipe = std::nullopt;
};

MaterializationResult materialize_recipe(const compact::EntryDescriptor& descriptor) noexcept;

struct Resolution {
    ResolutionReason reason = ResolutionReason::Disabled;
    const compact::EntryDescriptor* descriptor = nullptr;
};

enum class ExecutionAction : std::uint8_t { Fallback, ObserveOnly, ApplyRecipe };

struct DispatchResult {
    Resolution resolution{};
    ExecutionAction action = ExecutionAction::Fallback;
    std::optional<Recipe> recipe = std::nullopt;
};

ResolutionReason preflight(const Eligibility& eligibility) noexcept;
Resolution resolve(
    Mode mode, const std::optional<RegistryRequest>& request, const Eligibility& eligibility) noexcept;

using ResolverFunction = Resolution (*)(Mode, const std::optional<RegistryRequest>&, const Eligibility&) noexcept;
using MaterializerFunction = MaterializationResult (*)(const compact::EntryDescriptor&);

// Off never calls the resolver. Shadow and On resolve at most once. Only On
// materializes, and any materialization failure circuit-breaks and falls back
// before the public device operation is attempted.
DispatchResult resolve_for_dispatch(
    Mode mode,
    const std::optional<RegistryRequest>& request,
    const Eligibility& eligibility,
    ResolverFunction resolver = &resolve,
    MaterializerFunction materializer = &materialize_recipe) noexcept;

Resolution resolve_with_table_for_testing(
    Mode mode,
    const std::optional<RegistryRequest>& request,
    const Eligibility& eligibility,
    const compact::TableLock& lock,
    std::span<const compact::EntryDescriptor> entries) noexcept;

Mode current_mode() noexcept;
void reset_startup_mode_for_testing() noexcept;

bool circuit_break() noexcept;
bool is_circuit_broken() noexcept;
void reset_circuit_breaker_for_testing() noexcept;

struct StatsSnapshot {
    bool mode_is_frozen = false;
    Mode frozen_mode = Mode::Off;
    std::size_t entry_count = 0;
    std::uint64_t resolution_attempts = 0;
    std::uint64_t certified_hits = 0;
    std::uint64_t shadow_would_hits = 0;
    std::uint64_t selected_hits = 0;
    // The asynchronous public launch returned without throwing. This is not a
    // device synchronization or silicon-correctness signal.
    std::uint64_t launch_completed_hits = 0;
    std::uint64_t fallbacks = 0;
    std::uint64_t circuit_breaker_activations = 0;
    bool circuit_broken = false;
    std::array<std::uint64_t, kResolutionReasonCount> reasons{};
};

void record_resolution(Mode mode, const Resolution& resolution, ExecutionAction action) noexcept;
void record_launch_completed_hit() noexcept;
StatsSnapshot stats_snapshot() noexcept;
void reset_stats_for_testing() noexcept;

bool fail_closed_trace_capture_active(Mode mode, std::optional<bool> observed_active) noexcept;

class SelectedExecutionGuard {
public:
    explicit SelectedExecutionGuard(const bool* selected) noexcept;
    ~SelectedExecutionGuard() noexcept;

    SelectedExecutionGuard(const SelectedExecutionGuard&) = delete;
    SelectedExecutionGuard& operator=(const SelectedExecutionGuard&) = delete;

private:
    const bool* selected_;
    int uncaught_exceptions_;
};

// No fallback callable exists at this boundary. Once selected execution begins,
// an exception propagates and the guard circuit-breaks without a baseline retry.
template <typename Callable>
decltype(auto) execute_selected_call_once(const SelectedExecutionGuard&, Callable&& callable) {
    return std::forward<Callable>(callable)();
}

}  // namespace ttnn::experimental::all_gather_minimal_matmul_registry
