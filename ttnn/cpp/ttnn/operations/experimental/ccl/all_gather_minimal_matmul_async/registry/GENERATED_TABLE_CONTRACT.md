# AGMM generated table contract

This is the only supported integration boundary between the offline 32-chip
predictor pipeline and TT-metal. The exporter replaces
`agmm_registry_data.hpp` with deterministic C++ constants. TT-metal compiles
those constants into the native AGMM operation. Runtime Python, JSON parsing,
model inference, wrapper dispatch, environment-selected sidecars, and network
or filesystem lookup are not part of this contract.

## Required C++ surface

The generated header must define exactly these objects and accessors in
`ttnn::experimental::all_gather_minimal_matmul_registry::generated`:

```cpp
inline constexpr compact::TableLock kLock{/* explicit fields */};
inline constexpr std::array<compact::EntryDescriptor, N> kEntries{/* entries */};

static_assert(
    compact::validate_table_lock(kLock, kEntries) ==
    (N == 0 ? compact::TableValidationStatus::Empty
            : compact::TableValidationStatus::Valid));

inline constexpr const compact::TableLock& lock() noexcept { return kLock; }
inline constexpr std::span<const compact::EntryDescriptor> entries() noexcept { return kEntries; }
```

The checked-in empty fixture uses `N == 0`. A production export must have
`N > 0`, set `entry_count == N`, use the schema/ABI constants declared in
`agmm_registry_descriptor.hpp`, and explicitly initialize every field. It must
not depend on aggregate defaults for a populated lock or entry.

Each populated lock certifies exactly one Blackhole 32-chip 8x4 device domain.
`certified_device` must use architecture value `kBlackholeArchitecture`,
`device_count == 32`, `mesh_rows == 8`, `mesh_cols == 4`, and a nonzero worker
grid. Different worker grids are distinct runtime and program-legality cohorts
and therefore require distinct locks; they are not wildcarded.

Board capability class, ordered-mesh, fabric-topology and runtime-capability
digests are no longer part of the device key, and per-tensor memory-config and
tensor-topology digests are no longer part of the tensor key. They had no
versioned canonical preimage an offline promoter could reproduce, so binding
them made every entry permanently unreachable at runtime while guarding nothing
`materialize_recipe` does not already prove. Codegen ABI 2 is the first ABI
without them; ABI 1 tables cannot load.

All entries must:

- have a nonzero `entry_id`;
- use the lock's key, replay, and codegen ABI versions;
- have a device descriptor exactly equal to the lock's `certified_device`;
- be strictly increasing by `KeyDescriptor::operator<=>`; and
- therefore contain no duplicate exact key.

The constexpr validator makes violations compilation errors in the generated
header. The runtime validates the same typed contract before exact lookup and
fails closed on malformed or incompatible tables.

## Digest rules

Every digest in a populated lock is required and is a 32-byte SHA-256 value.
The exporter writes bytes as explicit unsigned hexadecimal literals in digest
order. The following domain-separated byte encodings define the two generated
identities:

- `entry_id = SHA256("ttnn-agmm-entry-v1\0" || encode(key) || encode(replay))`
- `content_sha256 = SHA256("ttnn-agmm-table-v1\0" || u64(N) ||
  entry_id[0] || ... || entry_id[N-1])`

`encode` visits fields in their declaration order in
`agmm_registry_descriptor.hpp`. Unsigned and signed integers use fixed-width
little-endian two's-complement bytes, booleans use one byte (`0` or `1`), enums
use their explicitly stored integer field, arrays encode every element without
a length prefix, and structs recursively encode their fields without C++
padding. Floats never enter the ABI directly; stored IEEE-754 values use their
exact `uint32_t` bits. No text rendering, locale, native struct bytes, generic
C++ hash, or process-local identifier is permitted.

The remaining lock digests are inert provenance. They must be present and
nonzero so a table's origin stays readable in a diff and in telemetry, and no
runtime code compares any of them against a live build, firmware, or source
tree:

- `semantic_source_sha256`: the AGMM source path/content manifest measured when
  the lock was promoted;
- `evidence_manifest_sha256`: the immutable silicon evidence manifest;
- `predictor_sha256`: the immutable selection-policy or predictor artifact; and
- `exporter_sha256`: the exporter implementation plus its schema/configuration.

`content_sha256` is different in kind: it is the lock's own integrity digest,
the emitter validates it, and the runtime still requires it nonzero.

The codegen repository must regenerate the header byte-for-byte and test both
generated digests independently before proposing it to TT-metal. Native code
does not claim to recompute SHA-256 at dispatch time.

`ReplayDescriptor::compute_kernel_config.throttle_level` is an explicit ABI
field, not a default. The exporter must emit one of the reviewed integer values
`0..5`, corresponding exactly to `NO_THROTTLE` and `LEVEL_1` through `LEVEL_5`.
Native materialization maps all six values one-for-one into
`DeviceComputeKernelConfig`; every other value rejects the replay before launch.

## Promotion gates

A nonempty generated table is necessary but not sufficient to select a recipe.
Production selection remains fail closed until TT-metal can build the complete
exact request from live tensors, mesh, and fabric facts; until then `resolve`
reports `incomplete_request` and the operation falls back unchanged. Shadow must
demonstrate exact hits first.

Only measured keys may be exported. A predictor ranking for an unseen key is a
candidate for silicon validation, never an executable recipe. Every exported
entry must carry a real per-entry evidence-record digest and an immutable
promotion decision naming its baseline, its measured latency, and its speedup;
the exporter recomputes the evidence manifest over those and rejects a
mismatch.
