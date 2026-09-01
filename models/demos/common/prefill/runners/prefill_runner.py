#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Disaggregated prefill runner — one entry point driving an N-rank serving pipeline.

Model-agnostic: PREFILL_MODEL selects the model, driven through a PrefillModelAdapter (see
../adapter.py and ADDING_A_PREFILL_MODEL.md). The adapter supplies how to build the model, allocate
the KV cache, run a chunk, and describe the cache's layout as a KV-chunk address table; this driver
wires rank topology, input, transport, and the per-chunk schedule, and never reimplements embed /
layers / forward.

Each rank under tt-run owns a contiguous layer slice and builds the same TtPrefillRuntime
(first_layer_idx / is_first_rank / is_last_rank). With >1 rank the cross-rank hidden state moves
device-to-device over fabric sockets (connected MGD + FABRIC_2D); N=1 needs no transport. Ranks run
decoupled: one warm-up barrier after compile, no per-chunk barrier.

Serving is request-driven and the loop is UNBOUNDED — rank 0's tokens + per-iter PrefillMetadata
arrive over the H2D socket from an external producer (prefill_producer.py / the scheduler).
KV-chunk-table migration and per-layer LayerAck run at any rank count: every rank joins the
all-gather that merges the chunk table and rank 0 publishes it, and pipeline layer completions route
to the master rank, which re-emits them into the same ack channel the scheduler connects to
single-rank. Only PREFILL_MOCK_MIGRATION stays single-rank.

Shutdown is graceful: the producer closes the stream with an all -1 PrefillMetadata sentinel that
each rank forwards downstream and then exits on. A rank blocked in the recv can only be released by a
transfer (the recv device op has no timeout), so SIGTERM/SIGKILL remains the hard fallback.
"""

import json
import os
import signal
import time

from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.adapter import DEFAULT_MODEL, PrefillRunParams, get_adapter
from models.demos.common.prefill.runners.migration import (
    migration_file_export_enabled,
    remove_stale_device_map_sidecars,
    serialize_device_map,
)
from models.demos.common.prefill.runners.runner_utils import (
    activation_global_spec,
    build_h2d_service,
    compute_layer_split,
    open_mesh_device,
)


def _apply_manifest_env():
    """If PREFILL_MANIFEST is set, load the shared run.json into the env vars the runner reads.
    setdefault, so an explicitly exported var still wins. Must run before the module-level env reads
    below (e.g. PREFILL_MAX_SEQ_LEN) or the values never take effect."""
    manifest_path = os.environ.get("PREFILL_MANIFEST")
    if not manifest_path:
        return

    # Name PREFILL_MANIFEST in the failure: a bare FileNotFoundError on run.json gives no hint what
    # pointed here, and a manifest that silently goes missing drops every rank to adapter defaults.
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"PREFILL_MANIFEST={manifest_path} does not exist")

    with open(manifest_path) as mp:
        manifest = json.load(mp)

    def sd(key, val):
        if val is not None:
            os.environ.setdefault(key, str(val))

    # A flat PREFILL_* map applied verbatim, so a rank binding can stay topology-only
    # (rank_bindings + mesh_graph_desc) and point at a per-model manifest for all model config.
    for key, val in manifest.get("env", {}).items():
        sd(key, val)


_apply_manifest_env()

# Both socket transports (H2D input on rank 0, D2D between ranks) share this push/sync worker grid and
# the same 3-word PrefillMetadata (slot_id, actual_start, actual_end). 1x1 costs nothing: compute +
# handoff gap is flat from 1x1 to 4x4, since the per-chunk overhead is the service's fabric/NoC presence.
SYNC_WORKER_CORES = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
METADATA_SIZE_BYTES = 12

# LayerAck D2H FIFO: 12 B records, so 4 KB is a PCIe-aligned single page with ample in-flight headroom.
LAYER_ACK_FIFO_SIZE_BYTES = int(os.environ.get("PREFILL_LAYER_ACK_FIFO_BYTES", 4 * 1024))

# End-of-stream sentinel: all three PrefillMetadata words -1 (0xFFFFFFFF on the wire). -1 is out of
# range for slot_id and both KV positions, so it cannot collide with a real chunk. Shared wire
# convention with the scheduler; see ADDING_A_PREFILL_MODEL.md.
SHUTDOWN_METADATA_WORD = -1

# H2D input (rank 0): tokens arrive SP-sharded on seq, replicated across TP.
H2D_MAPPER_CONFIG = ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()])

D2D_FIFO_SIZE_BYTES = int(os.environ.get("PREFILL_PP_D2D_FIFO_BYTES", 256))

ADAPTER = get_adapter(os.environ.get("PREFILL_MODEL", DEFAULT_MODEL))
MODEL_CFG = ADAPTER.model_config

# D2D socket transport (>1 rank): one sender/receiver pair per rank boundary carries the hidden state
# over inter-galaxy fabric, sharded seq across SP rows. The emb (TP) axis follows the adapter's residual
# layout (see pipeline_activation_emb_tp_sharded) so the receiver backing needs no reshard.
D2D_MAPPER_CONFIG = ttnn.MeshMapperConfig(
    placements=[
        ttnn.PlacementShard(2),
        ttnn.PlacementShard(3) if ADAPTER.pipeline_activation_emb_tp_sharded else ttnn.PlacementReplicate(),
    ]
)

_sp = int(os.environ.get("PREFILL_SP", 8))
_tp = int(os.environ.get("PREFILL_TP", 4))
GLOBAL_MESH_SHAPE = (_sp, _tp)
NUM_LAYERS = int(os.environ.get("PREFILL_NUM_LAYERS", 61))
CHUNK_SIZE = int(os.environ.get("PREFILL_CHUNK_SIZE", 5 * 1024))
# Per-user KV cache length. In request mode the external producer decides the chunk count, so this is
# the one cache-sizing knob; a chunk must not push a slot past it. Default holds 11 chunks.
MAX_SEQ_LEN = int(os.environ.get("PREFILL_MAX_SEQ_LEN", CHUNK_SIZE * 11))
NUM_USERS = int(os.environ.get("PREFILL_NUM_USERS", 2))
CAPACITY_FACTOR = int(os.environ.get("PREFILL_CAPACITY_FACTOR", 8))
_gate_mode_name = os.environ.get("PREFILL_GATE_FALLBACK_MODE", ADAPTER.default_gate_mode)
# When on (default), the last transformer layer runs kv-only: it fills the KV cache for migration and
# skips its Q/SDPA/wo, FFN/MoE, final norm, and LM head. In a pipeline only the last rank applies it.
KV_ONLY_LAST_LAYER = os.environ.get("PREFILL_KV_ONLY_LAST_LAYER", "1") == "1"
# Build the DFlash drafter context-KV cache during this prefill. The capability gate stops a non-dflash
# model from ever building a Kimi drafter; the explicit switch keeps it off even when a ckpt is on disk.
DFLASH_ENABLED = (
    ADAPTER.supports_dflash and os.environ.get("PREFILL_DFLASH", "0") == "1" and bool(os.environ.get("DFLASH_HF_MODEL"))
)
# Measurement-only: synchronize the device after each chunk's forward and log the isolated per-rank
# compute (CHUNK_COMPUTE). Off in production — the sync serializes dispatch and kills pipeline overlap.
SYNC_PER_CHUNK = os.environ.get("PREFILL_SYNC_PER_CHUNK", "0") == "1"
TIMING_DIR = os.environ.get("PREFILL_TIMING_DIR", "")
# Some models (e.g. Kimi) park the MoE routing all-gather's global semaphores in L1_SMALL so they don't
# pin the main-L1 floor and clash with the next layer's MLA static CBs; that needs the region here.
_L1_SMALL_SIZE = ADAPTER.l1_small_size
# Capture each rank's per-chunk forward as a (segmented) ttnn trace and replay it every chunk instead of
# re-dispatching op-by-op. Needs the mesh opened with a trace region; the segmented capture (sub-device
# swaps + per-layer acks) is handled by SubDeviceTraceController inside the runtime.
USE_TRACE = os.environ.get("PREFILL_USE_TRACE", "0") == "1"
_TRACE_REGION_SIZE = int(os.environ.get("PREFILL_TRACE_REGION_SIZE", 256 * 1024 * 1024)) if USE_TRACE else 0
# DFlash is not trace-compatible: the drafter tap / pack-unpack / KV finalize run outside the runtime's
# captured per-chunk segment, so a replayed trace would silently skip them. Fail loudly rather than
# produce meaningless drafter KV.
assert not (DFLASH_ENABLED and USE_TRACE), (
    "PREFILL_DFLASH=1 is incompatible with PREFILL_USE_TRACE=1: the DFlash drafter path is not "
    "trace-captured. Run DFlash with PREFILL_USE_TRACE=0."
)

os.environ.setdefault("PREFILL_TTNN_CACHE", ADAPTER.ttnn_cache_default)

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


# ---------------------------------------------------------------------------
# Layer-completion routing (pipeline / num_ranks > 1)
# ---------------------------------------------------------------------------

# When the completion ring is full, spin waiting for the router to drain rather than
# dropping/failing immediately. Bounded so a genuinely stalled router still surfaces.
LAYER_COMPLETION_PUSH_SPIN_TIMEOUT_S = float(os.environ.get("PREFILL_LAYER_COMPLETION_PUSH_TIMEOUT_S", 30.0))
LAYER_COMPLETION_PUSH_SPIN_LOG_EVERY_S = 10.0
LAYER_COMPLETION_PUSH_SPIN_SLEEP_S = 0.001  # tiny yield so the spin doesn't peg a core


def build_layer_completion_sink(producer, *, source_rank, num_layers):
    """Build the per-layer completion sink the runtime fires as ``sink(layer_idx, request_id)``.

    Pushes a full completion into `producer` (a host-local LayerCompletionQueue) keyed by
    ``seq = request_id * num_layers + layer_idx``, which the master router re-emits strictly in
    ascending order. That key must be globally dense: each rank owns a disjoint set of GLOBAL layer
    indices per request, so the union of every rank's seqs tiles [0, num_requests*num_layers) with no
    gaps or collisions. Hence `num_layers` is the GLOBAL layer total, never this rank's slice — pass
    the slice and every rank's local layer k collides on one seq, silently dropping all but one.

    `source_rank` is diagnostic payload. request_id is bound per prefill() call, so the sink reads no
    shared mutable state.
    """

    def on_layer_complete(layer_idx: int, request_id: int) -> None:
        # Hot path, fired once per layer inside model.forward: return on the common success and leave
        # the rare full-ring case to the spin below.
        seq = request_id * num_layers + layer_idx
        if producer.try_push(seq=seq, source_rank=source_rank, layer_idx=layer_idx, request_id=request_id):
            return

        # The ring is sized well above in-flight depth, so a full one means the router thread is
        # momentarily behind. Spin rather than drop; only a router that never catches up is an error.
        start = time.monotonic()
        next_log = start + LAYER_COMPLETION_PUSH_SPIN_LOG_EVERY_S
        logger.warning(
            f"[layer-completion] ring full (seq={seq}); spinning up to "
            f"{LAYER_COMPLETION_PUSH_SPIN_TIMEOUT_S:.0f}s for router to drain"
        )
        while True:
            if producer.try_push(seq=seq, source_rank=source_rank, layer_idx=layer_idx, request_id=request_id):
                logger.info(f"[layer-completion] ring drained after {time.monotonic() - start:.1f}s; pushed seq={seq}")
                return
            if _shutdown:
                # Abort on SIGTERM/SIGINT rather than ignoring it for the full timeout; teardown runs
                # in run_request_loop's finally. Raise, so a dropped completion is never silent.
                raise RuntimeError(f"layer-completion ring full (seq={seq}); shutdown requested while spinning")
            now = time.monotonic()
            if now - start >= LAYER_COMPLETION_PUSH_SPIN_TIMEOUT_S:
                logger.error(f"[layer-completion] gave up after {now - start:.1f}s spinning on full ring (seq={seq})")
                raise RuntimeError(
                    f"layer-completion ring full (seq={seq}); router not draining after "
                    f"{LAYER_COMPLETION_PUSH_SPIN_TIMEOUT_S:.0f}s"
                )
            if now >= next_log:
                logger.warning(f"[layer-completion] still spinning on full ring (seq={seq}) after {now - start:.0f}s")
                next_log += LAYER_COMPLETION_PUSH_SPIN_LOG_EVERY_S
            time.sleep(LAYER_COMPLETION_PUSH_SPIN_SLEEP_S)

    return on_layer_complete


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


def _is_shutdown_sentinel(meta: dict) -> bool:
    """True only for the all -1 end-of-stream sentinel (see SHUTDOWN_METADATA_WORD)."""
    return (
        meta["slot_id"] == SHUTDOWN_METADATA_WORD
        and meta["actual_start"] == SHUTDOWN_METADATA_WORD
        and meta["actual_end"] == SHUTDOWN_METADATA_WORD
    )


def _socket_next(h2d_service) -> tuple:
    """Block on the next producer push (rank 0 input): returns (tt_tokens, {slot_id, actual_start,
    actual_end}, tt_metadata). The device metadata tensor is returned rather than discarded because the
    model's per-layer ack send carries it."""
    import torch

    tt_tokens, tt_metadata = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
        h2d_service, metadata_size_bytes=METADATA_SIZE_BYTES
    )
    m = ttnn.to_torch(ttnn.get_device_tensors(tt_metadata)[0]).view(torch.int32).flatten()
    return tt_tokens, {"slot_id": int(m[0]), "actual_start": int(m[1]), "actual_end": int(m[2])}, tt_metadata


def build_d2d_pipeline_endpoints(mesh_device, rank: int, num_ranks: int, chunk_size: int, hidden_size: int):
    """Stand up this rank's persistent D2D endpoints: an inbound receiver from rank-1 (every rank but
    the first) and an outbound sender to rank+1 (every rank but the last).

    Order is inbound-then-outbound on every rank, and that ordering is what avoids deadlock:
    create_sender/create_receiver rendezvous point-to-point (no world barrier) and each MeshSocket
    ctor blocks until its peer's, so building inbound first chains the bring-up — rank 0's sender
    unblocks rank 1's receiver, freeing rank 1 to build its sender for rank 2, and so on. Both sides
    must pass the identical worker-core grid and global spec."""
    global_spec = activation_global_spec(chunk_size, hidden_size)

    def _common():
        # Fresh mapper per call: create_sender/create_receiver MOVE it out of a std::unique_ptr, so a
        # middle rank building both would hand the second create a consumed mapper.
        return dict(
            global_spec=global_spec,
            mapper=ttnn.create_mesh_mapper(mesh_device, D2D_MAPPER_CONFIG),
            fifo_size_bytes=D2D_FIFO_SIZE_BYTES,
            sender_worker_cores=SYNC_WORKER_CORES,
            receiver_worker_cores=SYNC_WORKER_CORES,
            metadata_size_bytes=METADATA_SIZE_BYTES,
            share_fabric_links=True,
            # The service asserts L1-only (d2d_stream_service.cpp:260).
            socket_buffer_type=ttnn.BufferType.L1,
        )

    inbound = None
    if rank > 0:
        logger.info(f"[pp rank {rank}] [d2d] creating inbound receiver from rank {rank - 1}")
        inbound = ttnn.D2DStreamService.create_receiver(
            receiver_mesh=mesh_device, sender_rank=rank - 1, receiver_rank=rank, **_common()
        )
    outbound = None
    if rank < num_ranks - 1:
        logger.info(f"[pp rank {rank}] [d2d] creating outbound sender to rank {rank + 1}")
        outbound = ttnn.D2DStreamService.create_sender(
            sender_mesh=mesh_device, sender_rank=rank, receiver_rank=rank + 1, **_common()
        )
    logger.info(
        f"[pp rank {rank}] [d2d] endpoints up (inbound={'yes' if inbound else 'no'} "
        f"outbound={'yes' if outbound else 'no'}, workers={SYNC_WORKER_CORES}, fifo={D2D_FIFO_SIZE_BYTES}B)"
    )
    return inbound, outbound


def _d2d_recv(inbound) -> tuple:
    """Drain the next chunk from the inbound receiver backing into a fresh device tensor and decode the
    inline metadata. The tensor already carries the embedding-output sharding, so it feeds
    runtime.prefill with no reshard. Pairs with the upstream rank's _d2d_send."""
    import torch

    t0 = time.perf_counter()
    act, metadata_device = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
        inbound, metadata_size_bytes=METADATA_SIZE_BYTES
    )
    m = ttnn.to_torch(ttnn.get_device_tensors(metadata_device)[0]).view(torch.int32).flatten()
    meta = {"slot_id": int(m[0]), "actual_start": int(m[1]), "actual_end": int(m[2])}
    logger.info(
        f"[pp] RECV-d2d [{meta['actual_start']},{meta['actual_end']}) slot={meta['slot_id']} "
        f"[xfer] sync={(time.perf_counter() - t0) * 1000.0:.2f}ms"
    )
    return act, meta, metadata_device


def _d2d_send(outbound, activation: ttnn.Tensor, rank: int, meta: dict, *, deallocate: bool = True) -> None:
    """Push this rank's output hidden state + metadata to the downstream receiver, then free it. The
    model already emits the activation in the sender backing's spec (outbound_socket_service_sync
    TT_FATALs on any mismatch), so no host-side relayout is needed.

    deallocate=False for the traced path's persistent _trace_output: the socket sync copies it into the
    sender backing on the CQ before the next replay is enqueued, and that replay writes the same buffer
    in place."""
    t0 = time.perf_counter()
    backing = outbound.get_backing_tensor()
    import torch

    words = [meta["slot_id"], meta["actual_start"], meta["actual_end"]]
    # The outbound op ships metadata as a replicated device tensor (3 uint32 words), not a Python list.
    md_tensor = ttnn.from_torch(
        torch.tensor(words, dtype=torch.int32).reshape(1, 1, 1, -1),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=backing.device(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(
            backing.device(),
            ttnn.MeshMapperConfig(placements=[ttnn.PlacementReplicate(), ttnn.PlacementReplicate()]),
        ),
    )
    ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(outbound, activation, metadata=md_tensor)
    if deallocate:
        ttnn.deallocate(activation)
    logger.info(
        f"[pp rank {rank}] SEND-d2d [{meta['actual_start']},{meta['actual_end']}) "
        f"[xfer] push={(time.perf_counter() - t0) * 1000.0:.2f}ms"
    )


def _forward_shutdown(d2d_out, rank: int, hidden_size: int) -> None:
    """Forward the shutdown sentinel downstream so that rank unblocks in its own recv, then release the
    outbound link so the transfer ships (mirroring _compute_and_send's tail). The payload is discarded
    once the downstream sees the sentinel, but outbound_socket_service_sync requires the input's
    per-shard spec to equal the sender backing's — hence a dummy built exactly like a real activation."""
    import torch

    dev = d2d_out.get_backing_tensor().device()
    dummy = ttnn.from_torch(
        torch.zeros(1, 1, CHUNK_SIZE, hidden_size),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(dev, D2D_MAPPER_CONFIG),
    )
    sentinel = {
        "slot_id": SHUTDOWN_METADATA_WORD,
        "actual_start": SHUTDOWN_METADATA_WORD,
        "actual_end": SHUTDOWN_METADATA_WORD,
    }
    _d2d_send(d2d_out, dummy, rank, sentinel)  # ships + frees the dummy
    d2d_out.release_fabric_links()
    logger.info(f"[pp rank {rank}] forwarded SHUTDOWN sentinel to rank {rank + 1}")


def _lease_reclaim(d2d_in, d2d_out) -> None:
    """Before a chunk: reclaim this rank's fabric links (the previous-iter D2D transfer has drained),
    then grant the inbound receiver so this chunk's activation drains into its backing. No-op without
    D2D (single rank). The outbound grant happens AFTER the push, in _compute_and_send."""
    if d2d_in is not None:
        d2d_in.wait_for_fabric_links()
    if d2d_out is not None:
        d2d_out.wait_for_fabric_links()
    if d2d_in is not None:
        d2d_in.release_fabric_links()


def _record_chunk_timing(rank: int, c: int, compute_start: float, compute_ms: float) -> None:
    """Append one chunk's timing to this rank's CSV via a single O_APPEND write (per-rank file => lone
    writer, atomic even on NFS). Write errors are swallowed: telemetry must never take down the run."""
    if not TIMING_DIR:
        return
    try:
        fd = os.open(os.path.join(TIMING_DIR, f"rank{rank}.csv"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, f"{rank},{c},{compute_start:.6f},{compute_ms:.3f}\n".encode())
        finally:
            os.close(fd)
    except OSError:
        pass


def _compute_and_send(
    runtime, kv_caches, rank: int, c: int, inp, meta: dict, d2d_out, d2h_service=None, record_dev=None
) -> float:
    """Run one chunk: prefill into the engine-owned kv_caches, forward the output downstream (non-last
    rank) and grant the outbound sender so it ships over fabric. Returns the compute-start epoch
    (NTP-comparable). CHUNK_START is logged BEFORE the forward so the slot/KV-range stays visible per
    rank even if prefill_chunk hangs, with the metadata trailing compute_start to keep the
    c=/compute_start= fields parseable by plot_pipeline_trace.py."""
    if SYNC_PER_CHUNK:
        ttnn.synchronize_device(runtime.mesh_device)
    t_start = time.time()
    t_perf = time.perf_counter()
    logger.info(
        f"[pp rank {rank}] CHUNK_START c={c} compute_start={t_start:.6f} "
        f"slot={meta['slot_id']} [{meta['actual_start']},{meta['actual_end']})"
    )
    out = runtime.prefill_chunk(
        inp,
        kv_caches,
        slot_id=meta["slot_id"],
        actual_start=meta["actual_start"],
        actual_end=meta["actual_end"],
        request_id=c,
        d2h_service=d2h_service,
        record_dev=record_dev,
    )
    if SYNC_PER_CHUNK:
        # Block on device completion so the delta is this rank's forward alone, not a downstream-start
        # proxy. Serializes dispatch, so measurement runs only.
        ttnn.synchronize_device(runtime.mesh_device)
        compute_ms = (time.perf_counter() - t_perf) * 1000.0
        logger.info(f"[pp rank {rank}] CHUNK_COMPUTE c={c} compute_ms={compute_ms:.3f}")
        _record_chunk_timing(rank, c, t_start, compute_ms)
    if not runtime.config.is_last_rank:
        # Traced, `out` is the runtime's persistent _trace_output and the next replay overwrites it in
        # place, so the send must copy without freeing. Eager, `out` is fresh.
        _d2d_send(d2d_out, out, rank, meta, deallocate=not runtime.config.use_trace)  # grant below ships it
    if d2d_out is not None:
        d2d_out.release_fabric_links()
    return t_start


def _drain_and_log_e2e(runtime, rank: int, d2d_out, first_compute_start, n_done: int, t0: float) -> None:
    """Per-rank teardown: drain the last outbound D2D forward, synchronize so the e2e clock reflects
    device completion, then log E2E_CLOCK (NTP-comparable epochs) and the chunk count. No cross-rank
    teardown barrier."""
    if d2d_out is not None:
        d2d_out.wait_for_fabric_links()
    ttnn.synchronize_device(runtime.mesh_device)
    # None when the loop exited before any chunk was computed (an immediate shutdown sentinel, or
    # SIGINT during the initial socket wait).
    fcs = f"{first_compute_start:.6f}" if first_compute_start is not None else "n/a"
    logger.info(f"[pp rank {rank}] E2E_CLOCK first_compute_start={fcs} last_compute_end={time.time():.6f}")
    logger.info(f"[pp rank {rank}] processed {n_done} chunks in {(time.perf_counter() - t0) * 1000.0:.2f} ms")


def run_request_loop(
    runtime,
    kv_caches,
    rank: int,
    num_ranks: int,
    *,
    hidden_size: int,
    h2d_service=None,
    d2d_in=None,
    d2d_out=None,
    d2h_service=None,
) -> None:
    """Production serving loop — UNBOUNDED. Rank 0 reads each chunk from the H2D socket (the producer
    decides the count), downstream ranks from D2D, until the stream closes with the all -1 shutdown
    sentinel or SIGTERM/SIGKILL arrives. The runner only serves: migration is issued from outside
    (migration_driver.py)."""
    cfg = runtime.config
    if cfg.is_first_rank and h2d_service is None:
        raise ValueError("request mode requires the H2D service on the first rank for input")
    logger.info(
        f"[pp rank {rank}/{num_ranks}] request (unbounded) loop start "
        f"(is_first={cfg.is_first_rank} is_last={cfg.is_last_rank} input={'h2d' if cfg.is_first_rank else 'd2d'})"
    )
    t0 = time.perf_counter()
    c = 0
    first = None
    while not _shutdown:
        _lease_reclaim(d2d_in, d2d_out)
        if cfg.is_first_rank:
            inp, meta, metadata_device = _socket_next(h2d_service)  # slot/start/end from the producer
        else:
            inp, meta, metadata_device = _d2d_recv(d2d_in)
        if _is_shutdown_sentinel(meta):
            # Drop the throwaway payload, hand the sentinel to the next rank so it unblocks too, then
            # fall through to the graceful drain.
            logger.info(f"[pp rank {rank}] SHUTDOWN sentinel received after {c} chunks; exiting request loop")
            ttnn.deallocate(inp)
            ttnn.deallocate(metadata_device)
            if d2d_out is not None:
                _forward_shutdown(d2d_out, rank, hidden_size)
            break
        t = _compute_and_send(
            runtime, kv_caches, rank, c, inp, meta, d2d_out, d2h_service=d2h_service, record_dev=metadata_device
        )
        if first is None:
            first = t
        c += 1
    _drain_and_log_e2e(runtime, rank, d2d_out, first, c, t0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _print_config() -> None:
    """Log every env var the runner (and its downstream model/runner_utils) reads, so each rank's config
    is visible per-log. Values are the resolved effective ones, not just what the environment set."""
    rows = [
        ("PREFILL_MODEL", ADAPTER.name),
        ("PREFILL_HF_MODEL", os.environ.get("PREFILL_HF_MODEL", ADAPTER.hf_model_default)),
        ("PREFILL_TTNN_CACHE", os.environ.get("PREFILL_TTNN_CACHE", ADAPTER.ttnn_cache_default)),
        ("resolved weight_cache_path", str(ADAPTER.weight_cache_path(GLOBAL_MESH_SHAPE))),
        ("PREFILL_SP", str(_sp)),
        ("PREFILL_TP", str(_tp)),
        ("PREFILL_NUM_LAYERS", str(NUM_LAYERS)),
        ("PREFILL_PP_LAYER_COUNTS", os.environ.get("PREFILL_PP_LAYER_COUNTS", "<even split>")),
        ("PREFILL_KV_ONLY_LAST_LAYER", str(KV_ONLY_LAST_LAYER)),
        (
            "DFLASH_ENABLED",
            f"{DFLASH_ENABLED} (adapter.supports_dflash={ADAPTER.supports_dflash}, "
            f"DFLASH_HF_MODEL={os.environ.get('DFLASH_HF_MODEL') or '<unset>'})",
        ),
        ("PREFILL_USE_TRACE", f"{USE_TRACE} (trace_region={_TRACE_REGION_SIZE >> 20} MB)"),
        ("PREFILL_CHUNK_SIZE", str(CHUNK_SIZE)),
        ("PREFILL_MAX_SEQ_LEN", str(MAX_SEQ_LEN)),
        ("PREFILL_NUM_USERS", str(NUM_USERS)),
        ("PREFILL_CAPACITY_FACTOR", str(CAPACITY_FACTOR)),
        ("PREFILL_GATE_FALLBACK_MODE", _gate_mode_name),
        ("PREFILL_FABRIC_MODE", os.environ.get("PREFILL_FABRIC_MODE", "<auto: 1d if sp<=8 else 2d>")),
        ("PREFILL_PP_D2D_FIFO_BYTES", str(D2D_FIFO_SIZE_BYTES)),
        ("PREFILL_H2D_SERVICE_ID", os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")),
        ("PREFILL_TRACE_DIR", os.environ.get("PREFILL_TRACE_DIR", ADAPTER.prefill_trace_default)),
        ("PREFILL_ENABLE_MIGRATION", os.environ.get("PREFILL_ENABLE_MIGRATION", "0")),
        ("PREFILL_MOCK_MIGRATION", os.environ.get("PREFILL_MOCK_MIGRATION", "0")),
        (
            "PREFILL_MIGRATION_TABLE_PATH",
            os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb"),
        ),
        ("PREFILL_MIGRATION_WAIT_READY_MS", os.environ.get("PREFILL_MIGRATION_WAIT_READY_MS", "120000")),
        ("PREFILL_MIGRATION_EXPORT_TO_FILE", os.environ.get("PREFILL_MIGRATION_EXPORT_TO_FILE", "0")),
        (
            "PREFILL_MIGRATION_DEVICE_MAP_PATH",
            os.environ.get("PREFILL_MIGRATION_DEVICE_MAP_PATH", "<transport-dependent default>"),
        ),
    ]
    sep = "=" * 70
    lines = [sep, "prefill_runner configuration", sep]
    lines += [f"  {label:<35} = {val}" for label, val in rows]
    lines.append(sep)
    logger.info("\n" + "\n".join(lines))


def _assert_ranks_agree_on_config(rank: int, num_ranks: int) -> None:
    """Fail fast when the ranks of a pipeline did not resolve the SAME model/shape config.

    Every PREFILL_* knob comes from this process's environment, and tt-run only guarantees per-rank
    delivery for what a rank binding puts in `global_env` (it auto-propagates only the
    TT_/ARCH_/WH_/TTNN_/DEEPSEEK_/MESH_ prefixes, and an `-x FOO` in --mpi-args lands in mpirun's FIRST
    application context, i.e. rank 0 only). So exporting PREFILL_MANIFEST in the launching shell sets
    rank 0 and silently leaves every other rank on adapter.py's DEFAULT_MODEL -- a different but equally
    valid model with an equally complete weight cache. Nothing errors: the pipeline runs one model's
    layers into another's and only the downstream ranks' KV fails PCC. A one-int allgather of a config
    fingerprint turns that into an immediate, named failure.
    """
    if num_ranks <= 1:
        return
    import zlib

    fields = {
        "PREFILL_MODEL": os.environ.get("PREFILL_MODEL") or f"<unset -> default:{DEFAULT_MODEL}>",
        "adapter": ADAPTER.name,
        "num_layers": NUM_LAYERS,
        "chunk_size": CHUNK_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
        "num_users": NUM_USERS,
        "mesh_shape": GLOBAL_MESH_SHAPE,
        "PREFILL_MIGRATION_EXPORT_TO_FILE": migration_file_export_enabled(),
    }
    fingerprint = "|".join(f"{k}={v}" for k, v in fields.items())
    digest = zlib.crc32(fingerprint.encode()) & 0x7FFFFFFF
    all_digests = ttnn.distributed_context_allgather_int(digest)
    if len(set(all_digests)) == 1:
        logger.info(f"[pp rank {rank}/{num_ranks}] config fingerprint {digest} agrees across all ranks ({fingerprint})")
        return
    disagreeing = [i for i, d in enumerate(all_digests) if d != all_digests[0]]
    raise RuntimeError(
        f"Pipeline ranks resolved DIFFERENT prefill configs: fingerprints {all_digests} "
        f"(ranks {disagreeing} differ from rank 0). This rank ({rank}) has {fingerprint}. "
        f"Each rank prints its own values above — compare PREFILL_MODEL first. "
        f"Fix: pin PREFILL_MODEL (and any other PREFILL_* the run depends on) in the rank binding's "
        f"global_env, which tt-run applies to EVERY rank. Exporting PREFILL_MANIFEST in the shell only "
        f"reaches rank 0."
    )


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    _print_config()

    # tt-run launches the MPI ranks but does not stand up the distributed context.
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()
    rank = int(ttnn.distributed_context_get_rank())
    num_ranks = int(ttnn.distributed_context_get_size())
    _assert_ranks_agree_on_config(rank, num_ranks)

    layer_split = compute_layer_split(NUM_LAYERS, num_ranks, ADAPTER.layer_split_boundaries(NUM_LAYERS))
    first_layer_idx, num_my_layers = layer_split[rank]
    is_first_rank = rank == 0
    is_last_rank = rank == num_ranks - 1
    logger.info(
        f"[pp rank {rank}/{num_ranks}] mesh={GLOBAL_MESH_SHAPE} layers=[{first_layer_idx}, "
        f"{first_layer_idx + num_my_layers}) is_first={is_first_rank} is_last={is_last_rank} "
        f"chunk_size={CHUNK_SIZE} max_seq_len={MAX_SEQ_LEN} num_users={NUM_USERS}"
    )

    mesh_device = open_mesh_device(
        GLOBAL_MESH_SHAPE, MODEL_CFG, l1_small_size=_L1_SMALL_SIZE, trace_region_size=_TRACE_REGION_SIZE
    )

    hf_config = ADAPTER.load_hf_config()
    hf_config.max_seq_len = MAX_SEQ_LEN

    params = PrefillRunParams(
        mesh_shape=GLOBAL_MESH_SHAPE,
        num_layers=num_my_layers,
        first_layer_idx=first_layer_idx,
        is_first_rank=is_first_rank,
        is_last_rank=is_last_rank,
        max_seq_len=MAX_SEQ_LEN,
        chunk_size=CHUNK_SIZE,
        num_users=NUM_USERS,
        capacity_factor=CAPACITY_FACTOR,
        num_links=2 if is_blackhole() else 1,  # Blackhole trains 2 fabric routing planes, others 1
        gate_mode_name=_gate_mode_name,
        # Only the final stage is headless, so single-rank inherits it via is_last_rank.
        kv_only_last_layer=is_last_rank and KV_ONLY_LAST_LAYER,
        # NOT gated on is_last_rank: every rank builds its owned fc slices, and the runtime derives the
        # last-rank KV tail from is_last_rank itself.
        dflash_enabled=DFLASH_ENABLED,
        weight_cache_path=ADAPTER.weight_cache_path(GLOBAL_MESH_SHAPE),
        sparse_kv_cache_format=ADAPTER.default_sparse_kv_cache_format,
        use_trace=USE_TRACE,
        overlap_shared_expert_with_dispatch=os.environ.get("PREFILL_OVERLAP_SHARED_EXPERT", "1") == "1",
    )

    runtime = ADAPTER.build_runtime(mesh_device=mesh_device, hf_config=hf_config, params=params)
    # The engine owns the KV cache(s): allocated once, handed to every runtime call as an opaque
    # container, freed with the mesh. The runner never unpacks it — that is what keeps it model-agnostic;
    # the model-specific runtime pulls out whichever caches it needs (e.g. a DSA model's index cache).
    kv_caches = ADAPTER.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)
    runtime.compile(kv_caches)

    _serve_request(runtime, kv_caches, mesh_device, hf_config, rank, num_ranks, is_first_rank)

    # Release captured traces + their sub-device managers BEFORE closing the mesh: the trace buffers
    # live inside the MoE-overlap SubDeviceManagers, so closing with both registered frees them in the
    # wrong order and segfaults in BankManager::deallocate_buffer. Optional hook.
    _release_trace = getattr(runtime, "release_trace", None)
    if _release_trace is not None:
        _release_trace()

    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    ttnn.close_mesh_device(mesh_device)
    logger.info(f"[pp rank {rank}] shutdown complete")


def _serve_request(runtime, kv_caches, mesh_device, hf_config, rank: int, num_ranks: int, is_first_rank: bool) -> None:
    """Stand up transport + migration + ack wiring, then run the unbounded request loop and tear it all
    down. num_ranks 1..N, chained over D2D.

    Migration (KV-chunk-table publish) runs at any rank count: every rank all-gathers its stage into the
    merged table and rank 0 builds + publishes it. Per-layer completions always go through a
    LayerCompletionRouter, which owns the scheduler channel locally at num_ranks == 1 and collects
    MPI-forwarded completions from subordinates above that."""
    single_rank = num_ranks == 1
    # DFlash concats the drafter's FC partial onto the hidden along the feature dim, so the D2D
    # activation is 2H wide when enabled.
    d2d_activation_width = hf_config.hidden_size * (2 if DFLASH_ENABLED else 1)

    ttnn.distributed_context_barrier()  # warm-up: all ranks finish compile before chunks flow

    # First rank only (downstream ranks read from D2D). compile() leaves a custom sub-device manager
    # loaded and the service's init program validates its cores against the default whole-chip
    # sub-device, so revert first.
    h2d_service = None
    if is_first_rank:
        mesh_device.clear_loaded_sub_device_manager()
        h2d_service = build_h2d_service(
            mesh_device,
            mesh_shape=GLOBAL_MESH_SHAPE,
            chunk_size=CHUNK_SIZE,
            mapper_config=H2D_MAPPER_CONFIG,
            worker_cores=SYNC_WORKER_CORES,
            metadata_size_bytes=METADATA_SIZE_BYTES,
        )
        service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
        descriptor_path = h2d_service.export_descriptor(service_id)
        logger.info(
            f"[pp rank {rank}] [h2d] descriptor service_id={service_id!r} -> {descriptor_path}; "
            f"drive it with prefill_producer.py / the scheduler."
        )

    d2d_in = d2d_out = None
    if num_ranks > 1:
        mesh_device.clear_loaded_sub_device_manager()
        d2d_in, d2d_out = build_d2d_pipeline_endpoints(mesh_device, rank, num_ranks, CHUNK_SIZE, d2d_activation_width)
        # The chained rendezvous finishes at staggered times per rank; without this barrier a rank can
        # reach the loop's first fabric-link lease while a peer is still in rendezvous, deadlocking the
        # lease handshake before any chunk flows.
        ttnn.distributed_context_barrier()

    # One ack wiring at every rank count, because at num_ranks == 1 the router never touches MPI
    # (layer_completion_router.cpp fetches the distributed context only when it has subordinates).
    service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
    ack_shm_name = f"/tt_prefill_layer_acks_{service_id}"
    master_rank = int(os.environ.get("PREFILL_MASTER_RANK", "0"))
    router = None
    d2h_service = None
    layer_ack_service = None
    producer = None

    def _unlink_stale_shm(name: str) -> None:
        # A run that didn't tear down cleanly leaves the segment behind, failing shm_open O_EXCL, so
        # blind removal is the recovery. Consequence: two concurrent runs sharing a
        # PREFILL_H2D_SERVICE_ID on one host evict each other's live channel.
        path = f"/dev/shm/{name.lstrip('/')}"
        if os.path.exists(path):
            logger.warning(f"[migration] removing stale shm {path} from a prior run")
            os.remove(path)

    # Rank 0 keeps the migration client alive for the process's lifetime: dropping the reference
    # destroys it and the worker loses the table it gated on. The runner publishes, never migrates.
    migration_endpoint = None
    _mock_migration = os.environ.get("PREFILL_MOCK_MIGRATION", "0") == "1"
    _migration_enabled = os.environ.get("PREFILL_ENABLE_MIGRATION", "0") == "1"
    _file_export = migration_file_export_enabled()

    # Mock integration (prefill_producer.py's PREFILL_PRODUCER_CHECK_PCC): publish the KV chunk table +
    # device map for an external device-less reader, with NO migration worker. Must stay OUTSIDE the
    # _migration_enabled block below, whose first step imports the _migration_client .so and joins a
    # cross-rank all-gather — mock has neither a client nor peers. Both writes here are local.
    if _mock_migration and not _migration_enabled:
        _mock_table_path = os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb")
        _mock_map_path = os.environ.get("PREFILL_MIGRATION_DEVICE_MAP_PATH", "/tmp/prefill_kv_device_map.json")
        runtime.build_kv_chunk_table(kv_caches, path=_mock_table_path)
        # fabric_node -> ASIC unique_id, so the producer resolves chips for read_dram_umd without
        # touching the ControlPlane. Stale rank-scoped siblings from a prior multi-rank run would merge
        # into the reader's map, so drop them first.
        remove_stale_device_map_sidecars(_mock_map_path)
        serialize_device_map(mesh_device, _mock_map_path)
        logger.info(
            f"[mock-migration] KV chunk table -> {_mock_table_path}, device map -> {_mock_map_path} "
            f"(no migration worker); prefill_producer can import them"
        )

    if _migration_enabled:
        # Bring-up must finish before the request loop opens (the worker gates on SetTable +
        # AssignDevMap), and is split by ownership:
        #   * ALL RANKS deliver their local device map + join the all-gather barrier. COLLECTIVE — a
        #     rank that skips it deadlocks the communicator.
        #   * The model RUNTIME builds + serializes the KV chunk table; it owns the cache layout.
        #   * RANK 0 ONLY publishes that table to the worker and blocks on WORKER_READY.
        # PREFILL_MIGRATION_EXPORT_TO_FILE=1 leaves both on disk instead, with no worker handshake.
        from models.demos.common.prefill.runners.migration import (
            KvCacheStage,
            allgather_kv_stage_layouts,
            deliver_device_map_and_gather_stage_layouts,
            export_device_map_file_and_gather_stage_layouts,
            migration_device_map_file_path,
            publish_serialized_table_and_wait_ready,
            rank_scoped_device_map_path,
        )

        # The layer-aware merge gathers each rank's range so the table spans all stages, so this range
        # must come through the adapter's boundaries — the same split main() built the MODEL with.
        # Without them a cross-layer-reuse model (GLM-5.2 snaps 39/39 to 38/40) describes a partition
        # its KV cache does not hold, mismapping every layer of the second stage.
        first_layer_idx, num_my_layers = compute_layer_split(
            NUM_LAYERS, num_ranks, ADAPTER.layer_split_boundaries(NUM_LAYERS)
        )[rank]
        table_path = os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb")
        wait_ready_ms = int(os.environ.get("PREFILL_MIGRATION_WAIT_READY_MS", "120000"))

        # Only rank 0 writes the merged table, but every rank (and every co-located producer) reads it
        # back, so multi-host needs shared storage. This check is rank-invariant, so all ranks raise
        # together before the stage-layout all-gather rather than deadlocking the survivors.
        if num_ranks > 1:
            _abs_table = os.path.abspath(table_path)
            if any(_abs_table == p or _abs_table.startswith(p + "/") for p in ("/tmp", "/dev/shm", "/run", "/var/tmp")):
                raise ValueError(
                    f"PREFILL_MIGRATION_TABLE_PATH={_abs_table} is on per-host storage; with num_ranks="
                    f"{num_ranks} the table rank 0 writes is invisible to the other hosts' readers. Point "
                    "it at shared/NFS storage (e.g. /data/...)."
                )

        # Drop a stale table before rank 0 rebuilds it, so a reader can never deserialize last run's.
        if is_first_rank and os.path.exists(table_path):
            logger.warning(f"[migration] removing stale KV chunk table {table_path} from a prior run")
            os.remove(table_path)

        # Same for the JSON device-map sidecars: a leftover rank-scoped file from a run with a different
        # rank count would silently merge into this run's map. Must stay BEFORE the all-gather barrier,
        # since every rank writes its fresh map only after it.
        if not _file_export:
            remove_stale_device_map_sidecars(
                os.environ.get("PREFILL_MIGRATION_DEVICE_MAP_PATH", "/tmp/prefill_kv_device_map.json")
            )

        # ALL RANKS join the stage-layout all-gather; rank 0 needs the merged layout to build the table.
        # Real migration also delivers this rank's local FNID->UMD map to its co-located worker first,
        # while mock has no worker and so never imports the client extension.
        #
        # The runtime, not the engine, describes the migratable caches: KvCaches is opaque and per-model.
        # One stage per cache, since a layout carries one DRAM base and one layer-index space.
        _multi_cache_runtime = hasattr(runtime, "kv_migration_stages")
        if _multi_cache_runtime:
            kv_stages = runtime.kv_migration_stages(kv_caches, first_layer_idx, num_my_layers)
        elif hasattr(runtime, "kv_migration_base_address"):
            kv_stages = [KvCacheStage(runtime.kv_migration_base_address(kv_caches), first_layer_idx, num_my_layers)]
        else:
            raise RuntimeError(
                f"migration enabled but runtime {type(runtime).__name__} implements neither "
                "kv_migration_stages nor kv_migration_base_address "
                "(see docs/ADDING_A_PREFILL_MODEL.md §2)."
            )
        _mock_migration = os.environ.get("PREFILL_MOCK_MIGRATION", "0") == "1"
        if _mock_migration:
            stage_layouts = allgather_kv_stage_layouts(mesh_device, kv_stages, GLOBAL_MESH_SHAPE)
        elif _file_export:
            stage_layouts = export_device_map_file_and_gather_stage_layouts(
                mesh_device, kv_stages, GLOBAL_MESH_SHAPE, migration_device_map_file_path()
            )
        else:
            stage_layouts = deliver_device_map_and_gather_stage_layouts(mesh_device, kv_stages, GLOBAL_MESH_SHAPE, rank)

        # Single-cache runtimes take the singular `stage_layout=`, and must keep taking it: their
        # single-rank guard counts stages in that layout, so handing over the outer per-cache list would
        # count caches (always 1) and silently stop rejecting multi-rank migration.
        _layout_kwarg = {"stage_layouts": stage_layouts} if _multi_cache_runtime else {"stage_layout": stage_layouts[0]}

        if _mock_migration:
            # The SAME merged table the real publish builds, minus the worker handshake. Checked before
            # is_first_rank so rank 0 takes this path too instead of blocking on a worker that isn't
            # running.
            #
            # EVERY rank serializes its OWN local map so each co-located producer resolves only its
            # host's chips: the merged table carries every host's fnids, and a producer merges the local
            # maps and skips layers owned elsewhere. Rank-scoped, or co-located ranks overwrite it.
            device_map_path = rank_scoped_device_map_path(
                os.environ.get("PREFILL_MIGRATION_DEVICE_MAP_PATH", "/tmp/prefill_kv_device_map.json"),
                rank,
                num_ranks,
            )
            serialize_device_map(mesh_device, device_map_path)
            if is_first_rank:
                # Identical to the real publish call below, minus the worker handshake.
                table_path = runtime.build_kv_chunk_table(
                    kv_caches,
                    table_path,
                    first_layer_idx=first_layer_idx,
                    num_my_layers=num_my_layers,
                    **_layout_kwarg,
                )
                logger.info(f"[mock-migration] merged KV chunk table -> {table_path} (no migration worker)")
            logger.info(f"[mock-migration] rank {rank}: local device map -> {device_map_path}")
        elif _file_export:
            # The files on disk are the handoff: no SET_TABLE, no WORKER_READY.
            if is_first_rank:
                table_path = runtime.build_kv_chunk_table(
                    kv_caches,
                    table_path,
                    first_layer_idx=first_layer_idx,
                    num_my_layers=num_my_layers,
                    **_layout_kwarg,
                )
                logger.info(f"[migration] merged KV chunk table -> {table_path} (file export; no worker handshake)")
            logger.info(f"[migration] rank {rank}: exported local device map -> {migration_device_map_file_path()}")
        else:
            # As the mock path above. Every device-less reader downstream depends on it:
            # migration_driver's --verify-migration and prefill_producer's source-KV PCC both resolve
            # chips through this file. Host-local by design, rank-scoped per co-located rank.
            device_map_path = rank_scoped_device_map_path(
                os.environ.get("PREFILL_MIGRATION_DEVICE_MAP_PATH", "/tmp/prefill_kv_device_map.json"),
                rank,
                num_ranks,
            )
            serialize_device_map(mesh_device, device_map_path)
            logger.info(f"[migration] rank {rank}: local device map -> {device_map_path}")

            if is_first_rank:
                # Build + serialize the merged table spanning all gathered stages, publish its path,
                # then block on WORKER_READY.
                table_path = runtime.build_kv_chunk_table(
                    kv_caches,
                    table_path,
                    first_layer_idx=first_layer_idx,
                    num_my_layers=num_my_layers,
                    **_layout_kwarg,
                )
                migration_endpoint = publish_serialized_table_and_wait_ready(
                    table_path=table_path,
                    wait_ready_timeout_ms=wait_ready_ms,
                )
            else:
                logger.info(
                    f"[migration] rank {rank}: delivered local device map + contributed stage "
                    f"(first_layer={first_layer_idx}, count={num_my_layers}); rank 0 sends the merged table."
                )

    elif os.environ.get("PREFILL_MOCK_MIGRATION", "0") == "1":
        # The mock publish for the migration-OFF case: one galaxy, so one complete table spanning all
        # NUM_LAYERS / NUM_USERS (both caches merged, for a sparse model).
        #
        # Single-rank only, and loudly so: only the real migration path merges stages, and that needs the
        # worker. With num_ranks>1 every rank would build a table covering only ITS layer slice and
        # publish over the same paths, racing serialize_device_map's `<path>.tmp` -> os.replace.
        if not single_rank:
            raise ValueError(
                f"PREFILL_MOCK_MIGRATION=1 is unsupported for num_ranks={num_ranks} (each rank would "
                "publish a table covering only its own layer slice; a merged mock table is not "
                "implemented); run single-rank or unset PREFILL_MOCK_MIGRATION."
            )
        table_path = os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb")
        runtime.build_kv_chunk_table(kv_caches, path=table_path)
        # The producer resolves chips for its device-less read_dram_umd through this map, so that it
        # never has to touch the ControlPlane.
        device_map_path = os.environ.get("PREFILL_MIGRATION_DEVICE_MAP_PATH", "/tmp/prefill_kv_device_map.json")
        serialize_device_map(mesh_device, device_map_path)
        logger.info(
            f"[mock-migration] KV chunk table -> {table_path}, device map -> {device_map_path} "
            f"(no migration worker); prefill_producer can import them"
        )

    # Which source feeds the ring: a device op writing one ack record per layer to a D2H socket, or
    # the runtime's host-side per-layer callback.
    use_d2h = os.environ.get("PREFILL_LAYER_ACK_D2H", "0") == "1"
    # D2H is NOT trace-capturable: its ack record is the per-chunk socket metadata tensor, whose
    # address changes every chunk, so a capture would bake in a stale one. Reject rather than quietly
    # downgrade, since PREFILL_LAYER_ACK_D2H=1 is an explicit ask; host-callback acks do work traced.
    if use_d2h and runtime.config.use_trace:
        raise ValueError(
            "PREFILL_LAYER_ACK_D2H=1 is incompatible with PREFILL_USE_TRACE=1: the D2H ack record is a "
            "per-chunk socket tensor whose address cannot be captured, so the replay would emit no acks. "
            "Run untraced for the D2H backend, or leave PREFILL_LAYER_ACK_D2H unset to ack under trace."
        )
    from ttnn._experimental.layer_completion import LayerCompletionQueue, LayerCompletionRouter

    # Each rank's router OWNS its own ring, so the name must be per-rank — _{rank} is appended even
    # to the env override, or co-located ranks unlink each other's live ring and collide on O_EXCL.
    ring_base = os.environ.get("PREFILL_LAYER_COMPLETION_RING", "/tt_prefill_layer_completion_ring")
    ring_shm_name = f"{ring_base}_{rank}"
    _unlink_stale_shm(ring_shm_name)
    if rank == master_rank:
        _unlink_stale_shm(ack_shm_name)
    # The router owns the host-local ring, and on the master the scheduler counter channel it
    # inject()s in order; subordinates MPI-forward to the master. Constructed BEFORE the ring source
    # below, because the router creates the ring and the source only connects to it.
    router = LayerCompletionRouter(
        rank=rank,
        world_size=num_ranks,
        master_rank=master_rank,
        ring_shm_name=ring_shm_name,
        scheduler_channel_shm_name=ack_shm_name if rank == master_rank else "",
        teardown_timeout_ms=30000,
    )
    if use_d2h:
        # The reader thread reconstructs (chunk, global-layer) from a per-rank record counter, which
        # holds because each rank emits exactly num_my_layers records per chunk in layer order.
        #
        # seq = chunk * NUM_LAYERS + first_layer_idx + k, so first_layer_idx/num_my_layers MUST be
        # the split the model was actually built with — hence the adapter's boundaries. A dense
        # model's are None (the even split); a cross-layer-reuse model's (GLM) differ, and an
        # unsnapped split hands the router overlapping or gapped seqs, which its reorder buffer
        # stalls head-of-line on instead of rejecting.
        first_layer_idx, num_my_layers = compute_layer_split(
            NUM_LAYERS, num_ranks, ADAPTER.layer_split_boundaries(NUM_LAYERS)
        )[rank]
        d2h_service = ttnn.D2HStreamService(
            mesh_device,
            global_spec=None,
            fifo_size_bytes=LAYER_ACK_FIFO_SIZE_BYTES,
            worker_cores=SYNC_WORKER_CORES,
            metadata_size_bytes=METADATA_SIZE_BYTES,
        )
        layer_ack_service = ttnn.LayerAckService(
            d2h_service,
            ring_shm_name,
            source_rank=rank,
            num_layers=NUM_LAYERS,
            first_layer_idx=first_layer_idx,
            local_layers=num_my_layers,
        )
        layer_ack_service.start()  # connects to the router-owned ring (created above)
        source_desc = "D2H device records"
    else:
        # The runtime fires on_layer_complete(layer_idx, request_id) per layer and the sink pushes
        # into the ring, no device D2H; layer_idx is already global (see build_layer_completion_sink).
        #
        # Named error rather than an AttributeError after a full model build + compile: a runtime
        # without the hook cannot report which layer completed, so it cannot feed the ring at all.
        if getattr(runtime, "set_layer_completion_sink", None) is None:
            raise RuntimeError(
                f"runtime {type(runtime).__name__} does not implement set_layer_completion_sink(sink), "
                "which the layer-ack path requires at every rank count "
                "(see docs/ADDING_A_PREFILL_MODEL.md)."
            )
        producer = LayerCompletionQueue.connect(ring_shm_name, connect_timeout_ms=30000)
        runtime.set_layer_completion_sink(
            build_layer_completion_sink(
                producer,
                source_rank=rank,
                num_layers=NUM_LAYERS,
            )
        )
        source_desc = "host on_layer_complete callback"
    logger.info(
        f"[migration] layer-completion routing up: rank={rank}/{num_ranks} master={master_rank} "
        f"ring={ring_shm_name} source={source_desc} "
        + (f"(owns scheduler channel {ack_shm_name})" if rank == master_rank else "(subordinate -> master)")
    )

    # Capture after the D2D endpoints AND the completion wiring exist (the capture splits at each
    # completion point) but before the loop, so the one-time cost stays out of it. No-op if captured.
    if getattr(runtime, "capture_trace", None) and runtime.config.use_trace:
        runtime.capture_trace(kv_caches)

    logger.info(f"[pp rank {rank}] setup complete, entering request loop")

    try:
        run_request_loop(
            runtime,
            kv_caches,
            rank,
            num_ranks,
            hidden_size=d2d_activation_width,
            h2d_service=h2d_service,
            d2d_in=d2d_in,
            d2d_out=d2d_out,
            d2h_service=d2h_service,
        )
    finally:
        # The request loop can raise (e.g. the sink's ring-full spin timing out), and without teardown
        # the shm segments and the router listener thread leak while a peer blocked in D2D recv
        # deadlocks the pipeline. Services must be released while the mesh + command queues are still
        # alive: their dtors free a command queue and service-core L1.
        import gc

        # The D2H reader thread reads the D2H service's sockets and pushes into the router-owned ring,
        # so join it before the service is dropped AND before router.stop() drains the ring, or the
        # last records never land. None under the host-callback backend.
        if layer_ack_service is not None:
            layer_ack_service.stop()
            layer_ack_service = None
        h2d_service = d2d_in = d2d_out = d2h_service = None
        gc.collect()
        if producer is not None:
            producer.shutdown()
        if router is not None:
            router.stop()  # joins the listener; the master's final ring-drain + inject happens HERE


if __name__ == "__main__":
    # Some galaxies ship an RLIMIT_NPROC soft limit small enough to starve the runner's threads. Guarded
    # because get/setrlimit raise when the limit is immutable, and that must not crash us before main().
    try:
        import resource

        _, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
    except (OSError, ValueError) as e:
        logger.warning(f"[prefill] could not raise RLIMIT_NPROC to the hard limit: {e}")

    main()
