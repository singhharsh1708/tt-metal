# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Which multi-turn resume offsets Kimi-K3 can actually serve, and why.

Multi-turn prefill resumes at whatever KV offset the previous turn ended on. Nothing in the model
*declares* a constraint on that offset beyond tile alignment -- KDA's `local T % KDA_CHUNK_SIZE`, the
MoE gate's `total_tokens % 32`, `update_padded_kv_cache`'s `kv_actual_global % 32` -- and all of those
are satisfied unconditionally, because the chunk tensor is always full width (5120 global, 640/chip)
no matter how many of its tokens are real.

The real constraint is emergent and silent. MLA's KV writer wants the rotated row order so its cache
writes stay contiguous (`rotated_chip_positions`, which mirrors the writer kernel). KDA's
sequence-parallel combinators -- `convolution_halo` and `_distributed_affine_prefix` in `tt/kda/ops.py`
-- compose their carries walking chips in DEVICE INDEX order. Those two agree only for some offsets:

    resume offset          chip order            split chip   KDA ordering work
    multiple of 5120       identity [0..7]       none         none
    multiple of 640        cyclic                none         small (walk in sequence order)
    otherwise (e.g. 1024)  cyclic                exactly one  structural (#54962)

A "split chip" holds two DISJOINT runs of sequence positions -- the chunk's first rows and its last
rows -- so a single per-chip affine summary cannot represent it. That is the case a cyclic reordering
cannot fix.

No assertion fires in any of these cases; a bad offset silently computes the wrong answer. This test
exists so the table cannot drift unnoticed, because the whole multi-turn alignment strategy (round the
resume point down to a supported offset and replay the remainder) is chosen from it.

Hardware-free: `rotated_chip_positions` is pure arithmetic.
"""

from __future__ import annotations

import pytest

from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions

SP = 8
CHUNK = 5120
CHUNK_LOCAL = CHUNK // SP  # 640


def _runs(positions: list[int]) -> int:
    """Number of maximal ascending-contiguous runs of global position on one chip."""
    count = 1
    for a, b in zip(positions, positions[1:]):
        if b != a + 1:
            count += 1
    return count


def _layout(resume: int):
    """(chip order in sequence order, chips whose rows are split) for a chunk starting at `resume`."""
    positions = rotated_chip_positions(resume, SP, CHUNK_LOCAL)
    order = sorted(range(SP), key=lambda c: min(positions[c]))
    split = [c for c in range(SP) if _runs(positions[c]) > 1]
    return order, split


def _is_cyclic_rotation(order: list[int]) -> bool:
    start = order[0]
    return order == [(start + i) % SP for i in range(SP)]


@pytest.mark.parametrize("resume", [0, CHUNK, 2 * CHUNK, 11 * CHUNK])
def test_chunk_aligned_resume_needs_no_reordering(resume: int) -> None:
    """A multiple of the full chunk leaves the rotation an identity: chip c holds block c, in order.

    This is the rung the producer is aimed at, precisely because KDA's device-order composition is
    already correct here and no KDA change is needed to serve it.
    """
    order, split = _layout(resume)
    assert order == list(range(SP)), f"resume={resume} should be identity, got {order}"
    assert split == [], f"resume={resume} should split no chip, got {split}"


@pytest.mark.parametrize("resume", [CHUNK_LOCAL, 2 * CHUNK_LOCAL, 5 * CHUNK_LOCAL, CHUNK + 3 * CHUNK_LOCAL])
def test_block_aligned_resume_is_a_pure_cyclic_rotation(resume: int) -> None:
    """A multiple of chunk_local rotates which chip comes first, but keeps every chip contiguous.

    Serving these needs the SP combinators to walk chips in sequence order rather than device order --
    mechanical, and expressible in the existing per-chip affine model because no chip is split.
    """
    order, split = _layout(resume)
    assert _is_cyclic_rotation(order), f"resume={resume} should be a cyclic rotation, got {order}"
    assert split == [], f"resume={resume} should split no chip, got {split}"
    if resume % CHUNK:
        assert order != list(range(SP)), f"resume={resume} is not chunk-aligned so it must not be identity"


@pytest.mark.parametrize("resume", [1024, 2048, 32, 4096, 5120 + 1024])
def test_merely_tile_aligned_resume_splits_exactly_one_chip(resume: int) -> None:
    """A 32-aligned offset that is NOT a multiple of 640 splits one chip into two disjoint runs.

    Exactly one, always: the chunk spans nine partial blocks over eight chips, so the first and last
    both land on the same chip. This is why the case is structural rather than an ordering bug, and it
    is the reason 1024- and 2048-token turns cannot be served directly today (#54962).
    """
    assert resume % 32 == 0 and resume % CHUNK_LOCAL != 0, "this case is tile- but not block-aligned"
    order, split = _layout(resume)
    assert len(split) == 1, f"resume={resume} should split exactly one chip, got {split}"
    assert _is_cyclic_rotation(order), f"resume={resume} chip order should still be cyclic, got {order}"


def test_every_chunk_covers_its_positions_exactly_once() -> None:
    """Whatever the offset, the chunk's rows are a permutation of [resume, resume+CHUNK).

    Guards the premise underneath all of the above: the rotation only ever REORDERS positions across
    chips, it never drops or duplicates one. If this failed, the alignment discussion would be moot
    because the chunk would not carry the tokens it claims to.
    """
    for resume in (0, 32, 640, 1024, 2048, 5120, 9280):
        positions = rotated_chip_positions(resume, SP, CHUNK_LOCAL)
        flat = sorted(p for chip in positions for p in chip)
        assert flat == list(range(resume, resume + CHUNK)), f"resume={resume} does not tile its range"


def test_kimi_k3_adapter_asks_for_an_alignment_this_ladder_calls_safe() -> None:
    """The adapter's declared resume granularity must be a rung that actually works.

    This is the join between the table above and what the producer does with it: the producer rounds
    a resume offset down to `multi_turn_resume_alignment`, so that value has to land somewhere the
    rotation leaves KDA's device-order composition correct. Asserting it here means a future tightening
    (to chunk_size // sp, say) cannot be made without the ladder agreeing it is serviceable.
    """
    from models.demos.common.prefill.adapter import get_adapter

    align = get_adapter("kimi_k3").multi_turn_resume_alignment(CHUNK)
    assert align % 32 == 0, f"alignment {align} violates the shared tile floor"

    # Every offset the producer can now produce must be identity-ordered and split-free, which is what
    # "correct with today's KDA" means. Sampling multiples of the alignment across a long conversation.
    for turns in range(1, 12):
        order, split = _layout(turns * align)
        assert order == list(range(SP)), f"resume={turns * align} is not identity-ordered: {order}"
        assert split == [], f"resume={turns * align} splits chips {split}"


def test_a_merely_tile_aligned_resume_would_not_be_serviceable() -> None:
    """The negative case, so the test above cannot pass by accident.

    If the adapter ever returned the shared 32, the offsets it produced would include split-chip cases
    that KDA silently miscomputes. Stated explicitly because that is the failure this whole mechanism
    exists to prevent, and it has no runtime symptom to catch it later.
    """
    bad = [t * 32 for t in range(1, 40) if (t * 32) % CHUNK_LOCAL]
    assert any(_layout(off)[1] for off in bad), "expected some 32-aligned offsets to split a chip"
