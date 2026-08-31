# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the stateless full-layer KDA reference transition."""

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kda.tests.helpers import make_config, random_weights
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    assert_accurate,
    assert_bit_identical,
    assert_equal,
)


@pytest.mark.parametrize("use_full_rank_gate", [False, True])
def test_split_forward_matches_full_forward_without_mutating_input_state(use_full_rank_gate: bool) -> None:
    config = make_config(use_full_rank_gate=use_full_rank_gate)
    weights = random_weights(config)
    hidden_states = torch.randn(1, 7, config.hidden_size, generator=torch.Generator().manual_seed(31))

    full_output, full_state = kda_forward_reference(hidden_states, weights, config)
    first_output, first_state = kda_forward_reference(hidden_states[:, :5], weights, config)
    input_state_snapshot = tuple(tensor.clone() for tensor in first_state.__dict__.values())
    last_output, split_state = kda_forward_reference(hidden_states[:, 5:], weights, config, first_state)

    assert_accurate(
        full_output,
        torch.cat((first_output, last_output), dim=1),
        name="split layer output",
        pcc_threshold=0.99999,
    )
    for index, (full_tensor, split_tensor) in enumerate(
        zip(full_state.__dict__.values(), split_state.__dict__.values())
    ):
        assert_accurate(full_tensor, split_tensor, name=f"split layer state {index}", pcc_threshold=0.99999)
    for input_tensor, snapshot in zip(first_state.__dict__.values(), input_state_snapshot):
        assert_equal(snapshot, input_tensor, name="reference input state unchanged")


def test_reference_layer_is_bit_identical() -> None:
    config = make_config()
    weights = random_weights(config)
    hidden_states = torch.randn(1, 32, config.hidden_size, generator=torch.Generator().manual_seed(3031))
    expected_output, expected_state = kda_forward_reference(hidden_states, weights, config)
    for iteration in range(2):
        actual_output, actual_state = kda_forward_reference(hidden_states, weights, config)
        assert_bit_identical(expected_output, actual_output, name=f"output iteration {iteration}")
        for field in expected_state.__dataclass_fields__:
            assert_bit_identical(
                getattr(expected_state, field),
                getattr(actual_state, field),
                name=f"{field} iteration {iteration}",
            )


def test_reference_rejects_missing_or_mismatched_weights(expect_error) -> None:
    config = make_config()
    hidden_states = torch.zeros(1, 1, config.hidden_size)
    weights = random_weights(config)
    del weights["q_proj.weight"]
    with expect_error(ValueError, "missing KDA weight: q_proj.weight"):
        kda_forward_reference(hidden_states, weights, config)

    weights = random_weights(config)
    weights["q_proj.weight"] = weights["q_proj.weight"][:-1]
    with expect_error(ValueError, "q_proj.weight shape"):
        kda_forward_reference(hidden_states, weights, config)


# ---------------------------------------------------------------------------
# Padded (multi-turn) forward
# ---------------------------------------------------------------------------
# A chunked prefill server hands every layer a full-width tile, so a turn whose length is not a whole
# number of tiles arrives padded. Attention can ignore the pad rows (causality masks them); KDA
# cannot, because it carries state forward -- pad tokens it does not know about are decayed and
# accumulated into the recurrent state and latch into the convolution history, corrupting every later
# turn. `valid_len` is what makes a pad step a no-op. See kda_forward_reference's docstring.


def _padded(real: torch.Tensor, pad_len: int, seed: int) -> torch.Tensor:
    """`real` followed by `pad_len` rows of NOISE, not zeros.

    Zero padding would make this test far weaker: zeros are close enough to neutral in several of the
    projections that a broken implementation could still score well. Noise pads make any leak into the
    carry obvious.
    """
    noise = torch.randn(
        real.shape[0], pad_len, real.shape[2], generator=torch.Generator().manual_seed(seed), dtype=real.dtype
    )
    return torch.cat((real, noise), dim=1)


@pytest.mark.parametrize("use_full_rank_gate", [False, True])
def test_padded_forward_leaves_the_same_state_as_the_unpadded_forward(use_full_rank_gate: bool) -> None:
    """`forward([real | pad], valid_len=len(real))` == `forward([real])`, in BOTH carries."""
    config = make_config(use_full_rank_gate=use_full_rank_gate)
    weights = random_weights(config)
    real = torch.randn(1, 5, config.hidden_size, generator=torch.Generator().manual_seed(101))
    padded = _padded(real, pad_len=3, seed=202)

    _, expected = kda_forward_reference(real, weights, config)
    padded_output, actual = kda_forward_reference(padded, weights, config, valid_len=real.shape[1])

    for name, expected_tensor, actual_tensor in zip(
        expected.__dict__.keys(), expected.__dict__.values(), actual.__dict__.values()
    ):
        assert_accurate(expected_tensor, actual_tensor, name=f"padded carry {name}", pcc_threshold=0.99999)
    # The real rows' outputs must also be untouched -- padding may not perturb what came before it.
    real_output, _ = kda_forward_reference(real, weights, config)
    assert_accurate(
        real_output, padded_output[:, : real.shape[1]], name="padded output over real rows", pcc_threshold=0.99999
    )


def test_padding_actually_corrupts_the_carry_when_valid_len_is_not_passed() -> None:
    """Guard against a vacuous test: without `valid_len` the pad rows DO change the state.

    If this ever starts failing, the case above stopped proving anything and the padding is being
    neutralised by accident (e.g. someone switched the pad rows to zeros).
    """
    config = make_config()
    weights = random_weights(config)
    real = torch.randn(1, 5, config.hidden_size, generator=torch.Generator().manual_seed(101))
    padded = _padded(real, pad_len=3, seed=202)

    _, unpadded_state = kda_forward_reference(real, weights, config)
    _, leaked_state = kda_forward_reference(padded, weights, config)

    assert not torch.allclose(
        unpadded_state.recurrent, leaked_state.recurrent, atol=1e-6
    ), "pad rows left the recurrent state unchanged even without valid_len; the padding is not adversarial"


@pytest.mark.parametrize("use_full_rank_gate", [False, True])
def test_two_padded_turns_match_one_unpadded_sequence(use_full_rank_gate: bool) -> None:
    """The multi-turn invariant itself.

    Two turns of 5 and 3 tokens, each delivered as its own 8-wide tile with the tail padded, chained
    through the carry, must produce the same state as prefilling the 8 real tokens in one go. This is
    the property a conversation depends on: turn 2 sees exactly the history turn 1 really had.
    """
    config = make_config(use_full_rank_gate=use_full_rank_gate)
    weights = random_weights(config)
    tile = 8
    turn_one = torch.randn(1, 5, config.hidden_size, generator=torch.Generator().manual_seed(303))
    turn_two = torch.randn(1, 3, config.hidden_size, generator=torch.Generator().manual_seed(404))

    _, reference_state = kda_forward_reference(torch.cat((turn_one, turn_two), dim=1), weights, config)

    _, after_one = kda_forward_reference(
        _padded(turn_one, tile - turn_one.shape[1], seed=505), weights, config, valid_len=turn_one.shape[1]
    )
    _, after_two = kda_forward_reference(
        _padded(turn_two, tile - turn_two.shape[1], seed=606),
        weights,
        config,
        after_one,
        valid_len=turn_two.shape[1],
    )

    for name, expected_tensor, actual_tensor in zip(
        reference_state.__dict__.keys(), reference_state.__dict__.values(), after_two.__dict__.values()
    ):
        assert_accurate(expected_tensor, actual_tensor, name=f"two-turn carry {name}", pcc_threshold=0.99999)
