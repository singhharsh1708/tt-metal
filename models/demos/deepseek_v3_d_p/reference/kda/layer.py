# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stateless full-layer reference transition for Kimi Delta Attention."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.ops import (
    causal_depthwise_conv_reference,
    kda_gate_reference,
    kda_recurrent_reference,
    sigmoid_gated_rms_norm_reference,
)
from models.demos.deepseek_v3_d_p.reference.kda.weights import validate_kda_weights


@dataclass(frozen=True)
class KDAReferenceState:
    """Caller-owned reference state; convolution histories exclude the current token."""

    recurrent: torch.Tensor
    q_convolution: torch.Tensor
    k_convolution: torch.Tensor
    v_convolution: torch.Tensor


def _initial_state(inputs: torch.Tensor, config: KDAConfig) -> KDAReferenceState:
    batch = inputs.shape[0]
    history = config.conv_kernel_size - 1
    return KDAReferenceState(
        recurrent=inputs.new_zeros(batch, config.num_heads, config.head_k_dim, config.head_v_dim),
        q_convolution=inputs.new_zeros(batch, history, config.q_dim),
        k_convolution=inputs.new_zeros(batch, history, config.k_dim),
        v_convolution=inputs.new_zeros(batch, history, config.v_dim),
    )


def kda_forward_reference(
    hidden_states: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    config: KDAConfig,
    state: KDAReferenceState | None = None,
    *,
    valid_len: int | None = None,
) -> tuple[torch.Tensor, KDAReferenceState]:
    """Execute the complete Kimi Delta Attention layer in pure Torch.

    ``valid_len`` marks the first ``valid_len`` tokens as real and the rest as padding. A chunked
    prefill server hands every layer a full-width tile, so any turn whose length is not a whole
    number of tiles arrives padded -- and KDA, unlike attention, carries state forward, so padding it
    cannot ignore is padding that corrupts every later turn. Two carries have to be defended:

    * the **convolution history**, which is otherwise the tail of the tile, i.e. pad rows;
    * the **recurrent state**, which is otherwise decayed and updated once per pad token.

    A pad step is made a no-op by setting its gate to 0 (``exp(0) == 1``, so no decay) and its beta to
    0 (so the delta-rule update contributes nothing). ``k`` is deliberately NOT zeroed: it is
    l2-normalised downstream, and a zero vector there is a division by zero whose NaN would propagate
    into the state that beta=0 was supposed to protect.

    With this set, ``forward([real | pad])`` returns the same state as ``forward([real])``. Outputs at
    padded positions are not meaningful and the caller discards them.
    """
    if hidden_states.ndim != 3 or hidden_states.shape[-1] != config.hidden_size:
        raise ValueError(f"hidden_states shape {tuple(hidden_states.shape)} must be [B,T,{config.hidden_size}]")
    if valid_len is not None and not 0 <= valid_len <= hidden_states.shape[1]:
        raise ValueError(f"valid_len {valid_len} outside [0, {hidden_states.shape[1]}]")
    validate_kda_weights(weights, config)
    state = _initial_state(hidden_states, config) if state is None else state
    hidden = hidden_states.float()

    q, q_state = causal_depthwise_conv_reference(
        F.linear(hidden, weights["q_proj.weight"].float()),
        weights["q_conv1d.weight"],
        state.q_convolution,
        valid_len=valid_len,
    )
    k, k_state = causal_depthwise_conv_reference(
        F.linear(hidden, weights["k_proj.weight"].float()),
        weights["k_conv1d.weight"],
        state.k_convolution,
        valid_len=valid_len,
    )
    v, v_state = causal_depthwise_conv_reference(
        F.linear(hidden, weights["v_proj.weight"].float()),
        weights["v_conv1d.weight"],
        state.v_convolution,
        valid_len=valid_len,
    )

    batch, sequence, _ = hidden.shape
    q = q.reshape(batch, sequence, config.num_heads, config.head_k_dim)
    k = k.reshape(batch, sequence, config.num_heads, config.head_k_dim)
    v = v.reshape(batch, sequence, config.num_heads, config.head_v_dim)
    raw_gate = F.linear(
        F.linear(hidden, weights["f_a_proj.weight"].float()), weights["f_b_proj.weight"].float()
    ).reshape(batch, sequence, config.num_heads, config.head_k_dim)
    gate = kda_gate_reference(raw_gate, weights["A_log"], weights["dt_bias"], config.gate_lower_bound)
    beta = torch.sigmoid(F.linear(hidden, weights["b_proj.weight"].float()))
    if valid_len is not None and valid_len < sequence:
        # Neutralise the pad steps: gate 0 => exp(0) == 1 => no decay, beta 0 => no delta-rule update.
        # Together they make `state = state * 1 + k (x) (0 * residual)` an identity on the state. See
        # the docstring for why `k` is left alone rather than zeroed.
        gate = gate.clone()
        beta = beta.clone()
        gate[:, valid_len:] = 0.0
        beta[:, valid_len:] = 0.0
    output, recurrent = kda_recurrent_reference(q, k, v, gate, beta, state.recurrent)

    if config.use_full_rank_gate:
        output_gate = F.linear(hidden, weights["g_proj.weight"].float())
    else:
        output_gate = F.linear(F.linear(hidden, weights["g_a_proj.weight"].float()), weights["g_b_proj.weight"].float())
    output_gate = output_gate.reshape(batch, sequence, config.num_heads, config.head_v_dim)
    output = sigmoid_gated_rms_norm_reference(output, output_gate, weights["o_norm.weight"], config.norm_eps).reshape(
        batch, sequence, config.v_dim
    )
    output = F.linear(output, weights["o_proj.weight"].float())

    return output, KDAReferenceState(recurrent, q_state, k_state, v_state)
