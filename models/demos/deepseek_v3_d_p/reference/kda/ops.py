# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure operation oracles for Kimi Delta Attention."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_depthwise_conv_reference(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    valid_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply causal depthwise convolution and SiLU with ``[B,W-1,D]`` history.

    ``valid_len`` marks the first ``valid_len`` rows of ``inputs`` as real and the rest as padding,
    for a chunk that is a full tile but only partly filled -- the shape of every turn but the first
    in a multi-turn conversation. Padding is always a SUFFIX in sequence order, so the only thing it
    can corrupt is the carry: the history returned to the next chunk is the tail of the window, which
    without this would be the pad rows. With it, the carry is taken from the last real row, so
    ``forward([real | pad])`` leaves the same state as ``forward([real])``.

    Outputs at padded positions are left as computed and are not meaningful; the caller discards them.
    """
    batch, _, channels = inputs.shape
    if weight.ndim != 3 or tuple(weight.shape[:2]) != (channels, 1):
        raise ValueError(f"convolution weight shape {tuple(weight.shape)} incompatible with D={channels}")
    kernel = weight.shape[-1]
    state_shape = (batch, kernel - 1, channels)
    if initial_state is None:
        history = inputs.new_zeros(state_shape)
    elif tuple(initial_state.shape) != state_shape:
        raise ValueError(f"convolution state shape {tuple(initial_state.shape)} != {state_shape}")
    else:
        history = initial_state

    window = torch.cat((history.float(), inputs.float()), dim=1)
    output = F.conv1d(window.transpose(1, 2), weight.float(), groups=channels).transpose(1, 2)
    if valid_len is None:
        carry_end = window.shape[1]
    else:
        if not 0 <= valid_len <= inputs.shape[1]:
            raise ValueError(f"valid_len {valid_len} outside [0, {inputs.shape[1]}]")
        # The window is [history | inputs], so the last real row sits at (kernel-1) + valid_len.
        carry_end = (kernel - 1) + valid_len
    final_state = (window[:, carry_end - (kernel - 1) : carry_end] if kernel > 1 else window[:, :0]).clone()
    return F.silu(output), final_state


def kda_gate_reference(
    raw_gate: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """Convert raw gate logits to negative per-key log decay."""
    heads, key_dim = raw_gate.shape[-2:]
    if a_log.numel() != heads:
        raise ValueError(f"A_log has {a_log.numel()} values, expected {heads}")
    if dt_bias.numel() != heads * key_dim:
        raise ValueError(f"dt_bias has {dt_bias.numel()} values, expected {heads * key_dim}")
    scale = a_log.float().reshape(1, 1, heads, 1).exp()
    bias = dt_bias.float().reshape(1, 1, heads, key_dim)
    gate_input = raw_gate.float() + bias
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(scale * gate_input)
    return -scale * F.softplus(gate_input)


def l2_norm_reference(inputs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Match FLA's ``x / sqrt(sum(x²) + eps)`` normalization."""
    inputs = inputs.float()
    return inputs * torch.rsqrt(inputs.square().sum(dim=-1, keepdim=True) + eps)


def kda_recurrent_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the token-ordered KDA recurrence in ``[B,T,H,D]`` layout."""
    batch, sequence, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    expected = {
        "k": (batch, sequence, heads, key_dim),
        "v": (batch, sequence, heads, value_dim),
        "gate": (batch, sequence, heads, key_dim),
        "beta": (batch, sequence, heads),
    }
    tensors = {"k": k, "v": v, "gate": gate, "beta": beta}
    for name, shape in expected.items():
        if tuple(tensors[name].shape) != shape:
            raise ValueError(f"{name} shape {tuple(tensors[name].shape)} != {shape}")

    state_shape = (batch, heads, key_dim, value_dim)
    if initial_state is None:
        state = torch.zeros(state_shape, device=q.device, dtype=torch.float32)
    elif tuple(initial_state.shape) != state_shape:
        raise ValueError(f"recurrent state shape {tuple(initial_state.shape)} != {state_shape}")
    else:
        state = initial_state.float().clone()

    q = l2_norm_reference(q) * (key_dim**-0.5)
    k = l2_norm_reference(k)
    v, gate, beta = v.float(), gate.float(), beta.float()
    output = torch.empty(batch, sequence, heads, value_dim, device=q.device, dtype=torch.float32)

    for token in range(sequence):
        q_t, k_t, v_t = q[:, token], k[:, token], v[:, token]
        state = state * gate[:, token].exp().unsqueeze(-1)
        residual = v_t - (k_t.unsqueeze(-1) * state).sum(dim=-2)
        state = state + k_t.unsqueeze(-1) * (beta[:, token].unsqueeze(-1) * residual).unsqueeze(-2)
        output[:, token] = torch.einsum("bhk,bhkv->bhv", q_t, state)

    return output, state


def sigmoid_gated_rms_norm_reference(
    inputs: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply RMSNorm followed by sigmoid output gating, per Kimi/FLA."""
    inputs = inputs.float()
    normalized = inputs * torch.rsqrt(inputs.square().mean(dim=-1, keepdim=True) + eps)
    return normalized * weight.float() * torch.sigmoid(gate.float())
