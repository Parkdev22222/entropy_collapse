# Copyright 2026 STEER-F authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Glue between verl's rollout batch and the STEER-F forecast.

Kept out of the patches on purpose.  The patches under ``patches/`` stay small
and mechanical; everything with real logic lives here, where it can be tested
on a CPU with no verl installed.

Two entry points, matching the two places verl offers:

* :func:`forecast_h_togo` — call inside the ``compute_log_prob`` (pi_old) pass,
  where the final hidden states already exist and no extra forward is needed.
* :func:`compute_a_h` — call in ``ray_trainer`` after the rollout batch is
  assembled, because the sibling baseline needs the whole prompt group and a
  micro-batch cannot see it.

Index alignment
---------------
verl slices response-aligned tensors as ``[:, -response_length - 1 : -1]``
(``dp_actor.py:244``), i.e. index ``i`` is the distribution that *predicted*
``responses[:, i]``.  The MTP heads need the hidden produced *after* consuming
``responses[:, i]``, which is one step later: ``[:, -response_length:]``.
:func:`slice_response_hidden` is the single place that offset is expressed, so
it can be fixed once if verl's layout ever changes.  See
``docs/steer_code_map.md`` §5 for the derivation.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from .entropy_forecast import HeadCalibration, entropy_advantage, h_togo
from .omega_tilde import SteerFConfig

__all__ = ["slice_response_hidden", "forecast_h_togo", "compute_a_h"]


def slice_response_hidden(hidden_states: torch.Tensor, response_length: int) -> torch.Tensor:
    """Take the hidden states that sit *after* each response token.

    Args:
        hidden_states: ``[B, S, H]`` for the full prompt+response sequence.
        response_length: ``T``.

    Returns:
        ``[B, T, H]`` where index ``i`` is the hidden produced after consuming
        ``responses[:, i]`` — the conditioning state for ``H_togo(s_t + y_t)``.
    """
    if hidden_states.dim() != 3:
        raise ValueError(f"hidden_states must be [B, S, H], got {tuple(hidden_states.shape)}")
    if response_length < 1:
        raise ValueError(f"response_length must be >= 1, got {response_length}")
    if hidden_states.shape[1] < response_length:
        raise ValueError(
            f"sequence length {hidden_states.shape[1]} shorter than response_length {response_length}"
        )
    return hidden_states[:, -response_length:, :]


@torch.no_grad()
def forecast_h_togo(
    hidden_states: torch.Tensor,
    lm_head,
    heads,
    response_length: int,
    cfg: SteerFConfig,
    calib: Optional[HeadCalibration] = None,
    temperature: float = 1.0,
    chunk_size: int = 4096,
    already_sliced: bool = False,
) -> torch.Tensor:
    """``H_togo`` for every response position of a micro-batch.

    Args:
        hidden_states: ``[B, S, H]`` full-sequence hidden states, or ``[B, T, H]``
            if ``already_sliced``.
        lm_head: the policy's unembedding (shared with the heads).
        heads: a :class:`steer_f.mtp_heads.MTPHeads`.
        response_length: ``T``.
        cfg: supplies ``kappa`` and ``gamma_h``.
        calib: per-head calibration from Phase 1.  ``None`` means uncalibrated,
            which biases distant heads upward — fine for a smoke test, not for
            a reported run (ablation A6 measures exactly this).
        temperature: must match the temperature used for the policy's own
            entropy (verl divides logits by it before computing ``entropys``),
            otherwise ``A_H`` and ``Omega`` live on different scales.
        chunk_size: positions per unembedding call.
        already_sliced: skip :func:`slice_response_hidden`.

    Returns:
        ``[B, T]`` float32.
    """
    h = hidden_states if already_sliced else slice_response_hidden(hidden_states, response_length)
    head_temp = temperature
    if calib is not None:
        # A single shared temperature is all forecast_entropy takes; per-head
        # temperatures require one call per head, which is what this loop does.
        per_head = []
        for k in range(cfg.kappa):
            ent_k = heads.forecast_entropy(
                h, lm_head, kappa=k + 1, chunk_size=chunk_size,
                temperature=temperature * float(calib.temperature[k]),
            )[k]
            per_head.append(ent_k)
        head_entropies = torch.stack(per_head)
    else:
        head_entropies = heads.forecast_entropy(
            h, lm_head, kappa=cfg.kappa, chunk_size=chunk_size, temperature=head_temp
        )

    return h_togo(head_entropies, kappa=cfg.kappa, gamma_h=cfg.gamma_h, calib=calib)


@torch.no_grad()
def compute_a_h(
    h_togo_vals: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    uid: Sequence,
    cfg: SteerFConfig,
) -> torch.Tensor:
    """``A_H`` for a full rollout batch, ready to drop into ``batch.batch["a_h"]``.

    Args:
        h_togo_vals: ``[B, T]`` from :func:`forecast_h_togo`, concatenated over
            micro-batches in rollout order.
        responses: ``[B, T]`` token ids.
        response_mask: ``[B, T]``.
        uid: ``batch.non_tensor_batch["uid"]`` — the prompt-group key verl
            already uses for GRPO advantages.
        cfg: supplies ``baseline``.

    Returns:
        ``[B, T]`` float32, unclipped (``compute_token_weights_steerf`` clips).
    """
    if h_togo_vals.shape != responses.shape:
        raise ValueError(
            f"h_togo {tuple(h_togo_vals.shape)} != responses {tuple(responses.shape)}"
        )
    return entropy_advantage(
        h_togo_vals.float(),
        group_index=uid,
        mask=response_mask,
        responses=responses,
        baseline=cfg.baseline,
    )
