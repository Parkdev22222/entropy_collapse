# Copyright 2026 STEER-F authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Medusa-style parallel Multi-Token-Prediction heads for STEER-F.

The heads exist to *forecast the entropy of the near future*, not to predict
tokens accurately.  That difference drives two design choices that would look
odd in a speculative-decoding setting:

1.  The primary export is :meth:`MTPHeads.forecast_entropy`, which never
    materialises the ``[K, B, T, V]`` logit tensor.  For Qwen2.5 (V=152k),
    K=8, B=8, T=3072 that tensor alone is ~1.2 TiB in bf16; entropy is a
    scalar per position, so it is computed head-by-head in token chunks and
    the logits are freed immediately.
2.  The last projection layer is zero-initialised (Medusa's trick).  With a
    residual connection this makes head *k* start life as an exact copy of
    the body's next-token head, so ``H_togo`` is a sane (if myopic) forecast
    from step 0 instead of the ~log(V) noise of a random head.

Reference for the head shape: Cai et al., "Medusa: Simple LLM Inference
Acceleration Framework with Multiple Decoding Heads" (2024).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

__all__ = ["MTPHeads", "mtp_ce_loss", "entropy_from_logits"]


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) of ``softmax(logits)`` along the last dim.

    Uses the ``logsumexp`` form ``H = lse(l) - <softmax(l), l>`` which is
    numerically stable without ever forming ``log(p)``.

    Args:
        logits: ``[..., V]``.

    Returns:
        ``[...]`` entropy in nats, always ``>= 0`` up to float error.
    """
    lse = torch.logsumexp(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    return lse - (probs * logits).sum(dim=-1)


class MTPHeads(nn.Module):
    """Predict the token distribution at offsets ``+1 .. +K`` in parallel.

    Head ``k`` (0-indexed) reads the policy's final hidden state at position
    ``t`` and predicts the distribution of ``y_{t+k+1}``.  Because the hidden
    state at position ``t`` is the one produced *after* consuming ``y_t``, the
    forecast is conditioned on the sampled token, i.e. it is the
    ``H_togo(s_t ⊕ y_t)`` of the STEER-F derivation rather than
    ``H_togo(s_t)``.  See ``docs/steer_code_map.md`` §5 for the index audit.

    Args:
        hidden_size: policy hidden width ``H``.
        vocab_size: kept for bookkeeping / checkpoint portability checks; the
            unembedding itself is injected at call time when ``tie_unembedding``
            is True.
        num_heads: ``K``.  Train with the largest ``K`` you might want; the
            heads are independent, so choosing ``kappa <= K`` at forecast time
            needs no retraining (plan §3.1).
        head_hidden: bottleneck width of the per-head MLP.
        tie_unembedding: when True (recommended start) no unembedding is
            allocated and ``lm_head`` must be supplied to :meth:`forward` /
            :meth:`forecast_entropy`.  When False an independent unembedding of
            shape ``[V, H]`` is owned by this module.
        residual: add the MLP output to its input.  Combined with
            ``zero_init_output`` this makes head ``k`` an identity-initialised
            copy of the +1 predictor.
        zero_init_output: zero-initialise the final ``Linear`` of every head.
        dtype / device: passed to the parameter constructors.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_heads: int = 8,
        head_hidden: int = 1024,
        tie_unembedding: bool = True,
        residual: bool = True,
        zero_init_output: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if head_hidden < 1:
            raise ValueError(f"head_hidden must be >= 1, got {head_hidden}")

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.head_hidden = head_hidden
        self.tie_unembedding = tie_unembedding
        self.residual = residual

        factory = {"dtype": dtype, "device": device}
        self.projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, head_hidden, **factory),
                    nn.SiLU(),
                    nn.Linear(head_hidden, hidden_size, **factory),
                )
                for _ in range(num_heads)
            ]
        )

        if tie_unembedding:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False, **factory)

        if zero_init_output:
            for proj in self.projs:
                nn.init.zeros_(proj[-1].weight)
                if proj[-1].bias is not None:
                    nn.init.zeros_(proj[-1].bias)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _resolve_lm_head(self, lm_head: Optional[Callable]) -> Callable:
        if lm_head is not None:
            return lm_head
        if self.lm_head is not None:
            return self.lm_head
        raise ValueError(
            "MTPHeads was built with tie_unembedding=True, so the policy's "
            "lm_head must be passed to forward()/forecast_entropy()."
        )

    def _project(self, k: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply head ``k``'s MLP, casting the input to the head's dtype."""
        proj = self.projs[k]
        param_dtype = proj[0].weight.dtype
        h = hidden_states.to(param_dtype)
        out = proj(h)
        if self.residual:
            out = out + h
        return out

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Return the full logit stack.

        Only use this on small tensors (unit tests, tiny debug batches) — for
        a real vocabulary the result is enormous.  Production code should call
        :meth:`forecast_entropy` or :meth:`ce_loss_inputs` instead.

        Args:
            hidden_states: ``[B, T, H]`` final hidden states of the policy.
            lm_head: unembedding callable, required iff ``tie_unembedding``.

        Returns:
            ``[K, B, T, V]`` logits, entry ``[k]`` predicting ``y_{t+k+1}``.
        """
        head = self._resolve_lm_head(lm_head)
        return torch.stack([head(self._project(k, hidden_states)) for k in range(self.num_heads)])

    @torch.no_grad()
    def forecast_entropy(
        self,
        hidden_states: torch.Tensor,
        lm_head: Optional[Callable] = None,
        kappa: Optional[int] = None,
        chunk_size: int = 4096,
        temperature: float = 1.0,
        out_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Per-head predictive entropy without materialising the logit stack.

        Args:
            hidden_states: ``[B, T, H]`` or ``[N, H]`` (rmpad / packed layout).
            lm_head: unembedding callable, required iff ``tie_unembedding``.
            kappa: use only the first ``kappa`` heads.  Defaults to all.
            chunk_size: number of positions per unembedding call.  Peak extra
                memory is ``chunk_size * V`` elements, independent of ``T``.
            temperature: divide logits before the softmax.  Applied *inside*
                the entropy so a per-head calibration temperature costs
                nothing extra; pass 1.0 and calibrate later if you prefer.
            out_dtype: dtype of the returned tensor (float32 recommended —
                entropies are summed over ``kappa`` and bf16 loses precision).

        Returns:
            ``[K', B, T]`` (or ``[K', N]`` for packed input) entropies in nats,
            where ``K' = kappa or num_heads``.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        n_heads = self.num_heads if kappa is None else int(kappa)
        if not (1 <= n_heads <= self.num_heads):
            raise ValueError(f"kappa must be in [1, {self.num_heads}], got {kappa}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

        head = self._resolve_lm_head(lm_head)
        packed = hidden_states.dim() == 2
        flat = hidden_states if packed else hidden_states.reshape(-1, hidden_states.shape[-1])
        n_pos = flat.shape[0]

        out = torch.empty(n_heads, n_pos, dtype=out_dtype, device=flat.device)
        for k in range(n_heads):
            for start in range(0, n_pos, chunk_size):
                stop = min(start + chunk_size, n_pos)
                proj = self._project(k, flat[start:stop])
                logits = head(proj).float()
                if temperature != 1.0:
                    logits = logits / temperature
                out[k, start:stop] = entropy_from_logits(logits).to(out_dtype)
                del logits, proj

        if packed:
            return out
        return out.reshape(n_heads, *hidden_states.shape[:-1])

    def ce_loss_inputs(self, hidden_states: torch.Tensor, k: int) -> torch.Tensor:
        """Projected hidden states for head ``k``, ready for an unembedding.

        Split out from the loss so callers can reuse fused
        ``linear + cross_entropy`` kernels (e.g. Liger / cut-cross-entropy)
        that take the projected hidden and the unembedding weight directly,
        which is what keeps the auxiliary loss affordable at V=152k.
        """
        if not (0 <= k < self.num_heads):
            raise ValueError(f"k must be in [0, {self.num_heads}), got {k}")
        return self._project(k, hidden_states)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, vocab_size={self.vocab_size}, "
            f"num_heads={self.num_heads}, head_hidden={self.head_hidden}, "
            f"tie_unembedding={self.tie_unembedding}, residual={self.residual}"
        )


def mtp_ce_loss(
    heads: MTPHeads,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    lm_head: Optional[Callable] = None,
    kappa: Optional[int] = None,
    chunk_size: int = 2048,
    head_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    """Cross-entropy warm-up / auxiliary loss for the MTP heads.

    Head ``k`` at position ``t`` is trained against ``labels[:, t + k + 1]``;
    positions whose target falls outside the sequence, or whose target is
    masked, contribute nothing.

    This is the loss used both by ``scripts/phase1_warmup_heads.py`` (policy
    frozen) and by the RL auxiliary term ``beta_mtp * L_mtp_ce`` of plan §4.1
    (policy live, which is what stops the forecaster going stale).

    Args:
        heads: the :class:`MTPHeads` module.
        hidden_states: ``[B, T, H]``, requires grad when training the policy.
        labels: ``[B, T]`` token ids aligned with ``hidden_states`` — that is,
            ``labels[b, t]`` is the token that *produced* ``hidden_states[b, t]``.
        mask: ``[B, T]`` 1 for real tokens, 0 for padding.
        lm_head: unembedding callable, required iff ``tie_unembedding``.
        kappa: train only the first ``kappa`` heads.  Defaults to all.
        chunk_size: positions per unembedding call, bounding peak memory.
        head_weights: optional ``[K]`` per-head loss weights.

    Returns:
        ``(loss, metrics)`` where ``loss`` is the mask-weighted mean CE over
        all heads and ``metrics`` maps ``"mtp/ce_k{k}"`` and
        ``"mtp/ntok_k{k}"`` to floats for logging.
    """
    if hidden_states.dim() != 3:
        raise ValueError(f"hidden_states must be [B, T, H], got {tuple(hidden_states.shape)}")
    head = heads._resolve_lm_head(lm_head)
    n_heads = heads.num_heads if kappa is None else int(kappa)
    if not (1 <= n_heads <= heads.num_heads):
        raise ValueError(f"kappa must be in [1, {heads.num_heads}], got {kappa}")

    _, seq_len, _ = hidden_states.shape
    mask_f = mask.to(hidden_states.dtype if hidden_states.is_floating_point() else torch.float32)

    total = hidden_states.new_zeros((), dtype=torch.float32)
    total_weight = 0.0
    metrics: dict[str, float] = {}

    for k in range(n_heads):
        offset = k + 1
        if offset >= seq_len:
            metrics[f"mtp/ce_k{k}"] = float("nan")
            metrics[f"mtp/ntok_k{k}"] = 0.0
            continue

        src_h = hidden_states[:, : seq_len - offset, :]
        tgt = labels[:, offset:]
        # A position counts only if both it and its target are real tokens.
        valid = (mask_f[:, : seq_len - offset] * mask_f[:, offset:]) > 0

        n_valid = int(valid.sum().item())
        if n_valid == 0:
            metrics[f"mtp/ce_k{k}"] = float("nan")
            metrics[f"mtp/ntok_k{k}"] = 0.0
            continue

        flat_h = src_h.reshape(-1, src_h.shape[-1])[valid.reshape(-1)]
        flat_t = tgt.reshape(-1)[valid.reshape(-1)]

        ce_sum = hidden_states.new_zeros((), dtype=torch.float32)
        for start in range(0, flat_h.shape[0], chunk_size):
            stop = min(start + chunk_size, flat_h.shape[0])
            logits = head(heads.ce_loss_inputs(flat_h[start:stop], k)).float()
            ce_sum = ce_sum + torch.nn.functional.cross_entropy(
                logits, flat_t[start:stop], reduction="sum"
            )
        ce_k = ce_sum / n_valid

        w = 1.0 if head_weights is None else float(head_weights[k])
        total = total + w * ce_k
        total_weight += w
        metrics[f"mtp/ce_k{k}"] = float(ce_k.detach())
        metrics[f"mtp/ntok_k{k}"] = float(n_valid)

    if total_weight == 0.0:
        return hidden_states.new_zeros((), dtype=torch.float32), metrics
    loss = total / total_weight
    metrics["mtp/ce_mean"] = float(loss.detach())
    return loss, metrics
