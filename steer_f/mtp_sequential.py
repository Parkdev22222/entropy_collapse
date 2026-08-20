# Copyright 2026 STEER-F authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Sequential (DeepSeek-V3 style) MTP heads.

The parallel heads in :mod:`steer_f.mtp_heads` all read the same hidden state
``h_t`` and predict offsets ``+1 .. +K`` independently.  That is cheap, but at a
branch point every sibling's forecast is a function of its own ``h_t`` alone, so
the only thing separating siblings is whatever one token of divergence did to
one hidden vector.  Measured on Phase-0 rollouts, ranking siblings at a branch
point by parallel-head ``H_togo`` against the entropy actually observed
downstream gave a pooled rho of 0.819, of which 0.711 was already available from
untrained (identity-initialised) heads — i.e. from local entropy alone.

Sequential heads close that gap by giving depth ``k`` the tokens that were
actually taken in between, which is the information that differs between
siblings:

    h^0_t = h_t                                             (policy hidden)
    h^k_t = Block_k( W_k [ norm(h^{k-1}_t) ; norm(Emb(y_{t+k})) ] )
    p^k   = softmax( lm_head(h^k_t) )      predicting y_{t+k+1}

Head 0 keeps the parallel form: it predicts ``y_{t+1}`` from ``h_t``, which is
what the policy's own unembedding already computes, so there is nothing to
condition on.

    TEACHER FORCING, AND WHAT IT COSTS CONCEPTUALLY.  Depth ``k`` consumes the
    realised ``y_{t+1..t+k}``.  During RLVR that is legitimate -- the update
    happens after the rollout completes, so the tokens exist -- but it moves
    ``H_togo`` away from "expected future entropy under the policy" and toward
    "entropy along this particular continuation".  The limit of that direction
    is the *oracle*: read the policy's own entropy at ``t+k`` straight out of
    the rollout, which verl already computes and which needs no heads at all.
    Sequential heads are therefore only worth their parameters if they beat
    both the parallel heads AND come close to the oracle while estimating the
    *expected* rather than the realised value.  ``scripts/phase1_rank_test.py``
    reports all three side by side; do not adopt this module without that
    comparison.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import nn

from .mtp_heads import entropy_from_logits

__all__ = ["SequentialMTPHeads", "seq_mtp_ce_loss"]


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, **factory):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, **factory))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype) * self.weight)


class SequentialMTPHeads(nn.Module):
    """Depth-``K`` sequential MTP predictor.

    Args:
        hidden_size: policy hidden width ``H``.
        vocab_size: bookkeeping / checkpoint portability.
        num_heads: ``K``.  Head ``k`` (0-indexed) predicts ``y_{t+k+1}``.
        head_hidden: bottleneck width inside each depth's block.
        tie_unembedding: share the policy's unembedding (recommended).
        tie_embedding: share the policy's *input* embedding for ``Emb(y_{t+k})``
            rather than allocating a second one.  Sharing keeps the checkpoint
            small and puts the token representation in the space the block
            already understands.
        zero_init_output: start each depth as a pass-through of its input
            hidden, so an untrained sequential head reduces to the parallel
            identity head and the two are comparable at initialisation.
        dtype / device: parameter factory arguments.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_heads: int = 8,
        head_hidden: int = 1024,
        tie_unembedding: bool = True,
        tie_embedding: bool = True,
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
        self.tie_embedding = tie_embedding

        factory = {"dtype": dtype, "device": device}

        # Depth 0 has no token to consume: it predicts y_{t+1} from h_t.
        self.head0 = nn.Sequential(
            nn.Linear(hidden_size, head_hidden, **factory),
            nn.SiLU(),
            nn.Linear(head_hidden, hidden_size, **factory),
        )

        # Depths 1..K-1 consume [h^{k-1} ; Emb(y_{t+k})].
        self.norm_h = nn.ModuleList(
            [_RMSNorm(hidden_size, **factory) for _ in range(num_heads - 1)]
        )
        self.norm_e = nn.ModuleList(
            [_RMSNorm(hidden_size, **factory) for _ in range(num_heads - 1)]
        )
        self.merge = nn.ModuleList(
            [nn.Linear(2 * hidden_size, hidden_size, bias=False, **factory)
             for _ in range(num_heads - 1)]
        )
        self.blocks = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(hidden_size, head_hidden, **factory),
                nn.SiLU(),
                nn.Linear(head_hidden, hidden_size, **factory),
            ) for _ in range(num_heads - 1)]
        )

        self.lm_head = None if tie_unembedding else nn.Linear(
            hidden_size, vocab_size, bias=False, **factory)
        self.embedding = None if tie_embedding else nn.Embedding(
            vocab_size, hidden_size, **factory)

        if zero_init_output:
            nn.init.zeros_(self.head0[-1].weight)
            if self.head0[-1].bias is not None:
                nn.init.zeros_(self.head0[-1].bias)
            for blk in self.blocks:
                nn.init.zeros_(blk[-1].weight)
                if blk[-1].bias is not None:
                    nn.init.zeros_(blk[-1].bias)
            # merge starts as "keep the previous hidden, ignore the token", so
            # depth k begins as a copy of depth k-1 rather than as noise.
            for mg in self.merge:
                with torch.no_grad():
                    mg.weight.zero_()
                    mg.weight[:, :hidden_size] = torch.eye(
                        hidden_size, dtype=mg.weight.dtype, device=mg.weight.device)

    # ------------------------------------------------------------------
    def _resolve(self, obj, attr, name):
        if obj is not None:
            return obj
        own = getattr(self, attr)
        if own is not None:
            return own
        raise ValueError(f"{name} must be supplied when tied to the policy")

    def _embed(self, tokens: torch.Tensor, embedding: Optional[Callable]) -> torch.Tensor:
        emb = self._resolve(embedding, "embedding", "embedding")
        return emb(tokens)

    def depth_hidden(
        self,
        hidden_states: torch.Tensor,
        tokens: torch.Tensor,
        kappa: Optional[int] = None,
        embedding: Optional[Callable] = None,
    ):
        """Hidden state of every depth.  Yields ``(k, h^k)`` for ``k < kappa``.

        Args:
            hidden_states: ``[B, T, H]`` policy hidden states, index ``i`` being
                the state after consuming ``tokens[:, i]``.
            tokens: ``[B, T]`` the response token ids, same alignment.
            kappa: number of depths to walk.
            embedding: the policy's input embedding when ``tie_embedding``.

        Positions near the end of the sequence have no ``y_{t+k}`` to consume;
        they are fed the token at the last valid index, and callers mask those
        columns out (``h_togo`` is only read where the response continues).
        """
        n = self.num_heads if kappa is None else int(kappa)
        if not (1 <= n <= self.num_heads):
            raise ValueError(f"kappa must be in [1, {self.num_heads}], got {kappa}")
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states must be [B, T, H], got {tuple(hidden_states.shape)}")
        if tokens.shape != hidden_states.shape[:2]:
            raise ValueError(
                f"tokens {tuple(tokens.shape)} must match hidden_states[:2] "
                f"{tuple(hidden_states.shape[:2])}"
            )

        t = hidden_states.shape[1]
        h = hidden_states + self.head0(hidden_states)
        yield 0, h

        for k in range(1, n):
            idx = torch.clamp(torch.arange(t, device=tokens.device) + k, max=t - 1)
            nxt = tokens.index_select(1, idx)                       # y_{t+k}
            e = self._embed(nxt, embedding).to(h.dtype)
            merged = self.merge[k - 1](
                torch.cat([self.norm_h[k - 1](h), self.norm_e[k - 1](e)], dim=-1)
            )
            h = merged + self.blocks[k - 1](merged)
            yield k, h

    @torch.no_grad()
    def forecast_entropy(
        self,
        hidden_states: torch.Tensor,
        lm_head: Optional[Callable] = None,
        tokens: Optional[torch.Tensor] = None,
        kappa: Optional[int] = None,
        chunk_size: int = 4096,
        temperature: float = 1.0,
        out_dtype: torch.dtype = torch.float32,
        embedding: Optional[Callable] = None,
    ) -> torch.Tensor:
        """``[K', B, T]`` per-depth predictive entropy, chunked over positions.

        Signature matches :meth:`steer_f.mtp_heads.MTPHeads.forecast_entropy`
        with one addition: ``tokens`` is required, because a sequential depth
        conditions on what was actually taken in between.
        """
        if tokens is None:
            raise ValueError("SequentialMTPHeads.forecast_entropy requires `tokens`")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        head = self._resolve(lm_head, "lm_head", "lm_head")

        b, t, _ = hidden_states.shape
        n = self.num_heads if kappa is None else int(kappa)
        out = torch.empty(n, b * t, dtype=out_dtype, device=hidden_states.device)

        for k, hk in self.depth_hidden(hidden_states, tokens, kappa=n, embedding=embedding):
            flat = hk.reshape(-1, hk.shape[-1])
            for start in range(0, flat.shape[0], chunk_size):
                stop = min(start + chunk_size, flat.shape[0])
                logits = head(flat[start:stop]).float()
                if temperature != 1.0:
                    logits = logits / temperature
                out[k, start:stop] = entropy_from_logits(logits).to(out_dtype)
                del logits
        return out.reshape(n, b, t)

    def ce_loss_inputs(self, hidden_states: torch.Tensor, k: int,
                       tokens: torch.Tensor, embedding: Optional[Callable] = None):
        """Projected hidden for depth ``k``, ready for an unembedding."""
        if not (0 <= k < self.num_heads):
            raise ValueError(f"k must be in [0, {self.num_heads}), got {k}")
        last = None
        for kk, hk in self.depth_hidden(hidden_states, tokens, kappa=k + 1, embedding=embedding):
            last = hk
        return last

    def extra_repr(self) -> str:
        return (f"hidden_size={self.hidden_size}, vocab_size={self.vocab_size}, "
                f"num_heads={self.num_heads}, head_hidden={self.head_hidden}, "
                f"sequential=True")


def seq_mtp_ce_loss(
    heads: "SequentialMTPHeads",
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    lm_head: Optional[Callable] = None,
    embedding: Optional[Callable] = None,
    kappa: Optional[int] = None,
    chunk_size: int = 2048,
    head_weights=None,
):
    """Cross-entropy for sequential heads.

    Mirrors :func:`steer_f.mtp_heads.mtp_ce_loss` but cannot reuse it: the
    parallel version flattens positions before projecting, and a sequential
    depth needs the sequence intact so it can consume ``y_{t+k}``.

    Depth ``k`` reads ``h^k_t`` and is scored against ``labels[:, t+k+1]``.  A
    position counts only when it and its target are both real tokens, so the
    per-depth token counts shrink with ``k`` exactly as in the parallel case
    and the two are directly comparable.

    Returns:
        ``(loss, metrics)`` with the same key names the parallel loss emits, so
        the warm-up script's monotonicity check works unchanged.
    """
    if hidden_states.dim() != 3:
        raise ValueError(f"hidden_states must be [B, T, H], got {tuple(hidden_states.shape)}")
    head = heads._resolve(lm_head, "lm_head", "lm_head")
    n_heads = heads.num_heads if kappa is None else int(kappa)
    _, seq_len, _ = hidden_states.shape
    mask_f = mask.to(torch.float32)

    total = hidden_states.new_zeros((), dtype=torch.float32)
    total_weight = 0.0
    metrics = {}

    for k, hk in heads.depth_hidden(hidden_states, labels, kappa=n_heads, embedding=embedding):
        offset = k + 1
        if offset >= seq_len:
            metrics[f"mtp/ce_k{k}"] = float("nan")
            metrics[f"mtp/ntok_k{k}"] = 0.0
            continue

        src = hk[:, : seq_len - offset, :]
        tgt = labels[:, offset:]
        valid = (mask_f[:, : seq_len - offset] * mask_f[:, offset:]) > 0
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            metrics[f"mtp/ce_k{k}"] = float("nan")
            metrics[f"mtp/ntok_k{k}"] = 0.0
            continue

        flat_h = src.reshape(-1, src.shape[-1])[valid.reshape(-1)]
        flat_t = tgt.reshape(-1)[valid.reshape(-1)]

        ce_sum = hidden_states.new_zeros((), dtype=torch.float32)
        for start in range(0, flat_h.shape[0], chunk_size):
            stop = min(start + chunk_size, flat_h.shape[0])
            logits = head(flat_h[start:stop]).float()
            ce_sum = ce_sum + torch.nn.functional.cross_entropy(
                logits, flat_t[start:stop], reduction="sum")
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
