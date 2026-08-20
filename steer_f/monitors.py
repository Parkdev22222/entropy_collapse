# Copyright 2026 STEER-F authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Diagnostics required by plan §4.1–§4.2, and the lambda safety valve.

Four things are watched during RL:

1.  **Forecast drift** — ``KL(policy_next_token || mtp_head_1)``.  The +1 head
    is predicting the same distribution the policy itself produces, so this KL
    is a direct, assumption-free staleness meter.  When it climbs past a
    threshold the forecast is no longer describing the current policy and
    :class:`LambdaDriftController` decays ``lambda``.
2.  **Branch-token entropy** — the mechanism check of gate G2.  If STEER-F
    works, the entropy of the top-``A_H`` decile should stay above the
    ``lambda=0`` run even when *average* entropy matches.
3.  **Token-weight distribution** — the real STEER has no discrete alpha, so
    the plan's "gamma / 1 / 1-over-gamma ratio" is reported as occupancy of
    the low / middle / high thirds of ``[token_weight_min, token_weight_max]``.
4.  **Branch recall** — do high-``A_H`` positions actually coincide with where
    sibling rollouts diverge?  Used as gate G1's secondary metric and re-run at
    RL checkpoints.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from .entropy_forecast import _group_rows, first_divergence

__all__ = [
    "LambdaDriftController",
    "mtp_policy_kl",
    "branch_token_entropy",
    "token_weight_distribution",
    "branch_recall_at_k",
]


class LambdaDriftController:
    """Decay ``lambda`` when the MTP forecast drifts away from the policy.

    The plan asks for "halve lambda when KL exceeds the threshold".  Two
    details are added because a bare trip-wire misbehaves in practice:

    * decisions use an EMA of the KL, so one noisy micro-batch cannot knock
      ``lambda`` down;
    * ``lambda`` is never restored automatically.  Drift means the forecaster
      has fallen behind the policy; silently re-enabling it on a lucky batch
      would make runs unreproducible.  :meth:`reset` is explicit.

    Args:
        lam_init: starting ``lambda``.
        threshold: KL above which a decay fires.
        decay: multiplier per trip.
        lam_min: at or below this, ``lambda`` snaps to 0 (fully off), so the
            controller reaches a clean "STEER-F disabled" state rather than
            trailing off into values too small to matter but not zero.
        ema: smoothing factor for the observed KL.
        patience: consecutive over-threshold observations before a trip.
    """

    def __init__(
        self,
        lam_init: float,
        threshold: float = 0.5,
        decay: float = 0.5,
        lam_min: float = 0.01,
        ema: float = 0.9,
        patience: int = 2,
    ) -> None:
        if not (0.0 <= ema < 1.0):
            raise ValueError(f"ema must be in [0, 1), got {ema}")
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        self.lam_init = float(lam_init)
        self.lam = float(lam_init)
        self.threshold = float(threshold)
        self.decay = float(decay)
        self.lam_min = float(lam_min)
        self.ema = float(ema)
        self.patience = int(patience)
        self.kl_ema: Optional[float] = None
        self.strikes = 0
        self.n_trips = 0

    def update(self, kl: float) -> dict:
        """Feed one KL observation; returns the metrics to log."""
        kl = float(kl)
        self.kl_ema = kl if self.kl_ema is None else self.ema * self.kl_ema + (1 - self.ema) * kl
        tripped = False
        if self.kl_ema > self.threshold:
            self.strikes += 1
            if self.strikes >= self.patience and self.lam > 0.0:
                self.lam *= self.decay
                if self.lam <= self.lam_min:
                    self.lam = 0.0
                self.strikes = 0
                self.n_trips += 1
                tripped = True
        else:
            self.strikes = 0
        return {
            "steerf/mtp_kl": kl,
            "steerf/mtp_kl_ema": self.kl_ema,
            "steerf/lam_active": self.lam,
            "steerf/lam_trips": float(self.n_trips),
            "steerf/lam_tripped": float(tripped),
        }

    def reset(self) -> None:
        """Restore ``lambda`` and clear the drift state."""
        self.lam = self.lam_init
        self.kl_ema = None
        self.strikes = 0

    def state_dict(self) -> dict:
        return {"lam": self.lam, "kl_ema": self.kl_ema, "strikes": self.strikes, "n_trips": self.n_trips}

    def load_state_dict(self, state: dict) -> None:
        self.lam = state["lam"]
        self.kl_ema = state["kl_ema"]
        self.strikes = state["strikes"]
        self.n_trips = state.get("n_trips", 0)


@torch.no_grad()
def mtp_policy_kl(
    policy_logits: torch.Tensor,
    head_projected: torch.Tensor,
    lm_head,
    max_positions: int = 4096,
    generator: Optional[torch.Generator] = None,
) -> float:
    """``KL(policy || head_1)`` averaged over a random subsample of positions.

    Both distributions describe the *same* random variable — the next token —
    so this is a like-for-like comparison and needs no alignment fudge.  It is
    subsampled because materialising two vocabulary-sized distributions for
    every position of every rollout is exactly the cost STEER-F is trying to
    avoid.

    Args:
        policy_logits: ``[N, V]`` next-token logits from the policy.
        head_projected: ``[N, H]`` head-1 output *before* the unembedding
            (i.e. ``MTPHeads.ce_loss_inputs(hidden, 0)``), aligned position for
            position with ``policy_logits``.
        lm_head: unembedding callable applied to ``head_projected``.
        max_positions: subsample size.
        generator: optional RNG for a reproducible subsample.

    Returns:
        Mean KL in nats.
    """
    if policy_logits.dim() != 2:
        raise ValueError(f"policy_logits must be [N, V], got {tuple(policy_logits.shape)}")
    if head_projected.shape[0] != policy_logits.shape[0]:
        raise ValueError(
            f"position mismatch: {head_projected.shape[0]} vs {policy_logits.shape[0]}"
        )

    n = policy_logits.shape[0]
    if n == 0:
        return 0.0
    if n > max_positions:
        idx = torch.randperm(n, device=policy_logits.device, generator=generator)[:max_positions]
        policy_logits = policy_logits[idx]
        head_projected = head_projected[idx]

    p_logprob = torch.log_softmax(policy_logits.float(), dim=-1)
    q_logprob = torch.log_softmax(lm_head(head_projected).float(), dim=-1)
    p = p_logprob.exp()
    return float((p * (p_logprob - q_logprob)).sum(dim=-1).mean())


def _top_k_selection(values: torch.Tensor, k: int) -> torch.Tensor:
    """Boolean mask selecting exactly ``k`` entries of a 1-D tensor.

    The obvious spelling — take the k-th largest as a threshold and keep
    everything ``>=`` it — silently returns far more than ``k`` when the values
    have a point mass, and ``A_H`` always does.  With ``baseline="sibling"``
    a rollout that has become unique is its own only sibling, so its ``A_H`` is
    *exactly* zero (``sibling_prefix_baseline``'s documented behaviour); past
    the first divergence that is almost every position.  Measured on 8 siblings
    with a shared 5-token prefix:

        T=  96   A_H == 0 at 93.8% of positions  ->  ">= thresh" keeps 96.7%
        T= 512   A_H == 0 at 98.8% of positions  ->  ">= thresh" keeps 99.4%
        T=1024   A_H == 0 at 99.4% of positions  ->  ">= thresh" keeps 99.7%

    A "top decile" holding 99.7% of tokens breaks both gates that rest on it:
    G1's binomial test gets a null selection rate of 0.997 and can never reach
    significance, and G2's `branch_entropy` / `nonbranch_entropy` split becomes
    99.7% against 0.3%, so `branch_entropy_gap` reports nothing about branch
    points.  Selecting by rank instead of by value fixes both.

    Ties are broken by ``torch.topk``'s index order, which is deterministic but
    arbitrary; when the tie block straddles the cut there is no better answer
    available from ``A_H`` alone.
    """
    sel = torch.zeros(values.numel(), dtype=torch.bool, device=values.device)
    sel[torch.topk(values, min(k, values.numel())).indices] = True
    return sel


@torch.no_grad()
def branch_token_entropy(
    entropys: torch.Tensor,
    a_h: torch.Tensor,
    mask: torch.Tensor,
    top_frac: float = 0.1,
) -> dict:
    """Entropy of the top-``A_H`` tokens versus everything else.

    Gate G2's mechanism check: STEER-F is supposed to preserve entropy
    *specifically at branch points*, so a rise here with a flat overall
    entropy is the signature of the method working as designed (rather than
    of a uniform entropy bonus).

    Args:
        entropys: ``[B, T]`` per-token policy entropy.
        a_h: ``[B, T]`` entropy advantage.
        mask: ``[B, T]`` response mask.
        top_frac: decile fraction, 0.1 per the plan.

    Returns:
        Metrics dict; values are ``nan`` when the batch has no valid tokens.
    """
    if not (0.0 < top_frac <= 1.0):
        raise ValueError(f"top_frac must be in (0, 1], got {top_frac}")
    valid = mask.bool()
    n = int(valid.sum())
    if n == 0:
        return {
            "steerf/branch_entropy": float("nan"),
            "steerf/nonbranch_entropy": float("nan"),
            "steerf/branch_entropy_gap": float("nan"),
        }

    ent_v = entropys[valid].float()
    a_v = a_h[valid].float()
    k = max(1, int(round(n * top_frac)))
    is_branch = _top_k_selection(a_v, k)

    branch = float(ent_v[is_branch].mean())
    other = float(ent_v[~is_branch].mean()) if int((~is_branch).sum()) > 0 else float("nan")
    return {
        "steerf/branch_entropy": branch,
        "steerf/nonbranch_entropy": other,
        "steerf/branch_entropy_gap": branch - other,
        "steerf/branch_token_frac": k / n,
        "steerf/a_h_std": float(a_v.std(unbiased=False)),
    }


@torch.no_grad()
def token_weight_distribution(
    token_weights: torch.Tensor,
    mask: torch.Tensor,
    token_weight_min: float,
    token_weight_max: float,
) -> dict:
    """Occupancy of the low / middle / high thirds of the weight range.

    Stock STEER maps the metric continuously, so there is no discrete alpha to
    count.  Thirds of ``[token_weight_min, token_weight_max]`` are the closest
    faithful analogue of the plan's "gamma / 1 / 1-over-gamma ratio", and they
    catch the same failure the plan's risk table names: a run where ~90% of
    tokens pile into one end has an effectively constant reweighting and is
    doing nothing.
    """
    if token_weight_max <= token_weight_min:
        raise ValueError(
            f"token_weight_max ({token_weight_max}) must exceed token_weight_min ({token_weight_min})"
        )
    valid = mask.bool()
    n = int(valid.sum())
    if n == 0:
        return {"steerf/tw_frac_low": float("nan"), "steerf/tw_frac_mid": float("nan"), "steerf/tw_frac_high": float("nan")}

    w = token_weights[valid].float()
    span = token_weight_max - token_weight_min
    lo_edge = token_weight_min + span / 3.0
    hi_edge = token_weight_min + 2.0 * span / 3.0
    low = float((w < lo_edge).float().mean())
    high = float((w >= hi_edge).float().mean())
    return {
        "steerf/tw_frac_low": low,
        "steerf/tw_frac_mid": 1.0 - low - high,
        "steerf/tw_frac_high": high,
        "steerf/tw_mean": float(w.mean()),
        "steerf/tw_std": float(w.std(unbiased=False)),
    }


@torch.no_grad()
def branch_recall_at_k(
    a_h: torch.Tensor,
    responses: torch.Tensor,
    mask: torch.Tensor,
    group_index: Sequence,
    top_frac: float = 0.1,
    support: Optional[torch.Tensor] = None,
) -> dict:
    """Do high-``A_H`` positions coincide with real sibling divergences?

    A position ``t`` of rollout ``i`` is a *true branch point* when at least
    one sibling that shared the prefix ``[0, t)`` chose a different token at
    ``t`` — i.e. some pairwise first-divergence equals exactly ``t``.

    Args:
        a_h: ``[B, T]`` entropy advantage.
        responses: ``[B, T]`` token ids.
        mask: ``[B, T]`` response mask.
        group_index: length-``B`` prompt-group keys.
        top_frac: decile fraction.
        support: optional ``[B, T]`` mask restricting BOTH the selection and
            the null.  Pass
            :func:`steer_f.entropy_forecast.sibling_support` to make this a
            test of how ``A_H`` *ranks* positions rather than of where it is
            defined at all.  Without it the statistic is dominated by the fact
            that ``A_H`` is nonzero only where siblings remain, which is also
            the only place branch points can occur — so it passes identically
            with untrained heads (0.4711 untrained vs 0.4693 trained on the
            Phase-0 rollouts) and carries no information about the forecast.

    Returns:
        ``recall`` (fraction of true branch points inside the top decile),
        ``baseline`` (what random selection would give, i.e. ``top_frac``),
        and ``lift`` = recall / baseline.  Gate G1 asks whether ``recall``
        beats ``baseline`` significantly; the counts needed for a binomial
        test are returned as ``n_branch`` / ``n_hit`` / ``n_valid``.
    """
    valid = mask.bool()
    if support is not None:
        valid = valid & support.bool()
    n_valid = int(valid.sum())
    if n_valid == 0:
        return {"recall": float("nan"), "baseline": top_frac, "lift": float("nan"),
                "n_branch": 0, "n_hit": 0, "n_valid": 0}

    is_branch = torch.zeros_like(mask, dtype=torch.bool)
    t = responses.shape[1]
    for rows in _group_rows(group_index):
        if len(rows) == 1:
            continue
        idx = torch.as_tensor(rows, dtype=torch.long, device=a_h.device)
        div = first_divergence(responses[idx], mask[idx])  # [g, g]
        positions = torch.arange(t, device=a_h.device).view(1, 1, t)
        hits = (div.unsqueeze(-1) == positions).any(dim=1)  # [g, T]
        is_branch[idx] = hits & mask[idx].bool()

    a_v = a_h[valid].float()
    k = max(1, int(round(n_valid * top_frac)))
    selected = torch.zeros_like(mask, dtype=torch.bool)
    selected[valid] = _top_k_selection(a_v, k)

    n_branch = int(is_branch.sum())
    n_hit = int((is_branch & selected).sum())
    recall = n_hit / n_branch if n_branch > 0 else float("nan")
    return {
        "recall": recall,
        "baseline": float(selected.sum()) / n_valid,
        "lift": recall / top_frac if n_branch > 0 else float("nan"),
        "n_branch": n_branch,
        "n_hit": n_hit,
        "n_valid": n_valid,
    }
