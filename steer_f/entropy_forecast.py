# Copyright 2026 STEER-F authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Future-entropy forecasting utilities: ``H_togo``, calibration, ``A_H``.

Definitions follow the STEER-F plan §0.2::

    H_togo^kappa(s) = sum_{k=1..kappa} gamma_H^k * H( p_MTP(y_{t+k} | s) )
    A_H(s_t, y_t)   = H_togo(s_t + y_t) - H_bar_togo(s_t)

``H_togo`` is produced from per-head entropies (see
:meth:`steer_f.mtp_heads.MTPHeads.forecast_entropy`), so nothing here needs a
vocabulary-sized tensor.

Two baselines for ``H_bar_togo`` are implemented, because the plan asks for
both and for Phase-1 to pick between them (ablation A5):

* :func:`sibling_prefix_baseline` — the faithful one.  Rollouts count as
  siblings at position ``t`` only while their responses are token-identical on
  ``[0, t)``, so the baseline really is a function of the shared state ``s_t``.
* :func:`group_mean_baseline` — the cheap approximation: average over the whole
  prompt group regardless of prefix.

Both return ``0`` where a rollout has no sibling other than itself, which is
the degenerate case called out in plan §8.4.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Sequence, Union

import torch

__all__ = [
    "HeadCalibration",
    "h_togo",
    "entropy_advantage",
    "sibling_prefix_baseline",
    "group_mean_baseline",
    "fit_head_calibration",
    "first_divergence",
]


@dataclass
class HeadCalibration:
    """Per-head correction of forecast entropy against measured entropy.

    Distant heads systematically *over*-estimate entropy: their predictive
    distribution is blurred by everything that can happen in between, so
    ``H(p_MTP(y_{t+k}|s))`` drifts toward ``log V`` as ``k`` grows.  Left
    uncorrected this makes "far future" look uniformly diverse and destroys
    the discriminative power of ``A_H`` (plan §3.4, risk table row 4).

    The correction is applied as::

        H_cal_k = scale_k * H( softmax(logits_k / temperature_k) ) + bias_k

    Fields are per-head sequences of length ``K``.  ``temperature`` is folded
    into the entropy computation itself (cheap); ``scale``/``bias`` are the
    residual affine fit.
    """

    temperature: Sequence[float]
    scale: Sequence[float]
    bias: Sequence[float]

    def __post_init__(self) -> None:
        n = len(self.temperature)
        if not (len(self.scale) == len(self.bias) == n):
            raise ValueError(
                "temperature/scale/bias must have equal length, got "
                f"{len(self.temperature)}/{len(self.scale)}/{len(self.bias)}"
            )
        if any(t <= 0 for t in self.temperature):
            raise ValueError("all temperatures must be > 0")

    def __len__(self) -> int:
        return len(self.temperature)

    @classmethod
    def identity(cls, num_heads: int) -> "HeadCalibration":
        return cls([1.0] * num_heads, [1.0] * num_heads, [0.0] * num_heads)

    def to_dict(self) -> dict:
        return {k: list(map(float, v)) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> "HeadCalibration":
        return cls(d["temperature"], d["scale"], d["bias"])

    def apply(self, head_entropies: torch.Tensor) -> torch.Tensor:
        """Apply ``scale``/``bias`` to a ``[K, ...]`` stack of head entropies.

        ``temperature`` is *not* applied here — it must be passed to the
        entropy computation itself (``forecast_entropy(temperature=...)``),
        since re-deriving it from an entropy value is not possible.
        """
        k = head_entropies.shape[0]
        if k > len(self):
            raise ValueError(f"calibration has {len(self)} heads, got {k} entropies")
        dev, dt = head_entropies.device, head_entropies.dtype
        scale = torch.tensor(list(self.scale[:k]), device=dev, dtype=dt).view(-1, *([1] * (head_entropies.dim() - 1)))
        bias = torch.tensor(list(self.bias[:k]), device=dev, dtype=dt).view(-1, *([1] * (head_entropies.dim() - 1)))
        return scale * head_entropies + bias


def h_togo(
    head_entropies: torch.Tensor,
    kappa: int,
    gamma_h: float,
    calib: Optional[HeadCalibration] = None,
    clamp_min: float = 0.0,
) -> torch.Tensor:
    """Discounted sum of per-head forecast entropies.

    Args:
        head_entropies: ``[K, ...]`` entropies in nats, head ``k`` (0-indexed)
            forecasting offset ``+k+1``.  Produced by
            :meth:`MTPHeads.forecast_entropy`.  Passing raw ``[K, B, T, V]``
            logits is *not* supported — compute entropies first, the logit
            tensor is too large to exist.
        kappa: horizon, uses heads ``0 .. kappa-1``.
        gamma_h: discount, weight of head ``k`` is ``gamma_h ** (k + 1)``.
        calib: optional per-head affine correction.
        clamp_min: floor on the result.  Entropies are non-negative, so a
            negative ``H_togo`` can only come from a bad calibration fit;
            clamping at 0 keeps it interpretable.  Pass ``-inf`` to disable.

    Returns:
        ``[...]`` — the leading head axis is consumed.  For the usual
        ``[K, B, T]`` input this is ``[B, T]``.
    """
    if head_entropies.dim() < 1:
        raise ValueError("head_entropies must have a leading head axis")
    k_avail = head_entropies.shape[0]
    if not (1 <= kappa <= k_avail):
        raise ValueError(f"kappa must be in [1, {k_avail}], got {kappa}")
    if gamma_h <= 0:
        raise ValueError(f"gamma_h must be > 0, got {gamma_h}")

    ent = head_entropies[:kappa]
    if calib is not None:
        ent = calib.apply(ent)

    weights = torch.tensor(
        [gamma_h ** (k + 1) for k in range(kappa)],
        device=ent.device,
        dtype=ent.dtype,
    ).view(-1, *([1] * (ent.dim() - 1)))
    out = (weights * ent).sum(dim=0)
    if clamp_min != float("-inf"):
        out = out.clamp_min(clamp_min)
    return out


def first_divergence(responses: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """First position at which each pair of rollouts differs.

    Args:
        responses: ``[G, T]`` token ids of the rollouts in one prompt group.
        mask: ``[G, T]`` validity mask.  A position where exactly one of the
            two rollouts is still alive counts as a divergence (one finished
            earlier); a position where neither is alive does not.

    Returns:
        ``[G, G]`` int64, entry ``(i, j)`` is the smallest ``t`` with
        ``responses[i, t] != responses[j, t]``, or ``T`` if they never differ.
        The diagonal is always ``T``.
    """
    if responses.dim() != 2:
        raise ValueError(f"responses must be [G, T], got {tuple(responses.shape)}")
    g, t = responses.shape
    differs = responses.unsqueeze(1) != responses.unsqueeze(0)  # [G, G, T]
    if mask is not None:
        m = mask.bool()
        both_dead = (~m).unsqueeze(1) & (~m).unsqueeze(0)
        alive_mismatch = m.unsqueeze(1) != m.unsqueeze(0)
        differs = (differs | alive_mismatch) & ~both_dead
    # argmax on a bool tensor returns the first True, but returns 0 when the
    # row is all-False — hence the explicit "never differs" fallback.
    any_diff = differs.any(dim=-1)
    first = differs.float().argmax(dim=-1)
    return torch.where(any_diff, first, torch.full_like(first, t)).to(torch.int64)


def sibling_prefix_baseline(
    h_togo_vals: torch.Tensor,
    responses: torch.Tensor,
    mask: torch.Tensor,
    group_index: Sequence,
) -> torch.Tensor:
    """``H_bar_togo(s_t)`` averaged over rollouts that still share the prefix.

    Rollout ``j`` is a sibling of rollout ``i`` at position ``t`` iff both are
    in the same prompt group and ``responses[i, :t] == responses[j, :t]``.
    Note the strict ``< t``: siblings must agree on everything *before* ``t``
    but are free to differ at ``t`` itself, which is exactly what makes the
    difference ``H_togo(s_t + y_t) - H_bar_togo(s_t)`` a branch score.

    Args:
        h_togo_vals: ``[B, T]``.
        responses: ``[B, T]`` token ids.
        mask: ``[B, T]`` response mask.
        group_index: length-``B`` sequence of prompt-group keys (verl's
            ``non_tensor_batch["uid"]``).

    Returns:
        ``[B, T]`` baseline.  Where a rollout is its own only sibling the
        baseline equals its own value, so ``A_H`` is exactly 0.
    """
    _check_bt(h_togo_vals, responses, mask, group_index)
    b, t = h_togo_vals.shape
    out = h_togo_vals.clone()

    for rows in _group_rows(group_index):
        if len(rows) == 1:
            continue  # baseline = self => A_H = 0
        idx = torch.as_tensor(rows, dtype=torch.long, device=h_togo_vals.device)
        div = first_divergence(responses[idx], mask[idx])  # [g, g]
        positions = torch.arange(t, device=h_togo_vals.device).view(1, 1, t)
        # sibling at t  <=>  no divergence strictly before t  <=>  div > t - 1
        sibling = div.unsqueeze(-1) > (positions - 1)  # [g, g, T]
        sibling = sibling & mask[idx].bool().unsqueeze(0)  # only alive siblings contribute
        sibling[torch.arange(len(rows)), torch.arange(len(rows)), :] = True  # self always counts

        vals = h_togo_vals[idx].unsqueeze(0)  # [1, g, T]
        weights = sibling.to(vals.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0)
        out[idx] = (weights * vals).sum(dim=1) / denom

    return out * mask.to(out.dtype)


def group_mean_baseline(
    h_togo_vals: torch.Tensor,
    mask: torch.Tensor,
    group_index: Sequence,
) -> torch.Tensor:
    """``H_bar_togo`` approximated by the whole prompt group's mean at each ``t``.

    Cheaper and prefix-agnostic: every rollout of the prompt contributes at
    every position, whether or not it shares the prefix.  This is the
    approximation the plan allows when prefix matching is too expensive, and
    the B-arm of ablation A5.

    Only rollouts still alive at ``t`` (``mask == 1``) enter the mean, so the
    baseline does not decay toward zero as short rollouts finish.
    """
    if h_togo_vals.shape != mask.shape:
        raise ValueError(f"shape mismatch: {tuple(h_togo_vals.shape)} vs {tuple(mask.shape)}")
    if len(group_index) != h_togo_vals.shape[0]:
        raise ValueError("group_index length must equal batch size")

    out = h_togo_vals.clone()
    m = mask.to(h_togo_vals.dtype)
    for rows in _group_rows(group_index):
        if len(rows) == 1:
            continue
        idx = torch.as_tensor(rows, dtype=torch.long, device=h_togo_vals.device)
        w = m[idx]
        denom = w.sum(dim=0).clamp_min(1.0)
        mean_t = (w * h_togo_vals[idx]).sum(dim=0) / denom
        # A rollout that is the only survivor at t must see itself as the mean
        # (otherwise it would score against an empty set).
        lone = w.sum(dim=0) <= 1
        out[idx] = torch.where(lone.unsqueeze(0), h_togo_vals[idx], mean_t.unsqueeze(0).expand_as(w))
    return out * m


def entropy_advantage(
    h_togo_vals: torch.Tensor,
    group_index: Sequence,
    mask: torch.Tensor,
    responses: Optional[torch.Tensor] = None,
    baseline: str = "sibling",
    clip_c: Optional[float] = None,
) -> torch.Tensor:
    """``A_H(s_t, y_t) = H_togo(s_t + y_t) - H_bar_togo(s_t)``.

    Args:
        h_togo_vals: ``[B, T]`` from :func:`h_togo`.
        group_index: length-``B`` prompt-group keys.
        mask: ``[B, T]`` response mask.
        responses: ``[B, T]`` token ids — required for ``baseline="sibling"``.
        baseline: ``"sibling"`` (prefix-matched) or ``"group"`` (group mean).
        clip_c: if given, clamp the result to ``[-clip_c, clip_c]``.  Leaving
            it ``None`` is the usual choice here because
            :func:`steer_f.omega_tilde.compute_omega_tilde` does the clipping.

    Returns:
        ``[B, T]``, zero outside ``mask``.
    """
    if baseline == "sibling":
        if responses is None:
            raise ValueError('baseline="sibling" requires responses')
        base = sibling_prefix_baseline(h_togo_vals, responses, mask, group_index)
    elif baseline == "group":
        base = group_mean_baseline(h_togo_vals, mask, group_index)
    else:
        raise ValueError(f'baseline must be "sibling" or "group", got {baseline!r}')

    a_h = (h_togo_vals - base) * mask.to(h_togo_vals.dtype)
    if clip_c is not None:
        if clip_c <= 0:
            raise ValueError(f"clip_c must be > 0, got {clip_c}")
        a_h = a_h.clamp(-clip_c, clip_c)
    return a_h


def fit_head_calibration(
    forecast_entropies: torch.Tensor,
    measured_entropies: torch.Tensor,
    fit_temperature: bool = False,
    temperature_grid: Optional[Sequence[float]] = None,
    forecast_fn=None,
) -> HeadCalibration:
    """Least-squares affine fit of forecast entropy onto measured entropy.

    For each head ``k`` solves ``measured ~= scale_k * forecast + bias_k``.
    ``measured`` in Phase 1 is the Monte-Carlo empirical entropy of the
    continuations at offset ``+k+1``.

    Args:
        forecast_entropies: ``[K, N]`` forecast entropy per head per sample.
        measured_entropies: ``[K, N]`` measured counterpart.
        fit_temperature: also search a per-head temperature.  Requires
            ``forecast_fn(k, temperature) -> [N]``, since a temperature cannot
            be applied post-hoc to an entropy value.
        temperature_grid: candidate temperatures when ``fit_temperature``.
        forecast_fn: callable re-computing head ``k``'s entropies at a given
            temperature.

    Returns:
        A :class:`HeadCalibration`.  Heads whose forecast has no variance get
        ``scale=1, bias=0`` rather than a division by zero.
    """
    if forecast_entropies.shape != measured_entropies.shape:
        raise ValueError(
            f"shape mismatch: {tuple(forecast_entropies.shape)} vs {tuple(measured_entropies.shape)}"
        )
    k_heads = forecast_entropies.shape[0]
    temps, scales, biases = [], [], []
    grid = list(temperature_grid) if temperature_grid else [0.7, 0.85, 1.0, 1.25, 1.5, 2.0]

    for k in range(k_heads):
        best = (float("inf"), 1.0, 1.0, 0.0)  # (sse, temp, scale, bias)
        candidates = grid if (fit_temperature and forecast_fn is not None) else [1.0]
        for temp in candidates:
            x = forecast_entropies[k].double()
            if fit_temperature and forecast_fn is not None:
                x = torch.as_tensor(forecast_fn(k, temp)).double()
            y = measured_entropies[k].double()
            scale, bias = _lstsq_affine(x, y)
            sse = float(((scale * x + bias - y) ** 2).sum())
            if sse < best[0]:
                best = (sse, temp, scale, bias)
        _, temp, scale, bias = best
        temps.append(temp)
        scales.append(scale)
        biases.append(bias)

    return HeadCalibration(temps, scales, biases)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _lstsq_affine(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    """Return ``(scale, bias)`` minimising ``||scale*x + bias - y||^2``."""
    n = x.numel()
    if n == 0:
        return 1.0, 0.0
    xm, ym = x.mean(), y.mean()
    var = ((x - xm) ** 2).sum()
    if float(var) < 1e-12:
        return 1.0, float(ym - xm)
    scale = float(((x - xm) * (y - ym)).sum() / var)
    return scale, float(ym - scale * xm)


def _group_rows(group_index: Sequence) -> list[list[int]]:
    """Map a sequence of group keys to lists of row indices, order-stable."""
    buckets: dict = {}
    for i, key in enumerate(group_index):
        buckets.setdefault(_hashable(key), []).append(i)
    return list(buckets.values())


def _hashable(key):
    """numpy scalars / 0-d arrays appear in verl's ``uid`` column."""
    if isinstance(key, torch.Tensor):
        return key.item() if key.numel() == 1 else tuple(key.flatten().tolist())
    try:
        hash(key)
        return key
    except TypeError:
        return str(key)


def _check_bt(h_togo_vals, responses, mask, group_index) -> None:
    if h_togo_vals.dim() != 2:
        raise ValueError(f"h_togo_vals must be [B, T], got {tuple(h_togo_vals.shape)}")
    if responses.shape != h_togo_vals.shape:
        raise ValueError(
            f"responses shape {tuple(responses.shape)} != h_togo shape {tuple(h_togo_vals.shape)}"
        )
    if mask.shape != h_togo_vals.shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} != h_togo shape {tuple(h_togo_vals.shape)}")
    if len(group_index) != h_togo_vals.shape[0]:
        raise ValueError(
            f"group_index length {len(group_index)} != batch size {h_togo_vals.shape[0]}"
        )
