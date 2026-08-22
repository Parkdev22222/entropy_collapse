# Copyright 2026 STEER-F authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Omega-tilde: STEER's local entropy-change estimate plus a visitation term.

    Omega_tilde = norm(Omega_local) + lambda * norm( dlogpi_hat * clip(A_H, -c, c) )
    dlogpi_hat  = eta * w * (1 - pi_sampled)

This module owns the *whole* patched ``compute_token_weights`` so that the
edit to verl's ``core_algos.py`` stays a three-line delegation (see
the delegation in ``verl/trainer/ppo/core_algos.py``).  That keeps the vendored verl tree
essentially pristine and — more usefully — makes the token-weighting logic
unit-testable without importing verl at all.

Two properties are load-bearing and are enforced by ``tests/``:

**(1) lambda = 0 is bit-identical to stock STEER.**  Guaranteed structurally,
by returning ``omega_local`` untouched before any normalisation runs, not by
hoping that ``norm`` happens to be a no-op.

**(2) The normalisation must not disturb STEER's mapping.**  This is subtler
than the plan anticipated, because the real STEER has no entropy band and no
discrete alpha (see ``docs/steer_code_map.md`` §3).  What it actually does is
map the metric to ``[token_weight_min, token_weight_max]`` by one of two
functions, and each has its own invariance:

=================  ==========================================================
mapping            invariant under
=================  ==========================================================
``linear=True``    any affine ``a*x + b``, ``a > 0`` (it is a min-max rescale)
``linear=False``   positive scaling ``a*x`` only, and only while the ``0.02``
                   floor on the denominator is slack (``exp`` is not
                   shift-invariant)
=================  ==========================================================

and on top of that, ``mode="symmetric"`` applies ``abs()`` to the metric, so
the point ``0`` carries meaning and *any* recentring changes which tokens are
treated as "large entropy change".

The intersection of those constraints is **positive scaling without
recentring**.  That is why :func:`normalize` defaults to ``"scale"``
(divide by RMS) rather than the plan's ``z_norm``: RMS scaling is the only
normalisation that leaves all four ``(mode, linear)`` combinations behaving
exactly as stock STEER when ``lambda -> 0``, while still putting the local and
visitation terms on a common footing.  ``"z"`` is implemented and selectable,
but it is only safe with ``mode="asymmetric", linear=True``; the config
validator says so out loud.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

__all__ = [
    "SteerFConfig",
    "clip_indicator",
    "local_omega_signed",
    "delta_logpi_hat",
    "visit_term",
    "normalize",
    "reshape_metric",
    "compute_omega_tilde",
    "branch_weight_correction",
    "compute_token_weights_steerf",
]

_EPS = 1e-8


@dataclass
class SteerFConfig:
    """Hyper-parameters of the future-entropy term.

    Attributes:
        lam: ``lambda``, weight of the visitation term.  ``0.0`` disables
            STEER-F entirely and is bit-identical to stock STEER.
        eta: learning-rate-like scale in ``dlogpi_hat``.  Only its product
            with ``lam`` matters after normalisation, so leave it at 1.0 and
            sweep ``lam``.
        clip_c: symmetric clip on ``A_H``, bounding how much a mis-forecast
            can move any single token.
        norm: ``"scale"`` (RMS, default and safe), ``"z"``, or ``"none"``.
        mapping: what the metric looks like when STEER's band mapping sees it —
            ``"minmax"`` (stock), ``"winsor"`` or ``"rank"``.  See
            :func:`reshape_metric`.
        winsor_q: tail fraction clamped at each end when ``mapping="winsor"``.
        baseline: ``"sibling"`` or ``"group"`` — see
            :func:`steer_f.entropy_forecast.entropy_advantage`.
        kappa / gamma_h: forecast horizon and discount.
        beta_mtp: weight of the MTP cross-entropy auxiliary loss.
        kl_drift_threshold: KL(policy || head-1) above which ``lam`` is halved.
        lam_decay: multiplier applied to ``lam`` on a drift trip.
        lam_min: floor below which the decayed ``lam`` is set to 0.
    """

    lam: float = 0.0
    eta: float = 1.0
    clip_c: float = 1.0
    norm: str = "scale"
    mapping: str = "minmax"
    winsor_q: float = 0.01
    baseline: str = "sibling"
    kappa: int = 4
    gamma_h: float = 0.85
    beta_mtp: float = 0.05
    kl_drift_threshold: float = 0.5
    lam_decay: float = 0.5
    lam_min: float = 0.01

    def validate(self, mode: str = "symmetric", linear: bool = True) -> list[str]:
        """Check the config against the mapping it will feed.

        Returns a list of human-readable warnings (empty when clean) and
        raises on outright invalid values.
        """
        if self.lam < 0:
            raise ValueError(f"lam must be >= 0, got {self.lam}")
        if self.clip_c <= 0:
            raise ValueError(f"clip_c must be > 0, got {self.clip_c}")
        if self.norm not in ("scale", "z", "none"):
            raise ValueError(f'norm must be "scale", "z" or "none", got {self.norm!r}')
        if self.baseline not in ("sibling", "group"):
            raise ValueError(f'baseline must be "sibling" or "group", got {self.baseline!r}')
        if self.kappa < 1:
            raise ValueError(f"kappa must be >= 1, got {self.kappa}")
        if self.gamma_h <= 0:
            raise ValueError(f"gamma_h must be > 0, got {self.gamma_h}")

        warnings: list[str] = []
        if self.norm == "z" and mode == "symmetric":
            warnings.append(
                'norm="z" recentres the metric, but mode="symmetric" takes abs() of it, '
                "so the zero pivot moves and token weights change even at lam=0. "
                'Use norm="scale" or mode="asymmetric".'
            )
        if self.norm == "z" and not linear:
            warnings.append(
                'norm="z" recentres the metric, but the exponential mapping (linear=False) '
                'is not shift-invariant. Use norm="scale".'
            )
        if self.mapping not in ("minmax", "winsor", "rank"):
            raise ValueError(
                f'mapping must be "minmax", "winsor" or "rank", got {self.mapping!r}')
        if not (0.0 <= self.winsor_q < 0.5):
            raise ValueError(f"winsor_q must be in [0, 0.5), got {self.winsor_q}")
        if self.mapping != "minmax":
            warnings.append(
                f'mapping="{self.mapping}" changes the metric STEER maps, so this arm is '
                "no longer bit-identical to stock STEER even at lam=0. That is the point "
                "of the option, but it means lam=0 is a second baseline, not the stock one."
            )
        if self.mapping == "rank" and not linear:
            warnings.append(
                'mapping="rank" discards the metric\'s zero point, and the exponential '
                "mapping (linear=False) is not shift-invariant, so the attenuation centre "
                "moves to the batch median. Use linear=True, or winsor."
            )
        if self.norm == "none" and self.lam > 0:
            warnings.append(
                'norm="none" with lam>0 mixes two terms of unrelated magnitude; '
                "the visitation term will either dominate or vanish."
            )
        return warnings


# ----------------------------------------------------------------------
# Omega, the local term
# ----------------------------------------------------------------------
def clip_indicator(
    advantages: torch.Tensor,
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    cliprange_low: float = 0.2,
    cliprange_high: float = 0.28,
) -> torch.Tensor:
    """``I_clip`` of the paper's Eq. 6 — 0 where the surrogate is clipped.

    A token whose importance ratio has left the trust region contributes no
    gradient, so its true entropy change is exactly zero.  Theorem 1 carries
    this factor; the released implementation drops it, and the difference is
    not cosmetic under the exponential mapping: a clipped token can otherwise
    be assigned the batch-max ``|Omega|``, and that maximum is the denominator
    every *other* token in the micro-batch is normalised by.  One token that
    the update will not move can therefore weaken the attenuation applied to
    all the tokens it will.

        I_clip = 0  if A > 0 and r > 1 + eps_high
        I_clip = 0  if A < 0 and r < 1 - eps_low
        I_clip = 1  otherwise

    At the first inner epoch ``r == 1`` exactly and this is the all-ones
    tensor; it only bites once the ratio has drifted across mini-batches.
    """
    ratio = torch.exp(torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0))
    clipped = ((advantages > 0) & (ratio > 1.0 + cliprange_high)) | (
        (advantages < 0) & (ratio < 1.0 - cliprange_low)
    )
    return (~clipped).to(advantages.dtype)


def local_omega_signed(
    advantages: torch.Tensor,
    entropys: torch.Tensor,
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    iclip: Optional[torch.Tensor] = None,
    use_ratio: bool = False,
) -> torch.Tensor:
    """STEER's first-order entropy-change estimate, in *signed* convention.

    Reproduces ``compute_token_weights``' arithmetic verbatim::

        x   = clamp(exp(log_prob), 1e-8, 1-1e-8)
        f   = x * (1 - x) * (log(x) + H)
        Omega_signed = -(A / clamp(exp(old_log_prob), 1e-8, 1)) * f

    The leading minus matches verl's ``mode="asymmetric"`` branch, whose sign
    convention is the one the STEER-F derivation assumes: ``Omega > 0`` means
    the update is predicted to *raise* entropy at this position.  The stock
    ``symmetric`` branch computes the negation of this and then takes
    ``abs()``, which is identical (``|-x| == |x|`` exactly in IEEE-754), so
    both stock modes are recovered without changing any number.

    Non-finite entries are zeroed, matching stock STEER, and are zeroed here —
    before the visitation term is added — so a NaN cannot leak through the sum.

    Args:
        advantages / entropys / old_log_prob / log_prob: ``[B, T]``.

    Args (continued):
        iclip: optional ``[B, T]`` clip indicator from :func:`clip_indicator`.
            ``None`` reproduces the released STEER implementation, which omits
            it; passing it reproduces Theorem 1.  The default is ``None``
            because the equivalence test, and the numbers the paper reports,
            are both against the released code.
        use_ratio: use ``r * A`` (Theorem 1) instead of ``A / pi_old`` (the
            released code).  Appendix G Step 3 is explicit that only the
            *sampled* action's logit moves, so the realised entropy change is
            the single inner-product term

                Omega = dH/dz_{s,a} * (z^{k+1}_{s,a} - z^k_{s,a})
                      = -(eta/L) I_clip * r * A * pi (1-pi) (log pi + H),

            with ``r = pi_theta / pi_old``.  The same factor arrives
            independently from Eq. 7: it is an expectation over ``a ~ pi_theta``
            while rollouts are drawn from ``pi_old``, so the importance-
            corrected single-sample estimator carries ``r``.  Both routes
            differ from the released code by exactly one factor of ``pi_theta``.

            It is not a rescaling.  Measured on real rollouts at the first
            inner epoch (``docs/omega_forms.json``, pooled over 20 groups so
            the batch max spans several queries), the released form hands the
            batch maximum -- Eq. 9's denominator for every other token -- to a
            token of probability ``2.0e-6``; the theorem form hands it to one
            of probability ``0.30``, which is where Figures 1-2 say the entropy
            change actually lives.  Downstream: max/p99 6.4 -> 3.7, tokens
            pinned within 1%% of ``w_max`` 83.4%% -> 66.7%%, weight std
            0.0124 -> 0.0180.  Rank correlation between the two is 0.945, so
            they also disagree about *which* tokens to attenuate.

            Default ``False``: the released form is what the lambda=0
            equivalence test and the paper's reported numbers are against.

    Returns:
        ``[B, T]``.
    """
    x = torch.exp(log_prob)
    x = torch.clamp(x, min=_EPS, max=1.0 - _EPS)

    x_one_minus_x = x * (1 - x)
    ln_x_plus_h = torch.log(x) + entropys
    f_x = x_one_minus_x * ln_x_plus_h

    if use_ratio:
        # r = pi_theta / pi_old, clamped the way verl clamps it elsewhere.
        coef = advantages * torch.exp(
            torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0))
    else:
        old_prob = torch.exp(old_log_prob)
        old_prob = torch.clamp(old_prob, min=_EPS, max=1.0)
        coef = advantages / old_prob

    omega = -coef * f_x
    if iclip is not None:
        # Theorem 1's I_clip: a clipped token's entropy change is exactly zero.
        omega = omega * iclip.to(omega.dtype)
    if not torch.isfinite(omega).all():
        print("[STEER-F] Warning: Found non-finite values in Omega")
        omega = torch.where(torch.isfinite(omega), omega, torch.zeros_like(omega))
    return omega


# ----------------------------------------------------------------------
# the visitation term
# ----------------------------------------------------------------------
def delta_logpi_hat(
    w: torch.Tensor,
    pi_sampled: torch.Tensor,
    eta: float = 1.0,
) -> torch.Tensor:
    """First-order shift in the sampled token's log-probability.

    One gradient step on the clipped surrogate moves the sampled token's
    logit by ``eta * w``; the softmax Jacobian turns that into
    ``eta * w * (1 - pi)`` on the log-probability.  Positive means the update
    concentrates mass on the token that was actually taken.
    """
    return eta * w * (1.0 - pi_sampled)


def visit_term(
    w: torch.Tensor,
    pi_sampled: torch.Tensor,
    a_h: torch.Tensor,
    eta: float = 1.0,
    clip_c: float = 1.0,
) -> torch.Tensor:
    """``dlogpi_hat * clip(A_H, -c, c)`` — the trajectory-entropy contribution.

    Sign reading, which matches ``local_omega_signed``:

    * ``dlogpi > 0, A_H > 0`` — the update concentrates mass on a branch whose
      future is *more* diverse than its siblings, so trajectory entropy rises.
    * ``dlogpi > 0, A_H < 0`` — mass moves onto a dead-end branch and the
      diversity carried by the abandoned siblings is lost.  This is precisely
      the collapse channel STEER's local ``Omega`` cannot see.

    ``A_H`` is clipped *before* multiplication so a single wild forecast is
    bounded in influence regardless of how large ``w`` happens to be.
    """
    if clip_c <= 0:
        raise ValueError(f"clip_c must be > 0, got {clip_c}")
    return delta_logpi_hat(w, pi_sampled, eta) * torch.clamp(a_h, -clip_c, clip_c)


# ----------------------------------------------------------------------
# normalisation
# ----------------------------------------------------------------------
def normalize(
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    mode: str = "scale",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Put a tensor on a unit scale, using masked statistics.

    Args:
        x: ``[B, T]``.
        mask: ``[B, T]``; statistics are computed over ``mask > 0`` only.
        mode:
            ``"scale"`` — divide by RMS.  Zero-preserving, so it is safe ahead
            of ``abs()`` and ahead of the exponential mapping.
            ``"z"`` — subtract the mean, divide by the std.  Only safe with
            ``mode="asymmetric", linear=True``.
            ``"none"`` — identity.
        eps: degeneracy threshold.

    Returns:
        ``[B, T]``, zeroed outside the mask.  A degenerate input (all-zero for
        ``"scale"``, zero-variance for ``"z"``) returns all zeros rather than
        dividing by ~0 — for both mappings an all-equal metric already lands
        every token on the same weight, so zeros lose nothing and cannot blow
        up (plan §8.3).
    """
    if mode == "none":
        return x if mask is None else x * mask.to(x.dtype)
    if mode not in ("scale", "z"):
        raise ValueError(f'mode must be "scale", "z" or "none", got {mode!r}')

    if mask is None:
        vals = x.reshape(-1)
    else:
        vals = x[mask.bool()]
    if vals.numel() == 0:
        return torch.zeros_like(x)

    vals = vals.float()
    if mode == "scale":
        denom = torch.sqrt((vals**2).mean())
        if float(denom) < eps:
            return torch.zeros_like(x)
        out = x / denom.to(x.dtype)
    else:
        std = vals.std(unbiased=False)
        if float(std) < eps:
            return torch.zeros_like(x)
        out = (x - vals.mean().to(x.dtype)) / std.to(x.dtype)

    return out if mask is None else out * mask.to(out.dtype)


# ----------------------------------------------------------------------
# metric reshaping — what the mapping actually sees
# ----------------------------------------------------------------------
def _average_ranks(x: torch.Tensor) -> torch.Tensor:
    """Average (tie-corrected) 0-based ranks of a 1-D tensor, as float64."""
    n = x.numel()
    order = torch.argsort(x)
    pos = torch.empty(n, dtype=torch.float64, device=x.device)
    pos[order] = torch.arange(n, dtype=torch.float64, device=x.device)
    vals, inv = torch.unique(x, return_inverse=True)
    tot = torch.zeros(vals.numel(), dtype=torch.float64, device=x.device).scatter_add_(0, inv, pos)
    cnt = torch.zeros(vals.numel(), dtype=torch.float64, device=x.device).scatter_add_(
        0, inv, torch.ones(n, dtype=torch.float64, device=x.device)
    )
    return (tot / cnt)[inv]


def _quantile_sorted(v: torch.Tensor, q: float) -> torch.Tensor:
    """``q``-quantile by index into the sorted values.

    ``torch.quantile`` refuses tensors above ~16M elements; a micro-batch is
    far below that, but sorting keeps this usable on a pooled offline batch too.
    """
    n = v.numel()
    idx = int(round(q * (n - 1)))
    return torch.sort(v).values[max(0, min(n - 1, idx))]


def reshape_metric(
    metric: torch.Tensor,
    response_mask: torch.Tensor,
    mapping: str = "minmax",
    winsor_q: float = 0.01,
    mode: str = "symmetric",
) -> torch.Tensor:
    """Transform the metric *before* STEER's min-max / exponential mapping.

    STEER rescales the metric onto ``[token_weight_min, token_weight_max]``
    using the micro-batch's own ``min`` and ``max``.  ``Omega`` carries a
    division by ``pi_old`` floored at ``1e-8``, so those two extremes are set
    by outliers: measured on real rollouts, ``max|Omega| / median|Omega|`` is
    3.4e4 and ``max / p99`` is 6.4 (``docs/omega_forms.json``).  The band is
    therefore spent on one or two tokens and the bulk lands on a point --
    ``tw_std`` 0.001 on a 0.1-wide band in the Phase-2 runs, i.e. the ordering
    information STEER computes never reaches the loss.

    Three options:

    ``"minmax"``
        Identity.  The stock behaviour, and the default, because it is what
        the ``lambda = 0`` equivalence test compares against.
    ``"winsor"``
        Clamp the metric to its ``[winsor_q, 1 - winsor_q]`` quantiles first.
        Smallest possible change: the shape of the bulk is untouched, only the
        tail's grip on the denominator is cut.
    ``"rank"``
        Replace the metric by its within-micro-batch normalised rank.  Band
        occupancy becomes uniform *by construction* and no tail can influence
        it.  The cost is that all magnitude information is discarded: the gap
        between the 1st and 2nd largest ``|Omega|`` is treated the same as the
        gap between two adjacent bulk tokens.

    Ranks are averaged over ties, so the point mass at ``Omega == 0`` (masked
    or clipped positions) maps to a single shared value rather than an
    arbitrary order.

    Args:
        metric: ``[B, T]``, already ``abs()``-ed when ``mode="symmetric"``.
        response_mask: ``[B, T]``; statistics come from ``mask > 0`` only.
        mapping: ``"minmax"``, ``"winsor"`` or ``"rank"``.
        winsor_q: tail fraction clamped at each end when ``mapping="winsor"``.
        mode: ``"symmetric"`` puts ranks on ``[0, 1]`` (the metric is
            non-negative and 0 is a meaningful floor); ``"asymmetric"`` puts
            them on ``[-1, 1]`` so the signed metric keeps a centre.  Under
            ``linear=True`` this choice is invisible -- the min-max rescale is
            affine-invariant -- and it only matters for the exponential
            mapping, which is not shift-invariant.

    Returns:
        ``[B, T]``, zero outside the mask.
    """
    if mapping == "minmax":
        return metric
    if mapping not in ("winsor", "rank"):
        raise ValueError(f'mapping must be "minmax", "winsor" or "rank", got {mapping!r}')
    if mode not in ("symmetric", "asymmetric"):
        raise ValueError(f'mode must be "symmetric" or "asymmetric", got {mode!r}')

    valid = response_mask.bool()
    vals = metric[valid]
    if vals.numel() == 0:
        return metric

    out = torch.zeros_like(metric)
    if mapping == "winsor":
        if not (0.0 <= winsor_q < 0.5):
            raise ValueError(f"winsor_q must be in [0, 0.5), got {winsor_q}")
        lo = _quantile_sorted(vals, winsor_q)
        hi = _quantile_sorted(vals, 1.0 - winsor_q)
        out[valid] = vals.clamp(lo, hi)
        return out

    n = vals.numel()
    if n == 1:
        return out
    r = (_average_ranks(vals) / (n - 1)).to(metric.dtype)
    out[valid] = r if mode == "symmetric" else (2.0 * r - 1.0)
    return out


# ----------------------------------------------------------------------
# the combination
# ----------------------------------------------------------------------
def compute_omega_tilde(
    omega_local: torch.Tensor,
    w: torch.Tensor,
    pi_sampled: torch.Tensor,
    a_h: torch.Tensor,
    response_mask: torch.Tensor,
    lam: float = 0.0,
    eta: float = 1.0,
    clip_c: float = 1.0,
    norm: str = "scale",
) -> tuple[torch.Tensor, dict]:
    """Combine the local and visitation terms.

    Args:
        omega_local: ``[B, T]`` signed ``Omega`` from :func:`local_omega_signed`.
        w: ``[B, T]`` ``clip(ratio) * advantage``.
        pi_sampled: ``[B, T]`` current probability of the sampled token.
        a_h: ``[B, T]`` entropy advantage, pre-clip.
        response_mask: ``[B, T]``.
        lam, eta, clip_c, norm: see :class:`SteerFConfig`.

    Returns:
        ``(omega_tilde, stats)``.  ``stats`` reports the relative magnitude of
        the two terms, which plan §4.2 asks to be logged, plus the fraction of
        ``A_H`` values that hit the clip — a saturated clip means ``clip_c`` is
        too tight to discriminate branches.

    Note:
        With ``lam == 0`` this returns ``omega_local`` itself (same object
        contents, no arithmetic applied), which is what makes the
        equivalence test exact rather than approximate.
    """
    if lam == 0.0:
        return omega_local, {
            "steerf/lam": 0.0,
            "steerf/visit_rel_mag": 0.0,
            "steerf/a_h_clip_frac": 0.0,
        }
    if lam < 0:
        raise ValueError(f"lam must be >= 0, got {lam}")

    vt = visit_term(w, pi_sampled, a_h, eta=eta, clip_c=clip_c)

    local_n = normalize(omega_local, response_mask, mode=norm)
    visit_n = normalize(vt, response_mask, mode=norm)
    omega_tilde = local_n + lam * visit_n

    valid = response_mask.bool()
    if valid.any():
        local_mag = float(local_n[valid].abs().mean())
        visit_mag = float((lam * visit_n)[valid].abs().mean())
        clip_frac = float((a_h[valid].abs() >= clip_c).float().mean())
    else:
        local_mag = visit_mag = clip_frac = 0.0

    stats = {
        "steerf/lam": float(lam),
        "steerf/local_term_mag": local_mag,
        "steerf/visit_term_mag": visit_mag,
        "steerf/visit_rel_mag": visit_mag / (local_mag + _EPS),
        "steerf/a_h_clip_frac": clip_frac,
    }
    return omega_tilde, stats


def branch_weight_correction(
    token_weights: torch.Tensor,
    visit: torch.Tensor,
    response_mask: torch.Tensor,
    lam: float,
    token_weight_min: float,
    token_weight_max: float,
    mode: str = "signed",
) -> tuple:
    """Adjust already-mapped weights at branch points.  ``apply="weight"``.

    The additive metric form (``apply="metric"``) puts the branch signal into
    ``Omega`` and lets STEER's min-max map it.  Measured on real rollouts, that
    map destroys the signal: ``Omega`` is heavy-tailed -- it carries a division
    by ``pi_old``, floored at 1e-8 -- so its extremes set the scale and the bulk
    of tokens land within ~5% of the weight range of each other.  A branch term
    that is nonzero at ~1% of positions comes out the far side as a mean
    difference of 0.0007 on a 0.1-wide band, i.e. nothing, while still costing
    the attenuating range 8-34% of its occupancy because it shifted the metric
    distribution the min-max is computed over.  Four metric-level formulations
    were compared offline and all four failed the same way
    (``docs/weight_forms.json``).

    Applying the correction *after* the mapping removes that competition
    entirely: STEER's weights are computed exactly as upstream computes them,
    and the branch adjustment is then a bounded nudge in weight space, whose
    size is set by ``lam`` alone rather than by the shape of ``Omega``'s tail.

        w = clamp( w_steer + lam * (w_max - w_min) * tanh(visit / rms(visit)),
                   w_min, w_max )

    ``tanh`` bounds a single token's adjustment to ``lam * (w_max - w_min)``, so
    ``lam = 0.5`` moves a branch token at most half the band.  ``rms`` is taken
    over the *support* -- positions where the branch score is actually defined
    -- because ``A_H`` is exactly zero wherever a rollout has no surviving
    siblings, and normalising over all positions would rescale the correction by
    the sparsity rather than by its magnitude.

    Two modes, and the difference is not cosmetic.

    ``mode="signed"`` grades the adjustment by the branch score, so siblings at
    one branch point are pushed apart: the one whose future looks richer than
    its siblings' average is attenuated less, the dead end more.  Because
    ``A_H`` is a deviation from that average it is zero-centred *by
    construction*, so this can never shift the branch tokens' mean weight -- it
    only spreads them.  Measured offline: mean branch-vs-other difference
    +0.002 on a 0.1 band, while the weight standard deviation rose 0.0034 ->
    0.0043.  Discrimination, not protection.

    ``mode="uniform"`` attenuates every position that has surviving siblings by
    the same amount, ignoring the sign and magnitude of the score entirely.
    This is the "branch points are where diversity lives, so slow all of them
    down" reading, and it is a genuinely weaker hypothesis: it needs no
    forecast, no MTP heads and no Phase 1, because "does this rollout still
    have siblings here" is answered by the rollout group alone.  If it matches
    ``signed``, the forecast is not carrying the effect.

    Args:
        token_weights: ``[B, T]`` output of STEER's mapping.
        visit: ``[B, T]`` ``dlogpi_hat * clip(A_H)``, the same quantity the
            metric form adds.  Only its support is used when ``mode="uniform"``.
        response_mask: ``[B, T]``.
        lam: correction size as a fraction of the weight band.
        token_weight_min / token_weight_max: the band to stay inside.
        mode: ``"signed"`` or ``"uniform"``.

    Returns:
        ``(weights, stats)``.
    """
    if mode not in ("signed", "uniform"):
        raise ValueError(f'mode must be "signed" or "uniform", got {mode!r}')
    valid = response_mask.bool()
    support = valid & (visit != 0)
    stats = {
        "steerf/branch_corr_frac": float(support.float().sum() / max(1, int(valid.sum()))),
    }
    if lam == 0.0 or not support.any():
        stats["steerf/branch_corr_mean_abs"] = 0.0
        return token_weights, stats

    band = token_weight_max - token_weight_min
    delta = torch.zeros_like(token_weights)

    if mode == "uniform":
        # Negative: lower weight is more attenuation, which is what "protect
        # this position" means under the asymmetric mapping.
        delta[support] = -lam * band
    else:
        rms = visit[support].float().pow(2).mean().sqrt()
        if not torch.isfinite(rms) or rms < _EPS:
            stats["steerf/branch_corr_mean_abs"] = 0.0
            return token_weights, stats
        delta[support] = lam * band * torch.tanh(visit[support].float() / rms)

    out = torch.clamp(token_weights + delta, token_weight_min, token_weight_max)
    out = out * response_mask.float()
    stats["steerf/branch_corr_mean_abs"] = float(delta[support].abs().mean())
    stats["steerf/branch_corr_max_abs"] = float(delta[support].abs().max())
    return out, stats


# ----------------------------------------------------------------------
# drop-in replacement for verl's compute_token_weights
# ----------------------------------------------------------------------
def compute_token_weights_steerf(
    advantages: torch.Tensor,
    entropys: torch.Tensor,
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    token_weight_min: float = 0.8,
    token_weight_max: float = 1.2,
    linear: bool = True,
    mode: str = "symmetric",
    # ---- STEER-F additions; all-default == stock STEER ----
    a_h: Optional[torch.Tensor] = None,
    cliprange_low: float = 0.2,
    cliprange_high: float = 0.28,
    lam: float = 0.0,
    eta: float = 1.0,
    clip_c: float = 1.0,
    norm: str = "scale",
    apply: str = "metric",
    mapping: str = "minmax",
    winsor_q: float = 0.01,
    use_iclip: bool = False,
    use_ratio: bool = False,
    return_stats: bool = False,
):
    """Token weights from ``Omega_tilde``, mapping logic unchanged from STEER.

    Everything after the metric is computed — the min-max / exponential
    mapping, the clamps, the mask handling — is a verbatim copy of
    ``verl.trainer.ppo.core_algos.compute_token_weights``.  Only the metric
    fed into it changes, which is exactly the substitution plan §4.1 asks for.

    Args:
        advantages, entropys, old_log_prob, log_prob, response_mask,
        token_weight_min, token_weight_max, linear, mode:
            identical to stock STEER.
        a_h: ``[B, T]`` entropy advantage.  ``None`` (or ``lam == 0``) means
            stock STEER.
        cliprange_low / cliprange_high: used to rebuild ``w = clip(ratio) * A``
            with the same clipping the policy loss will apply.
        lam, eta, clip_c, norm: see :class:`SteerFConfig`.
        mapping / winsor_q: reshape the metric before STEER's band mapping —
            see :func:`reshape_metric`.  ``"minmax"`` is stock and is what the
            ``lambda = 0`` equivalence test compares against.
        return_stats: also return the diagnostics dict.

    Returns:
        ``[B, T]`` token weights, or ``(weights, stats)`` if ``return_stats``.
    """
    with torch.no_grad():
        # use_iclip=False is the released STEER implementation and is what the
        # lambda=0 equivalence test and the paper's reported numbers are against;
        # True is Theorem 1 as written. See clip_indicator's docstring.
        iclip = (
            clip_indicator(advantages, old_log_prob, log_prob,
                           cliprange_low=cliprange_low, cliprange_high=cliprange_high)
            if use_iclip
            else None
        )
        omega = local_omega_signed(advantages, entropys, old_log_prob, log_prob,
                                   iclip=iclip, use_ratio=use_ratio)

        if apply not in ("metric", "weight", "branch"):
            raise ValueError(
                f'apply must be "metric", "weight" or "branch", got {apply!r}')

        steerf_on = lam != 0.0 and a_h is not None
        pending_visit = None
        if steerf_on:
            negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
            ratio = torch.exp(negative_approx_kl)
            w = torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high) * advantages
            pi_sampled = torch.clamp(torch.exp(log_prob), min=_EPS, max=1.0 - _EPS)
            if apply == "metric":  # noqa: SIM108
                metric, stats = compute_omega_tilde(
                    omega_local=omega,
                    w=w,
                    pi_sampled=pi_sampled,
                    a_h=a_h,
                    response_mask=response_mask,
                    lam=lam,
                    eta=eta,
                    clip_c=clip_c,
                    norm=norm,
                )
            else:
                # apply="weight": the metric stays exactly stock STEER's, so the
                # min-max below sees the distribution it was designed for, and
                # the branch signal is applied to the mapped weights instead.
                metric = omega
                pending_visit = visit_term(w, pi_sampled, a_h, eta=eta, clip_c=clip_c)
                stats = {"steerf/lam": float(lam),
                         "steerf/apply_weight": 1.0,
                         "steerf/apply_uniform": float(apply == "branch")}
        else:
            metric, stats = omega, {"steerf/lam": 0.0}

        if mode == "symmetric":
            metric = torch.abs(metric)
        elif mode != "asymmetric":
            raise ValueError(f'mode must be "symmetric" or "asymmetric", got {mode!r}')

        # The one line that changes what the stock mapping below is looking at.
        # mapping="minmax" returns `metric` itself, so the stock path is bit-identical.
        metric = reshape_metric(metric, response_mask, mapping=mapping,
                                winsor_q=winsor_q, mode=mode)
        stats["steerf/mapping_rank"] = float(mapping == "rank")
        stats["steerf/mapping_winsor"] = float(mapping == "winsor")

        # ---- below: unchanged from stock STEER ----
        valid_metric = metric[response_mask.bool()]
        if valid_metric.numel() == 0:
            empty = torch.zeros_like(response_mask, dtype=torch.float)
            return (empty, stats) if return_stats else empty

        metric_min = valid_metric.min()
        metric_max = valid_metric.max()

        token_weights = torch.zeros_like(metric, dtype=torch.float)
        valid_mask = response_mask.bool()
        if valid_mask.any():
            valid_metric = metric[valid_mask]

            if linear:
                if metric_max > metric_min:
                    scale_factor = (token_weight_max - token_weight_min) / (metric_max - metric_min)
                    if mode == "asymmetric":
                        valid_weights = token_weight_min + (valid_metric - metric_min) * scale_factor
                    else:
                        valid_weights = token_weight_max - (valid_metric - metric_min) * scale_factor
                else:
                    valid_weights = torch.full_like(
                        valid_metric, (token_weight_min + token_weight_max) / 2
                    )
                valid_weights = torch.clamp(valid_weights, min=token_weight_min, max=token_weight_max)
            else:
                if mode == "asymmetric":
                    abs_max = torch.maximum(
                        valid_metric.abs().max(),
                        torch.tensor(0.02, dtype=metric.dtype, device=metric.device),
                    )
                    k = -torch.log(
                        torch.tensor(token_weight_min, dtype=metric.dtype, device=metric.device)
                    ) / abs_max
                    valid_weights = torch.exp(k * valid_metric)
                    valid_weights = torch.clamp(
                        valid_weights, min=token_weight_min, max=token_weight_max
                    )
                else:
                    k = -torch.log(
                        torch.tensor(token_weight_min, dtype=metric.dtype, device=metric.device)
                    ) / torch.maximum(
                        metric_max,
                        torch.tensor(0.02, dtype=metric.dtype, device=metric.device),
                    )
                    valid_weights = torch.exp(-k * valid_metric)
                    valid_weights = torch.clamp(valid_weights, min=token_weight_min, max=1.0)

            token_weights[valid_mask] = valid_weights

        token_weights = token_weights * response_mask.float()

        if pending_visit is not None:
            token_weights, corr_stats = branch_weight_correction(
                token_weights,
                pending_visit,
                response_mask,
                lam=lam,
                token_weight_min=token_weight_min,
                token_weight_max=token_weight_max,
                mode="uniform" if apply == "branch" else "signed",
            )
            stats.update(corr_stats)

        # The tw_* metrics above cover every response token, but at this stage
        # ~2/3 of them sit in GRPO groups whose 8 rollouts all scored the same,
        # so their advantage -- and therefore Omega -- is exactly 0.  An
        # all-tied micro-batch collapses to a single point (1.000 under
        # "minmax" because metric_max hits the 0.02 floor, 0.700 under "rank"
        # because every tied position takes average rank 0.5), and that point
        # dominates the occupancy numbers while contributing nothing to the
        # update: pg_losses * token_weights is 0 wherever A == 0, and agg_loss
        # divides by response_mask rather than by the weights.  Reading tw_*
        # at face value therefore says "the reweighting is dead" for minmax and
        # "everything is maximally attenuated" for rank, and both are artefacts.
        # twg_* is the same occupancy restricted to the tokens that can
        # actually move the update.
        if return_stats:
            valid_n = response_mask.float().sum()
            grad_mask = response_mask.float() * (advantages != 0).float()
            grad_n = grad_mask.sum()
            stats["steerf/adv_zero_frac"] = (
                float(1.0 - grad_n / valid_n) if valid_n > 0 else float("nan")
            )
            # Omitted, not NaN, when a micro-batch carries no gradient at all:
            # ~2/3 of them do not, and a NaN would poison the mean verl takes
            # over micro-batches.
            if grad_n > 0:
                from steer_f.monitors import token_weight_distribution

                for key, val in token_weight_distribution(
                    token_weights, grad_mask, token_weight_min, token_weight_max
                ).items():
                    stats[key.replace("steerf/tw_", "steerf/twg_")] = val

        return (token_weights, stats) if return_stats else token_weights
