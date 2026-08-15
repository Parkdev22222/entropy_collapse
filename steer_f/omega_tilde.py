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
``patches/core_algos_steerf.patch``).  That keeps the vendored verl tree
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
    "local_omega_signed",
    "delta_logpi_hat",
    "visit_term",
    "normalize",
    "compute_omega_tilde",
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
        if self.norm == "none" and self.lam > 0:
            warnings.append(
                'norm="none" with lam>0 mixes two terms of unrelated magnitude; '
                "the visitation term will either dominate or vanish."
            )
        return warnings


# ----------------------------------------------------------------------
# Omega, the local term
# ----------------------------------------------------------------------
def local_omega_signed(
    advantages: torch.Tensor,
    entropys: torch.Tensor,
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
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

    Returns:
        ``[B, T]``.
    """
    x = torch.exp(log_prob)
    x = torch.clamp(x, min=_EPS, max=1.0 - _EPS)

    x_one_minus_x = x * (1 - x)
    ln_x_plus_h = torch.log(x) + entropys
    f_x = x_one_minus_x * ln_x_plus_h

    old_prob = torch.exp(old_log_prob)
    old_prob = torch.clamp(old_prob, min=_EPS, max=1.0)
    advantage_over_old_prob = advantages / old_prob

    omega = -advantage_over_old_prob * f_x
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
        return_stats: also return the diagnostics dict.

    Returns:
        ``[B, T]`` token weights, or ``(weights, stats)`` if ``return_stats``.
    """
    with torch.no_grad():
        omega = local_omega_signed(advantages, entropys, old_log_prob, log_prob)

        if lam != 0.0 and a_h is not None:
            negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
            ratio = torch.exp(negative_approx_kl)
            w = torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high) * advantages
            pi_sampled = torch.clamp(torch.exp(log_prob), min=_EPS, max=1.0 - _EPS)
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
            metric, stats = omega, {"steerf/lam": 0.0}

        if mode == "symmetric":
            metric = torch.abs(metric)
        elif mode != "asymmetric":
            raise ValueError(f'mode must be "symmetric" or "asymmetric", got {mode!r}')

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
        return (token_weights, stats) if return_stats else token_weights
