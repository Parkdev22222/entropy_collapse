# Copyright 2026 STEER-F authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Gate-G1 statistics: within-problem correlation and branch-recall testing.

Split out of ``scripts/phase1_validate.py`` deliberately.  A gate decision
should not depend on arithmetic that only ever runs inside a GPU script — the
pass/fail rule is the experiment's contract with itself, so it lives in a
tested module and the script only feeds it data.

The primary metric follows plan §3.3: Spearman rho computed *within* each
problem and then averaged through the Fisher z transform.  Pooling across
problems instead would let between-problem differences in difficulty inflate
the correlation, which is the classic way a forecaster looks predictive
without being useful inside the decision it actually informs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

__all__ = [
    "spearman",
    "fisher_z_mean",
    "within_problem_spearman",
    "binomial_test_greater",
    "GateResult",
    "evaluate_gate_g1",
    "select_kappa_gamma",
]


def _rankdata(values: Sequence[float]) -> list[float]:
    """Average-tie ranks, matching ``scipy.stats.rankdata``."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation.  Returns ``nan`` when either side is constant."""
    if len(x) != len(y):
        raise ValueError(f"length mismatch: {len(x)} vs {len(y)}")
    n = len(x)
    if n < 3:
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if vx < 1e-12 or vy < 1e-12:
        return float("nan")
    return cov / (vx * vy)


def fisher_z_mean(rhos: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
    """Average correlations in Fisher z space, then map back.

    Correlations are not additive; averaging them directly under-weights strong
    ones.  ``nan`` entries (constant-input problems) are dropped rather than
    propagated, and rho exactly +-1 is nudged inside the domain so a single
    degenerate problem cannot send the mean to infinity.
    """
    pairs = [
        (r, 1.0 if weights is None else float(weights[i]))
        for i, r in enumerate(rhos)
        if r == r  # drop nan
    ]
    if not pairs:
        return float("nan")
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return float("nan")
    acc = 0.0
    for r, w in pairs:
        r = max(-0.999999, min(0.999999, float(r)))
        acc += w * math.atanh(r)
    return math.tanh(acc / total_w)


def within_problem_spearman(
    forecasts: Sequence[float],
    truths: Sequence[float],
    problem_ids: Sequence,
    min_per_problem: int = 5,
) -> dict:
    """Per-problem Spearman, Fisher-z averaged (plan §3.3 primary metric).

    Args:
        forecasts / truths: paired observations, one per prefix.
        problem_ids: which problem each observation came from.
        min_per_problem: problems with fewer prefixes are skipped — a rho over
            3 points is noise, and including it widens the CI for nothing.

    Returns:
        ``rho_mean`` (Fisher-z average), ``rho_pooled`` (the naive version, for
        contrast), the ``per_problem`` list, and the counts.
    """
    if not (len(forecasts) == len(truths) == len(problem_ids)):
        raise ValueError("forecasts, truths and problem_ids must have equal length")

    buckets: dict = {}
    for f, t, p in zip(forecasts, truths, problem_ids):
        buckets.setdefault(p, ([], []))
        buckets[p][0].append(float(f))
        buckets[p][1].append(float(t))

    per_problem, used = [], []
    for pid, (fs, ts) in buckets.items():
        if len(fs) < min_per_problem:
            continue
        r = spearman(fs, ts)
        per_problem.append({"problem": pid, "n": len(fs), "rho": r})
        if r == r:
            used.append(pid)

    return {
        "rho_mean": fisher_z_mean([p["rho"] for p in per_problem],
                                  [p["n"] for p in per_problem]),
        "rho_pooled": spearman(list(forecasts), list(truths)),
        "per_problem": per_problem,
        "n_problems": len(per_problem),
        "n_problems_used": len(used),
        "n_observations": len(forecasts),
    }


def binomial_test_greater(successes: int, trials: int, p_null: float) -> float:
    """One-sided binomial p-value for ``successes > trials * p_null``.

    Exact, so it stays valid at the small ``n_branch`` counts Phase 1 produces,
    where a normal approximation would be optimistic.
    """
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError(f"invalid counts: {successes}/{trials}")
    if not (0.0 <= p_null <= 1.0):
        raise ValueError(f"p_null must be in [0, 1], got {p_null}")
    if trials == 0:
        return float("nan")
    if p_null == 0.0:
        return 0.0 if successes > 0 else 1.0
    if p_null == 1.0:
        return 1.0

    total = 0.0
    for k in range(successes, trials + 1):
        log_c = (
            math.lgamma(trials + 1) - math.lgamma(k + 1) - math.lgamma(trials - k + 1)
            + k * math.log(p_null) + (trials - k) * math.log1p(-p_null)
        )
        total += math.exp(log_c)
    return min(1.0, total)


@dataclass
class GateResult:
    """Outcome of a gate check, with the numbers that decided it."""

    passed: bool
    criteria: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"GATE: {'PASS' if self.passed else 'FAIL'}"]
        for name, (ok, detail) in self.criteria.items():
            lines.append(f"  [{'x' if ok else ' '}] {name}: {detail}")
        lines.extend(f"  ! {n}" for n in self.notes)
        return "\n".join(lines)


def evaluate_gate_g1(
    rho_mean: float,
    recall: float,
    n_branch: int,
    n_hit: int,
    selection_rate: float,
    rho_threshold: float = 0.2,
    alpha: float = 0.05,
) -> GateResult:
    """Gate G1 exactly as plan §3 states it.

    Two criteria, both required:

    1. within-problem ``rho(H_togo, GT_future_entropy) >= 0.2``
       (ExTra's MTE was judged useful at 0.229).
    2. branch recall@10% significantly above chance, ``p < 0.05``, where chance
       is the *actual* selection rate rather than a nominal 0.1 — rounding and
       ties make the realised top-decile slightly larger or smaller, and
       testing against the nominal value would bias the verdict.
    """
    rho_ok = (rho_mean == rho_mean) and rho_mean >= rho_threshold
    p_value = binomial_test_greater(n_hit, n_branch, selection_rate)
    recall_ok = (p_value == p_value) and p_value < alpha and recall > selection_rate

    notes = []
    if n_branch < 30:
        notes.append(
            f"only {n_branch} true branch points — the recall test has little power; "
            "sample more prefixes before treating a FAIL as informative"
        )
    if rho_mean != rho_mean:
        notes.append("rho_mean is nan: every problem was constant or below min_per_problem")

    return GateResult(
        passed=bool(rho_ok and recall_ok),
        criteria={
            "within-problem rho >= %.2f" % rho_threshold: (
                rho_ok, f"rho_mean={rho_mean:.4f}"
            ),
            "branch recall@10%% above chance (p<%.2f)" % alpha: (
                recall_ok,
                f"recall={recall:.4f} vs chance={selection_rate:.4f}, "
                f"p={p_value:.4g} ({n_hit}/{n_branch})",
            ),
        },
        notes=notes,
    )


def select_kappa_gamma(
    grid: Sequence[dict],
    elbow_tolerance: float = 0.01,
    min_kappa: int = 2,
) -> dict:
    """Pick ``(kappa, gamma_H)`` from a grid of correlation results.

    Plan §3.4 asks for "the elbow".  Made concrete: among the ``gamma_H`` whose
    best rho is within ``elbow_tolerance`` of the global best, take the
    *smallest* kappa that already reaches within ``elbow_tolerance`` of that
    gamma's best.  Preferring the shortest horizon that is statistically tied
    with the longest keeps forecast cost down and reduces exposure to the
    distant-head over-estimation bias, instead of chasing a rho difference the
    sample size cannot resolve.

    ``min_kappa`` defaults to 2 because ``kappa = 1`` is degenerate rather than
    merely short.  Head 0 maps the hidden state at position ``t`` to the
    distribution of ``y_{t+1}``, which is what the policy's own unembedding
    computes; with ``tie_unembedding`` and ``zero_init_output`` the head is
    initialised as a copy of exactly that predictor.  So ``H_togo(kappa=1)`` is
    approximately the policy's own next-token entropy, and the measured
    counterpart at offset +1 *is* that entropy — the correlation is an identity,
    not a forecast.  Measured on Qwen2.5-1.5B it came out at rho = 0.997 for
    every gamma (gamma cancels from a rank correlation when kappa = 1), against
    0.71-0.81 for kappa >= 2, so an unconstrained elbow search always lands
    there.  Selecting it would reduce ``A_H`` to a deviation of *local* entropy
    from the sibling mean, i.e. exactly the myopia STEER-F exists to correct.

    Args:
        grid: dicts with keys ``kappa``, ``gamma_h``, ``rho_mean``.
        elbow_tolerance: how much rho may be given up for a shorter horizon.
        min_kappa: smallest horizon eligible for selection.  Pass 1 only to
            reproduce the degenerate behaviour deliberately.

    Returns:
        The winning grid entry, annotated with ``rho_best`` and ``rho_giveup``.
    """
    usable = [g for g in grid if g.get("rho_mean") == g.get("rho_mean")]
    if min_kappa > 1:
        usable = [g for g in usable if g["kappa"] >= min_kappa]
    if not usable:
        raise ValueError("grid contains no finite rho_mean values")

    rho_best = max(g["rho_mean"] for g in usable)
    contenders = [g for g in usable if g["rho_mean"] >= rho_best - elbow_tolerance]
    best_gamma = max(set(g["gamma_h"] for g in contenders),
                     key=lambda gh: max(g["rho_mean"] for g in contenders if g["gamma_h"] == gh))

    same_gamma = [g for g in usable if g["gamma_h"] == best_gamma]
    gamma_best = max(g["rho_mean"] for g in same_gamma)
    eligible = [g for g in same_gamma if g["rho_mean"] >= gamma_best - elbow_tolerance]
    winner = min(eligible, key=lambda g: g["kappa"])

    return {**winner, "rho_best": rho_best, "rho_giveup": rho_best - winner["rho_mean"]}
