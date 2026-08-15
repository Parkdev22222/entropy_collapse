"""Gate-G1 statistics.  Spearman/binomial are cross-checked against scipy."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from steer_f.validation import (  # noqa: E402
    binomial_test_greater,
    evaluate_gate_g1,
    fisher_z_mean,
    select_kappa_gamma,
    spearman,
    within_problem_spearman,
)

scipy_stats = pytest.importorskip("scipy.stats", reason="cross-check needs scipy")


# ----------------------------------------------------------------------
# spearman
# ----------------------------------------------------------------------
def test_spearman_perfect_monotone():
    assert spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_is_rank_based_not_linear():
    """Monotone but wildly non-linear must still score 1.0."""
    assert spearman([1, 2, 3, 4], [1, 10, 1000, 1e6]) == pytest.approx(1.0)


def test_spearman_matches_scipy():
    import random

    rng = random.Random(0)
    for _ in range(20):
        n = rng.randint(5, 40)
        x = [rng.gauss(0, 1) for _ in range(n)]
        y = [xi * rng.gauss(1, 1) + rng.gauss(0, 1) for xi in x]
        assert spearman(x, y) == pytest.approx(float(scipy_stats.spearmanr(x, y).statistic), abs=1e-9)


def test_spearman_handles_ties_like_scipy():
    x = [1, 1, 2, 2, 3, 3]
    y = [5, 5, 1, 9, 2, 2]
    assert spearman(x, y) == pytest.approx(float(scipy_stats.spearmanr(x, y).statistic), abs=1e-9)


def test_spearman_constant_input_is_nan():
    assert math.isnan(spearman([1, 1, 1, 1], [1, 2, 3, 4]))


def test_spearman_too_few_points_is_nan():
    assert math.isnan(spearman([1, 2], [3, 4]))


def test_spearman_rejects_length_mismatch():
    with pytest.raises(ValueError):
        spearman([1, 2, 3], [1, 2])


# ----------------------------------------------------------------------
# fisher z
# ----------------------------------------------------------------------
def test_fisher_z_mean_of_identical_values_is_that_value():
    assert fisher_z_mean([0.3, 0.3, 0.3]) == pytest.approx(0.3)


def test_fisher_z_mean_exceeds_the_arithmetic_mean():
    """The reason the plan asks for Fisher z rather than a plain average."""
    rhos = [0.1, 0.9]
    assert fisher_z_mean(rhos) > sum(rhos) / len(rhos)


def test_fisher_z_drops_nan_problems():
    assert fisher_z_mean([0.5, float("nan"), 0.5]) == pytest.approx(0.5)


def test_fisher_z_all_nan_is_nan():
    assert math.isnan(fisher_z_mean([float("nan"), float("nan")]))


def test_fisher_z_survives_a_perfect_correlation():
    """rho=1 must not send the mean to infinity."""
    out = fisher_z_mean([1.0, 0.0])
    assert math.isfinite(out) and 0.0 < out < 1.0


def test_fisher_z_weighting_favours_larger_problems():
    unweighted = fisher_z_mean([0.1, 0.9])
    weighted = fisher_z_mean([0.1, 0.9], weights=[100, 1])
    assert weighted < unweighted


def test_fisher_z_empty_is_nan():
    assert math.isnan(fisher_z_mean([]))


# ----------------------------------------------------------------------
# within-problem aggregation
# ----------------------------------------------------------------------
def test_within_problem_separates_a_simpson_reversal():
    """The exact confound the within-problem design exists to remove.

    Inside each problem the relationship is negative, but the problems sit at
    different offsets so the pooled correlation comes out positive.  Pooling
    would report a useful forecaster where there is none.
    """
    forecasts = [1, 2, 3, 11, 12, 13, 21, 22, 23]
    truths = [3, 2, 1, 13, 12, 11, 23, 22, 21]
    pids = ["a"] * 3 + ["b"] * 3 + ["c"] * 3
    out = within_problem_spearman(forecasts, truths, pids, min_per_problem=3)
    assert out["rho_mean"] == pytest.approx(-1.0, abs=1e-3)
    assert out["rho_pooled"] > 0.7


def test_within_problem_skips_undersized_problems():
    out = within_problem_spearman(
        [1, 2, 3, 4, 5, 6, 1, 2], [1, 2, 3, 4, 5, 6, 1, 2],
        ["a"] * 6 + ["b"] * 2, min_per_problem=5,
    )
    assert out["n_problems"] == 1


def test_within_problem_reports_counts():
    out = within_problem_spearman(list(range(10)), list(range(10)), ["a"] * 10)
    assert out["n_observations"] == 10 and out["n_problems"] == 1


def test_within_problem_rejects_length_mismatch():
    with pytest.raises(ValueError):
        within_problem_spearman([1, 2], [1, 2, 3], ["a", "a", "a"])


# ----------------------------------------------------------------------
# binomial test
# ----------------------------------------------------------------------
def test_binomial_matches_scipy():
    for successes, trials, p in [(8, 10, 0.1), (15, 100, 0.1), (3, 5, 0.5), (0, 10, 0.3)]:
        expected = float(scipy_stats.binomtest(successes, trials, p, alternative="greater").pvalue)
        assert binomial_test_greater(successes, trials, p) == pytest.approx(expected, rel=1e-9)


def test_binomial_at_chance_is_not_significant():
    assert binomial_test_greater(10, 100, 0.1) > 0.05


def test_binomial_far_above_chance_is_significant():
    assert binomial_test_greater(40, 100, 0.1) < 1e-9


def test_binomial_zero_trials_is_nan():
    assert math.isnan(binomial_test_greater(0, 0, 0.1))


def test_binomial_rejects_impossible_counts():
    with pytest.raises(ValueError):
        binomial_test_greater(11, 10, 0.1)


# ----------------------------------------------------------------------
# gate G1
# ----------------------------------------------------------------------
def test_gate_passes_when_both_criteria_hold():
    res = evaluate_gate_g1(rho_mean=0.25, recall=0.4, n_branch=200, n_hit=80, selection_rate=0.1)
    assert res.passed
    assert "PASS" in res.summary()


def test_gate_fails_on_weak_correlation():
    res = evaluate_gate_g1(rho_mean=0.15, recall=0.4, n_branch=200, n_hit=80, selection_rate=0.1)
    assert not res.passed


def test_gate_fails_on_chance_level_recall():
    res = evaluate_gate_g1(rho_mean=0.5, recall=0.1, n_branch=200, n_hit=20, selection_rate=0.1)
    assert not res.passed


def test_gate_fails_on_nan_correlation():
    res = evaluate_gate_g1(rho_mean=float("nan"), recall=0.9, n_branch=100, n_hit=90,
                           selection_rate=0.1)
    assert not res.passed
    assert any("nan" in n for n in res.notes)


def test_gate_warns_when_underpowered():
    """A FAIL on 12 branch points is not evidence of absence — say so."""
    res = evaluate_gate_g1(rho_mean=0.3, recall=0.3, n_branch=12, n_hit=4, selection_rate=0.1)
    assert any("power" in n for n in res.notes)


def test_gate_tests_against_the_realised_selection_rate():
    """Chance is the realised top-decile size, not the nominal 0.1."""
    tight = evaluate_gate_g1(rho_mean=0.3, recall=0.2, n_branch=100, n_hit=20, selection_rate=0.1)
    loose = evaluate_gate_g1(rho_mean=0.3, recall=0.2, n_branch=100, n_hit=20, selection_rate=0.19)
    assert tight.passed and not loose.passed


# ----------------------------------------------------------------------
# (kappa, gamma_H) selection
# ----------------------------------------------------------------------
def _grid(entries):
    return [{"kappa": k, "gamma_h": g, "rho_mean": r} for k, g, r in entries]


def test_selection_prefers_the_shortest_tied_horizon():
    """kappa=2 and kappa=8 are statistically tied, so take kappa=2."""
    grid = _grid([(2, 0.85, 0.300), (4, 0.85, 0.305), (8, 0.85, 0.307)])
    out = select_kappa_gamma(grid, elbow_tolerance=0.01)
    assert out["kappa"] == 2
    assert out["rho_giveup"] == pytest.approx(0.007)


def test_selection_takes_a_longer_horizon_when_it_really_helps():
    grid = _grid([(2, 0.85, 0.10), (4, 0.85, 0.20), (8, 0.85, 0.35)])
    assert select_kappa_gamma(grid, elbow_tolerance=0.01)["kappa"] == 8


def test_selection_picks_the_best_gamma():
    grid = _grid([(4, 0.7, 0.10), (4, 0.85, 0.30), (4, 1.0, 0.12)])
    assert select_kappa_gamma(grid)["gamma_h"] == 0.85


def test_selection_ignores_nan_entries():
    grid = _grid([(2, 0.85, float("nan")), (4, 0.85, 0.30)])
    assert select_kappa_gamma(grid)["kappa"] == 4


def test_selection_rejects_an_all_nan_grid():
    with pytest.raises(ValueError):
        select_kappa_gamma(_grid([(2, 0.85, float("nan"))]))
