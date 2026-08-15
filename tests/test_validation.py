"""Phase 1 지표·게이트 로직 테스트 (GPU 불필요)."""

import math

import numpy as np
import pytest

from steer_f.validation import (
    GridResult,
    branch_recall_at_k,
    evaluate_gate_g1,
    fisher_z_mean,
    select_kappa_gamma,
    spearman,
    within_problem_spearman,
)


# ------------------------------------------------------------ spearman
def test_spearman_monotone():
    assert spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_is_rank_based_not_linear():
    """비선형 단조 변환에도 ρ = 1 이어야 한다."""
    x = [1, 2, 3, 4, 5, 6]
    y = [math.exp(v) for v in x]
    assert spearman(x, y) == pytest.approx(1.0)


def test_spearman_handles_ties():
    rho = spearman([1, 1, 2, 2, 3, 3], [1, 1, 2, 2, 3, 3])
    assert rho == pytest.approx(1.0)


def test_spearman_constant_or_short_is_nan():
    assert math.isnan(spearman([1, 1, 1, 1], [1, 2, 3, 4]))
    assert math.isnan(spearman([1, 2], [3, 4]))


def test_spearman_drops_non_finite():
    assert spearman([1, 2, 3, float("nan")], [1, 2, 3, 99]) == pytest.approx(1.0)


# ------------------------------------------------------------- fisher z
def test_fisher_z_mean_between_min_and_max():
    v = fisher_z_mean([0.1, 0.5, 0.9])
    assert 0.1 < v < 0.9


def test_fisher_z_mean_pulls_above_arithmetic_mean():
    """Fisher z 평균은 큰 상관에 더 무게를 준다."""
    rhos = [0.1, 0.9]
    assert fisher_z_mean(rhos) > float(np.mean(rhos))


def test_fisher_z_mean_ignores_nan():
    assert fisher_z_mean([float("nan"), 0.4]) == pytest.approx(0.4, abs=1e-6)
    assert math.isnan(fisher_z_mean([float("nan")]))


def test_fisher_z_mean_handles_perfect_correlation():
    assert fisher_z_mean([1.0, 1.0]) == pytest.approx(1.0, abs=1e-4)


# -------------------------------------------------- within-problem rho
def test_within_problem_beats_pooled_when_offsets_differ():
    """문제마다 오프셋이 다르면 pooled 상관은 부풀거나 깨진다 — 그래서 문제 내에서 잰다."""
    rng = np.random.default_rng(0)
    pids, pred, tgt = [], [], []
    for p, offset in enumerate([0.0, 50.0, 100.0]):
        x = rng.normal(size=20)
        pids += [p] * 20
        pred += list(x + offset)
        tgt += list(-x + offset * 3)  # 문제 내에서는 음의 상관
    rho, per = within_problem_spearman(pids, pred, tgt)
    assert rho < -0.9
    assert len(per) == 3
    pooled = spearman(pred, tgt)
    assert pooled > 0  # pooled는 부호마저 뒤집힌다


def test_within_problem_skips_small_groups():
    pids = [0] * 10 + [1] * 2
    pred = list(range(10)) + [0.0, 1.0]
    tgt = list(range(10)) + [1.0, 0.0]
    rho, per = within_problem_spearman(pids, pred, tgt, min_prefixes=5)
    assert list(per) == [0]
    # Fisher z 변환에서 ±1은 발산하므로 ±0.999999로 클리핑된다.
    assert rho == pytest.approx(1.0, abs=1e-5)


# ------------------------------------------------------------- recall
def test_branch_recall_perfect_ranking():
    scores = list(range(100))
    is_branch = [i >= 90 for i in range(100)]
    out = branch_recall_at_k(scores, is_branch, top_frac=0.1, n_permutations=500)
    assert out["recall"] == pytest.approx(1.0)
    assert out["baseline"] == pytest.approx(0.1)
    assert out["lift"] == pytest.approx(10.0)
    assert out["p_value"] < 0.05


def test_branch_recall_random_ranking_not_significant():
    rng = np.random.default_rng(3)
    scores = rng.normal(size=400)
    is_branch = rng.random(400) < 0.1
    out = branch_recall_at_k(scores, is_branch, top_frac=0.1, n_permutations=1000, seed=1)
    assert out["p_value"] > 0.05


def test_branch_recall_no_branch_tokens():
    out = branch_recall_at_k([1.0, 2.0, 3.0], [False, False, False])
    assert math.isnan(out["recall"])
    assert out["n_branch"] == 0


def test_branch_recall_is_deterministic():
    rng = np.random.default_rng(5)
    s, b = rng.normal(size=200), rng.random(200) < 0.2
    a1 = branch_recall_at_k(s, b, seed=7, n_permutations=300)
    a2 = branch_recall_at_k(s, b, seed=7, n_permutations=300)
    assert a1 == a2


# --------------------------------------------------------- grid select
def test_select_kappa_prefers_smaller_at_equal_rho():
    results = [
        GridResult(kappa=2, gamma_h=0.85, rho=0.295),
        GridResult(kappa=4, gamma_h=0.85, rho=0.300),
        GridResult(kappa=8, gamma_h=0.85, rho=0.301),
    ]
    best = select_kappa_gamma(results, elbow_tol=0.01)
    assert best.kappa == 2


def test_select_kappa_picks_clear_winner():
    results = [
        GridResult(kappa=2, gamma_h=0.85, rho=0.10),
        GridResult(kappa=4, gamma_h=0.85, rho=0.30),
    ]
    assert select_kappa_gamma(results, elbow_tol=0.01).kappa == 4


def test_select_kappa_ties_break_on_rho():
    results = [
        GridResult(kappa=4, gamma_h=0.7, rho=0.30),
        GridResult(kappa=4, gamma_h=1.0, rho=0.305),
    ]
    assert select_kappa_gamma(results).gamma_h == 1.0


def test_select_kappa_requires_finite():
    with pytest.raises(ValueError):
        select_kappa_gamma([GridResult(kappa=1, gamma_h=1.0, rho=float("nan"))])


# --------------------------------------------------------------- gate
def test_gate_g1_passes():
    g = evaluate_gate_g1(0.25, {"p_value": 0.01, "lift": 2.0})
    assert g.passed and "Phase 2" in g.summary()


def test_gate_g1_fails_on_low_rho():
    g = evaluate_gate_g1(0.15, {"p_value": 0.01, "lift": 2.0})
    assert not g.passed
    assert any("rho" in r for r in g.reasons)


def test_gate_g1_fails_on_insignificant_recall():
    g = evaluate_gate_g1(0.4, {"p_value": 0.30, "lift": 1.1})
    assert not g.passed
    assert any("p-value" in r for r in g.reasons)


def test_gate_g1_fails_on_lift_below_one():
    g = evaluate_gate_g1(0.4, {"p_value": 0.001, "lift": 0.8})
    assert not g.passed


def test_gate_g1_threshold_is_the_plan_value():
    assert evaluate_gate_g1(0.2, {"p_value": 0.01, "lift": 2.0}).passed
    assert not evaluate_gate_g1(0.199, {"p_value": 0.01, "lift": 2.0}).passed
