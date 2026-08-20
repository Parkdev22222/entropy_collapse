"""Diagnostics and the lambda safety valve (plan §4.1 drift guard, §4.2 logging)."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from steer_f.monitors import (  # noqa: E402
    LambdaDriftController,
    branch_recall_at_k,
    branch_token_entropy,
    mtp_policy_kl,
    token_weight_distribution,
)
from steer_f.mtp_heads import MTPHeads  # noqa: E402


# ----------------------------------------------------------------------
# drift controller
# ----------------------------------------------------------------------
def test_quiet_kl_leaves_lambda_alone():
    ctrl = LambdaDriftController(lam_init=1.0, threshold=0.5, ema=0.0)
    for _ in range(20):
        ctrl.update(0.1)
    assert ctrl.lam == 1.0


def test_sustained_drift_halves_lambda():
    ctrl = LambdaDriftController(lam_init=1.0, threshold=0.5, decay=0.5, ema=0.0, patience=2)
    ctrl.update(2.0)
    assert ctrl.lam == 1.0, "one strike must not trip (patience=2)"
    ctrl.update(2.0)
    assert ctrl.lam == 0.5


def test_lambda_snaps_to_zero_below_the_floor():
    ctrl = LambdaDriftController(lam_init=0.02, threshold=0.5, decay=0.5, lam_min=0.01,
                                 ema=0.0, patience=1)
    ctrl.update(5.0)
    assert ctrl.lam == 0.0


def test_lambda_never_recovers_on_its_own():
    """Reproducibility guard: only an explicit reset restores lambda."""
    ctrl = LambdaDriftController(lam_init=1.0, threshold=0.5, ema=0.0, patience=1)
    ctrl.update(9.0)
    assert ctrl.lam == 0.5
    for _ in range(50):
        ctrl.update(0.0)
    assert ctrl.lam == 0.5
    ctrl.reset()
    assert ctrl.lam == 1.0


def test_ema_absorbs_a_single_spike():
    ctrl = LambdaDriftController(lam_init=1.0, threshold=0.5, ema=0.9, patience=1)
    for _ in range(10):
        ctrl.update(0.05)
    ctrl.update(3.0)  # one bad micro-batch
    assert ctrl.lam == 1.0


def test_strikes_reset_when_kl_recovers():
    ctrl = LambdaDriftController(lam_init=1.0, threshold=0.5, ema=0.0, patience=3)
    ctrl.update(2.0)
    ctrl.update(2.0)
    ctrl.update(0.0)
    ctrl.update(2.0)
    ctrl.update(2.0)
    assert ctrl.lam == 1.0


def test_controller_state_roundtrips():
    ctrl = LambdaDriftController(lam_init=1.0, threshold=0.5, ema=0.0, patience=1)
    ctrl.update(9.0)
    restored = LambdaDriftController(lam_init=1.0)
    restored.load_state_dict(ctrl.state_dict())
    assert restored.state_dict() == ctrl.state_dict()


@pytest.mark.parametrize("kwargs", [{"ema": 1.0}, {"decay": 0.0}, {"decay": 1.0}, {"patience": 0}])
def test_controller_rejects_invalid_config(kwargs):
    with pytest.raises(ValueError):
        LambdaDriftController(lam_init=1.0, **kwargs)


# ----------------------------------------------------------------------
# MTP / policy KL
# ----------------------------------------------------------------------
def test_kl_is_zero_when_head_matches_policy():
    """Zero-init heads reproduce the policy exactly, so drift starts at 0."""
    torch.manual_seed(0)
    h, v = 16, 40
    heads = MTPHeads(hidden_size=h, vocab_size=v, num_heads=2, head_hidden=8)
    lm_head = nn.Linear(h, v, bias=False)
    hidden = torch.randn(30, h)
    kl = mtp_policy_kl(lm_head(hidden), heads.ce_loss_inputs(hidden, 0), lm_head)
    assert kl == pytest.approx(0.0, abs=1e-5)


def test_kl_is_positive_when_head_disagrees():
    torch.manual_seed(0)
    h, v = 16, 40
    heads = MTPHeads(hidden_size=h, vocab_size=v, num_heads=2, head_hidden=8,
                     zero_init_output=False)
    lm_head = nn.Linear(h, v, bias=False)
    hidden = torch.randn(30, h)
    assert mtp_policy_kl(lm_head(hidden), heads.ce_loss_inputs(hidden, 0), lm_head) > 0


def test_kl_subsamples_large_inputs():
    torch.manual_seed(0)
    h, v = 8, 20
    lm_head = nn.Linear(h, v, bias=False)
    hidden = torch.randn(5000, h)
    kl = mtp_policy_kl(lm_head(hidden), hidden, lm_head, max_positions=100)
    assert torch.isfinite(torch.tensor(kl))


def test_kl_handles_empty_input():
    lm_head = nn.Linear(8, 20, bias=False)
    assert mtp_policy_kl(torch.zeros(0, 20), torch.zeros(0, 8), lm_head) == 0.0


def test_kl_rejects_misaligned_positions():
    lm_head = nn.Linear(8, 20, bias=False)
    with pytest.raises(ValueError):
        mtp_policy_kl(torch.zeros(5, 20), torch.zeros(7, 8), lm_head)


# ----------------------------------------------------------------------
# branch-token entropy
# ----------------------------------------------------------------------
def test_branch_entropy_isolates_the_top_decile():
    ent = torch.arange(20).float().view(1, 20)
    a_h = torch.arange(20).float().view(1, 20)  # top A_H == top entropy here
    out = branch_token_entropy(ent, a_h, torch.ones(1, 20), top_frac=0.1)
    assert out["steerf/branch_entropy"] == pytest.approx(18.5)  # tokens 18, 19
    assert out["steerf/branch_entropy_gap"] > 0


def test_branch_entropy_ignores_padding():
    ent = torch.cat([torch.ones(1, 10), torch.full((1, 10), 99.0)], dim=1)
    a_h = torch.cat([torch.arange(10).float().view(1, 10), torch.zeros(1, 10)], dim=1)
    mask = torch.cat([torch.ones(1, 10), torch.zeros(1, 10)], dim=1)
    out = branch_token_entropy(ent, a_h, mask, top_frac=0.5)
    assert out["steerf/branch_entropy"] == pytest.approx(1.0)


def test_branch_entropy_empty_mask_is_nan_not_a_crash():
    out = branch_token_entropy(torch.rand(2, 5), torch.rand(2, 5), torch.zeros(2, 5))
    assert out["steerf/branch_entropy"] != out["steerf/branch_entropy"]  # nan


def test_branch_entropy_rejects_bad_fraction():
    with pytest.raises(ValueError):
        branch_token_entropy(torch.rand(1, 4), torch.rand(1, 4), torch.ones(1, 4), top_frac=0.0)


# ----------------------------------------------------------------------
# token-weight distribution
# ----------------------------------------------------------------------
def test_weight_thirds_sum_to_one():
    w = torch.rand(4, 25) * 0.4 + 0.8
    out = token_weight_distribution(w, torch.ones(4, 25), 0.8, 1.2)
    total = out["steerf/tw_frac_low"] + out["steerf/tw_frac_mid"] + out["steerf/tw_frac_high"]
    assert total == pytest.approx(1.0)


def test_weight_thirds_detect_a_degenerate_run():
    """The failure the plan's risk table calls 'band misconfigured'."""
    w = torch.full((2, 10), 0.81)
    out = token_weight_distribution(w, torch.ones(2, 10), 0.8, 1.2)
    assert out["steerf/tw_frac_low"] == pytest.approx(1.0)


def test_weight_distribution_rejects_inverted_range():
    with pytest.raises(ValueError):
        token_weight_distribution(torch.ones(1, 3), torch.ones(1, 3), 1.2, 0.8)


# ----------------------------------------------------------------------
# branch recall (gate G1 secondary metric)
# ----------------------------------------------------------------------
def test_perfect_a_h_gives_perfect_recall():
    """A_H that peaks exactly at the divergence recovers every branch point."""
    responses = torch.tensor([
        [1, 1, 1, 5, 5, 5, 5, 5, 5, 5],
        [1, 1, 1, 9, 9, 9, 9, 9, 9, 9],
    ])
    a_h = torch.zeros(2, 10)
    a_h[:, 3] = 10.0  # the true branch position
    out = branch_recall_at_k(a_h, responses, torch.ones(2, 10), ["p", "p"], top_frac=0.2)
    assert out["n_branch"] == 2
    assert out["recall"] == pytest.approx(1.0)
    assert out["lift"] > 1.0


def test_uninformative_a_h_gives_chance_recall():
    responses = torch.tensor([[1, 1, 5, 5], [1, 1, 9, 9]])
    a_h = torch.zeros(2, 4)
    out = branch_recall_at_k(a_h, responses, torch.ones(2, 4), ["p", "p"], top_frac=0.5)
    assert 0.0 <= out["recall"] <= 1.0


def test_recall_reports_counts_for_a_binomial_test():
    """Gate G1 needs n_hit / n_branch to compute its p-value."""
    responses = torch.tensor([[1, 2, 3], [1, 9, 9]])
    out = branch_recall_at_k(torch.rand(2, 3), responses, torch.ones(2, 3), ["p", "p"])
    assert set(["n_branch", "n_hit", "n_valid"]).issubset(out)
    assert out["n_hit"] <= out["n_branch"]


def test_recall_with_no_siblings_is_nan():
    out = branch_recall_at_k(torch.rand(1, 4), torch.ones(1, 4, dtype=torch.long),
                             torch.ones(1, 4), ["p"])
    assert out["n_branch"] == 0
    assert out["recall"] != out["recall"]  # nan


def test_recall_empty_mask_is_safe():
    out = branch_recall_at_k(torch.rand(2, 4), torch.ones(2, 4, dtype=torch.long),
                             torch.zeros(2, 4), ["p", "p"])
    assert out["n_valid"] == 0


# ----------------------------------------------------------------------
# tie handling in the top-decile selection
#
# A_H carries a point mass at exactly 0: with baseline="sibling" a rollout that
# has become unique is its own only sibling, so its A_H is exactly zero
# (sibling_prefix_baseline's documented behaviour), and past the first
# divergence that is nearly every position. Selecting with ">= kth largest"
# then keeps ~99% of tokens instead of the top decile, which silently disables
# gate G1's binomial test and gate G2's branch/non-branch entropy split.
# ----------------------------------------------------------------------
def test_selection_is_exactly_top_frac_under_a_zero_point_mass():
    """Realised selection stays at top_frac even when most A_H values tie at 0."""
    b, t = 8, 200
    a_h = torch.zeros(b, t)
    a_h[:, :4] = torch.randn(b, 4)          # only the sibling-rich prefix is nonzero
    responses = torch.arange(b * t).reshape(b, t) % 7
    responses[:, :3] = 1                     # shared prefix, then divergence
    mask = torch.ones(b, t)

    out = branch_recall_at_k(a_h, responses, mask, ["p"] * b, top_frac=0.1)

    # `baseline` is the realised selection rate, which G1 uses as the null.
    assert out["baseline"] == pytest.approx(0.1, abs=0.005), (
        f"top decile holds {out['baseline']:.1%} of tokens; a tie-tolerant "
        "threshold would make G1's binomial test unpassable"
    )


def test_branch_entropy_split_is_not_degenerate_under_ties():
    b, t = 8, 200
    a_h = torch.zeros(b, t)
    a_h[:, :4] = torch.randn(b, 4)
    entropys = torch.rand(b, t)

    out = branch_token_entropy(entropys, a_h, torch.ones(b, t), top_frac=0.1)

    assert out["steerf/branch_token_frac"] == pytest.approx(0.1, abs=0.005)
    # Both sides must be populated, else the gap compares a set with nothing.
    assert out["steerf/nonbranch_entropy"] == out["steerf/nonbranch_entropy"]  # not nan


def test_selection_picks_the_largest_values_first():
    """Rank-based selection must still prefer high A_H over the tie block."""
    a_h = torch.zeros(1, 100)
    a_h[0, :5] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    out = branch_token_entropy(torch.rand(1, 100), a_h, torch.ones(1, 100), top_frac=0.1)
    assert out["steerf/branch_token_frac"] == pytest.approx(0.1, abs=0.01)


# ----------------------------------------------------------------------
# Restricting the recall test to the sibling support
# ----------------------------------------------------------------------
def test_support_restricts_both_selection_and_null():
    """Without a support mask the null is diluted by positions A_H cannot rank."""
    from steer_f.entropy_forecast import sibling_support

    responses = torch.tensor([
        [1, 1, 5, 7, 8, 9, 3, 3],
        [1, 1, 9, 2, 4, 6, 3, 3],
    ])
    mask = torch.ones(2, 8)
    a_h = torch.zeros(2, 8)
    a_h[:, 2] = torch.tensor([1.0, -1.0])   # nonzero only where siblings remain

    sup = sibling_support(responses, mask, ["p", "p"])
    # positions 0..2 share the prefix; from 3 on each rollout is alone
    assert bool(sup[0, 2]) and not bool(sup[0, 5])

    wide = branch_recall_at_k(a_h, responses, mask, ["p", "p"], top_frac=0.5)
    narrow = branch_recall_at_k(a_h, responses, mask, ["p", "p"], top_frac=0.5, support=sup)
    assert narrow["n_valid"] < wide["n_valid"]


def test_support_mask_of_all_false_is_safe():
    from steer_f.entropy_forecast import sibling_support

    responses = torch.tensor([[1, 2, 3, 4]])
    out = branch_recall_at_k(torch.rand(1, 4), responses, torch.ones(1, 4), ["p"],
                             support=torch.zeros(1, 4, dtype=torch.bool))
    assert out["n_valid"] == 0


def test_sibling_support_is_false_where_a_h_is_structurally_zero():
    from steer_f.entropy_forecast import sibling_prefix_baseline, sibling_support

    responses = torch.tensor([[1, 1, 4, 4, 4], [1, 1, 9, 9, 9]])
    mask = torch.ones(2, 5)
    h = torch.randn(2, 5)
    base = sibling_prefix_baseline(h, responses, mask, ["p", "p"])
    a_h = (h - base) * mask
    sup = sibling_support(responses, mask, ["p", "p"])
    # every position outside the support must have A_H exactly zero
    assert torch.all(a_h[~sup] == 0)
