"""Plan §8.4 — H_togo, the two A_H baselines, and the singleton-sibling case."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from steer_f.entropy_forecast import (  # noqa: E402
    HeadCalibration,
    entropy_advantage,
    first_divergence,
    fit_head_calibration,
    group_mean_baseline,
    h_togo,
    permute_a_h_within_siblings,
    sibling_prefix_baseline,
)


# ----------------------------------------------------------------------
# H_togo
# ----------------------------------------------------------------------
def test_h_togo_is_the_discounted_sum():
    ent = torch.tensor([[[1.0]], [[2.0]], [[4.0]]])  # K=3, B=1, T=1
    got = h_togo(ent, kappa=3, gamma_h=0.5)
    assert float(got) == pytest.approx(0.5 * 1 + 0.25 * 2 + 0.125 * 4)


def test_h_togo_kappa_truncates():
    ent = torch.tensor([[[1.0]], [[2.0]], [[4.0]]])
    assert float(h_togo(ent, kappa=1, gamma_h=1.0)) == pytest.approx(1.0)
    assert float(h_togo(ent, kappa=2, gamma_h=1.0)) == pytest.approx(3.0)


def test_h_togo_consumes_the_head_axis():
    assert h_togo(torch.rand(8, 4, 16), kappa=4, gamma_h=0.85).shape == (4, 16)


def test_h_togo_clamps_at_zero_by_default():
    """A bad calibration fit must not produce a negative entropy."""
    calib = HeadCalibration([1.0], [1.0], [-100.0])
    assert float(h_togo(torch.ones(1, 1, 1), kappa=1, gamma_h=1.0, calib=calib)) == 0.0


def test_h_togo_clamp_can_be_disabled():
    calib = HeadCalibration([1.0], [1.0], [-100.0])
    out = h_togo(torch.ones(1, 1, 1), kappa=1, gamma_h=1.0, calib=calib, clamp_min=float("-inf"))
    assert float(out) < 0


@pytest.mark.parametrize("kwargs", [{"kappa": 0}, {"kappa": 9}, {"gamma_h": 0.0}])
def test_h_togo_rejects_invalid(kwargs):
    kw = {"kappa": 4, "gamma_h": 0.85, **kwargs}
    with pytest.raises(ValueError):
        h_togo(torch.rand(8, 2, 3), **kw)


def test_discount_suppresses_distant_heads():
    """The bias that gamma_H exists to control: far heads over-estimate."""
    ent = torch.ones(8, 1, 1) * torch.arange(1, 9).float().view(-1, 1, 1)
    discounted = float(h_togo(ent, kappa=8, gamma_h=0.7))
    undiscounted = float(h_togo(ent, kappa=8, gamma_h=1.0))
    assert discounted < undiscounted


# ----------------------------------------------------------------------
# calibration
# ----------------------------------------------------------------------
def test_calibration_recovers_a_known_affine_map():
    torch.manual_seed(0)
    forecast = torch.rand(3, 200) * 5
    true_scale, true_bias = torch.tensor([0.5, 0.8, 1.3]), torch.tensor([0.1, -0.4, 2.0])
    measured = forecast * true_scale.unsqueeze(1) + true_bias.unsqueeze(1)
    calib = fit_head_calibration(forecast, measured)
    for k in range(3):
        assert calib.scale[k] == pytest.approx(float(true_scale[k]), abs=1e-4)
        assert calib.bias[k] == pytest.approx(float(true_bias[k]), abs=1e-4)


def test_calibration_corrects_distant_head_overestimation():
    """The concrete failure of plan §3.4: uncorrected far heads look diverse."""
    forecast = torch.stack([torch.rand(100) + k for k in range(4)])  # inflated with k
    measured = torch.stack([torch.rand(100) for _ in range(4)])      # truth is flat
    calib = fit_head_calibration(forecast, measured)
    corrected = calib.apply(forecast)
    spread_before = float(forecast.mean(dim=1).std())
    spread_after = float(corrected.mean(dim=1).std())
    assert spread_after < spread_before


def test_calibration_handles_a_constant_forecast():
    calib = fit_head_calibration(torch.ones(2, 50), torch.rand(2, 50))
    assert all(s == 1.0 for s in calib.scale)


def test_calibration_roundtrips_through_dict():
    calib = HeadCalibration([1.0, 1.2], [0.9, 0.8], [0.1, -0.1])
    assert HeadCalibration.from_dict(calib.to_dict()).to_dict() == calib.to_dict()


def test_identity_calibration_is_a_noop():
    ent = torch.rand(4, 2, 3)
    torch.testing.assert_close(HeadCalibration.identity(4).apply(ent), ent)


def test_calibration_rejects_bad_temperature():
    with pytest.raises(ValueError):
        HeadCalibration([0.0], [1.0], [0.0])


# ----------------------------------------------------------------------
# first divergence
# ----------------------------------------------------------------------
def test_first_divergence_finds_the_branch_point():
    responses = torch.tensor([[1, 2, 3, 4], [1, 2, 9, 9], [1, 7, 7, 7]])
    div = first_divergence(responses)
    assert int(div[0, 1]) == 2
    assert int(div[0, 2]) == 1
    assert int(div[1, 2]) == 1


def test_first_divergence_diagonal_is_sequence_length():
    responses = torch.tensor([[1, 2, 3], [4, 5, 6]])
    div = first_divergence(responses)
    assert int(div[0, 0]) == 3 and int(div[1, 1]) == 3


def test_first_divergence_is_symmetric():
    torch.manual_seed(0)
    div = first_divergence(torch.randint(0, 5, (4, 10)))
    assert torch.equal(div, div.T)


def test_differing_lengths_count_as_divergence():
    """One rollout stopping early is a real branch, not a tie."""
    responses = torch.tensor([[1, 2, 3, 4], [1, 2, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    assert int(first_divergence(responses, mask)[0, 1]) == 2


# ----------------------------------------------------------------------
# sibling baseline — the plan §8.4 case
# ----------------------------------------------------------------------
def test_singleton_group_yields_zero_advantage():
    """Plan §8.4: with no sibling, the baseline is self and A_H must be 0."""
    h = torch.rand(1, 6)
    a_h = entropy_advantage(
        h, group_index=["p0"], mask=torch.ones(1, 6),
        responses=torch.zeros(1, 6, dtype=torch.long), baseline="sibling",
    )
    assert torch.equal(a_h, torch.zeros(1, 6))


def test_singleton_group_yields_zero_advantage_group_baseline():
    h = torch.rand(1, 6)
    a_h = entropy_advantage(h, ["p0"], torch.ones(1, 6), baseline="group")
    assert torch.equal(a_h, torch.zeros(1, 6))


def test_identical_rollouts_yield_zero_advantage():
    """Never-diverging siblings share every state, so no branch score exists."""
    responses = torch.ones(4, 8, dtype=torch.long)
    h = torch.rand(4, 8)
    a_h = entropy_advantage(h, ["p"] * 4, torch.ones(4, 8), responses, baseline="sibling")
    # every rollout sees the same 4-member sibling set at every position
    torch.testing.assert_close(a_h, h - h.mean(dim=0, keepdim=True))


def test_sibling_set_shrinks_after_divergence():
    """Before the branch all four are siblings; after it, only the sub-branch."""
    responses = torch.tensor([
        [1, 1, 5, 5],
        [1, 1, 5, 6],
        [1, 1, 7, 7],
        [1, 1, 7, 8],
    ])
    h = torch.tensor([
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 3.0, 3.0],
        [1.0, 1.0, 5.0, 5.0],
        [1.0, 1.0, 7.0, 7.0],
    ])
    mask = torch.ones(4, 4)
    base = sibling_prefix_baseline(h, responses, mask, ["p"] * 4)
    # t=2: nothing has diverged before t=2, so all four are siblings
    assert float(base[0, 2]) == pytest.approx((1 + 3 + 5 + 7) / 4)
    # t=3: rows 0,1 diverged from 2,3 at t=2, so siblings are {0,1} and {2,3}
    assert float(base[0, 3]) == pytest.approx((1 + 3) / 2)
    assert float(base[2, 3]) == pytest.approx((5 + 7) / 2)


def test_sibling_baseline_uses_prefix_strictly_before_t():
    """Siblings must agree on [0, t) but may differ AT t — that is the branch."""
    responses = torch.tensor([[1, 2], [1, 3]])
    h = torch.tensor([[0.0, 2.0], [0.0, 4.0]])
    base = sibling_prefix_baseline(h, responses, torch.ones(2, 2), ["p", "p"])
    # they differ at t=1 but share the prefix [0,1), so they are siblings at t=1
    assert float(base[0, 1]) == pytest.approx(3.0)


def test_prompt_groups_do_not_mix():
    responses = torch.ones(4, 4, dtype=torch.long)
    h = torch.tensor([[1.0] * 4, [3.0] * 4, [10.0] * 4, [30.0] * 4])
    base = sibling_prefix_baseline(h, responses, torch.ones(4, 4), ["a", "a", "b", "b"])
    assert float(base[0, 0]) == pytest.approx(2.0)
    assert float(base[2, 0]) == pytest.approx(20.0)


def test_masked_positions_are_zeroed():
    responses = torch.ones(2, 5, dtype=torch.long)
    mask = torch.tensor([[1.0, 1, 1, 0, 0], [1.0, 1, 1, 1, 1]])
    base = sibling_prefix_baseline(torch.rand(2, 5), responses, mask, ["p", "p"])
    assert torch.equal(base[0, 3:], torch.zeros(2))


def test_dead_siblings_do_not_enter_the_baseline():
    """A finished rollout must not drag the baseline toward zero."""
    responses = torch.ones(2, 4, dtype=torch.long)
    h = torch.tensor([[2.0, 2.0, 2.0, 2.0], [6.0, 6.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1, 1, 1], [1.0, 1, 0, 0]])
    base = sibling_prefix_baseline(h, responses, mask, ["p", "p"])
    assert float(base[0, 0]) == pytest.approx(4.0)   # both alive
    assert float(base[0, 2]) == pytest.approx(2.0)   # row 1 finished


# ----------------------------------------------------------------------
# group baseline (ablation A5 arm B)
# ----------------------------------------------------------------------
def test_group_baseline_averages_the_whole_group():
    h = torch.tensor([[1.0, 1.0], [3.0, 3.0]])
    base = group_mean_baseline(h, torch.ones(2, 2), ["p", "p"])
    torch.testing.assert_close(base, torch.full((2, 2), 2.0))


def test_group_baseline_ignores_prefix_identity():
    """The A5 contrast: group baseline scores across branches, sibling doesn't."""
    responses = torch.tensor([[1, 5, 5], [1, 7, 7]])
    h = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 9.0]])
    mask = torch.ones(2, 3)
    sib = entropy_advantage(h, ["p", "p"], mask, responses, baseline="sibling")
    grp = entropy_advantage(h, ["p", "p"], mask, responses, baseline="group")
    assert float(sib[0, 2]) == 0.0        # diverged at t=1, so alone at t=2
    assert float(grp[0, 2]) == pytest.approx(-4.0)


def test_group_baseline_lone_survivor_scores_zero():
    h = torch.tensor([[1.0, 1.0], [3.0, 3.0]])
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    a_h = entropy_advantage(h, ["p", "p"], mask, baseline="group")
    assert float(a_h[0, 1]) == 0.0


# ----------------------------------------------------------------------
# entropy_advantage plumbing
# ----------------------------------------------------------------------
def test_entropy_advantage_clips_when_asked():
    h = torch.tensor([[0.0, 0.0], [100.0, 100.0]])
    a_h = entropy_advantage(h, ["p", "p"], torch.ones(2, 2), baseline="group", clip_c=1.0)
    assert float(a_h.abs().max()) == pytest.approx(1.0)


def test_entropy_advantage_sums_to_zero_within_a_sibling_set():
    """A_H is a centred advantage, so it carries no group-level offset."""
    torch.manual_seed(0)
    h = torch.rand(4, 6)
    a_h = entropy_advantage(h, ["p"] * 4, torch.ones(4, 6),
                            torch.ones(4, 6, dtype=torch.long), baseline="sibling")
    torch.testing.assert_close(a_h.sum(dim=0), torch.zeros(6), atol=1e-5, rtol=1e-5)


def test_sibling_baseline_requires_responses():
    with pytest.raises(ValueError, match="responses"):
        entropy_advantage(torch.rand(2, 3), ["p", "p"], torch.ones(2, 3), baseline="sibling")


def test_unknown_baseline_rejected():
    with pytest.raises(ValueError):
        entropy_advantage(torch.rand(2, 3), ["p", "p"], torch.ones(2, 3), baseline="bogus")


@pytest.mark.parametrize("bad", ["shape", "group_len"])
def test_shape_mismatches_are_caught(bad):
    h = torch.rand(2, 5)
    responses = torch.ones(2, 5, dtype=torch.long) if bad != "shape" else torch.ones(2, 4, dtype=torch.long)
    group = ["p", "p"] if bad != "group_len" else ["p"]
    with pytest.raises(ValueError):
        sibling_prefix_baseline(h, responses, torch.ones(2, 5), group)


def test_numpy_style_group_keys_work():
    """verl hands us a numpy object array of uuid strings."""
    import numpy as np

    uids = np.array(["a", "a", "b"], dtype=object)
    a_h = entropy_advantage(
        torch.rand(3, 4), uids, torch.ones(3, 4),
        torch.ones(3, 4, dtype=torch.long), baseline="sibling",
    )
    assert a_h.shape == (3, 4)
    assert torch.equal(a_h[2], torch.zeros(4))  # "b" is a singleton


# ----------------------------------------------------------------------
# oracle_h_togo — the control the MTP forecast has to beat
# ----------------------------------------------------------------------
def test_oracle_h_togo_is_the_discounted_forward_sum():
    from steer_f.entropy_forecast import oracle_h_togo

    ent = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
    mask = torch.ones(1, 4)
    out = oracle_h_togo(ent, mask, kappa=2, gamma_h=0.5)
    # position 0: .5*2 + .25*4 = 2.0 ; position 1: .5*4 + .25*8 = 4.0
    assert out[0, 0] == pytest.approx(2.0)
    assert out[0, 1] == pytest.approx(4.0)
    # position 2 has only one future token left, position 3 none
    assert out[0, 2] == pytest.approx(0.5 * 8.0)
    assert out[0, 3] == pytest.approx(0.0)


def test_oracle_h_togo_ignores_masked_future():
    from steer_f.entropy_forecast import oracle_h_togo

    ent = torch.tensor([[1.0, 2.0, 99.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])       # third token is padding
    out = oracle_h_togo(ent, mask, kappa=2, gamma_h=1.0)
    assert out[0, 0] == pytest.approx(2.0)        # padding contributes nothing
    assert out[0, 2] == pytest.approx(0.0)        # masked position is zeroed


def test_oracle_h_togo_rejects_bad_shapes():
    from steer_f.entropy_forecast import oracle_h_togo

    with pytest.raises(ValueError):
        oracle_h_togo(torch.rand(2, 3), torch.rand(2, 4), kappa=1, gamma_h=1.0)
    with pytest.raises(ValueError):
        oracle_h_togo(torch.rand(2, 3), torch.ones(2, 3), kappa=0, gamma_h=1.0)


def test_oracle_a_h_is_zero_without_siblings():
    """Same structural property as the MTP path, so the two are comparable."""
    from steer_f.entropy_forecast import entropy_advantage, oracle_h_togo

    responses = torch.tensor([[1, 2, 3, 4]])
    mask = torch.ones(1, 4)
    h = oracle_h_togo(torch.rand(1, 4), mask, kappa=2, gamma_h=0.85)
    a = entropy_advantage(h, ["p"], mask, responses=responses, baseline="sibling")
    assert torch.all(a == 0)


# ----------------------------------------------------------------------
# permute_a_h_within_siblings -- the control arm
#
# The arm is only a control if it destroys the forecast-to-rollout pairing and
# nothing else, so every invariant it has to keep is asserted here rather than
# read off a training log 47 GPU-hours later.
# ----------------------------------------------------------------------
def _tree_batch(n=8, cuts=(3, 6), T=10):
    """n rollouts sharing a prefix, splitting at each cut. Returns (resp, mask, h)."""
    resp = torch.zeros(n, T, dtype=torch.long)
    for i in range(n):
        path = [(i >> (len(cuts) - 1 - k)) & 1 for k in range(len(cuts))]
        for p in range(T):
            depth = sum(1 for c in cuts if p >= c)
            resp[i, p] = 1000 * p + int("".join(map(str, path[:depth])) or "0", 2)
    mask = torch.ones(n, T)
    # H_togo is a deterministic function of the causal prefix, as it is in the
    # real forecast -- that is what makes A_H vanish away from a cut.
    h = torch.zeros(n, T)
    for i in range(n):
        for p in range(T):
            key = tuple(resp[i, : p + 1].tolist())
            h[i, p] = (abs(hash(key)) % 10_000) / 10_000.0
    return resp, mask, h


def _a_h_tree(**kw):
    resp, mask, h = _tree_batch(**kw)
    uid = ["g"] * resp.shape[0]
    a_h = entropy_advantage(h, group_index=uid, mask=mask,
                            responses=resp, baseline="sibling")
    return a_h, resp, mask, uid


def test_permute_preserves_the_multiset_of_a_h():
    a_h, resp, mask, uid = _a_h_tree()
    got = permute_a_h_within_siblings(
        a_h, resp, mask, uid, generator=torch.Generator().manual_seed(0))
    # column by column: a sibling class never spans two positions
    for t in range(a_h.shape[1]):
        assert torch.allclose(a_h[:, t].sort().values, got[:, t].sort().values, atol=1e-7)


def test_permute_preserves_the_support():
    """`support` sets rms AND where delta lands, so it must survive exactly."""
    a_h, resp, mask, uid = _a_h_tree()
    got = permute_a_h_within_siblings(
        a_h, resp, mask, uid, generator=torch.Generator().manual_seed(1))
    assert bool(((a_h != 0) == (got != 0)).all())


def test_permute_preserves_the_zero_centring():
    """A_H sums to 0 over a sibling class; a permutation cannot shift the mean."""
    a_h, resp, mask, uid = _a_h_tree()
    got = permute_a_h_within_siblings(
        a_h, resp, mask, uid, generator=torch.Generator().manual_seed(2))
    assert torch.allclose(a_h.sum(dim=0), got.sum(dim=0), atol=1e-6)


def test_permute_actually_moves_values():
    """A control that silently no-ops would look like a perfect replication."""
    a_h, resp, mask, uid = _a_h_tree()
    got = permute_a_h_within_siblings(
        a_h, resp, mask, uid, generator=torch.Generator().manual_seed(0))
    assert not torch.allclose(a_h, got)
    assert int((a_h != got).sum()) > 0


def test_permute_only_touches_branch_columns():
    """Off a cut, siblings share y_t, so A_H is 0 for all of them and the
    permutation is the identity on zeros -- no branch detection needed."""
    a_h, resp, mask, uid = _a_h_tree(cuts=(3, 6))
    got = permute_a_h_within_siblings(
        a_h, resp, mask, uid, generator=torch.Generator().manual_seed(0))
    changed = sorted(set((a_h != got).nonzero()[:, 1].tolist()))
    assert changed and set(changed).issubset({3, 6})


def test_permute_is_identity_without_siblings():
    a_h = torch.tensor([[0.3, -0.2, 0.5]])
    resp = torch.tensor([[1, 2, 3]])
    mask = torch.ones(1, 3)
    got = permute_a_h_within_siblings(a_h, resp, mask, ["only"])
    assert torch.equal(got, a_h)


def test_permute_is_zero_outside_the_mask():
    a_h, resp, mask, uid = _a_h_tree()
    mask[:, 7:] = 0.0
    a_h = a_h * mask
    got = permute_a_h_within_siblings(
        a_h, resp, mask, uid, generator=torch.Generator().manual_seed(3))
    assert bool((got[:, 7:] == 0).all())


def test_permute_is_reproducible_from_the_generator_alone():
    a_h, resp, mask, uid = _a_h_tree()
    first = permute_a_h_within_siblings(
        a_h, resp, mask, uid, generator=torch.Generator().manual_seed(7))
    torch.rand(1000)  # ambient RNG moves; the arm must not
    second = permute_a_h_within_siblings(
        a_h, resp, mask, uid, generator=torch.Generator().manual_seed(7))
    assert torch.equal(first, second)


def test_permute_does_not_leak_across_prompt_groups():
    a_h, resp, mask, _ = _a_h_tree()
    uid = ["a"] * 4 + ["b"] * 4
    got = permute_a_h_within_siblings(
        a_h, resp, mask, uid, generator=torch.Generator().manual_seed(4))
    for rows in ([0, 1, 2, 3], [4, 5, 6, 7]):
        assert torch.allclose(a_h[rows].sum(dim=0), got[rows].sum(dim=0), atol=1e-6)


def test_compute_a_h_is_unchanged_when_the_arm_is_off():
    """Regression guard: the treatment path must be bit-identical."""
    from steer_f.omega_tilde import SteerFConfig
    from steer_f.verl_integration import compute_a_h

    _, resp, mask, uid = _a_h_tree()
    _, _, h = _tree_batch()
    plain = compute_a_h(h, resp, mask, uid, SteerFConfig(lam=0.25))
    direct = entropy_advantage(h.float(), group_index=uid, mask=mask,
                               responses=resp, baseline="sibling")
    assert torch.equal(plain, direct)

    shuffled = compute_a_h(h, resp, mask, uid,
                           SteerFConfig(lam=0.25, permute_a_h=True),
                           generator=torch.Generator().manual_seed(0))
    assert not torch.equal(shuffled, direct)
