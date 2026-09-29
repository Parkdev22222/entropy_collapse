"""Plan §8.3 — A_H clip boundaries, normalisation degeneracy, sign semantics."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from steer_f.omega_tilde import (  # noqa: E402
    SteerFConfig,
    branch_weight_correction,
    compute_omega_tilde,
    compute_token_weights_steerf,
    delta_logpi_hat,
    local_omega_signed,
    normalize,
    visit_term,
)


# ----------------------------------------------------------------------
# A_H clipping
# ----------------------------------------------------------------------
def test_a_h_clip_bounds_the_visit_term():
    """A wild forecast cannot produce an unbounded contribution."""
    w = torch.ones(1, 5)
    pi = torch.zeros(1, 5)
    a_h = torch.tensor([[-1e6, -2.0, 0.0, 2.0, 1e6]])
    out = visit_term(w, pi, a_h, eta=1.0, clip_c=1.0)
    torch.testing.assert_close(out, torch.tensor([[-1.0, -1.0, 0.0, 1.0, 1.0]]))


@pytest.mark.parametrize("clip_c", [0.1, 1.0, 5.0])
def test_clip_boundary_is_inclusive_and_exact(clip_c):
    w = torch.ones(1, 3)
    pi = torch.zeros(1, 3)
    a_h = torch.tensor([[-clip_c, clip_c, clip_c * 10]])
    out = visit_term(w, pi, a_h, clip_c=clip_c)
    torch.testing.assert_close(out, torch.tensor([[-clip_c, clip_c, clip_c]]))


def test_clip_c_must_be_positive():
    with pytest.raises(ValueError):
        visit_term(torch.ones(1, 2), torch.zeros(1, 2), torch.zeros(1, 2), clip_c=0.0)


def test_a_h_clip_fraction_is_reported():
    """The saturation diagnostic must fire when clip_c is too tight."""
    _, stats = compute_omega_tilde(
        omega_local=torch.zeros(1, 4),
        w=torch.ones(1, 4),
        pi_sampled=torch.zeros(1, 4),
        a_h=torch.tensor([[10.0, 10.0, 0.0, 0.0]]),
        response_mask=torch.ones(1, 4),
        lam=1.0,
        clip_c=1.0,
    )
    assert stats["steerf/a_h_clip_frac"] == pytest.approx(0.5)


# ----------------------------------------------------------------------
# normalisation
# ----------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["scale", "z"])
def test_zero_variance_batch_is_safe(mode):
    """Plan §8.3: a constant metric must not divide by ~0."""
    x = torch.full((2, 6), 3.0)
    out = normalize(x, torch.ones(2, 6), mode=mode)
    assert torch.isfinite(out).all()
    if mode == "z":
        assert torch.equal(out, torch.zeros_like(x))


def test_all_zero_batch_is_safe():
    out = normalize(torch.zeros(2, 6), torch.ones(2, 6), mode="scale")
    assert torch.equal(out, torch.zeros(2, 6))


def test_empty_mask_is_safe():
    for mode in ("scale", "z", "none"):
        out = normalize(torch.randn(2, 6), torch.zeros(2, 6), mode=mode)
        assert torch.isfinite(out).all()
        assert torch.equal(out, torch.zeros(2, 6))


def test_scale_norm_is_zero_preserving():
    """The property that makes norm="scale" safe ahead of abs()."""
    x = torch.tensor([[-2.0, 0.0, 4.0, 1.0]])
    out = normalize(x, torch.ones(1, 4), mode="scale")
    assert out[0, 1] == 0.0
    assert torch.sign(out).equal(torch.sign(x))


def test_scale_norm_gives_unit_rms():
    x = torch.randn(4, 20) * 7.0
    out = normalize(x, torch.ones(4, 20), mode="scale")
    assert float((out**2).mean().sqrt()) == pytest.approx(1.0, abs=1e-5)


def test_z_norm_gives_zero_mean_unit_std():
    x = torch.randn(4, 20) * 7.0 + 3.0
    out = normalize(x, torch.ones(4, 20), mode="z")
    assert float(out.mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(out.std(unbiased=False)) == pytest.approx(1.0, abs=1e-5)


def test_statistics_ignore_masked_positions():
    """Padding must not drag the scale down."""
    x = torch.cat([torch.full((1, 5), 4.0), torch.zeros(1, 95)], dim=1)
    mask = torch.cat([torch.ones(1, 5), torch.zeros(1, 95)], dim=1)
    out = normalize(x, mask, mode="scale")
    assert float(out[0, :5].mean()) == pytest.approx(1.0, abs=1e-5)


def test_normalize_rejects_unknown_mode():
    with pytest.raises(ValueError):
        normalize(torch.randn(2, 3), torch.ones(2, 3), mode="minmax")


# ----------------------------------------------------------------------
# sign semantics — the part that decides whether the extension helps or hurts
# ----------------------------------------------------------------------
def test_delta_logpi_sign_follows_the_update_direction():
    """Positive advantage concentrates mass on the sampled token."""
    assert float(delta_logpi_hat(torch.tensor(1.0), torch.tensor(0.5))) > 0
    assert float(delta_logpi_hat(torch.tensor(-1.0), torch.tensor(0.5))) < 0


def test_delta_logpi_vanishes_for_a_saturated_token():
    """A token already at probability 1 cannot move further."""
    assert float(delta_logpi_hat(torch.tensor(5.0), torch.tensor(1.0))) == pytest.approx(0.0)


def test_visit_term_signs_match_the_derivation():
    """Concentrating onto a low-entropy branch must read as entropy loss."""
    w = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
    pi = torch.zeros(1, 4)
    a_h = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    out = visit_term(w, pi, a_h, clip_c=2.0)
    # (up-weight, diverse branch) -> entropy up; (up-weight, dead end) -> down
    torch.testing.assert_close(out, torch.tensor([[1.0, -1.0, -1.0, 1.0]]))


def test_local_omega_sign_convention_matches_asymmetric_mode():
    """local_omega_signed must equal upstream's asymmetric metric exactly."""
    sys.path.insert(0, str(Path(__file__).parent))
    from reference_steer import compute_token_weights  # noqa: F401  (import check)

    g = torch.Generator().manual_seed(0)
    log_prob = -torch.rand(2, 8, generator=g) * 4
    old_log_prob = log_prob + 0.01
    advantages = torch.randn(2, 8, generator=g)
    entropys = torch.rand(2, 8, generator=g) * 2

    x = torch.clamp(torch.exp(log_prob), 1e-8, 1 - 1e-8)
    f_x = x * (1 - x) * (torch.log(x) + entropys)
    expected = -(advantages / torch.clamp(torch.exp(old_log_prob), 1e-8, 1.0)) * f_x
    got = local_omega_signed(advantages, entropys, old_log_prob, log_prob)
    assert torch.equal(got, expected)


def test_non_finite_omega_is_zeroed():
    """A NaN must be contained before it can reach the sum."""
    out = local_omega_signed(
        advantages=torch.tensor([[float("nan"), 1.0]]),
        entropys=torch.tensor([[1.0, 1.0]]),
        old_log_prob=torch.tensor([[-1.0, -1.0]]),
        log_prob=torch.tensor([[-1.0, -1.0]]),
    )
    assert torch.isfinite(out).all()
    assert float(out[0, 0]) == 0.0


# ----------------------------------------------------------------------
# combination
# ----------------------------------------------------------------------
def test_lambda_scales_the_visit_contribution_monotonically():
    omega = torch.randn(2, 10)
    w = torch.randn(2, 10)
    pi = torch.rand(2, 10)
    a_h = torch.randn(2, 10)
    mask = torch.ones(2, 10)
    prev = None
    for lam in (0.25, 0.5, 1.0, 2.0):
        _, stats = compute_omega_tilde(omega, w, pi, a_h, mask, lam=lam)
        mag = stats["steerf/visit_rel_mag"]
        if prev is not None:
            assert mag > prev
        prev = mag


def test_relative_magnitudes_are_comparable_after_normalisation():
    """Two terms of wildly different raw scale end up within an order of magnitude.

    This is what the normalisation is *for*: without it, lambda would have to
    absorb an unknown scale ratio that changes across models and training
    steps, making a single swept lambda meaningless.
    """
    omega = torch.randn(4, 50) * 1e-6
    w = torch.randn(4, 50) * 1e3
    pi = torch.rand(4, 50)
    a_h = torch.randn(4, 50)
    _, stats = compute_omega_tilde(omega, w, pi, a_h, torch.ones(4, 50), lam=1.0, clip_c=5.0)
    assert 0.1 < stats["steerf/visit_rel_mag"] < 10.0


def test_negative_lambda_rejected():
    with pytest.raises(ValueError):
        compute_omega_tilde(
            torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2),
            torch.zeros(1, 2), torch.ones(1, 2), lam=-1.0,
        )


def test_masked_positions_stay_zero():
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    out, _ = compute_omega_tilde(
        omega_local=torch.randn(1, 4),
        w=torch.randn(1, 4),
        pi_sampled=torch.rand(1, 4),
        a_h=torch.randn(1, 4),
        response_mask=mask,
        lam=1.0,
    )
    assert torch.equal(out[0, 2:], torch.zeros(2))


# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs", [{"lam": -1}, {"clip_c": 0}, {"norm": "bogus"}, {"baseline": "bogus"},
               {"kappa": 0}, {"gamma_h": 0}],
)
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        SteerFConfig(**kwargs).validate()


def test_config_defaults_are_the_stock_steer_arm():
    cfg = SteerFConfig()
    assert cfg.lam == 0.0 and cfg.validate() == []


# ----------------------------------------------------------------------
# apply="weight": the branch correction lands after STEER's mapping
#
# The metric-level form has to survive a per-micro-batch min-max whose scale is
# set by Omega's extremes, and Omega carries a division by pi_old floored at
# 1e-8. Offline on real rollouts that reduced a branch signal covering ~1% of
# positions to a mean weight difference of 0.0007 on a 0.1-wide band, while
# still costing the attenuating range up to 34% of its occupancy.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["symmetric", "asymmetric"])
@pytest.mark.parametrize("linear", [True, False])
def test_apply_weight_is_still_bit_identical_at_lambda_zero(mode, linear):
    torch.manual_seed(7)
    b, t = 4, 16
    kw = dict(
        advantages=torch.randn(b, t), entropys=torch.rand(b, t),
        old_log_prob=-torch.rand(b, t), log_prob=-torch.rand(b, t),
        response_mask=torch.ones(b, t), token_weight_min=0.9,
        token_weight_max=1.0, linear=linear, mode=mode,
    )
    a = compute_token_weights_steerf(**kw, a_h=torch.randn(b, t), lam=0.0, apply="metric")
    c = compute_token_weights_steerf(**kw, a_h=torch.randn(b, t), lam=0.0, apply="weight")
    assert torch.equal(a, c)


def test_apply_weight_leaves_non_branch_tokens_exactly_alone():
    """The whole point: STEER's treatment of the other 99% must be untouched."""
    torch.manual_seed(11)
    b, t = 4, 32
    a_h = torch.zeros(b, t)
    a_h[:, 3] = torch.randn(b)          # branch signal at one position only
    kw = dict(
        advantages=torch.randn(b, t), entropys=torch.rand(b, t),
        old_log_prob=-torch.rand(b, t), log_prob=-torch.rand(b, t),
        response_mask=torch.ones(b, t), token_weight_min=0.9,
        token_weight_max=1.0, linear=True, mode="asymmetric",
    )
    base = compute_token_weights_steerf(**kw, a_h=a_h, lam=0.0, apply="weight")
    corr = compute_token_weights_steerf(**kw, a_h=a_h, lam=0.5, apply="weight")

    off = [i for i in range(t) if i != 3]
    assert torch.allclose(base[:, off], corr[:, off]), (
        "apply='weight' changed a token with no branch score"
    )
    assert not torch.allclose(base[:, 3], corr[:, 3]), "branch token was not adjusted"


def test_apply_metric_does_disturb_non_branch_tokens():
    """Contrast: the shipped metric form moves tokens that have no A_H at all."""
    torch.manual_seed(11)
    b, t = 4, 32
    a_h = torch.zeros(b, t)
    a_h[:, 3] = torch.randn(b)
    kw = dict(
        advantages=torch.randn(b, t), entropys=torch.rand(b, t),
        old_log_prob=-torch.rand(b, t), log_prob=-torch.rand(b, t),
        response_mask=torch.ones(b, t), token_weight_min=0.9,
        token_weight_max=1.0, linear=True, mode="asymmetric",
    )
    base = compute_token_weights_steerf(**kw, a_h=a_h, lam=0.0, apply="metric")
    corr = compute_token_weights_steerf(**kw, a_h=a_h, lam=0.5, apply="metric")
    off = [i for i in range(t) if i != 3]
    assert not torch.allclose(base[:, off], corr[:, off])


def test_apply_weight_correction_is_bounded_by_lambda():
    torch.manual_seed(3)
    b, t, w_min, w_max = 2, 24, 0.9, 1.0
    a_h = torch.randn(b, t)
    kw = dict(
        advantages=torch.randn(b, t), entropys=torch.rand(b, t),
        old_log_prob=-torch.rand(b, t), log_prob=-torch.rand(b, t),
        response_mask=torch.ones(b, t), token_weight_min=w_min,
        token_weight_max=w_max, linear=True, mode="asymmetric",
    )
    base = compute_token_weights_steerf(**kw, a_h=a_h, lam=0.0, apply="weight")
    for lam in (0.25, 0.5):
        corr = compute_token_weights_steerf(**kw, a_h=a_h, lam=lam, apply="weight")
        assert (corr - base).abs().max() <= lam * (w_max - w_min) + 1e-6
        assert corr.min() >= w_min - 1e-6 and corr.max() <= w_max + 1e-6


def test_apply_rejects_unknown_mode():
    with pytest.raises(ValueError):
        compute_token_weights_steerf(
            advantages=torch.randn(2, 4), entropys=torch.rand(2, 4),
            old_log_prob=-torch.rand(2, 4), log_prob=-torch.rand(2, 4),
            response_mask=torch.ones(2, 4), a_h=torch.randn(2, 4),
            lam=0.5, apply="nonsense")


# ----------------------------------------------------------------------
# apply="branch": uniform protection, no forecast used
#
# A_H is a deviation from the sibling mean, so it is zero-centred by
# construction and the signed correction can only *spread* branch tokens, never
# shift their mean. Uniform protection is the weaker hypothesis that needs no
# forecast at all -- only "does this rollout still have siblings here".
# ----------------------------------------------------------------------
def _kw(b=4, t=24, w_min=0.9, w_max=1.0):
    torch.manual_seed(5)
    return dict(
        advantages=torch.randn(b, t), entropys=torch.rand(b, t),
        old_log_prob=-torch.rand(b, t), log_prob=-torch.rand(b, t),
        response_mask=torch.ones(b, t), token_weight_min=w_min,
        token_weight_max=w_max, linear=True, mode="asymmetric",
    )


def test_uniform_branch_shifts_the_mean_where_signed_cannot():
    b, t = 4, 24
    a_h = torch.zeros(b, t)
    a_h[:, 5] = torch.tensor([1.0, -1.0, 0.7, -0.7])   # zero-centred, as A_H is
    kw = _kw(b, t)
    base = compute_token_weights_steerf(**kw, a_h=a_h, lam=0.0, apply="weight")
    signed = compute_token_weights_steerf(**kw, a_h=a_h, lam=0.5, apply="weight")
    uniform = compute_token_weights_steerf(**kw, a_h=a_h, lam=0.5, apply="branch")

    base_m, sgn_m, uni_m = base[:, 5].mean(), signed[:, 5].mean(), uniform[:, 5].mean()
    # signed moves the spread, not the mean
    assert signed[:, 5].std() > base[:, 5].std()
    # uniform moves the mean down (down == more attenuation == more protection)
    assert uni_m < base_m - 1e-6, "uniform protection did not attenuate branch tokens"
    assert abs(float(sgn_m - base_m)) < abs(float(uni_m - base_m))


def test_uniform_branch_ignores_the_sign_of_the_score():
    b, t = 2, 16
    kw = _kw(b, t)
    a1 = torch.zeros(b, t); a1[:, 4] = 1.0
    a2 = torch.zeros(b, t); a2[:, 4] = -1.0     # opposite sign, same support
    u1 = compute_token_weights_steerf(**kw, a_h=a1, lam=0.5, apply="branch")
    u2 = compute_token_weights_steerf(**kw, a_h=a2, lam=0.5, apply="branch")
    assert torch.allclose(u1, u2), "uniform mode used the score's sign"


def test_uniform_branch_still_leaves_non_branch_tokens_untouched():
    b, t = 3, 20
    a_h = torch.zeros(b, t); a_h[:, 7] = torch.randn(b)
    kw = _kw(b, t)
    base = compute_token_weights_steerf(**kw, a_h=a_h, lam=0.0, apply="branch")
    corr = compute_token_weights_steerf(**kw, a_h=a_h, lam=0.5, apply="branch")
    off = [i for i in range(t) if i != 7]
    assert torch.allclose(base[:, off], corr[:, off])


@pytest.mark.parametrize("mode", ["symmetric", "asymmetric"])
@pytest.mark.parametrize("linear", [True, False])
def test_uniform_branch_is_bit_identical_at_lambda_zero(mode, linear):
    kw = _kw()
    kw["mode"] = mode; kw["linear"] = linear
    a = compute_token_weights_steerf(**kw, a_h=torch.randn(4, 24), lam=0.0, apply="metric")
    c = compute_token_weights_steerf(**kw, a_h=torch.randn(4, 24), lam=0.0, apply="branch")
    assert torch.equal(a, c)


def test_branch_correction_rejects_unknown_inner_mode():
    with pytest.raises(ValueError):
        branch_weight_correction(torch.ones(2, 4), torch.randn(2, 4),
                                 torch.ones(2, 4), lam=0.5,
                                 token_weight_min=0.9, token_weight_max=1.0,
                                 mode="nope")


# ----------------------------------------------------------------------
# I_clip — Theorem 1 carries it, the released implementation drops it
# ----------------------------------------------------------------------
def test_clip_indicator_matches_equation_6():
    from steer_f.omega_tilde import clip_indicator

    # ratio = exp(log_prob - old_log_prob); build it exactly
    old = torch.zeros(1, 4)
    logp = torch.tensor([[0.5, 0.5, -0.5, -0.5]])       # r = e^±0.5 = 1.649 / 0.607
    adv = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    out = clip_indicator(adv, old, logp, cliprange_low=0.2, cliprange_high=0.28)
    # A>0 & r>1.28 -> clipped ; A<0 & r>1 -> kept ; A>0 & r<1 -> kept ;
    # A<0 & r<0.8 -> clipped
    torch.testing.assert_close(out, torch.tensor([[0.0, 1.0, 1.0, 0.0]]))


def test_clip_indicator_is_all_ones_at_ratio_one():
    """First inner epoch: log_prob == old_log_prob, so nothing is clipped."""
    from steer_f.omega_tilde import clip_indicator

    lp = -torch.rand(3, 9)
    out = clip_indicator(torch.randn(3, 9), lp, lp)
    assert torch.equal(out, torch.ones(3, 9))


def test_iclip_zeroes_omega_exactly_where_clipped():
    from steer_f.omega_tilde import clip_indicator, local_omega_signed

    old = torch.zeros(1, 4)
    logp = torch.tensor([[0.5, 0.5, -0.5, -0.5]])
    adv = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    ent = torch.rand(1, 4)
    ind = clip_indicator(adv, old, logp)
    with_clip = local_omega_signed(adv, ent, old, logp, iclip=ind)
    without = local_omega_signed(adv, ent, old, logp)
    assert float(with_clip[0, 0]) == 0.0 and float(with_clip[0, 3]) == 0.0
    torch.testing.assert_close(with_clip[0, 1:3], without[0, 1:3])


def test_use_iclip_defaults_off_so_equivalence_is_preserved():
    """The default must stay the released behaviour, or lambda=0 stops matching."""
    torch.manual_seed(1)
    b, t = 3, 20
    kw = dict(
        advantages=torch.randn(b, t) * 3, entropys=torch.rand(b, t),
        old_log_prob=-torch.rand(b, t), log_prob=-torch.rand(b, t) * 3,
        response_mask=torch.ones(b, t), token_weight_min=0.7,
        token_weight_max=1.0, linear=False, mode="symmetric",
    )
    a_h = torch.randn(b, t)     # the SAME tensor for both calls, or this
                                # compares two different branch scores
    default = compute_token_weights_steerf(**kw, a_h=a_h, lam=0.5)
    explicit_off = compute_token_weights_steerf(**kw, a_h=a_h, lam=0.5, use_iclip=False)
    assert torch.equal(default, explicit_off)


def test_use_iclip_actually_changes_the_weights():
    """A ratio spread wide enough to clip must move the result."""
    torch.manual_seed(2)
    b, t = 3, 40
    kw = dict(
        advantages=torch.randn(b, t) * 3, entropys=torch.rand(b, t),
        old_log_prob=torch.zeros(b, t), log_prob=torch.randn(b, t),  # ratio far from 1
        response_mask=torch.ones(b, t), token_weight_min=0.7,
        token_weight_max=1.0, linear=False, mode="symmetric",
    )
    a_h = torch.randn(b, t)
    off = compute_token_weights_steerf(**kw, a_h=a_h, lam=0.0)
    on = compute_token_weights_steerf(**kw, a_h=a_h, lam=0.0, use_iclip=True)
    assert not torch.allclose(off, on), "I_clip had no effect on a clipping batch"


# ----------------------------------------------------------------------
# use_ratio — Theorem 1's r*A vs the released code's A/pi_old
# ----------------------------------------------------------------------
def test_use_ratio_is_exactly_pi_theta_times_the_released_form():
    """Appendix G: the two differ by precisely one factor of pi_theta."""
    from steer_f.omega_tilde import local_omega_signed

    torch.manual_seed(3)
    b, t = 4, 30
    adv, ent = torch.randn(b, t) * 2, torch.rand(b, t) + 0.1
    old, logp = -torch.rand(b, t) * 4, -torch.rand(b, t) * 4
    released = local_omega_signed(adv, ent, old, logp)
    theorem = local_omega_signed(adv, ent, old, logp, use_ratio=True)
    pi = torch.exp(logp).clamp(1e-8, 1 - 1e-8)
    torch.testing.assert_close(theorem, released * pi, rtol=1e-4, atol=1e-7)


def test_use_ratio_is_one_at_the_first_inner_epoch_times_pi():
    """log_prob == old_log_prob, so r == 1 and Omega == -A * f(x) exactly."""
    from steer_f.omega_tilde import local_omega_signed

    torch.manual_seed(4)
    b, t = 2, 12
    adv, ent = torch.randn(b, t), torch.rand(b, t)
    lp = -torch.rand(b, t) * 3
    got = local_omega_signed(adv, ent, lp, lp, use_ratio=True)
    x = torch.exp(lp).clamp(1e-8, 1 - 1e-8)
    want = -adv * (x * (1 - x) * (torch.log(x) + ent))
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-8)


def test_use_ratio_shrinks_the_tail_that_sets_the_eq9_denominator():
    """A rare token must stop dominating max|Omega|. This is the whole point."""
    from steer_f.omega_tilde import local_omega_signed

    torch.manual_seed(5)
    b, t = 4, 64
    adv, ent = torch.randn(b, t) * 2, torch.rand(b, t) + 0.5
    lp = torch.log(torch.rand(b, t) * 0.5 + 1e-3)
    lp[0, 0] = torch.log(torch.tensor(2e-6))      # the pathological token
    released = local_omega_signed(adv, ent, lp, lp).abs()
    theorem = local_omega_signed(adv, ent, lp, lp, use_ratio=True).abs()
    # released: the 2e-6 token sets the batch max. theorem: it must not.
    assert released.flatten().argmax() == 0
    assert theorem.flatten().argmax() != 0
    r_tail = released.max() / released.median()
    t_tail = theorem.max() / theorem.median()
    assert t_tail < r_tail, f"tail not reduced: {t_tail} vs {r_tail}"


def test_use_ratio_defaults_off_so_equivalence_is_preserved():
    torch.manual_seed(6)
    b, t = 3, 20
    a_h = torch.randn(b, t)
    kw = dict(
        advantages=torch.randn(b, t) * 3, entropys=torch.rand(b, t),
        old_log_prob=-torch.rand(b, t), log_prob=-torch.rand(b, t) * 3,
        response_mask=torch.ones(b, t), token_weight_min=0.7,
        token_weight_max=1.0, linear=False, mode="symmetric", a_h=a_h, lam=0.5,
    )
    assert torch.equal(compute_token_weights_steerf(**kw),
                       compute_token_weights_steerf(**kw, use_ratio=False))
    assert not torch.equal(compute_token_weights_steerf(**kw),
                           compute_token_weights_steerf(**kw, use_ratio=True))


# ----------------------------------------------------------------------
# twg_* : occupancy over the tokens that can actually move the update
#
# At the start of a math RL run the policy fails almost every problem, so most
# GRPO groups score identically, their advantage is 0, and Omega is 0 for every
# token in them.  Those tokens cannot affect the update (pg_losses *
# token_weights is 0 wherever A == 0, and agg_loss divides by response_mask),
# but they dominate the tw_* occupancy metrics and make the reweighting look
# either dead or maximally attenuating depending on the mapping.
# ----------------------------------------------------------------------
def _stats(**kw):
    base = dict(
        entropys=torch.rand(kw.pop("_b"), kw.pop("_t")),
        token_weight_min=0.7, token_weight_max=1.0,
        linear=False, mode="symmetric", a_h=None, lam=0.0, return_stats=True,
    )
    base.update(kw)
    return compute_token_weights_steerf(**base)


def test_all_tied_micro_batch_collapses_to_one_point():
    """The mechanism twg_* exists to expose, pinned for both mappings.

    Every advantage 0 => every Omega 0 => the whole micro-batch ties.  minmax
    hits the 0.02 floor on metric_max and maps everything to 1.0; rank gives
    every tied position average rank 0.5, so metric_max is 0.5 and the
    exponential mapping sends everything to exactly token_weight_min.
    """
    b, t = 8, 30
    kw = dict(
        advantages=torch.zeros(b, t), entropys=torch.rand(b, t),
        old_log_prob=-torch.rand(b, t), log_prob=-torch.rand(b, t),
        response_mask=torch.ones(b, t), token_weight_min=0.7,
        token_weight_max=1.0, linear=False, mode="symmetric", a_h=None, lam=0.0,
    )
    w_minmax = compute_token_weights_steerf(**kw, mapping="minmax")
    w_rank = compute_token_weights_steerf(**kw, mapping="rank")
    assert torch.allclose(w_minmax, torch.ones(b, t)), w_minmax.unique()
    assert torch.allclose(w_rank, torch.full((b, t), 0.7), atol=1e-6), w_rank.unique()


@pytest.mark.parametrize("mapping", ["minmax", "winsor", "rank"])
def test_twg_is_omitted_when_no_token_carries_gradient(mapping):
    """NaN would poison the mean verl takes over micro-batches, so omit."""
    b, t = 4, 12
    _, stats = _stats(
        _b=b, _t=t, advantages=torch.zeros(b, t),
        old_log_prob=-torch.rand(b, t), log_prob=-torch.rand(b, t),
        response_mask=torch.ones(b, t), mapping=mapping,
    )
    assert stats["steerf/adv_zero_frac"] == pytest.approx(1.0)
    assert not [k for k in stats if k.startswith("steerf/twg_")]


def test_twg_matches_the_distribution_over_nonzero_advantage_tokens():
    from steer_f.monitors import token_weight_distribution

    torch.manual_seed(0)
    b, t = 8, 40
    adv = torch.randn(b, t) * 2
    adv[:5] = 0.0                      # 5 of 8 sequences carry no gradient
    mask = torch.ones(b, t)
    kw = dict(
        advantages=adv, entropys=torch.rand(b, t), old_log_prob=-torch.rand(b, t),
        log_prob=-torch.rand(b, t), response_mask=mask, token_weight_min=0.7,
        token_weight_max=1.0, linear=False, mode="symmetric", a_h=None, lam=0.0,
        mapping="rank",
    )
    w, stats = compute_token_weights_steerf(**kw, return_stats=True)

    assert stats["steerf/adv_zero_frac"] == pytest.approx(5 / 8)
    expected = token_weight_distribution(w, mask * (adv != 0).float(), 0.7, 1.0)
    for key, val in expected.items():
        assert stats[key.replace("steerf/tw_", "steerf/twg_")] == pytest.approx(val)


def test_twg_and_tw_disagree_across_a_realistic_batch():
    """The whole point, reproduced the way the live run produces it.

    The floor collapse does NOT come from mixed micro-batches -- there the tied
    zero-Omega block sits at some interior average rank and maps to an interior
    weight.  It comes from micro-batches that are tied *in their entirety*,
    which is the common case when a micro-batch is exactly one GRPO group and
    that group scored 8/8 the same.  Averaged over micro-batches the way verl
    averages logged metrics, tw_ then reports a jammed floor while twg_ -- which
    only sees the micro-batches that carry gradient -- reports a real spread.
    """
    from steer_f.monitors import token_weight_distribution

    torch.manual_seed(1)
    b, t, n_dead, n_live = 8, 60, 6, 2      # 75% dead, as measured on the real run
    tw_low, twg_low, twg_high = [], [], []
    for i in range(n_dead + n_live):
        adv = torch.zeros(b, t) if i < n_dead else torch.randn(b, t) * 2
        mask = torch.ones(b, t)
        w, stats = compute_token_weights_steerf(
            advantages=adv, entropys=torch.rand(b, t),
            old_log_prob=-torch.rand(b, t), log_prob=-torch.rand(b, t),
            response_mask=mask, token_weight_min=0.7, token_weight_max=1.0,
            linear=False, mode="symmetric", a_h=None, lam=0.0,
            mapping="rank", return_stats=True,
        )
        tw_low.append(token_weight_distribution(w, mask, 0.7, 1.0)["steerf/tw_frac_low"])
        if "steerf/twg_frac_low" in stats:
            twg_low.append(stats["steerf/twg_frac_low"])
            twg_high.append(stats["steerf/twg_frac_high"])

    mean = lambda x: sum(x) / len(x)
    assert len(twg_low) == n_live, "dead micro-batches must not contribute"
    # tw_ averages in the fully-tied batches, every one of which is pinned at 0.7.
    assert mean(tw_low) >= n_dead / (n_dead + n_live)
    # twg_ sees only the live ones, where the band is genuinely occupied.
    assert mean(twg_low) < 0.6
    assert mean(twg_high) > 0.1


@pytest.mark.parametrize("mapping", ["minmax", "winsor", "rank"])
def test_twg_does_not_perturb_the_returned_weights(mapping):
    """Stats are observational: return_stats must not change the tensor."""
    torch.manual_seed(2)
    b, t = 4, 25
    adv = torch.randn(b, t)
    adv[0] = 0.0
    kw = dict(
        advantages=adv, entropys=torch.rand(b, t), old_log_prob=-torch.rand(b, t),
        log_prob=-torch.rand(b, t), response_mask=torch.ones(b, t),
        token_weight_min=0.7, token_weight_max=1.0, linear=False,
        mode="symmetric", a_h=None, lam=0.0, mapping=mapping,
    )
    bare = compute_token_weights_steerf(**kw)
    withstats, _ = compute_token_weights_steerf(**kw, return_stats=True)
    assert torch.equal(bare, withstats)
