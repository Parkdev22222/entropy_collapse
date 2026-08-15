"""Plan §8.3 — A_H clip boundaries, normalisation degeneracy, sign semantics."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from steer_f.omega_tilde import (  # noqa: E402
    SteerFConfig,
    compute_omega_tilde,
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
