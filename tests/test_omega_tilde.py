"""계획서 §8.3: A_H 클리핑 경계, z_norm 0-분산 안전성, 결합 규칙."""

import pytest
import torch

from steer_f.omega_tilde import compute_omega_tilde, normalize


def test_normalize_zero_variance_returns_zeros():
    x = torch.full((4, 5), 3.14)
    out = normalize(x, torch.ones_like(x))
    assert torch.count_nonzero(out) == 0


def test_normalize_all_masked_returns_zeros():
    x = torch.randn(4, 5)
    out = normalize(x, torch.zeros_like(x))
    assert torch.count_nonzero(out) == 0


def test_normalize_ignores_masked_values():
    """마스크 밖의 거대한 값이 통계를 오염시키면 안 된다."""
    x = torch.randn(2, 6)
    mask = torch.ones(2, 6)
    mask[:, 4:] = 0
    x_polluted = x.clone()
    x_polluted[:, 4:] = 1e6
    torch.testing.assert_close(normalize(x, mask), normalize(x_polluted, mask))


def test_normalize_zeros_outside_mask():
    x = torch.randn(3, 8) + 5.0
    mask = torch.zeros(3, 8)
    mask[:, :4] = 1
    out = normalize(x, mask)
    assert torch.count_nonzero(out[:, 4:]) == 0


def test_normalize_robust_mode():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(200, 1, generator=g)
    x[0] = 1e6  # 이상치 하나
    z = normalize(x, torch.ones_like(x), how="z")
    r = normalize(x, torch.ones_like(x), how="robust")
    # 이상치가 z 스케일을 삼켜 나머지가 0 근처로 압축된다. robust는 안 그렇다.
    # (code_map §6.1의 min-max 이상치 민감성 완화 옵션이 실제로 작동하는지 확인)
    assert r[1:].abs().mean() > z[1:].abs().mean() * 5


def test_normalize_unknown_mode():
    with pytest.raises(ValueError):
        normalize(torch.randn(2, 2), None, how="nope")


def test_a_h_clipping_bound():
    b, t, c = 3, 6, 0.5
    mask = torch.ones(b, t)
    _, stats = compute_omega_tilde(
        omega_local=torch.randn(b, t),
        w=torch.randn(b, t),
        pi_sampled=torch.rand(b, t) * 0.8 + 0.1,
        h_togo_vals=torch.full((b, t), 100.0),   # 극단적으로 큰 차이
        baseline_h_togo=torch.zeros(b, t),
        response_mask=mask,
        lam=1.0,
        clip_c=c,
        return_stats=True,
    )
    assert stats["steerf/a_h_absmean"] == pytest.approx(c)
    assert stats["steerf/a_h_clip_frac"] == pytest.approx(1.0)


def test_symmetric_mode_uses_magnitude():
    """symmetric에서 미래 항의 부호를 뒤집어도 결과가 같아야 한다 (크기만 사용)."""
    b, t = 3, 6
    kw = dict(
        omega_local=torch.rand(b, t),
        w=torch.randn(b, t),
        pi_sampled=torch.rand(b, t) * 0.8 + 0.1,
        response_mask=torch.ones(b, t),
        mode="symmetric",
        lam=1.0,
    )
    h = torch.rand(b, t) * 2
    base = torch.rand(b, t) * 2
    a = compute_omega_tilde(h_togo_vals=h, baseline_h_togo=base, **kw)
    b_ = compute_omega_tilde(h_togo_vals=base, baseline_h_togo=h, **kw)  # A_H 부호 반전
    torch.testing.assert_close(a, b_)


def test_asymmetric_mode_keeps_sign():
    b, t = 3, 6
    kw = dict(
        omega_local=torch.randn(b, t),
        w=torch.ones(b, t),
        pi_sampled=torch.full((b, t), 0.5),
        response_mask=torch.ones(b, t),
        mode="asymmetric",
        lam=1.0,
    )
    h = torch.rand(b, t) * 2
    base = torch.rand(b, t) * 2
    a = compute_omega_tilde(h_togo_vals=h, baseline_h_togo=base, **kw)
    b_ = compute_omega_tilde(h_togo_vals=base, baseline_h_togo=h, **kw)
    assert not torch.allclose(a, b_)


def test_lambda_zero_passthrough_identity():
    om = torch.randn(4, 5)
    out = compute_omega_tilde(
        omega_local=om, w=torch.randn(4, 5), pi_sampled=torch.rand(4, 5),
        h_togo_vals=None, baseline_h_togo=None, lam=0.0,
    )
    assert out is om


def test_nonzero_lambda_requires_forecast():
    with pytest.raises(ValueError):
        compute_omega_tilde(
            omega_local=torch.randn(2, 3), w=torch.randn(2, 3), pi_sampled=torch.rand(2, 3),
            h_togo_vals=None, baseline_h_togo=None, lam=0.5,
        )


def test_future_frac_scales_with_lambda():
    b, t = 4, 8
    kw = dict(
        omega_local=torch.randn(b, t),
        w=torch.randn(b, t),
        pi_sampled=torch.rand(b, t) * 0.8 + 0.1,
        h_togo_vals=torch.rand(b, t) * 3,
        baseline_h_togo=torch.rand(b, t) * 3,
        response_mask=torch.ones(b, t),
        return_stats=True,
    )
    _, small = compute_omega_tilde(lam=0.1, **kw)
    _, big = compute_omega_tilde(lam=2.0, **kw)
    assert big["steerf/future_frac"] > small["steerf/future_frac"]


def test_non_finite_visit_term_is_zeroed():
    b, t = 2, 4
    w = torch.randn(b, t)
    w[0, 0] = float("inf")
    out = compute_omega_tilde(
        omega_local=torch.randn(b, t), w=w, pi_sampled=torch.rand(b, t) * 0.5 + 0.2,
        h_togo_vals=torch.rand(b, t), baseline_h_togo=torch.rand(b, t),
        response_mask=torch.ones(b, t), lam=1.0,
    )
    assert torch.isfinite(out).all()


def test_invalid_mode():
    with pytest.raises(ValueError):
        compute_omega_tilde(
            omega_local=torch.randn(2, 2), w=torch.randn(2, 2), pi_sampled=torch.rand(2, 2),
            h_togo_vals=torch.rand(2, 2), baseline_h_togo=torch.rand(2, 2), lam=1.0, mode="bogus",
        )
