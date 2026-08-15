"""verl 통합 헬퍼의 형상/의미 계약 테스트 (verl·GPU 불필요, 모의 텐서 사용)."""

import pytest
import torch
import torch.nn as nn

from steer_f.mtp_heads import MTPHeads
from steer_f.verl_integration import (
    attach_entropy_advantage,
    compute_h_togo_packed,
    measure_forecast_drift,
    mtp_auxiliary_loss,
)

H, V, K = 24, 41, 4


def make_heads():
    return MTPHeads(hidden_size=H, num_heads=K, head_hidden=32, vocab_size=V), nn.Linear(H, V, bias=False)


def test_compute_h_togo_packed_layout():
    heads, lm_head = make_heads()
    packed = torch.randn(37, H)
    out = compute_h_togo_packed(heads, packed, lm_head, kappa=2, gamma_h=0.85)
    assert out.shape == (37,)
    assert torch.isfinite(out).all()


def test_compute_h_togo_batched_layout():
    heads, lm_head = make_heads()
    out = compute_h_togo_packed(heads, torch.randn(3, 9, H), lm_head, kappa=3, gamma_h=0.7)
    assert out.shape == (3, 9)


def test_compute_h_togo_is_positive_and_discount_monotone():
    """엔트로피는 음이 아니므로 H_togo는 κ에 대해 단조 증가한다 (γ_H > 0)."""
    heads, lm_head = make_heads()
    hidden = torch.randn(2, 6, H)
    a = compute_h_togo_packed(heads, hidden, lm_head, kappa=1, gamma_h=1.0)
    b = compute_h_togo_packed(heads, hidden, lm_head, kappa=3, gamma_h=1.0)
    assert torch.all(a >= 0)
    assert torch.all(b >= a - 1e-5)


def test_attach_entropy_advantage_fills_baseline():
    b, t, g = 4, 5, 2
    td = {
        "h_togo": torch.rand(b, t) * 3,
        "responses": torch.randint(0, 7, (b, t)),
        "response_mask": torch.ones(b, t),
    }
    stats = attach_entropy_advantage(td, group_size=g)
    assert td["baseline_h_togo"].shape == (b, t)
    assert "steerf/h_togo_mean" in stats
    assert 0.0 <= stats["steerf/a_h_zero_frac"] <= 1.0


def test_attach_entropy_advantage_accepts_dataproto_like():
    class Fake:
        def __init__(self, td):
            self.batch = td

    td = {
        "h_togo": torch.rand(2, 4),
        "responses": torch.ones(2, 4, dtype=torch.long),
        "response_mask": torch.ones(2, 4),
    }
    attach_entropy_advantage(Fake(td), group_size=2)
    assert "baseline_h_togo" in td


def test_attach_entropy_advantage_derives_mask_from_attention():
    b, t = 2, 4
    td = {
        "h_togo": torch.rand(b, t),
        "responses": torch.ones(b, t, dtype=torch.long),
        "attention_mask": torch.ones(b, t + 3),  # prompt + response
    }
    attach_entropy_advantage(td, group_size=2)
    assert td["baseline_h_togo"].shape == (b, t)


def test_identical_rollouts_give_zero_advantage():
    """완전히 같은 rollout끼리는 A_H = 0.

    baseline은 **shift된** h_togo와 같아진다 (entropy_advantage가 y_t 조건 정렬을
    위해 한 칸 당기기 때문). 따라서 h 자체가 아니라 A_H가 0인지로 확인한다.
    """
    b, t = 4, 5
    h = torch.rand(1, t).expand(b, t).contiguous()
    td = {
        "h_togo": h,
        "responses": torch.ones(b, t, dtype=torch.long),
        "response_mask": torch.ones(b, t),
    }
    stats = attach_entropy_advantage(td, group_size=4)
    assert stats["steerf/a_h_zero_frac"] == pytest.approx(1.0)
    shifted = torch.cat([h[:, 1:], h[:, -1:]], dim=1)
    torch.testing.assert_close(td["baseline_h_togo"], shifted, atol=1e-5, rtol=1e-5)


def test_mtp_auxiliary_loss_scales_with_beta():
    heads, lm_head = make_heads()
    hidden = torch.randn(2, 8, H)
    ids = torch.randint(0, V, (2, 8))
    mask = torch.ones(2, 8)
    l1, s1 = mtp_auxiliary_loss(heads, hidden, lm_head, ids, mask, beta_mtp=0.05)
    l2, _ = mtp_auxiliary_loss(heads, hidden, lm_head, ids, mask, beta_mtp=0.10)
    assert float(l2.detach()) == pytest.approx(2 * float(l1.detach()), rel=1e-4)
    assert "steerf/mtp_ce" in s1


def test_mtp_auxiliary_loss_zero_beta_is_freeze():
    """β=0은 헤드 freeze와 동치 (Ablation A7) — 손실 0, 통계 없음."""
    heads, lm_head = make_heads()
    loss, stats = mtp_auxiliary_loss(
        heads, torch.randn(2, 6, H), lm_head, torch.randint(0, V, (2, 6)), torch.ones(2, 6), beta_mtp=0.0
    )
    assert float(loss) == 0.0 and stats == {}


def test_mtp_auxiliary_loss_is_differentiable_wrt_heads_only():
    heads, lm_head = make_heads()
    hidden = torch.randn(2, 8, H, requires_grad=True)
    loss, _ = mtp_auxiliary_loss(
        heads, hidden, lm_head, torch.randint(0, V, (2, 8)), torch.ones(2, 8), beta_mtp=1.0
    )
    loss.backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in heads.projs.parameters())


def test_measure_forecast_drift_zero_when_head_matches_policy():
    """0-초기화 residual 헤드는 본체와 같은 분포 → 정렬이 맞으면 KL ≈ 0."""
    heads, lm_head = make_heads()
    hidden = torch.randn(2, 7, H)
    policy_logits = lm_head(hidden)
    # 헤드 +1(위치 t)은 정책의 위치 t+1과 비교된다. 헤드가 항등이면 KL은
    # H(t) vs H(t+1) 차이만큼 나온다. 히든을 상수로 두면 정확히 0이어야 한다.
    const = torch.randn(1, 1, H).expand(2, 7, H).contiguous()
    kl = measure_forecast_drift(heads, const, lm_head, lm_head(const))
    assert kl == pytest.approx(0.0, abs=1e-4)
    assert measure_forecast_drift(heads, hidden, lm_head, policy_logits) >= 0.0


def test_measure_forecast_drift_short_sequence():
    heads, lm_head = make_heads()
    assert measure_forecast_drift(heads, torch.randn(2, 1, H), lm_head, torch.randn(2, 1, V)) == 0.0


def test_measure_forecast_drift_all_masked():
    heads, lm_head = make_heads()
    kl = measure_forecast_drift(
        heads, torch.randn(2, 5, H), lm_head, torch.randn(2, 5, V), response_mask=torch.zeros(2, 5)
    )
    assert kl == 0.0
