"""계획서 §8.4 + 예보 유틸 전반: sibling baseline, 정렬(shift), 캘리브레이션."""

import pytest
import torch

from steer_f.entropy_forecast import (
    HeadCalibration,
    entropy_advantage,
    h_togo,
    masked_zscore,
    sibling_mask,
)


# ----------------------------------------------------------------- h_togo
def test_h_togo_discounted_sum():
    k, b, t = 3, 2, 4
    ent = torch.ones(k, b, t)
    out = h_togo(ent, kappa=3, gamma_h=0.5)
    expected = 0.5 + 0.25 + 0.125
    torch.testing.assert_close(out, torch.full((b, t), expected))


def test_h_togo_kappa_truncates():
    ent = torch.arange(4, dtype=torch.float32).view(4, 1, 1) + 1.0
    out = h_togo(ent, kappa=2, gamma_h=1.0)
    torch.testing.assert_close(out, torch.tensor([[3.0]]))  # 1 + 2


def test_h_togo_kappa_bounds():
    with pytest.raises(ValueError):
        h_togo(torch.ones(2, 1, 1), kappa=3, gamma_h=1.0)
    with pytest.raises(ValueError):
        h_togo(torch.ones(2, 1, 1), kappa=0, gamma_h=1.0)


def test_h_togo_from_logits():
    logits = torch.zeros(2, 1, 1, 8)  # 균등분포 → H = log 8
    out = h_togo(logits, kappa=2, gamma_h=1.0, is_logits=True)
    torch.testing.assert_close(out, torch.full((1, 1), 2 * torch.log(torch.tensor(8.0))))


# -------------------------------------------------------- calibration
def test_calibration_recovers_affine_relation():
    pred = torch.randn(3, 500) * 2 + 5
    scale = torch.tensor([0.5, 0.8, 1.3]).view(3, 1)
    bias = torch.tensor([-1.0, 0.0, 2.0]).view(3, 1)
    target = pred * scale + bias
    calib = HeadCalibration.fit(pred, target)
    torch.testing.assert_close(torch.tensor(calib.scale), scale.view(-1), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(torch.tensor(calib.bias), bias.view(-1), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(calib.apply(pred), target, atol=1e-3, rtol=1e-3)


def test_calibration_corrects_far_head_overestimation():
    """먼 헤드가 엔트로피를 과대추정하는 편향을 보정이 실제로 줄이는지."""
    truth = torch.rand(2, 400) * 2
    pred = torch.stack([truth[0], truth[1] * 1.0 + 1.5])  # 헤드 2가 +1.5 과대추정
    before = (pred - truth).abs().mean(dim=1)
    calib = HeadCalibration.fit(pred, truth)
    after = (calib.apply(pred) - truth).abs().mean(dim=1)
    assert after[1] < before[1] * 0.1


def test_calibration_degenerate_input_is_identity():
    calib = HeadCalibration.fit(torch.full((2, 10), 1.0), torch.randn(2, 10))
    assert calib.scale == [1.0, 1.0] and calib.bias == [0.0, 0.0]


def test_calibration_identity_and_roundtrip():
    c = HeadCalibration.identity(4)
    x = torch.randn(4, 3, 3)
    torch.testing.assert_close(c.apply(x), x)
    assert HeadCalibration.from_dict(c.to_dict()).scale == c.scale


def test_calibration_too_few_heads():
    with pytest.raises(ValueError):
        HeadCalibration.identity(2).apply(torch.randn(4, 3))


# ------------------------------------------------------- sibling mask
def test_sibling_mask_diverging_rollouts():
    # G=2, 위치 2에서 갈라짐
    ids = torch.tensor([[1, 2, 3, 4], [1, 2, 9, 9]])
    sib = sibling_mask(ids, group_size=2)
    assert sib.shape == (1, 2, 2, 4)
    # t=0,1,2 에서는 prefix(y_<t)가 동일 → sibling
    assert sib[0, 0, 1, 0] and sib[0, 0, 1, 1] and sib[0, 0, 1, 2]
    # t=3 에서는 y_2가 이미 다르므로 sibling 아님
    assert not sib[0, 0, 1, 3]
    # 자기 자신은 항상 sibling
    assert sib[0, 0, 0].all() and sib[0, 1, 1].all()


def test_sibling_mask_padding_counts_as_divergence():
    ids = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    sib = sibling_mask(ids, group_size=2, response_mask=mask)
    assert sib[0, 0, 1, 2]        # t=2: prefix(0,1) 둘 다 유효 & 동일
    assert not sib[0, 0, 1, 3]    # t=3: y_2가 한쪽에서 패딩 → 갈라짐


def test_sibling_mask_requires_divisible_batch():
    with pytest.raises(ValueError):
        sibling_mask(torch.zeros(3, 4, dtype=torch.long), group_size=2)


# --------------------------------------------------- entropy advantage
def test_shift_aligns_branch_value_to_sampled_token():
    """A_H는 'y_t를 조건으로 한' 미래 가치여야 한다 → h_togo를 한 칸 당긴다."""
    h = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    a_h, _ = entropy_advantage(h, group_size=1, baseline="none", shift=True)
    torch.testing.assert_close(a_h, torch.tensor([[1.0, 2.0, 3.0, 3.0]]))
    a_h_ns, _ = entropy_advantage(h, group_size=1, baseline="none", shift=False)
    torch.testing.assert_close(a_h_ns, h)


def test_shift_does_not_pull_from_padding():
    h = torch.tensor([[1.0, 2.0, 99.0, 99.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    a_h, _ = entropy_advantage(h, group_size=1, baseline="none", response_mask=mask)
    # t=1의 다음 위치가 패딩이므로 자기 값 유지, 패딩 위치는 0
    torch.testing.assert_close(a_h, torch.tensor([[2.0, 2.0, 0.0, 0.0]]))


def test_single_sibling_gives_zero_advantage():
    """계획서 §8.4: prefix 공유 그룹이 자기 자신뿐이면 A_H = 0."""
    ids = torch.tensor([[1, 5, 6, 7], [1, 8, 9, 3]])  # t=1부터 갈라짐
    h = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    a_h, _ = entropy_advantage(h, response_ids=ids, group_size=2, baseline="sibling")
    # t>=2 에서는 서로 sibling이 아니므로 baseline = 자기 자신 → A_H = 0
    assert a_h[0, 2] == 0.0 and a_h[0, 3] == 0.0
    assert a_h[1, 2] == 0.0 and a_h[1, 3] == 0.0


def test_sibling_advantage_sums_to_zero_when_all_siblings():
    """전원이 sibling인 위치에서는 A_H의 그룹 합이 0."""
    ids = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    h = torch.rand(4, 4)
    a_h, _ = entropy_advantage(h, response_ids=ids, group_size=4, baseline="sibling")
    torch.testing.assert_close(a_h.sum(dim=0), torch.zeros(4), atol=1e-5, rtol=1e-5)


def test_group_baseline_matches_sibling_when_all_identical():
    ids = torch.ones(4, 5, dtype=torch.long)
    h = torch.rand(4, 5)
    a_sib, _ = entropy_advantage(h, response_ids=ids, group_size=4, baseline="sibling")
    a_grp, _ = entropy_advantage(h, response_ids=ids, group_size=4, baseline="group")
    torch.testing.assert_close(a_sib, a_grp, atol=1e-5, rtol=1e-5)


def test_group_and_sibling_differ_when_rollouts_diverge():
    ids = torch.tensor([[1, 2, 3], [1, 9, 9], [4, 4, 4], [4, 4, 4]])
    h = torch.rand(4, 3) * 5
    a_sib, _ = entropy_advantage(h, response_ids=ids, group_size=4, baseline="sibling")
    a_grp, _ = entropy_advantage(h, response_ids=ids, group_size=4, baseline="group")
    assert not torch.allclose(a_sib, a_grp)


def test_sibling_requires_ids():
    with pytest.raises(ValueError):
        entropy_advantage(torch.rand(2, 3), group_size=2, baseline="sibling")


def test_unknown_baseline():
    with pytest.raises(ValueError):
        entropy_advantage(torch.rand(2, 3), group_size=1, baseline="bogus")


def test_masked_zscore_zero_variance():
    x = torch.full((3, 4), 2.0)
    assert torch.count_nonzero(masked_zscore(x, torch.ones_like(x))) == 0
