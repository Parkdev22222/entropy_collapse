"""계획서 §8.1 필수 테스트: λ=0에서 STEER-F ≡ 순수 STEER.

`tests/reference_steer.py`는 STEER 원본에서 자동 추출한 사본이다
(scripts/extract_reference.py). 이 테스트가 깨지면 통합 계약이 깨진 것이다.
"""

import itertools

import pytest
import torch

from steer_f.token_weights import compute_token_weights_steerf

from .reference_steer import compute_token_weights as reference_compute_token_weights


def make_batch(seed: int, b: int = 6, t: int = 24, ragged: bool = True):
    g = torch.Generator().manual_seed(seed)
    advantages = torch.randn(b, t, generator=g)
    entropys = torch.rand(b, t, generator=g) * 3.0
    old_log_prob = -torch.rand(b, t, generator=g) * 6.0
    log_prob = old_log_prob + torch.randn(b, t, generator=g) * 0.1
    log_prob = log_prob.clamp(max=-1e-4)
    response_mask = torch.ones(b, t)
    if ragged:
        for i in range(b):
            keep = int(t * (0.4 + 0.6 * torch.rand(1, generator=g).item()))
            response_mask[i, keep:] = 0.0
    return advantages, entropys, old_log_prob, log_prob, response_mask


MODES = ["symmetric", "asymmetric"]
LINEARS = [True, False]


@pytest.mark.parametrize("seed,mode,linear", list(itertools.product([0, 1, 2, 3], MODES, LINEARS)))
def test_lambda_zero_bitwise_identical(seed, mode, linear):
    """λ=0 경로는 원본과 비트 단위로 동일해야 한다."""
    adv, ent, olp, lp, mask = make_batch(seed)
    kw = dict(
        advantages=adv,
        entropys=ent,
        old_log_prob=olp,
        log_prob=lp,
        response_mask=mask,
        token_weight_min=0.8,
        token_weight_max=1.2,
        linear=linear,
        mode=mode,
    )
    ref = reference_compute_token_weights(**kw)
    got = compute_token_weights_steerf(lam=0.0, **kw)
    assert torch.equal(ref, got), (ref - got).abs().max()


def test_lambda_zero_ignores_forecast_inputs():
    """λ=0이면 예보 텐서를 넘겨도 결과가 바뀌지 않는다."""
    adv, ent, olp, lp, mask = make_batch(7)
    kw = dict(
        advantages=adv, entropys=ent, old_log_prob=olp, log_prob=lp,
        response_mask=mask, token_weight_min=0.8, token_weight_max=1.2,
    )
    ref = reference_compute_token_weights(**kw)
    got = compute_token_weights_steerf(
        h_togo_vals=torch.rand_like(adv) * 10,
        baseline_h_togo=torch.rand_like(adv),
        lam=0.0,
        **kw,
    )
    assert torch.equal(ref, got)


@pytest.mark.parametrize("mode", MODES)
def test_zscore_is_affine_invariant_under_linear_mapping(mode):
    """code_map §6: linear=True의 min-max 매핑은 아핀 변환에 불변이다.

    따라서 λ=0에서 z-정규화를 켜든 끄든 α가 같다. 이 성질이 깨지면 Ω̃ 도입 시
    "밴드 재산정"이 실제로 필요해진다는 신호이므로 명시적으로 잠가 둔다.
    """
    adv, ent, olp, lp, mask = make_batch(11)
    kw = dict(
        advantages=adv, entropys=ent, old_log_prob=olp, log_prob=lp,
        response_mask=mask, token_weight_min=0.8, token_weight_max=1.2,
        linear=True, mode=mode,
    )
    ref = reference_compute_token_weights(**kw)

    # λ를 극미값으로 두어 z-norm 경로를 강제로 통과시킨다.
    got = compute_token_weights_steerf(
        h_togo_vals=torch.zeros_like(adv),
        baseline_h_togo=torch.zeros_like(adv),
        lam=1e-12,
        **kw,
    )
    torch.testing.assert_close(ref, got, atol=2e-4, rtol=2e-4)


def test_nonzero_lambda_changes_weights():
    """λ>0이고 A_H가 비자명하면 실제로 가중치가 달라져야 한다 (훅이 죽어있지 않은지)."""
    adv, ent, olp, lp, mask = make_batch(5)
    kw = dict(
        advantages=adv, entropys=ent, old_log_prob=olp, log_prob=lp,
        response_mask=mask, token_weight_min=0.8, token_weight_max=1.2,
    )
    ref = reference_compute_token_weights(**kw)
    g = torch.Generator().manual_seed(99)
    got = compute_token_weights_steerf(
        h_togo_vals=torch.rand(adv.shape, generator=g) * 4,
        baseline_h_togo=torch.rand(adv.shape, generator=g) * 4,
        lam=1.0,
        **kw,
    )
    valid = mask.bool()
    assert not torch.allclose(ref[valid], got[valid])


def test_stats_reported_for_nonzero_lambda():
    adv, ent, olp, lp, mask = make_batch(13)
    _, stats = compute_token_weights_steerf(
        advantages=adv, entropys=ent, old_log_prob=olp, log_prob=lp,
        response_mask=mask, token_weight_min=0.8, token_weight_max=1.2,
        h_togo_vals=torch.rand_like(adv) * 4,
        baseline_h_togo=torch.rand_like(adv) * 4,
        lam=0.5,
        return_stats=True,
    )
    assert 0.0 <= stats["steerf/future_frac"] <= 1.0
    assert stats["steerf/lam"] == 0.5
    assert "steerf/a_h_absmean" in stats


def test_all_masked_batch_returns_zeros():
    adv, ent, olp, lp, _ = make_batch(3)
    mask = torch.zeros_like(adv)
    ref = reference_compute_token_weights(
        advantages=adv, entropys=ent, old_log_prob=olp, log_prob=lp, response_mask=mask
    )
    got = compute_token_weights_steerf(
        advantages=adv, entropys=ent, old_log_prob=olp, log_prob=lp, response_mask=mask,
        h_togo_vals=torch.rand_like(adv), baseline_h_togo=torch.rand_like(adv), lam=1.0,
    )
    assert torch.equal(ref, got)
    assert torch.count_nonzero(got) == 0
