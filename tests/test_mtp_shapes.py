"""계획서 §8.2: MTP 헤드 shape/dtype/디바이스 정합성."""

import pytest
import torch
import torch.nn as nn

from steer_f.mtp_heads import MTPHeads, entropy_from_logits

H, V, K = 32, 61, 4


def make(dtype=torch.float32, num_heads=K, residual=True):
    heads = MTPHeads(hidden_size=H, num_heads=num_heads, head_hidden=48, vocab_size=V, residual=residual).to(dtype)
    lm_head = nn.Linear(H, V, bias=False).to(dtype)
    return heads, lm_head


def test_forward_logits_shape():
    heads, lm_head = make()
    hidden = torch.randn(3, 7, H)
    logits = heads.forward_logits(hidden, lm_head)
    assert logits.shape == (K, 3, 7, V)


def test_forward_logits_packed_layout():
    """verl rmpad 경로의 (N, H) packed 히든도 그대로 통과해야 한다."""
    heads, lm_head = make()
    logits = heads.forward_logits(torch.randn(29, H), lm_head)
    assert logits.shape == (K, 29, V)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_forward_entropy_matches_logits_path(dtype):
    heads, lm_head = make(dtype)
    hidden = torch.randn(2, 5, H, dtype=dtype)
    ent_chunked = heads.forward_entropy(hidden, lm_head, chunk_size=3)
    ent_direct = entropy_from_logits(heads.forward_logits(hidden, lm_head))
    assert ent_chunked.shape == (K, 2, 5)
    assert ent_chunked.dtype == torch.float32  # 엔트로피는 항상 fp32로 축약
    torch.testing.assert_close(ent_chunked, ent_direct, atol=1e-2 if dtype is torch.bfloat16 else 1e-5, rtol=1e-2)


def test_forward_entropy_respects_kappa():
    heads, lm_head = make()
    ent = heads.forward_entropy(torch.randn(2, 5, H), lm_head, kappa=2)
    assert ent.shape == (2, 2, 5)
    with pytest.raises(ValueError):
        heads.forward_entropy(torch.randn(2, 5, H), lm_head, kappa=K + 1)


def test_zero_init_residual_matches_base_model():
    """residual + 0 초기화 → 학습 전 예보가 본체 next-token 분포와 동일."""
    heads, lm_head = make(residual=True)
    hidden = torch.randn(2, 5, H)
    logits = heads.forward_logits(hidden, lm_head)
    base = lm_head(hidden)
    for k in range(K):
        torch.testing.assert_close(logits[k], base)


def test_ce_loss_targets_are_shifted():
    """헤드 k의 타깃이 y_{t+k}인지 — 완전 예측 가능한 시퀀스로 확인."""
    heads, lm_head = make(num_heads=2)
    b, t = 2, 9
    input_ids = torch.arange(t).unsqueeze(0).expand(b, t) % V
    hidden = torch.randn(b, t, H)
    loss, per_head = heads.ce_loss(hidden, lm_head, input_ids)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert set(per_head) == {"mtp/ce_head_1", "mtp/ce_head_2"}


def test_ce_loss_with_mask_and_short_sequence():
    heads, lm_head = make(num_heads=3)
    hidden = torch.randn(2, 2, H)
    input_ids = torch.randint(0, V, (2, 2))
    mask = torch.tensor([[1, 1], [1, 0]])
    loss, per_head = heads.ce_loss(hidden, lm_head, input_ids, mask=mask)
    # T=2 이므로 헤드 1만 타깃이 존재한다.
    assert set(per_head) == {"mtp/ce_head_1"}
    assert torch.isfinite(loss)


def test_ce_loss_no_valid_targets_returns_zero():
    heads, lm_head = make()
    hidden = torch.randn(2, 6, H)
    loss, per_head = heads.ce_loss(hidden, lm_head, torch.randint(0, V, (2, 6)), mask=torch.zeros(2, 6))
    assert float(loss) == 0.0 and per_head == {}


def test_entropy_from_logits_bounds():
    logits = torch.randn(50, V)
    ent = entropy_from_logits(logits)
    assert torch.all(ent >= -1e-5)
    assert torch.all(ent <= torch.log(torch.tensor(float(V))) + 1e-5)
    uniform = entropy_from_logits(torch.zeros(1, V))
    torch.testing.assert_close(uniform, torch.log(torch.tensor([float(V)])))


def test_invalid_num_heads():
    with pytest.raises(ValueError):
        MTPHeads(hidden_size=H, num_heads=0)
