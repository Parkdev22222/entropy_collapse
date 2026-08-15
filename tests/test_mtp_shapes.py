"""Plan §8.2 — MTP head shape / dtype / device coherence.

Uses a tiny vocabulary so the full logit stack is affordable; the production
path (`forecast_entropy`) is checked against that stack for agreement, which
is what lets us trust the chunked version at V=152k where the stack cannot
exist.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from steer_f.mtp_heads import MTPHeads, entropy_from_logits, mtp_ce_loss  # noqa: E402

H, V, K = 32, 50, 4


def make_heads(**kw):
    kw.setdefault("num_heads", K)
    kw.setdefault("head_hidden", 16)
    return MTPHeads(hidden_size=H, vocab_size=V, **kw)


def test_forward_shape():
    heads = make_heads()
    lm_head = nn.Linear(H, V, bias=False)
    out = heads(torch.randn(3, 7, H), lm_head)
    assert out.shape == (K, 3, 7, V)


def test_forecast_entropy_shape_and_range():
    heads = make_heads()
    lm_head = nn.Linear(H, V, bias=False)
    ent = heads.forecast_entropy(torch.randn(3, 7, H), lm_head)
    assert ent.shape == (K, 3, 7)
    assert (ent >= -1e-5).all(), "entropy must be non-negative"
    assert (ent <= torch.log(torch.tensor(float(V))) + 1e-4).all(), "entropy exceeds log(V)"


def test_forecast_entropy_matches_full_stack():
    """The chunked, logit-free path equals the naive one."""
    heads = make_heads()
    lm_head = nn.Linear(H, V, bias=False)
    hidden = torch.randn(2, 9, H)
    naive = entropy_from_logits(heads(hidden, lm_head).float())
    chunked = heads.forecast_entropy(hidden, lm_head, chunk_size=4)
    torch.testing.assert_close(chunked, naive, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 1000])
def test_chunking_is_invariant(chunk_size):
    heads = make_heads()
    lm_head = nn.Linear(H, V, bias=False)
    hidden = torch.randn(2, 9, H)
    ref = heads.forecast_entropy(hidden, lm_head, chunk_size=10_000)
    got = heads.forecast_entropy(hidden, lm_head, chunk_size=chunk_size)
    torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-6)


def test_packed_rmpad_layout():
    """verl runs with remove_padding, so [N, H] input must work."""
    heads = make_heads()
    lm_head = nn.Linear(H, V, bias=False)
    ent = heads.forecast_entropy(torch.randn(23, H), lm_head)
    assert ent.shape == (K, 23)


def test_kappa_selects_prefix_of_heads():
    heads = make_heads()
    lm_head = nn.Linear(H, V, bias=False)
    hidden = torch.randn(2, 5, H)
    full = heads.forecast_entropy(hidden, lm_head)
    part = heads.forecast_entropy(hidden, lm_head, kappa=2)
    assert part.shape == (2, 2, 5)
    torch.testing.assert_close(part, full[:2])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dtype_handling(dtype):
    """bf16 params with a bf16 body; entropies still come back in fp32."""
    heads = make_heads(dtype=dtype)
    lm_head = nn.Linear(H, V, bias=False).to(dtype)
    ent = heads.forecast_entropy(torch.randn(2, 5, H, dtype=dtype), lm_head)
    assert ent.dtype == torch.float32
    assert torch.isfinite(ent).all()


def test_hidden_dtype_mismatch_is_tolerated():
    """FSDP mixed precision can hand us fp32 hidden with bf16 head params."""
    heads = make_heads(dtype=torch.bfloat16)
    lm_head = nn.Linear(H, V, bias=False).to(torch.bfloat16)
    ent = heads.forecast_entropy(torch.randn(2, 5, H, dtype=torch.float32), lm_head)
    assert torch.isfinite(ent).all()


def test_zero_init_makes_heads_agree_with_next_token_predictor():
    """The Medusa zero-init prior: at step 0 every head IS the +1 head.

    This is what stops H_togo being ~log(V) noise before warm-up converges,
    and it means a run started from an untrained head degrades to a myopic
    forecast rather than to garbage.
    """
    heads = make_heads(residual=True, zero_init_output=True)
    lm_head = nn.Linear(H, V, bias=False)
    hidden = torch.randn(2, 5, H)
    ent = heads.forecast_entropy(hidden, lm_head)
    direct = entropy_from_logits(lm_head(hidden).float())
    for k in range(K):
        torch.testing.assert_close(ent[k], direct, rtol=1e-5, atol=1e-5)


def test_untied_unembedding_needs_no_lm_head():
    heads = make_heads(tie_unembedding=False)
    ent = heads.forecast_entropy(torch.randn(2, 5, H))
    assert ent.shape == (K, 2, 5)


def test_tied_unembedding_without_lm_head_raises():
    heads = make_heads(tie_unembedding=True)
    with pytest.raises(ValueError, match="lm_head"):
        heads.forecast_entropy(torch.randn(2, 5, H))


def test_ce_loss_runs_and_backprops():
    heads = make_heads()
    lm_head = nn.Linear(H, V, bias=False)
    hidden = torch.randn(3, 12, H, requires_grad=True)
    labels = torch.randint(0, V, (3, 12))
    mask = torch.ones(3, 12)
    mask[1, 8:] = 0

    loss, metrics = mtp_ce_loss(heads, hidden, labels, mask, lm_head)
    assert torch.isfinite(loss)
    loss.backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert metrics["mtp/ntok_k0"] > metrics["mtp/ntok_k3"], (
        "further heads must see strictly fewer valid targets"
    )


def test_ce_loss_targets_are_offset_correctly():
    """Head k must be trained on y_{t+k+1}, not y_{t+1}.

    Built as an oracle: a deterministic sequence where token t+k+1 is fully
    determined by t, so a correct offset can drive the loss to ~0 while an
    off-by-one cannot.
    """
    torch.manual_seed(0)
    seq_len, k_heads = 20, 3
    labels = torch.arange(seq_len).remainder(V).unsqueeze(0)
    mask = torch.ones(1, seq_len)
    hidden = torch.zeros(1, seq_len, H)
    hidden[0, torch.arange(seq_len), torch.arange(seq_len) % H] = 1.0

    heads = MTPHeads(hidden_size=H, vocab_size=V, num_heads=k_heads, head_hidden=64,
                     zero_init_output=False)
    lm_head = nn.Linear(H, V, bias=False)
    opt = torch.optim.Adam(list(heads.parameters()) + list(lm_head.parameters()), lr=5e-2)
    for _ in range(400):
        opt.zero_grad()
        loss, _ = mtp_ce_loss(heads, hidden, labels, mask, lm_head)
        loss.backward()
        opt.step()

    _, metrics = mtp_ce_loss(heads, hidden, labels, mask, lm_head)
    for k in range(k_heads):
        assert metrics[f"mtp/ce_k{k}"] < 0.1, f"head {k} failed to learn its offset: {metrics}"


def test_ce_loss_masks_padding_targets():
    """A position whose *target* is padding must not contribute."""
    heads = make_heads()
    lm_head = nn.Linear(H, V, bias=False)
    hidden = torch.randn(1, 10, H)
    labels = torch.randint(0, V, (1, 10))
    mask = torch.ones(1, 10)
    mask[0, 5:] = 0
    _, metrics = mtp_ce_loss(heads, hidden, labels, mask, lm_head, kappa=1)
    # positions 0..3 have a live target at t+1 (position 4 targets masked t=5)
    assert metrics["mtp/ntok_k0"] == 4.0


def test_ce_loss_empty_mask_is_safe():
    heads = make_heads()
    lm_head = nn.Linear(H, V, bias=False)
    loss, metrics = mtp_ce_loss(
        heads, torch.randn(1, 6, H), torch.randint(0, V, (1, 6)), torch.zeros(1, 6), lm_head
    )
    assert float(loss) == 0.0


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        MTPHeads(hidden_size=H, vocab_size=V, num_heads=0)
    heads = make_heads()
    lm_head = nn.Linear(H, V, bias=False)
    with pytest.raises(ValueError):
        heads.forecast_entropy(torch.randn(2, 3, H), lm_head, kappa=K + 1)
    with pytest.raises(ValueError):
        heads.forecast_entropy(torch.randn(2, 3, H), lm_head, temperature=0.0)


def test_entropy_from_logits_matches_uniform():
    logits = torch.zeros(5, V)
    torch.testing.assert_close(
        entropy_from_logits(logits),
        torch.full((5,), float(torch.log(torch.tensor(float(V))))),
        rtol=1e-6, atol=1e-6,
    )


def test_entropy_from_logits_matches_deterministic():
    logits = torch.full((3, V), -1e4)
    logits[:, 0] = 1e4
    torch.testing.assert_close(entropy_from_logits(logits), torch.zeros(3), atol=1e-5, rtol=1e-5)
