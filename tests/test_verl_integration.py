"""The verl glue: index alignment and the batch-level A_H assembly.

The index offset tested here is the one silent-failure mode of the whole
project — an off-by-one makes H_togo condition on the wrong token, A_H become
noise, and gate G1 fail for a reason no correlation plot would reveal.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from steer_f.entropy_forecast import HeadCalibration  # noqa: E402
from steer_f.mtp_heads import MTPHeads  # noqa: E402
from steer_f.omega_tilde import SteerFConfig  # noqa: E402
from steer_f.verl_integration import (  # noqa: E402
    compute_a_h,
    forecast_h_togo,
    slice_response_hidden,
)

H, V, K = 16, 32, 4


def make_heads(**kw):
    return MTPHeads(hidden_size=H, vocab_size=V, num_heads=K, head_hidden=8, **kw)


# ----------------------------------------------------------------------
# index alignment
# ----------------------------------------------------------------------
def test_slice_takes_the_last_response_length_positions():
    prompt_len, resp_len = 5, 3
    hidden = torch.arange(prompt_len + resp_len).float().view(1, -1, 1).expand(1, -1, H)
    out = slice_response_hidden(hidden, resp_len)
    assert out.shape == (1, resp_len, H)
    torch.testing.assert_close(out[0, :, 0], torch.tensor([5.0, 6.0, 7.0]))


def test_slice_is_one_step_later_than_verls_logprob_slice():
    """verl uses [-T-1:-1] for log-probs; the MTP conditioning state is [-T:].

    Asserting the exact one-step relationship documents the offset rather than
    leaving it implicit in a slice expression.
    """
    seq, resp_len = 9, 4
    hidden = torch.arange(seq).float().view(1, seq, 1).expand(1, seq, H)
    verl_logprob_slice = hidden[:, -resp_len - 1 : -1, :]
    steerf_slice = slice_response_hidden(hidden, resp_len)
    torch.testing.assert_close(steerf_slice[0, :, 0] - verl_logprob_slice[0, :, 0],
                               torch.ones(resp_len))


def test_slice_rejects_a_too_short_sequence():
    with pytest.raises(ValueError):
        slice_response_hidden(torch.randn(1, 3, H), response_length=5)


def test_slice_rejects_wrong_rank():
    with pytest.raises(ValueError):
        slice_response_hidden(torch.randn(7, H), response_length=3)


def test_full_response_prompt_free_case():
    hidden = torch.randn(2, 4, H)
    torch.testing.assert_close(slice_response_hidden(hidden, 4), hidden)


# ----------------------------------------------------------------------
# forecast
# ----------------------------------------------------------------------
def test_forecast_shape_matches_the_response_grid():
    heads, lm_head = make_heads(), nn.Linear(H, V, bias=False)
    cfg = SteerFConfig(kappa=3, gamma_h=0.85)
    out = forecast_h_togo(torch.randn(2, 10, H), lm_head, heads, response_length=6, cfg=cfg)
    assert out.shape == (2, 6)
    assert (out >= 0).all()


def test_forecast_respects_kappa():
    heads, lm_head = make_heads(), nn.Linear(H, V, bias=False)
    hidden = torch.randn(2, 8, H)
    small = forecast_h_togo(hidden, lm_head, heads, 5, SteerFConfig(kappa=1, gamma_h=1.0))
    large = forecast_h_togo(hidden, lm_head, heads, 5, SteerFConfig(kappa=4, gamma_h=1.0))
    assert (large >= small - 1e-5).all(), "adding heads cannot reduce an undiscounted sum"


def test_forecast_applies_calibration():
    heads, lm_head = make_heads(), nn.Linear(H, V, bias=False)
    hidden = torch.randn(2, 8, H)
    cfg = SteerFConfig(kappa=2, gamma_h=1.0)
    plain = forecast_h_togo(hidden, lm_head, heads, 5, cfg)
    shrunk = forecast_h_togo(
        hidden, lm_head, heads, 5, cfg,
        calib=HeadCalibration([1.0, 1.0], [0.5, 0.5], [0.0, 0.0]),
    )
    torch.testing.assert_close(shrunk, plain * 0.5, rtol=1e-4, atol=1e-4)


def test_forecast_accepts_pre_sliced_hidden():
    heads, lm_head = make_heads(), nn.Linear(H, V, bias=False)
    cfg = SteerFConfig(kappa=2, gamma_h=0.9)
    hidden = torch.randn(2, 6, H)
    a = forecast_h_togo(hidden, lm_head, heads, 6, cfg)
    b = forecast_h_togo(hidden, lm_head, heads, 6, cfg, already_sliced=True)
    torch.testing.assert_close(a, b)


def test_temperature_raises_forecast_entropy():
    """Same temperature convention as verl's own entropys — a scale check."""
    heads, lm_head = make_heads(zero_init_output=False), nn.Linear(H, V, bias=False)
    hidden = torch.randn(2, 6, H) * 3
    cfg = SteerFConfig(kappa=2, gamma_h=1.0)
    cold = forecast_h_togo(hidden, lm_head, heads, 6, cfg, temperature=0.5)
    hot = forecast_h_togo(hidden, lm_head, heads, 6, cfg, temperature=2.0)
    assert (hot > cold).all()


# ----------------------------------------------------------------------
# batch-level A_H
# ----------------------------------------------------------------------
def test_compute_a_h_returns_the_batch_grid():
    cfg = SteerFConfig(baseline="sibling")
    a_h = compute_a_h(
        torch.rand(4, 6),
        torch.randint(0, V, (4, 6)),
        torch.ones(4, 6),
        ["p0", "p0", "p1", "p1"],
        cfg,
    )
    assert a_h.shape == (4, 6) and a_h.dtype == torch.float32


def test_compute_a_h_honours_the_baseline_choice():
    responses = torch.tensor([[1, 5, 5], [1, 9, 9]])
    h = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 9.0]])
    mask = torch.ones(2, 3)
    sib = compute_a_h(h, responses, mask, ["p", "p"], SteerFConfig(baseline="sibling"))
    grp = compute_a_h(h, responses, mask, ["p", "p"], SteerFConfig(baseline="group"))
    assert float(sib[0, 2]) == 0.0
    assert float(grp[0, 2]) != 0.0


def test_compute_a_h_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        compute_a_h(torch.rand(2, 5), torch.zeros(2, 4, dtype=torch.long),
                    torch.ones(2, 5), ["p", "p"], SteerFConfig())


def test_a_h_is_finite_for_a_realistic_batch():
    """A rollout batch with ragged lengths and shared prefixes must stay finite."""
    torch.manual_seed(0)
    b, t = 8, 12
    responses = torch.randint(0, V, (b, t))
    responses[:, :4] = responses[0, :4]  # a shared prefix, as GRPO groups have
    mask = torch.ones(b, t)
    mask[3, 7:] = 0
    mask[5, 9:] = 0
    a_h = compute_a_h(torch.rand(b, t) * 5, responses, mask,
                      ["p0"] * 4 + ["p1"] * 4, SteerFConfig())
    assert torch.isfinite(a_h).all()
    assert torch.equal(a_h[3, 7:], torch.zeros(t - 7))


def test_end_to_end_forecast_to_a_h():
    """hidden states -> H_togo -> A_H, the shape contract the patches rely on."""
    torch.manual_seed(0)
    heads, lm_head = make_heads(zero_init_output=False), nn.Linear(H, V, bias=False)
    cfg = SteerFConfig(kappa=3, gamma_h=0.85, baseline="sibling")
    b, prompt_len, resp_len = 4, 5, 7

    hidden = torch.randn(b, prompt_len + resp_len, H)
    responses = torch.randint(0, V, (b, resp_len))
    responses[:, :2] = responses[0, :2]
    mask = torch.ones(b, resp_len)

    h_vals = forecast_h_togo(hidden, lm_head, heads, resp_len, cfg)
    a_h = compute_a_h(h_vals, responses, mask, ["p"] * b, cfg)

    assert a_h.shape == (b, resp_len)
    assert torch.isfinite(a_h).all()
    # A_H is a centred advantage within each sibling set
    torch.testing.assert_close(a_h[:, 0].sum(), torch.tensor(0.0), atol=1e-4, rtol=1e-4)
