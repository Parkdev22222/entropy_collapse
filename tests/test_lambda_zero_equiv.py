"""Gate test: at lambda=0, STEER-F must BE stock STEER — bit for bit.

Plan §8.1.  This is the single most important test in the repo: every STEER-F
result is reported as a delta against the lambda=0 arm, so if that arm is not
literally the baseline, every number downstream is measuring the wrong thing.

`tests/reference_steer.py` is an untouched copy of upstream's
`compute_token_weights`, so this compares against the real implementation
rather than against a re-derivation of it.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from reference_steer import compute_token_weights as steer_reference  # noqa: E402

from steer_f.omega_tilde import (  # noqa: E402
    SteerFConfig,
    compute_omega_tilde,
    compute_token_weights_steerf,
)

MODES = ["symmetric", "asymmetric"]
LINEARS = [True, False]


def make_batch(seed=0, b=4, t=16, dtype=torch.float32, device="cpu"):
    """A batch with the statistics STEER actually sees at training time."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    log_prob = -torch.rand(b, t, generator=g).to(dtype) * 6.0
    old_log_prob = log_prob + torch.randn(b, t, generator=g).to(dtype) * 0.05
    advantages = torch.randn(b, t, generator=g).to(dtype)
    entropys = torch.rand(b, t, generator=g).to(dtype) * 3.0
    response_mask = torch.ones(b, t, dtype=torch.float, device=device)
    # ragged tails, as in a real rollout batch
    response_mask[1, 10:] = 0
    response_mask[3, 4:] = 0
    return dict(
        advantages=advantages.to(device),
        entropys=entropys.to(device),
        old_log_prob=old_log_prob.to(device),
        log_prob=log_prob.to(device),
        response_mask=response_mask,
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("linear", LINEARS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lambda_zero_is_bit_identical(mode, linear, seed):
    """lam=0 reproduces upstream exactly, for every (mode, mapping) combo."""
    batch = make_batch(seed=seed)
    expected = steer_reference(
        **batch, token_weight_min=0.8, token_weight_max=1.2, linear=linear, mode=mode
    )
    got = compute_token_weights_steerf(
        **batch, token_weight_min=0.8, token_weight_max=1.2, linear=linear, mode=mode, lam=0.0
    )
    assert torch.equal(got, expected), (
        f"max abs diff {(got - expected).abs().max().item():.3e} "
        f"for mode={mode}, linear={linear}, seed={seed}"
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("linear", LINEARS)
def test_a_h_present_but_lambda_zero_still_identical(mode, linear):
    """Supplying A_H must have no effect whatsoever while lambda is 0."""
    batch = make_batch(seed=7)
    a_h = torch.randn_like(batch["advantages"]) * 2.0
    expected = steer_reference(
        **batch, token_weight_min=0.9, token_weight_max=1.1, linear=linear, mode=mode
    )
    got = compute_token_weights_steerf(
        **batch, token_weight_min=0.9, token_weight_max=1.1, linear=linear, mode=mode,
        a_h=a_h, lam=0.0,
    )
    assert torch.equal(got, expected)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("linear", LINEARS)
def test_a_h_all_zero_matches_stock(mode, linear):
    """A genuinely zero A_H (no branch signal) reduces to stock STEER.

    Unlike the lam=0 short-circuit this exercises the *full* arithmetic path —
    normalisation, the visit term, the sum — so it verifies the algebra rather
    than the early return.  With RMS normalisation and a zero visit term the
    metric is a positive rescale of Omega, which both mappings are invariant
    to, so weights must match to floating-point tolerance.
    """
    batch = make_batch(seed=11)
    a_h = torch.zeros_like(batch["advantages"])
    expected = steer_reference(
        **batch, token_weight_min=0.8, token_weight_max=1.2, linear=linear, mode=mode
    )
    got = compute_token_weights_steerf(
        **batch, token_weight_min=0.8, token_weight_max=1.2, linear=linear, mode=mode,
        a_h=a_h, lam=1.0, norm="scale",
    )
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("linear", LINEARS)
def test_rms_norm_alone_does_not_change_weights(mode, linear):
    """RMS rescaling of the metric is invisible to both mappings.

    This is the invariance argument that justifies norm="scale" as the default
    (see the module docstring of steer_f/omega_tilde.py).  If this test ever
    fails, the exponential mapping's 0.02 denominator floor has started to
    bind and the normalisation is no longer behaviour-neutral.
    """
    batch = make_batch(seed=3)
    expected = steer_reference(
        **batch, token_weight_min=0.8, token_weight_max=1.2, linear=linear, mode=mode
    )
    scaled = compute_token_weights_steerf(
        **batch, token_weight_min=0.8, token_weight_max=1.2, linear=linear, mode=mode,
        a_h=torch.zeros_like(batch["advantages"]), lam=1e-12, norm="scale",
    )
    torch.testing.assert_close(scaled, expected, rtol=1e-5, atol=1e-6)


def test_nonzero_lambda_actually_changes_weights():
    """Guard against a silent no-op: lam>0 with real A_H must do something."""
    batch = make_batch(seed=5)
    g = torch.Generator().manual_seed(5)
    a_h = torch.randn(batch["advantages"].shape, generator=g)
    base = compute_token_weights_steerf(**batch, a_h=a_h, lam=0.0)
    perturbed = compute_token_weights_steerf(**batch, a_h=a_h, lam=1.0)
    assert not torch.allclose(base, perturbed), "lam=1.0 left the weights unchanged"


def test_short_circuit_returns_same_tensor_object():
    """lam=0 must not even touch the tensor — the strongest form of identity."""
    omega = torch.randn(2, 5)
    out, stats = compute_omega_tilde(
        omega_local=omega,
        w=torch.randn(2, 5),
        pi_sampled=torch.rand(2, 5),
        a_h=torch.randn(2, 5),
        response_mask=torch.ones(2, 5),
        lam=0.0,
    )
    assert out is omega
    assert stats["steerf/lam"] == 0.0


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("linear", LINEARS)
def test_config_validate_flags_unsafe_norm(mode, linear):
    """The config must warn wherever z-norm would break the lam=0 equivalence."""
    warnings = SteerFConfig(lam=0.5, norm="z").validate(mode=mode, linear=linear)
    unsafe = (mode == "symmetric") or (not linear)
    assert bool(warnings) == unsafe, f"mode={mode}, linear={linear}: {warnings}"
    assert SteerFConfig(lam=0.5, norm="scale").validate(mode=mode, linear=linear) == []


def test_z_norm_would_break_symmetric_equivalence():
    """Demonstrate the failure the default is chosen to avoid.

    Documents *why* norm="scale" is the default: with symmetric mode the
    metric passes through abs(), so recentring moves the pivot and the token
    weights change even though the visit term contributes nothing.
    """
    batch = make_batch(seed=13)
    expected = steer_reference(**batch, mode="symmetric", linear=True)
    z = compute_token_weights_steerf(
        **batch, mode="symmetric", linear=True,
        a_h=torch.zeros_like(batch["advantages"]), lam=1e-9, norm="z",
    )
    assert not torch.allclose(z, expected, rtol=1e-3, atol=1e-3)
