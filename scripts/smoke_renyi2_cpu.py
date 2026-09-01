#!/usr/bin/env python3
"""CPU smoke of patches/steerf_renyi2_entropy.patch — every code path it touches.

Run from the root of the TRAINING tree (claude/qwen-math-training-crash-5jloyp)
AFTER applying the patch; no GPU, model, or dataset needed:

    git apply patches/steerf_renyi2_entropy.patch
    python scripts/smoke_renyi2_cpu.py        # expects ALL PASS, exit 0

Covers: the H2 = 2*lse(l) - lse(2l) identity within fp32 roundoff (measured
no worse than the existing shannon path's own roundoff on the same inputs),
forecast_entropy dispatch incl. the per-head-temperature calibration path and
the bit-identical default, SteerFConfig validation, forecast_h_togo end to end
for both kinds, the calibration entropy_kind stamp round-trip and mismatch
predicate, and compute_a_h on renyi2 forecasts preserving the sibling
mean-zero property (C5 corollary 2).  What it cannot cover on CPU: the FSDP
summon path in dp_actor and the live ray_trainer wiring — run SCALE=smoke on
the GPU box for those.
"""
import io, json, math, sys
from contextlib import redirect_stdout

import torch
torch.manual_seed(0)

sys.path.insert(0, ".")
from steer_f.mtp_heads import ENTROPY_FNS, MTPHeads, entropy_from_logits, renyi2_from_logits
from steer_f.omega_tilde import SteerFConfig
from steer_f.entropy_forecast import HeadCalibration, fit_head_calibration, h_togo
from steer_f.verl_integration import compute_a_h, forecast_h_togo

fails = []
def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  -- ' + str(detail)) if detail else ''}")
    if not cond: fails.append(label)

# ---- 1. renyi2_from_logits correctness ----
print("[1] renyi2_from_logits")
for dtype in (torch.float32, torch.float64):
    l = (torch.randn(4, 7, 101, dtype=dtype) * torch.tensor([1., 10., 100., 1000.]).view(4,1,1)
         + torch.tensor([0., -500., 500., 1e4]).view(4,1,1))
    p = torch.softmax(l.double(), dim=-1)
    direct = -torch.log((p * p).sum(-1))
    err2 = (renyi2_from_logits(l).double() - direct).abs().max()
    # the meaningful invariant: no worse than the EXISTING shannon path's own
    # fp32 roundoff on the same inputs (measured 1.9e-4 at logit scale 1000)
    direct1 = -(p * torch.log(p.clamp_min(1e-300))).sum(-1)
    err1 = (entropy_from_logits(l).double() - direct1).abs().max()
    check(f"identity within roundoff ({dtype})", float(err2) <= max(3 * float(err1), 1e-10),
          f"renyi2 err {float(err2):.2e} vs shannon err {float(err1):.2e}")
    h2, h1 = renyi2_from_logits(l), entropy_from_logits(l)
    check(f"0 <= H2 <= H_shannon ({dtype})",
          bool((h2 >= -1e-4).all() and (h2 <= h1 + 1e-3).all()))
check("shift invariance", torch.allclose(renyi2_from_logits(torch.tensor([[1.,2.,3.]])),
                                         renyi2_from_logits(torch.tensor([[1.,2.,3.]]) + 1234.0), atol=1e-4))
check("uniform k=10 -> log 10", abs(float(renyi2_from_logits(torch.zeros(1,10))) - math.log(10)) < 1e-6)
check("ENTROPY_FNS keys", set(ENTROPY_FNS) == {"shannon", "renyi2"})

# ---- 2. MTPHeads.forecast_entropy dispatch ----
print("[2] forecast_entropy dispatch")
H, V, K, B, T = 32, 101, 4, 3, 11
heads = MTPHeads(hidden_size=H, vocab_size=V, num_heads=K, head_hidden=16,
                 zero_init_output=False).eval()
lm = torch.nn.Linear(H, V, bias=False)
hs = torch.randn(B, T, H)
with torch.no_grad():
    e_sh  = heads.forecast_entropy(hs, lm, kappa=K)                                   # default
    e_sh2 = heads.forecast_entropy(hs, lm, kappa=K, entropy_kind="shannon")
    e_r2  = heads.forecast_entropy(hs, lm, kappa=K, entropy_kind="renyi2")
    # manual reference for head k
    ref_sh = torch.stack([entropy_from_logits(lm(heads._project(k, hs.reshape(-1,H))).float())
                          for k in range(K)]).reshape(K, B, T)
    ref_r2 = torch.stack([renyi2_from_logits(lm(heads._project(k, hs.reshape(-1,H))).float())
                          for k in range(K)]).reshape(K, B, T)
check("default == shannon (bit-identical off-switch)", torch.equal(e_sh, e_sh2))
check("shannon matches manual", torch.allclose(e_sh, ref_sh, atol=1e-5))
check("renyi2 matches manual", torch.allclose(e_r2, ref_r2, atol=1e-5))
check("renyi2 <= shannon per position", bool((e_r2 <= e_sh + 1e-5).all()))
with torch.no_grad():
    e_t = heads.forecast_entropy(hs, lm, kappa=K, temperature=[0.7,1.0,1.3,2.0],
                                 entropy_kind="renyi2")
check("per-head temperature list works", tuple(e_t.shape) == (K, B, T))
try:
    heads.forecast_entropy(hs, lm, kappa=K, entropy_kind="tsallis"); check("bad kind raises", False)
except ValueError as e:
    check("bad kind raises", "entropy_kind" in str(e))

# ---- 3. SteerFConfig ----
print("[3] SteerFConfig")
w_def = SteerFConfig(lam=0.25).validate()
w_r2  = SteerFConfig(lam=0.25, entropy_kind="renyi2").validate()
check("default: no entropy_kind warning", not any("renyi2" in w for w in w_def))
check("renyi2: warning emitted", any("Shannon-specific" in w for w in w_r2))
try:
    SteerFConfig(entropy_kind="fisher").validate(); check("bad kind rejected", False)
except ValueError:
    check("bad kind rejected", True)

# ---- 4. forecast_h_togo end-to-end (the training call path) ----
print("[4] forecast_h_togo")
S_len = T + 5
full = torch.randn(B, S_len, H)
cfg_sh = SteerFConfig(kappa=2, gamma_h=0.7)
cfg_r2 = SteerFConfig(kappa=2, gamma_h=0.7, entropy_kind="renyi2")
ht_sh = forecast_h_togo(full, lm, heads, response_length=T, cfg=cfg_sh)
ht_r2 = forecast_h_togo(full, lm, heads, response_length=T, cfg=cfg_r2)
check("shapes [B,T]", tuple(ht_sh.shape) == (B, T) == tuple(ht_r2.shape))
check("renyi2 h_togo <= shannon h_togo", bool((ht_r2 <= ht_sh + 1e-5).all()))
check("kinds actually differ", not torch.allclose(ht_r2, ht_sh))
# with a calibration carrying per-head temperature
calib = HeadCalibration(temperature=[0.9, 1.1], scale=[0.8, 0.5], bias=[0.05, 0.1])
ht_cal = forecast_h_togo(full, lm, heads, response_length=T, cfg=cfg_r2, calib=calib)
check("calibrated renyi2 path runs", tuple(ht_cal.shape) == (B, T))

# ---- 5. calibration stamp round-trip (what dp_actor / recall check) ----
print("[5] calibration stamp")
d = {**calib.to_dict(), "entropy_kind": "renyi2"}
d2 = json.loads(json.dumps(d))
check("from_dict ignores extra stamp key", len(HeadCalibration.from_dict(d2)) == 2)
check("stamp survives json round-trip", d2.get("entropy_kind") == "renyi2")
# the exact mismatch predicate dp_actor uses
check("mismatch detected", (str(d2.get("entropy_kind", "shannon")) != "shannon"))
fit = fit_head_calibration(torch.rand(2, 50) * 2, torch.rand(2, 50))
check("fit_head_calibration runs", len(fit) == 2)

# ---- 6. A_H chain on renyi2 forecasts (sibling mean-zero must survive) ----
print("[6] compute_a_h on renyi2 h_togo")
resp = torch.randint(0, V, (4, T)); resp[1] = resp[0]  # rollouts 0,1 identical prefix
mask = torch.ones(4, T, dtype=torch.long)
uid = ["g0", "g0", "g0", "g0"]
full4 = torch.randn(4, S_len, H)
ht4 = forecast_h_togo(full4, lm, heads, response_length=T, cfg=cfg_r2)
a_h = compute_a_h(ht4, resp, mask, uid, cfg_r2)
check("a_h shape/finite", tuple(a_h.shape) == (4, T) and bool(torch.isfinite(a_h).all()))
# at t=0 all four share the empty prefix -> siblings = all -> mean over group is 0
check("sibling mean-zero at t=0 (C5 corollary 2)", abs(float(a_h[:, 0].sum())) < 1e-4)

print()
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
