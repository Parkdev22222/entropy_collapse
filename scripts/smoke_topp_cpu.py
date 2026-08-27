#!/usr/bin/env python3
"""CPU smoke of patches/steerf_topp_entropy.patch — nucleus (top-p) entropy.

Run from the root of the TRAINING tree, AFTER applying BOTH patches in order
(the top-p patch builds on the renyi2 one and will not apply without it):

    git apply patches/steerf_renyi2_entropy.patch
    git apply patches/steerf_topp_entropy.patch
    python scripts/smoke_renyi2_cpu.py     # both must pass
    python scripts/smoke_topp_cpu.py

Covers: the nucleus definition (smallest set reaching top_p, crossing token
included, renormalised, ratios preserved), the k_cap saturation guard, the
top_p=1.0 short-circuit, no NaN from the 0*-inf trap that logit-space masking
would create, exact equality with the unpatched path at top_p=1.0, the tail
rejection property the switch exists for, forecast_h_togo end to end, and
SteerFConfig validation. No GPU, model or dataset needed.
"""
import math, sys
import torch
torch.manual_seed(0)
sys.path.insert(0, ".")
from steer_f.mtp_heads import (ENTROPY_FNS, MTPHeads, entropy_from_logits,
                               entropy_of_logits, nucleus_probs, renyi2_from_logits)
from steer_f.omega_tilde import SteerFConfig
from steer_f.verl_integration import forecast_h_togo

fails=[]
def check(l,c,d=""):
    print(f"  {'PASS' if c else 'FAIL'}  {l}{('  -- '+str(d)) if d else ''}")
    if not c: fails.append(l)

print("[1] nucleus_probs")
l = torch.tensor([[math.log(x) for x in (0.5,0.25,0.15,0.06,0.04)]])
p9 = nucleus_probs(l, 0.9, k_cap=8)[0]
nz = (p9 > 0).sum().item()
# 0.5 + 0.25 + 0.15 = 0.90 already reaches 0.9, so the nucleus is 3 tokens.
check("nucleus = smallest set reaching top_p (3 of 5)", nz == 3, f"kept {nz}")
check("renormalised to 1", abs(float(p9.sum()) - 1.0) < 1e-6)
check("kept mass ratios preserved", abs(float(p9[0]/p9[1]) - 2.0) < 1e-5)
l2 = torch.tensor([[math.log(x) for x in (0.5, 0.25, 0.14, 0.07, 0.04)]])
check("crossing token included when sum falls short",
      int((nucleus_probs(l2, 0.9, 8) > 0).sum()) == 4,
      "0.5+0.25+0.14=0.89 < 0.9 so the 4th is pulled in")
check("top_p tiny keeps exactly 1", int((nucleus_probs(l, 1e-9, 8) > 0).sum()) == 1)
big = torch.randn(3, 7, 500) * 3
check("top_p=1.0 keeps everything", int((nucleus_probs(big, 1.0, 500) > 0).sum()) == 3*7*500)
try:
    nucleus_probs(torch.randn(2, 5000), 0.999, k_cap=4); check("k_cap saturation raises", False)
except ValueError as e:
    check("k_cap saturation raises", "k_cap" in str(e))

print("\n[2] entropy_of_logits -- no NaN, exact at top_p=1")
for kind in ("shannon", "renyi2"):
    ref = ENTROPY_FNS[kind](big)
    check(f"{kind} top_p=1.0 bit-identical", torch.equal(entropy_of_logits(big, kind, 1.0), ref))
    e = entropy_of_logits(big, kind, 0.9, k_cap=500)
    check(f"{kind} nucleus finite (no 0*-inf NaN)", bool(torch.isfinite(e).all()))
    check(f"{kind} nucleus <= full", bool((e <= ref + 1e-5).all()),
          f"max excess {float((e-ref).max()):.2e}")
# manual cross-check
man_p = nucleus_probs(l, 0.9, 8)[0]
man = -(man_p[man_p>0] * man_p[man_p>0].log()).sum()
check("shannon nucleus matches manual", abs(float(entropy_of_logits(l,'shannon',0.9,8)[0]) - float(man)) < 1e-6)

print("\n[3] the property this exists for: tail rejection with head preserved")
V = 20000
def make(head, tail_mass):
    n = V - len(head)
    p = [h*(1-tail_mass) for h in head] + [tail_mass/n]*n
    return torch.tensor([[math.log(x) for x in p]])
A, B = make([0.7,0.2,0.1], 0.005), make([0.7,0.2,0.1], 0.05)   # 머리 동일, 꼬리만 다름
C, D = make([0.7,0.2,0.1], 0.02), make([0.34,0.22,0.16,0.16,0.12], 0.02)  # 실질 선택지 다름
def gap(f, x, y): return abs(float(f(x)) - float(f(y)))
full = lambda z: entropy_from_logits(z)[0]
nuc  = lambda z: entropy_of_logits(z, "shannon", 0.95, 4096)[0]
r2   = lambda z: renyi2_from_logits(z)[0]
for nm, f in (("Shannon(full)", full), ("Renyi-2(full)", r2), ("Shannon(top-p .95)", nuc)):
    n_, s_ = gap(f,A,B), gap(f,C,D)
    print(f"    {nm:<20} 잡음 {n_:7.4f}   신호 {s_:7.4f}   S/N {s_/max(n_,1e-9):>12.1f}")
check("nucleus kills tail noise", gap(nuc,A,B) < 0.01*gap(full,A,B), f"{gap(nuc,A,B):.2e}")
check("nucleus keeps head signal", gap(nuc,C,D) > 0.9*gap(full,C,D))

print("\n[4] forecast_h_togo end-to-end")
H,Vv,K,B_,T = 32,401,4,3,9
heads = MTPHeads(hidden_size=H, vocab_size=Vv, num_heads=K, head_hidden=16,
                 zero_init_output=False).eval()
lm = torch.nn.Linear(H, Vv, bias=False)
hs = torch.randn(B_, T+4, H)
base = forecast_h_togo(hs, lm, heads, T, SteerFConfig(kappa=2, gamma_h=0.7))
same = forecast_h_togo(hs, lm, heads, T, SteerFConfig(kappa=2, gamma_h=0.7, entropy_top_p=1.0))
nucl = forecast_h_togo(hs, lm, heads, T, SteerFConfig(kappa=2, gamma_h=0.7,
                                                      entropy_top_p=0.9, entropy_top_k_cap=Vv))
check("default == top_p 1.0 (off-switch)", torch.equal(base, same))
check("nucleus differs", not torch.allclose(base, nucl))
check("nucleus <= full", bool((nucl <= base + 1e-5).all()))
check("finite", bool(torch.isfinite(nucl).all()))

print("\n[5] config validation")
check("default: no top_p warning", not any("top_p" in w for w in SteerFConfig(lam=.25).validate()))
check("top_p<1 warns", any("entropy_top_p" in w for w in SteerFConfig(lam=.25, entropy_top_p=0.9).validate()))
for bad in (0.0, 1.5, -1.0):
    try: SteerFConfig(entropy_top_p=bad).validate(); check(f"reject top_p={bad}", False)
    except ValueError: check(f"reject top_p={bad}", True)

print()
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
