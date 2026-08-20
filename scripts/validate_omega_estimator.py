#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Measure which Omega actually predicts token-level entropy change.

This is the paper's own validation (Table 1 / Table 8: MSE, PCC, SRCC between
an estimator and ground-truth per-token entropy change) applied to the question
the offline distribution study could not settle: the released implementation
computes ``A/pi_old`` where Theorem 1 computes ``r*A``, and distribution shape
alone cannot say which is the better estimator.  Ground truth can.

Procedure, one micro-batch:

    1. Forward the policy, record H(s) and log pi at every response position.
    2. Compute Omega under each candidate form. At this point log_prob ==
       old_log_prob by construction, so r == 1 and I_clip is all ones -- the
       two forms differ by exactly one factor of pi_theta and nothing else.
    3. Take ONE gradient step on the GRPO surrogate.
    4. Forward again, same inputs, and record H(s) at the same positions.
    5. Ground truth is dH = H_after - H_before. Score each Omega against it.

Two deliberate choices:

PLAIN SGD, NOT ADAM.  Appendix G Step 3 derives the logit update as
``z^{k+1} - z^k = eta * (policy gradient)``.  That is a raw scaled-gradient
step.  Adam rescales each coordinate by its own second-moment estimate, which
breaks the proportionality the theorem is built on, so testing under Adam would
be testing something the theorem never claimed.  SGD is the faithful test.

CPU, FLOAT32.  Two reasons, both load-bearing.  The measured quantity is a
difference of entropies after a step of size eta ~ 1e-4; in bfloat16 that
difference is mostly rounding noise, so float32 is not optional here.  And the
GPU is running the Phase 2 grid -- a float32 1.5B model with gradients would
not fit beside it, and evicting a training run to measure an estimator is the
wrong trade.  On CPU this is a couple of minutes per group and costs the grid
nothing.

Note on scale: PCC and SRCC are scale-free, so eta/L cancels and only the
*shape* of each estimator is tested.  MSE is not, so eta/L is applied there --
which is what makes MSE the check on whether the estimator has the right
magnitude, not just the right ranking.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from steer_f.omega_tilde import _EPS  # noqa: E402

from compare_omega_forms import read_groups, srcc  # noqa: E402


def pcc(x, y):
    x = x.double() - x.double().mean()
    y = y.double() - y.double().mean()
    d = x.norm() * y.norm()
    return float((x * y).sum() / d) if float(d) > 0 else float("nan")


def entropies_and_logp(model, ids, resp, P, T, chunk=1):
    """H(s) and log pi(a_t|s) at every response position, row at a time."""
    B = ids.shape[0]
    ent = torch.zeros(B, T)
    logp = torch.zeros(B, T)
    for i in range(0, B, chunk):
        sl = slice(i, min(i + chunk, B))
        out = model(input_ids=ids[sl], use_cache=False)
        lg = out.logits[:, P - 1: P - 1 + T, :].float()
        lsm = torch.log_softmax(lg, -1)
        ent[sl] = -(lsm.exp() * lsm).sum(-1)
        logp[sl] = lsm.gather(-1, resp[sl].unsqueeze(-1)).squeeze(-1)
        del out, lg, lsm
    return ent, logp


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--rollouts", required=True)
    p.add_argument("--out", default="docs/omega_ground_truth.json")
    p.add_argument("--eta", type=float, default=1e-4,
                   help="learning rate of the single SGD step. Theorem 1 is "
                        "first order, and its remainder is O(eta^2), so this "
                        "trades signal against approximation error.")
    p.add_argument("--groups", type=int, default=6)
    p.add_argument("--max-response-tokens", type=int, default=192)
    p.add_argument("--max-rollouts", type=int, default=4)
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(16)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.pad_token or tok.eos_token

    groups = read_groups(args.rollouts, args.groups)
    print(f"[gt] {len(groups)} mixed-outcome groups, eta={args.eta}", flush=True)

    rows = defaultdict(list)
    for gi, (prompt, rs) in enumerate(groups):
        # A fresh model per group: the step must start from the same weights
        # every time, or group g would be measuring the policy that groups
        # 1..g-1 already moved.
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32).train()
        model.config.use_cache = False

        pid = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=512)["input_ids"][0]
        toks, accs = [], []
        for resp_txt, a in rs[: args.max_rollouts]:
            ids_ = tok(resp_txt, return_tensors="pt", truncation=True,
                       max_length=args.max_response_tokens,
                       add_special_tokens=False)["input_ids"][0]
            if ids_.numel() >= 2:
                toks.append(ids_)
                accs.append(a)
        if len(set(accs)) < 2:
            del model
            continue
        T = max(t.numel() for t in toks)
        B = len(toks)
        P = pid.numel()
        resp = torch.full((B, T), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((B, T))
        for i, t in enumerate(toks):
            resp[i, : t.numel()] = t
            mask[i, : t.numel()] = 1.0
        ids = torch.cat([pid.unsqueeze(0).expand(B, -1), resp], 1)

        with torch.no_grad():
            H0, logp0 = entropies_and_logp(model, ids, resp, P, T)

        r_ = torch.tensor([1.0 if a else -1.0 for a in accs])
        adv = ((r_ - r_.mean()) / (r_.std(unbiased=False) + _EPS)).unsqueeze(1).expand(B, T)

        pi = torch.exp(logp0).clamp(_EPS, 1 - _EPS)
        d = -pi * (1 - pi) * (torch.log(pi) + H0)      # Eq. 8, sign as in Eq. 7
        L = float(mask.sum())
        forms = {
            "released": (adv / pi) * d,                # A / pi_old, r == 1 here
            "theorem_r": adv * d,                      # r * A, r == 1 here
        }

        # --- one SGD step on the GRPO surrogate. r == 1 at this point, so the
        # clipped min() is inactive and the surrogate reduces to A * log pi.
        model.zero_grad(set_to_none=True)
        loss = torch.zeros(())
        for i in range(B):
            out = model(input_ids=ids[i: i + 1], use_cache=False)
            lp = torch.log_softmax(out.logits[0, P - 1: P - 1 + T, :].float(), -1)
            lp = lp.gather(-1, resp[i].unsqueeze(-1)).squeeze(-1)
            (-(adv[i] * lp * mask[i]).sum() / L).backward()
            loss = loss + (-(adv[i] * lp * mask[i]).sum() / L).detach()
            del out, lp
        with torch.no_grad():
            for p_ in model.parameters():
                if p_.grad is not None:
                    p_.add_(p_.grad, alpha=-args.eta)

        model.eval()
        with torch.no_grad():
            H1, _ = entropies_and_logp(model, ids, resp, P, T)
        dH = (H1 - H0)[mask.bool()]
        del model

        for name, om in forms.items():
            om = torch.nan_to_num(om, 0.0, 0.0, 0.0)[mask.bool()]
            scaled = om * (args.eta / L)               # Theorem 1's -(eta/L)
            rows[name].append({
                "mse": float(((scaled - dH) ** 2).mean()),
                "pcc": pcc(om, dH),
                "srcc": srcc(om, dH),
            })
        print(f"[gt] group {gi+1}/{len(groups)}  n={int(mask.sum())} tokens  "
              f"|dH|mean={float(dH.abs().mean()):.3e}", flush=True)

    if not rows:
        print("[gt] no usable groups")
        return 1
    summary = {n: {k: float(torch.tensor([r[k] for r in v]).mean())
                   for k in ("mse", "pcc", "srcc")} for n, v in rows.items()}
    Path(args.out).write_text(json.dumps(
        {"summary": summary, "eta": args.eta, "per_group": dict(rows)}, indent=2))

    print(f"\n{'form':<14}{'MSE':>14}{'PCC':>10}{'SRCC':>10}   "
          f"(paper reports PCC ~+0.4, SRCC ~+0.6..0.7)")
    for n, s in summary.items():
        print(f"{n:<14}{s['mse']:>14.3e}{s['pcc']:>10.3f}{s['srcc']:>10.3f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
