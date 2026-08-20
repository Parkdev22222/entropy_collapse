#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Compare ways of folding the branch score into STEER's token weights.

The shipped form is additive::

    Omega_tilde = norm(Omega_local) + lam * norm(dlogpi_hat * clip(A_H))

and the first lambda>0 run showed what that costs.  Against its own lambda=0
baseline at step 100, seed 1: the attenuating third of the weight range lost
occupancy (13.9% -> 12.5%), and global entropy fell further rather than less
(0.612 -> 0.395), while the branch/non-branch entropy gap did open (0.112).
The branch term is not adding protection, it is *relocating* it -- the weights
live in a fixed [w_min, w_max] band mapped by a per-micro-batch min-max, so a
second additive term competes with `Omega_local` for the same budget.

A multiplicative form should not compete, because it scales what STEER already
decided instead of adding a rival signal:

    Omega_tilde = norm(Omega_local) * (1 + lam * f(A_H))

That is the hypothesis this script tests, offline, on real rollouts and with no
training.  It reconstructs the exact inputs `compute_token_weights` sees at the
first PPO epoch -- where `log_prob == old_log_prob` by construction, so the
ratio is 1 -- and reports, for each candidate:

    tw_frac_low   does the attenuating range keep the occupancy lambda=0 gives
                  it?  This is the budget question.
    branch vs
    non-branch w  are branch tokens actually attenuated more?  This is whether
                  the form does its job at all.
    non-branch
    drift         how far non-branch weights move from their lambda=0 values.
                  Small is the point: STEER's global behaviour should survive.
    tw_std        collapse here means the min-max mapping compressed everything
                  into one point, the pathology that made gate G0 fail once.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from steer_f.entropy_forecast import HeadCalibration, sibling_support  # noqa: E402
from steer_f.mtp_heads import MTPHeads, entropy_from_logits  # noqa: E402
from steer_f.omega_tilde import (  # noqa: E402
    SteerFConfig,
    branch_weight_correction,
    delta_logpi_hat,
    local_omega_signed,
    normalize,
    visit_term,
)
from steer_f.verl_integration import compute_a_h, forecast_h_togo  # noqa: E402

_EPS = 1e-8


def steer_mapping(metric, mask, w_min, w_max, mode="asymmetric"):
    """STEER's metric -> weight map, linear branch, copied from core_algos."""
    metric = metric.clone()
    metric[~mask.bool()] = 0.0
    if mode == "symmetric":
        metric = torch.abs(metric)
    valid = metric[mask.bool()]
    w = torch.zeros_like(metric, dtype=torch.float)
    if valid.numel() == 0:
        return w
    lo, hi = valid.min(), valid.max()
    if (hi - lo) < 1e-8:
        w[mask.bool()] = (w_min + w_max) / 2
        return w
    scale = (w_max - w_min) / (hi - lo)
    w = w_min + (metric - lo) * scale
    return torch.clamp(w, w_min, w_max) * mask


def forms(omega, mask, a_h, dlogpi, lam, clip_c=1.0, norm="scale"):
    """Candidate metrics.  All reduce to norm(omega) at lam = 0."""
    base = normalize(omega, mask, mode=norm)
    if lam == 0.0:
        return {"lam0": base}
    ah = a_h.clamp(-clip_c, clip_c)
    ah_n = normalize(ah, mask, mode=norm)
    out = {
        # what ships today
        "A_additive": base + lam * normalize(dlogpi * ah, mask, mode=norm),
        # scale STEER's own decision up at branch points, sign untouched
        "B_mult_abs": base * (1.0 + lam * ah_n.abs()),
        # amplify only where the future is richer than the siblings' average
        "C_mult_pos": base * (1.0 + lam * ah_n.clamp(min=0.0)),
        # amplify only the attenuating direction: protect, never accelerate
        "D_mult_attenuate": torch.where(
            omega < 0, base * (1.0 + lam * ah_n.abs()), base
        ),
    }
    return out


def build_argparser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--heads", required=True)
    p.add_argument("--rollouts", required=True)
    p.add_argument("--calib", default=None)
    p.add_argument("--out", default="docs/weight_forms.json")
    p.add_argument("--lams", default="0.25,0.5,1.0")
    p.add_argument("--kappa", type=int, default=4)
    p.add_argument("--gamma-h", type=float, default=0.85)
    p.add_argument("--w-min", type=float, default=0.9)
    p.add_argument("--w-max", type=float, default=1.0)
    p.add_argument("--max-groups", type=int, default=48)
    p.add_argument("--max-response-tokens", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device).eval()
    lm = model.get_output_embeddings()

    ck = torch.load(args.heads, map_location="cpu", weights_only=False)
    heads = MTPHeads(**ck["config"]).to(args.device, dtype=torch.bfloat16)
    heads.load_state_dict(ck["state_dict"]); heads.eval()
    calib = None
    if args.calib and Path(args.calib).exists():
        d = json.loads(Path(args.calib).read_text())
        calib = HeadCalibration(scale=torch.tensor(d["scale"]),
                                bias=torch.tensor(d["bias"]),
                                temperature=torch.tensor(
                                    d.get("temperature", [1.0] * len(d["scale"]))))

    by_prompt = defaultdict(list)
    with open(args.rollouts) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pr, rs, acc = r.get("prompt"), r.get("response"), r.get("acc")
            if pr and rs and acc is not None:
                by_prompt[pr].append((rs, bool(acc)))
    groups = [(p, v) for p, v in by_prompt.items()
              if len(v) >= 4 and len(set(a for _, a in v)) > 1]
    groups.sort(key=lambda kv: -len(kv[1]))
    groups = groups[: args.max_groups]
    print(f"[forms] {len(groups)} mixed-outcome groups")

    cfg = SteerFConfig(kappa=args.kappa, gamma_h=args.gamma_h)
    lams = [float(x) for x in args.lams.split(",")]
    acc_stats = defaultdict(lambda: defaultdict(list))

    for gi, (prompt, rs) in enumerate(groups):
        pid = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=1024)["input_ids"][0]
        toks, accs = [], []
        for resp, a in rs:
            ids = tok(resp, return_tensors="pt", truncation=True,
                      max_length=args.max_response_tokens,
                      add_special_tokens=False)["input_ids"][0]
            if ids.numel() >= 2:
                toks.append(ids); accs.append(a)
        if len(set(accs)) < 2:
            continue
        T = max(t.numel() for t in toks); B = len(toks); P = pid.numel()
        resp = torch.full((B, T), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((B, T), dtype=torch.long)
        for i, t in enumerate(toks):
            resp[i, : t.numel()] = t; mask[i, : t.numel()] = 1
        ids = torch.cat([pid.unsqueeze(0).expand(B, -1), resp], 1).to(args.device)
        att = torch.cat([torch.ones((B, P), dtype=torch.long), mask], 1).to(args.device)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=att,
                        output_hidden_states=True, use_cache=False)
        logits = out.logits[:, P - 1: P - 1 + T, :].float()
        ent = entropy_from_logits(logits).cpu()
        logp = torch.log_softmax(logits, -1).gather(
            -1, resp.to(args.device).unsqueeze(-1)).squeeze(-1).cpu()

        # GRPO advantage: group-standardised reward, broadcast over tokens.
        r = torch.tensor([1.0 if a else -1.0 for a in accs])
        adv = ((r - r.mean()) / (r.std(unbiased=False) + _EPS)).unsqueeze(1).expand(B, T)

        h = forecast_h_togo(out.hidden_states[-1][:, P:, :], lm, heads,
                            response_length=T, cfg=cfg, calib=calib,
                            chunk_size=2048).cpu().float()
        a_h = compute_a_h(h, resp, mask, [gi] * B, cfg)
        sup = sibling_support(resp, mask, [gi] * B)

        # First PPO epoch: log_prob == old_log_prob, so ratio == 1 and w == adv.
        omega = local_omega_signed(adv, ent, logp, logp)
        pi = torch.exp(logp).clamp(_EPS, 1 - _EPS)
        dlog = delta_logpi_hat(adv, pi, eta=1.0)

        mb = mask.bool()
        w0 = steer_mapping(normalize(omega, mask), mask, args.w_min, args.w_max)
        vt = visit_term(adv, pi, a_h, eta=1.0, clip_c=1.0)
        for lam in lams:
            cand = forms(omega, mask, a_h, dlog, lam)
            weights = {n: steer_mapping(m, mask, args.w_min, args.w_max)
                       for n, m in cand.items()}
            # E: the post-mapping correction -- STEER's metric untouched, the
            # branch signal applied to the mapped weights instead.
            weights["E_post_mapping"] = branch_weight_correction(
                w0.clone(), vt, mask, lam=lam,
                token_weight_min=args.w_min, token_weight_max=args.w_max)[0]
            for name, w in weights.items():
                third = (args.w_max - args.w_min) / 3
                s = acc_stats[lam][name]
                s.append({
                    "frac_low": float((w[mb] < args.w_min + third).float().mean()),
                    "frac_high": float((w[mb] > args.w_max - third).float().mean()),
                    "mean": float(w[mb].mean()),
                    "std": float(w[mb].std()),
                    "w_branch": float(w[sup & mb].mean()) if (sup & mb).any() else float("nan"),
                    "w_other": float(w[~sup & mb].mean()) if (~sup & mb).any() else float("nan"),
                    "drift_other": float((w - w0)[~sup & mb].abs().mean()),
                })
        s = acc_stats[0.0]["lam0"]
        third = (args.w_max - args.w_min) / 3
        s.append({
            "frac_low": float((w0[mb] < args.w_min + third).float().mean()),
            "frac_high": float((w0[mb] > args.w_max - third).float().mean()),
            "mean": float(w0[mb].mean()), "std": float(w0[mb].std()),
            "w_branch": float(w0[sup & mb].mean()) if (sup & mb).any() else float("nan"),
            "w_other": float(w0[~sup & mb].mean()) if (~sup & mb).any() else float("nan"),
            "drift_other": 0.0,
        })
        if (gi + 1) % 8 == 0:
            print(f"[forms] {gi+1}/{len(groups)}", flush=True)

    def agg(rows):
        keys = rows[0].keys()
        return {k: float(torch.tensor([r[k] for r in rows]).nanmean()) for k in keys}

    result = {str(l): {n: agg(rs) for n, rs in d.items() if rs}
              for l, d in acc_stats.items()}
    base = result["0.0"]["lam0"]
    print(f"\nbaseline lambda=0:  frac_low={base['frac_low']:.3f}  "
          f"w_branch={base['w_branch']:.4f}  w_other={base['w_other']:.4f}  "
          f"std={base['std']:.4f}")
    print(f"\n{'lam':>5} {'form':18} {'frac_low':>9} {'w_branch':>9} {'w_other':>9} "
          f"{'branch-other':>13} {'drift_other':>12} {'std':>8}")
    for lam in lams:
        for name, v in sorted(result[str(lam)].items()):
            print(f"{lam:>5} {name:18} {v['frac_low']:>9.3f} {v['w_branch']:>9.4f} "
                  f"{v['w_other']:>9.4f} {v['w_branch']-v['w_other']:>13.5f} "
                  f"{v['drift_other']:>12.5f} {v['std']:>8.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"baseline": base, "by_lambda": result,
         "reading": "branch-other < 0 means branch tokens are attenuated more "
                    "(the intent). drift_other near 0 means STEER's treatment of "
                    "the other 99% of tokens is preserved. frac_low should stay "
                    "near the lambda=0 value, else the branch term is spending "
                    "the attenuation budget rather than adding to it."},
        indent=2))
    print(f"[forms] wrote {args.out}")


if __name__ == "__main__":
    main()
