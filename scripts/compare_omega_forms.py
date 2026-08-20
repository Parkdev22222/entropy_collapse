#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Compare the released STEER Omega against Theorem 1's Omega, on real rollouts.

Theorem 1 (Eq. 7, and Eq. 17 in Appendix G) states

    Omega(s) = -(eta/L) E_{a~pi_theta(.|s)}[ I_clip * A/pi_old
                 * pi_theta (1 - pi_theta) (log pi_theta + H(s)) ] .

The released implementation evaluates the *integrand* at the one sampled token
and stops there.  Two things go missing, and they are not the same size.

1. THE SAMPLING MEASURE.  Eq. 7 is an expectation over a ~ pi_theta.  Appendix G
   Step 4 reaches it by summing over the vocabulary and absorbing one factor of
   pi_theta into the measure -- the summand carries pi_theta^2, the integrand
   carries pi_theta.  Plugging the sampled token into the integrand is therefore
   a Monte-Carlo estimate that assumes the sample came from pi_theta.  It did
   not: rollouts come from pi_old.  The importance-corrected estimator is

       Omega_hat = r * [integrand],   r = pi_theta / pi_old .

   The same factor arrives by a second, independent route.  Step 3 is explicit
   that only the sampled action's logit moves ("for non-sampled actions a' at
   the same state s, the corresponding logits z_{s,a'} remain unchanged"), so
   the realised entropy change is the single inner-product term

       Omega = dH/dz_{s,a} * (z^{k+1}_{s,a} - z^k_{s,a})
             = -(eta/L) I_clip * r * A * pi (1-pi) (log pi + H) .

   Expectation-with-importance-weight and exact-realised-update agree.  Both
   differ from the released code by exactly one factor of pi_theta, i.e. the
   code computes A/pi_old where the theorem computes r*A = A*pi_theta/pi_old.

   This is not a cosmetic constant.  pi_theta spans many orders of magnitude
   across a response, so dividing by it inflates rare tokens without bound.  The
   released code floors pi_old at 1e-8, which caps the damage at 1e8 rather than
   removing it.  Since Eq. 9 normalises by max_B |Omega|, one such token sets
   the denominator for every other token in the micro-batch.

2. THE 1/L NORMALISATION.  L = sum_i |o_i| is per *group* (per query), and
   max_B |Omega| in Eq. 9 is over the whole batch, which mixes groups of very
   different total length.  So 1/L does not cancel: a token in a long-response
   group should be scaled down relative to a token in a short one, and dropping
   it biases attenuation toward whichever queries happen to generate more text.

This script measures both, offline, with no training.  It reconstructs the
inputs `compute_token_weights` sees at the first PPO epoch, where log_prob ==
old_log_prob by construction and hence r == 1 exactly -- the cleanest possible
setting, because there the released form and Theorem 1 differ by precisely the
factor pi_theta and nothing else.  I_clip is also all-ones there, which is why
it is reported but expected to be inert at this point in the update.

Reported per form:

    tail        max|Omega| / median|Omega|, and p99.9/p50.  This is the
                quantity Eq. 9's denominator is exposed to.
    pi_at_max   the token probability of the argmax-|Omega| token.  If the
                batch maximum is always set by a ~1e-6 token, the mapping is
                being scaled by a sampling artefact, not by an entropy risk.
    tw_std      weight spread after the Eq. 9 mapping.  Collapse here means
                every token got the same weight and STEER did nothing.
    frac_pinned fraction of tokens within 1% of w_max -- the compression
                pathology directly.
    srcc        rank correlation with the released form.  If this is low, the
                two forms disagree about *which* tokens to attenuate, and the
                choice is load-bearing rather than a rescaling.
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

_THIRD = 3.0


def delta(pi, entropy):
    """Eq. 8: delta(pi, H) = -pi (1-pi) (log pi + H).  Shared by every form."""
    pi = pi.clamp(_EPS, 1.0 - _EPS)
    return -pi * (1.0 - pi) * (torch.log(pi) + entropy)


def eq9_weights(omega, mask, w_min=0.7):
    """Eq. 9 with lambda_min = exp(-alpha): w = exp(-alpha |Omega|/max|Omega|)."""
    a = omega.abs() * mask
    m = a.max()
    if not torch.isfinite(m) or m <= 0:
        return torch.ones_like(omega)
    import math

    return torch.exp(-(-math.log(w_min)) * (a / m)) * mask + (1 - mask)


def srcc(x, y):
    """Spearman rank correlation, on the flattened masked vectors."""
    def rank(v):
        idx = v.argsort()
        r = torch.empty_like(idx, dtype=torch.float64)
        r[idx] = torch.arange(v.numel(), dtype=torch.float64)
        return r

    rx, ry = rank(x.double()), rank(y.double())
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = rx.norm() * ry.norm()
    return float((rx * ry).sum() / d) if float(d) > 0 else float("nan")


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--rollouts", required=True,
                   help="verl dump jsonl ({input,output,acc}) or converted "
                        "({prompt,response,acc})")
    p.add_argument("--out", default="docs/omega_forms.json")
    p.add_argument("--w-min", type=float, default=0.7)
    p.add_argument("--max-groups", type=int, default=24)
    p.add_argument("--max-response-tokens", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def read_groups(path, max_groups):
    by_prompt = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pr = r.get("prompt") or r.get("input")
            rs = r.get("response") or r.get("output")
            acc = r.get("acc")
            if acc is None and r.get("score") is not None:
                acc = float(r["score"]) > 0.5
            if pr and rs and acc is not None:
                by_prompt[pr].append((rs, bool(acc)))
    groups = [(p, v) for p, v in by_prompt.items()
              if len(v) >= 4 and len(set(a for _, a in v)) > 1]
    groups.sort(key=lambda kv: -len(kv[1]))
    return groups[:max_groups]


def main(argv=None):
    args = build_argparser().parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device).eval()

    groups = read_groups(args.rollouts, args.max_groups)
    print(f"[omega] {len(groups)} mixed-outcome groups", flush=True)

    stats = defaultdict(list)
    pooled = defaultdict(list)
    for gi, (prompt, rs) in enumerate(groups):
        pid = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=1024)["input_ids"][0]
        toks, accs = [], []
        for resp, a in rs:
            ids = tok(resp, return_tensors="pt", truncation=True,
                      max_length=args.max_response_tokens,
                      add_special_tokens=False)["input_ids"][0]
            if ids.numel() >= 2:
                toks.append(ids)
                accs.append(a)
        if len(set(accs)) < 2:
            continue
        T = max(t.numel() for t in toks)
        B = len(toks)
        P = pid.numel()
        resp = torch.full((B, T), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((B, T))
        for i, t in enumerate(toks):
            resp[i, : t.numel()] = t
            mask[i, : t.numel()] = 1.0
        ids = torch.cat([pid.unsqueeze(0).expand(B, -1), resp], 1).to(args.device)
        att = torch.cat([torch.ones((B, P), dtype=torch.long),
                         mask.long()], 1).to(args.device)
        # Row at a time: B x T x |V| in fp32 is several GB at these lengths, and
        # this runs alongside training on the same GPU.
        ent = torch.zeros(B, T)
        logp = torch.zeros(B, T)
        for i in range(B):
            with torch.no_grad():
                out = model(input_ids=ids[i: i + 1], attention_mask=att[i: i + 1],
                            use_cache=False)
                lg = out.logits[0, P - 1: P - 1 + T, :].float()
                lsm = torch.log_softmax(lg, -1)
                ent[i] = (-(lsm.exp() * lsm).sum(-1)).cpu()
                logp[i] = lsm.gather(
                    -1, resp[i].to(args.device).unsqueeze(-1)).squeeze(-1).cpu()
            del out, lg, lsm

        r_ = torch.tensor([1.0 if a else -1.0 for a in accs])
        adv = ((r_ - r_.mean()) / (r_.std(unbiased=False) + _EPS)).unsqueeze(1).expand(B, T)

        pi = torch.exp(logp)
        d = delta(pi, ent)
        # First PPO epoch: log_prob == old_log_prob, so ratio == 1 exactly.
        ratio = torch.ones_like(pi)
        pi_old = pi

        # Eq. 6.  Inert at ratio == 1; carried so the accounting is complete.
        iclip = torch.ones_like(pi)
        # L is per group (sum of response lengths within the group), the paper's
        # definition -- not the micro-batch total.
        L = float(mask.sum())

        forms = {
            # what the released implementation computes
            "released": adv / pi_old.clamp(_EPS, 1.0) * d,
            # Theorem 1: importance-corrected == exact realised update
            "theorem_r": adv * ratio * iclip * d,
            # and with the paper's 1/L
            "theorem_r_overL": adv * ratio * iclip * d / L,
        }

        # Pool across groups: Eq. 9's max is over the batch B, and a real
        # micro-batch holds several queries. Per-group statistics cannot see
        # the 1/L effect at all, because L is constant inside a group.
        for name, om in forms.items():
            pooled[name].append(torch.nan_to_num(om, 0.0, 0.0, 0.0)[mask.bool()])
        pooled["_pi"].append(pi[mask.bool()])

        mb = mask.bool()
        ref = None
        for name, om in forms.items():
            om = torch.nan_to_num(om, 0.0, 0.0, 0.0) * mask
            a = om[mb].abs()
            w = eq9_weights(om, mask, args.w_min)[mb]
            flat = om.abs().flatten()
            amax = int(flat.argmax())
            band = 1.0 - args.w_min
            rec = {
                "tail_max_p50": float(a.max() / (a.median() + 1e-30)),
                "tail_p999_p50": float(torch.quantile(a.double(), 0.999)
                                       / (a.median() + 1e-30)),
                "pi_at_max": float(pi.flatten()[amax]),
                "tw_std": float(w.std()),
                "tw_mean": float(w.mean()),
                "frac_pinned": float((w > 1.0 - 0.01 * band).float().mean()),
                "frac_low": float((w < args.w_min + band / _THIRD).float().mean()),
            }
            if ref is None:
                ref = a
                rec["srcc_vs_released"] = 1.0
            else:
                rec["srcc_vs_released"] = srcc(a, ref)
            stats[name].append(rec)
        if (gi + 1) % 4 == 0:
            print(f"[omega] {gi+1}/{len(groups)}", flush=True)

    # ---- pooled: one flat vector per form, as a micro-batch actually sees it
    pi_all = torch.cat(pooled.pop("_pi"))
    ones = torch.ones_like(pi_all)
    pooled_summary, ref = {}, None
    for name, chunks in pooled.items():
        om = torch.cat(chunks)
        a = om.abs()
        w = eq9_weights(om, ones, args.w_min)
        band = 1.0 - args.w_min
        pooled_summary[name] = {
            "tail_max_p50": float(a.max() / (a.median() + 1e-30)),
            "tail_max_p99": float(a.max() / (torch.quantile(a.double(), 0.99) + 1e-30)),
            "pi_at_max": float(pi_all[int(a.argmax())]),
            "tw_std": float(w.std()),
            "tw_mean": float(w.mean()),
            "frac_pinned": float((w > 1.0 - 0.01 * band).float().mean()),
            "frac_low": float((w < args.w_min + band / _THIRD).float().mean()),
            "srcc_vs_released": 1.0 if ref is None else srcc(a, ref),
        }
        if ref is None:
            ref = a

    keys = ["tail_max_p50", "tail_p999_p50", "pi_at_max", "tw_std", "tw_mean",
            "frac_pinned", "frac_low", "srcc_vs_released"]
    summary = {n: {k: float(torch.tensor([r[k] for r in rows]).mean()) for k in keys}
               for n, rows in stats.items()}
    Path(args.out).write_text(json.dumps(
        {"per_group": summary, "pooled": pooled_summary,
         "n_groups": len(next(iter(stats.values()), []))}, indent=2))

    print(f"\n{'form':<20}{'max/p50':>12}{'p99.9/p50':>12}{'pi@max':>11}"
          f"{'tw_std':>10}{'pinned':>9}{'frac_low':>10}{'SRCC':>8}")
    for n, s in summary.items():
        print(f"{n:<20}{s['tail_max_p50']:>12.1f}{s['tail_p999_p50']:>12.1f}"
              f"{s['pi_at_max']:>11.2e}{s['tw_std']:>10.4f}"
              f"{s['frac_pinned']:>9.3f}{s['frac_low']:>10.4f}"
              f"{s['srcc_vs_released']:>8.3f}")
    print(f"\n--- pooled across groups (this is what a micro-batch sees) ---")
    print(f"{'form':<20}{'max/p50':>12}{'max/p99':>10}{'pi@max':>11}"
          f"{'tw_std':>10}{'pinned':>9}{'frac_low':>10}{'SRCC':>8}")
    for n, s_ in pooled_summary.items():
        print(f"{n:<20}{s_['tail_max_p50']:>12.1f}{s_['tail_max_p99']:>10.1f}"
              f"{s_['pi_at_max']:>11.2e}{s_['tw_std']:>10.4f}"
              f"{s_['frac_pinned']:>9.3f}{s_['frac_low']:>10.4f}"
              f"{s_['srcc_vs_released']:>8.3f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
