#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Does a high-A_H branch actually lead to the right answer?

STEER-F protects branch points whose future looks *diverse*.  Diversity is a
proxy: high forecast entropy can mean "several valid routes remain" or it can
mean "the policy has no idea what comes next".  Nothing in `A_H` distinguishes
those, and the reweighting is computed without ever looking at the reward.

The counter-argument is that it does not have to: the token weight only scales
the gradient (`w` in [0.9, 1.0]), while GRPO's advantage sets its sign, so a
confused branch that ends in a wrong answer is pushed down regardless.  That
argument has two holes worth measuring rather than debating:

  * at ~9.5% accuracy roughly 45% of eight-rollout groups are all-wrong, so
    their advantage is exactly zero and the reward corrects nothing there;
  * the protection is sign-agnostic, so shielding a genuinely confused branch
    also slows the policy's ability to resolve it.

This script measures the alignment directly, on rollouts that already carry the
verifier's verdict.  At every branch point where the siblings' *outcomes*
disagree, it asks whether the sibling with the higher `A_H` is the one that
went on to be correct.  Chance is 50%.

    python scripts/analyze_ah_vs_correctness.py \
        --model <path> --heads checkpoints/mtp_heads.pt \
        --rollouts rollout_data/.../rollouts.jsonl \
        --kappa 2 --gamma-h 0.7 --out docs/ah_vs_correctness.json

Three predictors are compared, because the interesting question is about the
*quantity*, not only about the forecaster that estimates it:

    mtp     A_H from the warmed-up MTP heads      (what training actually uses)
    oracle  A_H from the realised downstream entropy, read off the rollout
    local   the token's own entropy, no future term at all

If `oracle` is near chance, future entropy simply is not aligned with
correctness at this scale, and no better forecaster would rescue it — a result
for §Limitations either way.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from steer_f.entropy_forecast import (  # noqa: E402
    HeadCalibration,
    first_divergence,
    oracle_h_togo,
)
from steer_f.mtp_heads import MTPHeads, entropy_from_logits  # noqa: E402
from steer_f.omega_tilde import SteerFConfig  # noqa: E402
from steer_f.validation import binomial_test_greater  # noqa: E402
from steer_f.verl_integration import compute_a_h, forecast_h_togo  # noqa: E402


def build_argparser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True)
    p.add_argument("--heads", default=None, help="omit to score only oracle/local")
    p.add_argument("--rollouts", required=True)
    p.add_argument("--calib", default=None)
    p.add_argument("--out", default="docs/ah_vs_correctness.json")
    p.add_argument("--kappa", type=int, default=2)
    p.add_argument("--gamma-h", type=float, default=0.7)
    p.add_argument("--max-groups", type=int, default=120,
                   help="only mixed-outcome groups are usable, so ask for plenty")
    p.add_argument("--max-response-tokens", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16")
    return p


def load_mixed_groups(path, max_groups):
    """Prompt groups whose rollouts did not all get the same verdict.

    A group where every rollout is wrong (or every one right) carries no
    information about which branch was better -- and, not coincidentally, it is
    also a group where GRPO's advantage is exactly zero.
    """
    by_prompt = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            prompt = r.get("prompt") or r.get("input")
            resp = r.get("response") or r.get("output")
            acc = r.get("acc")
            if not prompt or not resp or acc is None:
                continue
            by_prompt[prompt].append((resp, bool(acc)))

    mixed, all_same = [], 0
    for prompt, rs in by_prompt.items():
        if len(rs) < 2:
            continue
        verdicts = {a for _, a in rs}
        if len(verdicts) < 2:
            all_same += 1
            continue
        mixed.append((prompt, rs))
    mixed.sort(key=lambda kv: -len(kv[1]))
    return mixed[:max_groups], all_same, len(by_prompt)


def main(argv=None):
    args = build_argparser().parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype).to(args.device).eval()
    lm = model.get_output_embeddings()

    heads = calib = None
    if args.heads and Path(args.heads).exists():
        ck = torch.load(args.heads, map_location="cpu", weights_only=False)
        if ck.get("untrained"):
            sys.exit("[align] refusing an UNTRAINED head checkpoint")
        heads = MTPHeads(**ck["config"]).to(args.device, dtype=dtype)
        heads.load_state_dict(ck["state_dict"])
        heads.eval()
        if args.calib and Path(args.calib).exists():
            d = json.loads(Path(args.calib).read_text())
            calib = HeadCalibration(
                scale=torch.tensor(d["scale"]), bias=torch.tensor(d["bias"]),
                temperature=torch.tensor(d.get("temperature", [1.0] * len(d["scale"]))))

    groups, all_same, n_prompts = load_mixed_groups(args.rollouts, args.max_groups)
    print(f"[align] {n_prompts} prompts, {all_same} with a unanimous verdict "
          f"(no signal, and zero GRPO advantage), {len(groups)} usable")
    if not groups:
        sys.exit("[align] no mixed-outcome group — nothing to measure")

    cfg = SteerFConfig(kappa=args.kappa, gamma_h=args.gamma_h)
    names = ["oracle", "local"] + (["mtp"] if heads is not None else [])
    wins = {n: 0 for n in names}
    trials = {n: 0 for n in names}
    n_branch_used = 0

    for gi, (prompt, rs) in enumerate(groups):
        pid = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=1024)["input_ids"][0]
        toks, accs = [], []
        for resp, acc in rs:
            ids = tok(resp, return_tensors="pt", truncation=True,
                      max_length=args.max_response_tokens,
                      add_special_tokens=False)["input_ids"][0]
            if ids.numel() >= 2:
                toks.append(ids)
                accs.append(acc)
        if len(set(accs)) < 2:
            continue

        T = max(t.numel() for t in toks)
        B, P = len(toks), pid.numel()
        resp_ids = torch.full((B, T), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((B, T), dtype=torch.long)
        for i, t in enumerate(toks):
            resp_ids[i, : t.numel()] = t
            mask[i, : t.numel()] = 1

        ids = torch.cat([pid.unsqueeze(0).expand(B, -1), resp_ids], 1).to(args.device)
        att = torch.cat([torch.ones((B, P), dtype=torch.long), mask], 1).to(args.device)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=att,
                        output_hidden_states=True, use_cache=False)
        ent = entropy_from_logits(out.logits[:, P - 1: P - 1 + T, :].float()).cpu()

        scores = {
            "oracle": compute_a_h(
                oracle_h_togo(ent, mask, args.kappa, args.gamma_h),
                resp_ids, mask, [gi] * B, cfg),
            # No future term at all: A_H built from the token's own entropy.
            "local": compute_a_h(ent, resp_ids, mask, [gi] * B, cfg),
        }
        if heads is not None:
            h = forecast_h_togo(out.hidden_states[-1][:, P:, :], lm, heads,
                                response_length=T, cfg=cfg, calib=calib,
                                chunk_size=2048).cpu().float()
            scores["mtp"] = compute_a_h(h, resp_ids, mask, [gi] * B, cfg)

        div = first_divergence(resp_ids, mask)
        pos = torch.arange(T).view(1, 1, T)
        is_branch = ((div.unsqueeze(-1) == pos).any(1)) & mask.bool()

        acc_t = torch.tensor(accs)
        for t in range(T):
            rows = [i for i in range(B) if is_branch[i, t]]
            if len(rows) < 2:
                continue
            right = [i for i in rows if acc_t[i]]
            wrong = [i for i in rows if not acc_t[i]]
            if not right or not wrong:
                continue          # outcomes agree here — nothing to discriminate
            n_branch_used += 1
            for n in names:
                s = scores[n]
                # every correct-vs-incorrect sibling pair at this branch
                for i in right:
                    for j in wrong:
                        trials[n] += 1
                        if float(s[i, t]) > float(s[j, t]):
                            wins[n] += 1
        if (gi + 1) % 10 == 0:
            print(f"[align] {gi+1}/{len(groups)} groups, "
                  f"{n_branch_used} discriminating branch points", flush=True)

    print(f"\n[align] {n_branch_used} branch points where sibling outcomes differed")
    print(f"{'predictor':10s} {'win rate':>9} {'pairs':>8} {'p (> chance)':>13}")
    result = {}
    for n in names:
        if trials[n] == 0:
            continue
        rate = wins[n] / trials[n]
        p = binomial_test_greater(wins[n], trials[n], 0.5)
        result[n] = {"win_rate": rate, "wins": wins[n], "pairs": trials[n], "p_value": p}
        print(f"{n:10s} {rate:>9.4f} {trials[n]:>8d} {p:>13.3g}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "n_branch_points": n_branch_used,
        "n_groups": len(groups),
        "unanimous_groups_skipped": all_same,
        "kappa": args.kappa, "gamma_h": args.gamma_h,
        "results": result,
        "reading": "win_rate is P(higher A_H sibling is the correct one); "
                   "0.5 is chance. Near 0.5 means future-entropy diversity is "
                   "not aligned with correctness at this scale.",
    }, indent=2))
    print(f"[align] wrote {args.out}")


if __name__ == "__main__":
    main()
