#!/usr/bin/env python3
"""Phase 1 — the sibling-ranking test that replaces G1's criterion 2.

G1 as specified asks whether high-A_H positions coincide with sibling
divergences.  That question cannot be answered: A_H is *built* from sibling
divergences, so it is exactly zero wherever siblings agree and nonzero wherever
they do not.  Measured on Phase-0 rollouts, |A_H| separated branch from
non-branch positions with AUC = 1.0000 for BOTH warmed-up and randomly
initialised heads, and recall@10% saturated at n_hit == k for both.  The
criterion tests the sibling bookkeeping, never the forecast.

What STEER-F actually needs from A_H is a *ranking among siblings at a branch
point*: which continuation has the more diverse future.  That is what this
script measures, against the entropy the policy really had downstream, taken
straight from the rollout.  Three predictors are compared:

  oracle      discounted sum of the policy's realised entropy at t+1..t+kappa.
              Needs no heads and no extra forward -- verl already computes the
              entropy tensor.  This is the ceiling.
  parallel    steer_f.mtp_heads.MTPHeads
  sequential  steer_f.mtp_sequential.SequentialMTPHeads

plus an untrained control for whichever head type is given, because identity-
initialised heads reduce H_togo to a multiple of local entropy and already
score well; the control is what separates "the forecast helps" from "local
entropy was enough".
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from steer_f.entropy_forecast import first_divergence, sibling_prefix_baseline
from steer_f.mtp_heads import MTPHeads, entropy_from_logits
from steer_f.mtp_sequential import SequentialMTPHeads
from steer_f.validation import within_problem_spearman
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_branch_recall import load_groups


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--rollouts", required=True)
    p.add_argument("--heads", action="append", default=[],
                   help="NAME=PATH, repeatable")
    p.add_argument("--kappa", type=int, default=2)
    p.add_argument("--gamma-h", type=float, default=0.7)
    p.add_argument("--max-groups", type=int, default=40)
    p.add_argument("--max-response-tokens", type=int, default=512)
    p.add_argument("--min-siblings", type=int, default=3)
    p.add_argument("--out", default="docs/phase1_rank_test.json")
    return p


def load_heads(path, device, dtype):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cls = SequentialMTPHeads if ck.get("sequential") else MTPHeads
    h = cls(**ck["config"]).to(device, dtype=dtype)
    h.load_state_dict(ck["state_dict"]); h.eval()
    ctrl = cls(**ck["config"]).to(device, dtype=dtype).eval()   # same init, untrained
    return h, ctrl, bool(ck.get("sequential"))


def main(argv=None):
    args = build_argparser().parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev, dt = "cuda", torch.bfloat16
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt).to(dev).eval()
    lm, emb = model.get_output_embeddings(), model.get_input_embeddings()

    variants = {}
    for spec in args.heads:
        name, path = spec.split("=", 1)
        h, c, seq = load_heads(path, dev, dt)
        variants[name] = (h, seq)
        variants[name + "-untrained"] = (c, seq)

    groups = load_groups(args.rollouts, 2, args.max_groups)
    KAP, GAM = args.kappa, args.gamma_h
    pairs = {k: ([], [], []) for k in list(variants) + ["oracle"]}

    for gi, (pr, resps) in enumerate(groups):
        pid = tok(pr, return_tensors="pt", truncation=True, max_length=1024)["input_ids"][0]
        rid = [tok(r, return_tensors="pt", truncation=True,
                   max_length=args.max_response_tokens,
                   add_special_tokens=False)["input_ids"][0] for r in resps]
        rid = [r for r in rid if r.numel() >= 2]
        if len(rid) < args.min_siblings:
            continue
        T = max(r.numel() for r in rid); B = len(rid); P = pid.numel()
        resp = torch.full((B, T), tok.pad_token_id, dtype=torch.long)
        msk = torch.zeros((B, T), dtype=torch.long)
        for i, r in enumerate(rid):
            resp[i, :r.numel()] = r; msk[i, :r.numel()] = 1
        ids = torch.cat([pid.unsqueeze(0).expand(B, -1), resp], 1).to(dev)
        att = torch.cat([torch.ones((B, P), dtype=torch.long), msk], 1).to(dev)
        with torch.no_grad():
            o = model(input_ids=ids, attention_mask=att,
                      output_hidden_states=True, use_cache=False)
        hidden = o.hidden_states[-1][:, P:, :]
        ent = entropy_from_logits(o.logits[:, P - 1:P - 1 + T, :].float()).cpu()

        preds = {}
        ho = torch.zeros_like(ent)
        for k in range(1, KAP + 1):
            sh = torch.zeros_like(ent); sh[:, :T - k] = ent[:, k:] * msk[:, k:]
            ho += (GAM ** k) * sh
        preds["oracle"] = ho
        rd = resp.to(dev)
        for name, (h, seq) in variants.items():
            with torch.no_grad():
                e = (h.forecast_entropy(hidden, lm, tokens=rd, kappa=KAP, embedding=emb)
                     if seq else h.forecast_entropy(hidden, lm, kappa=KAP))
            v = torch.zeros_like(ent)
            for k in range(KAP):
                v += (GAM ** (k + 1)) * e[k].cpu().float()
            preds[name] = v * msk

        div = first_divergence(resp, msk)
        pos = torch.arange(T).view(1, 1, T)
        isb = ((div.unsqueeze(-1) == pos).any(1)) & msk.bool()
        for t in range(T):
            rows = [i for i in range(B) if isb[i, t]]
            if len(rows) < args.min_siblings:
                continue
            for name, v in preds.items():
                F, Tr, I = pairs[name]
                F += [float(v[i, t]) for i in rows]
                Tr += [float(ho[i, t]) for i in rows]
                I += [(gi, t)] * len(rows)

    out = {}
    for name, (F, Tr, I) in pairs.items():
        if not F:
            continue
        r = within_problem_spearman(F, Tr, I, min_per_problem=args.min_siblings)
        out[name] = {"rho_pooled": r["rho_pooled"], "rho_mean": r["rho_mean"],
                     "n_branch_points": r["n_problems"], "n_obs": len(F)}
    print(f"{'predictor':26s} {'rho_pooled':>11} {'rho_mean':>10} {'branch pts':>11}")
    for n, v in sorted(out.items(), key=lambda kv: -kv[1]["rho_pooled"]):
        print(f"{n:26s} {v['rho_pooled']:>11.4f} {v['rho_mean']:>10.4f} {v['n_branch_points']:>11d}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"kappa": KAP, "gamma_h": GAM, "results": out}, indent=2))
    print(f"[rank-test] wrote {args.out}")


if __name__ == "__main__":
    main()
