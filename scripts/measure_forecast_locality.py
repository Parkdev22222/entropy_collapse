#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
# Licensed under the Apache License, Version 2.0
"""Is the future term reducible to a local one?

    python3 scripts/measure_forecast_locality.py \
        --model Qwen/Qwen2.5-Math-1.5B \
        --heads checkpoints/mtp_heads_Qwen2.5-Math-1.5B-paper.pt \
        --calib checkpoints/mtp_calibration_Qwen2.5-Math-1.5B-paper.json \
        --rollouts rollout_data/.../rollouts.jsonl \
        --kappa 2 --gamma-h 0.7 \
        --out docs/forecast_locality.json

WHY THIS EXISTS -- the sharpest objection to the paper
    kappa=2 with gamma_H=0.7 and the measured calibration gives

        H_togo = 0.7206 * H_1 + 0.0636 * H_2 + 0.106

    so head 2 contributes 8% and the "future" horizon is effectively one step.
    A reviewer will therefore ask: is A_H anything a purely local quantity
    could not have produced? If the answer is no, the MTP heads, Phase 1 and
    the calibration are machinery around a local signal, and the paper's
    two-channel framing collapses to a re-weighting of the first channel.

THE TEST
    Apply the SAME operator to two different inputs, on the same positions:

        A_H      = sibling_baseline( shift( H_togo ) )       the treatment
        A_local  = sibling_baseline( shift( H_local ) )      the local twin

    where H_local is the policy's own next-token entropy H(pi(.|s_t)) -- the
    quantity the local channel already sees, and the one STEER's Omega is built
    from. Because both go through compute_a_h with the same config, mask and
    grouping, any difference between them is the forecast's contribution and
    nothing else.

    Reported on the sibling support (where A_H can be nonzero at all -- outside
    it both are identically zero and including those positions measures the
    support rather than the forecast; that mistake once scored an untrained
    forecaster at 0.4711 against 0.4693 trained):

      corr(H_togo, H_local)   levels. Expected to be high -- H_1 IS a
                              one-step-ahead entropy. High here is not a
                              problem by itself, which is the point of the
                              next two.
      corr(A_H, A_local)      the ranking that actually drives the weights.
                              THIS is the number that answers the objection.
      R^2 of A_H ~ A_local    how much of the treatment a local proxy explains.
      sign agreement          how often the two would attenuate the same
                              sibling. The weights are monotone in the sign.

    Low rank correlation with high level correlation is the informative
    outcome, and the mechanism is not subtle: H_local is a property of the
    position, shared by every sibling that reaches it, so the sibling baseline
    subtracts most of it away. What survives in A_H is the part of the forecast
    that depends on WHICH token was taken -- exactly the visitation quantity.

WHAT WOULD FALSIFY THE PAPER'S CLAIM
    corr(A_H, A_local) near 1 with R^2 near 1. Then the future channel is a
    local channel wearing a hat, and the honest move is to say so and report
    the cheaper local twin as the method.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from steer_f.entropy_forecast import HeadCalibration, sibling_support  # noqa: E402
from steer_f.mtp_heads import MTPHeads  # noqa: E402
from steer_f.omega_tilde import SteerFConfig  # noqa: E402
from steer_f.verl_integration import compute_a_h, forecast_h_togo  # noqa: E402


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--heads", required=True)
    p.add_argument("--rollouts", required=True,
                   help="JSONL with prompt/response (scripts/phase0_collect_rollouts.py)")
    p.add_argument("--calib", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--kappa", type=int, default=2)
    p.add_argument("--gamma-h", type=float, default=0.7)
    p.add_argument("--baseline", default="sibling", choices=["sibling", "group"])
    p.add_argument("--max-groups", type=int, default=48)
    p.add_argument("--min-group-size", type=int, default=2)
    p.add_argument("--max-response-tokens", type=int, default=1024)
    p.add_argument("--max-prompt-tokens", type=int, default=1024)
    p.add_argument("--chunk-size", type=int, default=2048)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "bfloat16", "float16"])
    return p


def load_groups(path, min_size, max_groups):
    by_prompt = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prompt = rec.get("prompt") or rec.get("input")
            response = rec.get("response") or rec.get("output")
            if prompt and response:
                by_prompt[prompt].append(response)
    groups = [(p, r) for p, r in by_prompt.items() if len(r) >= min_size]
    groups.sort(key=lambda kv: -len(kv[1]))
    return groups[:max_groups]


# ------------------------------------------------------------- statistics
def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double() - x.double().mean()
    y = y.double() - y.double().mean()
    d = x.norm() * y.norm()
    return float((x @ y) / d) if d > 0 else float("nan")


def ranks(x: torch.Tensor) -> torch.Tensor:
    """Tie-corrected ranks, so Spearman is Pearson on these."""
    order = torch.argsort(x)
    sorted_x = x[order]
    r = torch.empty_like(x, dtype=torch.float64)
    n = x.numel()
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    return pearson(ranks(x), ranks(y))


def summarise(a_h: torch.Tensor, a_loc: torch.Tensor,
              h_togo: torch.Tensor, h_loc: torch.Tensor) -> dict:
    r = pearson(a_h, a_loc)
    both_nonzero = (a_h != 0) & (a_loc != 0)
    sign_agree = float(((a_h > 0) == (a_loc > 0))[both_nonzero].double().mean()) \
        if int(both_nonzero.sum()) else float("nan")
    return {
        "n_positions": int(a_h.numel()),
        "levels_pearson_h_togo_vs_h_local": pearson(h_togo, h_loc),
        "levels_spearman_h_togo_vs_h_local": spearman(h_togo, h_loc),
        "advantage_pearson_a_h_vs_a_local": r,
        "advantage_spearman_a_h_vs_a_local": spearman(a_h, a_loc),
        "advantage_r_squared": r * r if math.isfinite(r) else float("nan"),
        "sign_agreement": sign_agree,
        "a_h_absmean": float(a_h.abs().mean()),
        "a_local_absmean": float(a_loc.abs().mean()),
        "h_togo_mean": float(h_togo.mean()),
        "h_local_mean": float(h_loc.mean()),
    }


def main(argv=None):
    args = build_argparser().parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True).to(args.device).eval()

    ckpt = torch.load(args.heads, map_location="cpu", weights_only=False)
    if ckpt.get("untrained"):
        sys.exit("[locality] --heads is an UNTRAINED checkpoint; this measures nothing")
    heads = MTPHeads(**ckpt["config"]).to(args.device, dtype=dtype)
    heads.load_state_dict(ckpt["state_dict"])
    heads.eval()
    if args.kappa > ckpt["config"]["num_heads"]:
        sys.exit(f"[locality] kappa={args.kappa} > trained heads K={ckpt['config']['num_heads']}")

    calib = None
    if args.calib and Path(args.calib).exists():
        d = json.loads(Path(args.calib).read_text())
        calib = HeadCalibration(
            scale=torch.tensor(d["scale"]), bias=torch.tensor(d["bias"]),
            temperature=torch.tensor(d.get("temperature", [1.0] * len(d["scale"]))))
        print(f"[locality] calibration from {args.calib}")
    else:
        print("[locality] NO calibration -- distant heads biased upward (ablation A6)")

    cfg = SteerFConfig(kappa=args.kappa, gamma_h=args.gamma_h, baseline=args.baseline)
    groups = load_groups(args.rollouts, args.min_group_size, args.max_groups)
    if not groups:
        sys.exit(f"[locality] no prompt group of size >= {args.min_group_size}")
    print(f"[locality] {len(groups)} groups")

    keep_a_h, keep_a_loc, keep_htg, keep_hloc = [], [], [], []
    keep_branch = []

    for gi, (prompt, responses) in enumerate(groups):
        prompt_ids = tok(prompt, return_tensors="pt", truncation=True,
                         max_length=args.max_prompt_tokens)["input_ids"][0]
        resp_ids = [tok(r, return_tensors="pt", truncation=True,
                        max_length=args.max_response_tokens,
                        add_special_tokens=False)["input_ids"][0] for r in responses]
        resp_ids = [r for r in resp_ids if r.numel() >= 2]
        if len(resp_ids) < args.min_group_size:
            continue

        T, B = max(r.numel() for r in resp_ids), len(resp_ids)
        resp = torch.full((B, T), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((B, T), dtype=torch.long)
        for i, r in enumerate(resp_ids):
            resp[i, :r.numel()] = r
            mask[i, :r.numel()] = 1

        P = prompt_ids.numel()
        ids = torch.cat([prompt_ids.unsqueeze(0).expand(B, -1), resp], dim=1).to(args.device)
        attn = torch.cat([torch.ones((B, P), dtype=torch.long), mask], dim=1).to(args.device)

        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=attn,
                        output_hidden_states=True, use_cache=False)
            hidden = out.hidden_states[-1]
            # H(pi(.|s_t)) for the response positions. Position t of the
            # response is predicted by the logits at index P+t-1, the same
            # off-by-one verl applies when it logs actor/entropy.
            logits = out.logits[:, P - 1: P - 1 + T, :].float()
            logp = torch.log_softmax(logits, dim=-1)
            h_local = -(logp.exp() * logp).sum(-1).cpu()
            del out, logits, logp

        h_togo = forecast_h_togo(hidden, model.get_output_embeddings(), heads,
                                 response_length=T, cfg=cfg, calib=calib,
                                 chunk_size=args.chunk_size).cpu().float()

        uid = [gi] * B
        a_h = compute_a_h(h_togo, resp, mask, uid, cfg)
        # The local twin: same operator, same grouping, same mask.
        a_loc = compute_a_h(h_local, resp, mask, uid, cfg)

        support = sibling_support(resp, B, mask).bool() & mask.bool()
        if not bool(support.any()):
            continue
        branch = (a_h != 0) & support

        keep_a_h.append(a_h[support])
        keep_a_loc.append(a_loc[support])
        keep_htg.append(h_togo[support])
        keep_hloc.append(h_local[support])
        keep_branch.append(branch[support])
        if (gi + 1) % 8 == 0:
            print(f"[locality] {gi + 1}/{len(groups)} groups", flush=True)

    if not keep_a_h:
        sys.exit("[locality] every group was dropped -- responses too short?")

    a_h = torch.cat(keep_a_h)
    a_loc = torch.cat(keep_a_loc)
    htg = torch.cat(keep_htg)
    hloc = torch.cat(keep_hloc)
    br = torch.cat(keep_branch)

    res = {
        "config": {"model": args.model, "heads": args.heads, "calib": args.calib,
                   "kappa": args.kappa, "gamma_h": args.gamma_h,
                   "baseline": args.baseline, "n_groups": len(keep_a_h)},
        "sibling_support": summarise(a_h, a_loc, htg, hloc),
        "branch_points_only": summarise(a_h[br], a_loc[br], htg[br], hloc[br])
        if int(br.sum()) > 2 else {"n_positions": int(br.sum())},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))

    print(f"\n[locality] -> {args.out}\n")
    for scope in ("sibling_support", "branch_points_only"):
        s = res[scope]
        if s.get("n_positions", 0) <= 2:
            print(f"  {scope}: too few positions ({s.get('n_positions')})")
            continue
        print(f"  {scope}  (n = {s['n_positions']})")
        print(f"    levels     r(H_togo, H_local)      = "
              f"{s['levels_pearson_h_togo_vs_h_local']:+.3f}  "
              f"(rho {s['levels_spearman_h_togo_vs_h_local']:+.3f})")
        print(f"    ADVANTAGE  r(A_H,    A_local)      = "
              f"{s['advantage_pearson_a_h_vs_a_local']:+.3f}  "
              f"(rho {s['advantage_spearman_a_h_vs_a_local']:+.3f})")
        print(f"               R^2 explained by local  = {s['advantage_r_squared']:.3f}")
        print(f"               sign agreement          = {s['sign_agreement']:.3f}")
        print()
    print("  Read it as: high levels correlation is expected and harmless; the")
    print("  claim survives when the ADVANTAGE correlation and R^2 are low, because")
    print("  that is the quantity the token weights are monotone in.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
