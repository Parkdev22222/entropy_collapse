#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Phase 1 — the second half of gate G1: branch recall@10% on real rollouts.

`scripts/phase1_validate.py` measures criterion 1 (within-problem Spearman) and
then hands `evaluate_gate_g1` `recall=nan, n_branch=0` with a printed note that
the recall half needs a real rollout group.  Nothing computed it, so G1's
criterion 2 was structurally unsatisfiable and the gate could never pass.  This
script closes that hole.

Why it needs rollouts rather than the prefixes phase1_validate samples: a
*true branch point* is defined by siblings — position `t` of rollout `i` counts
when some rollout sharing the prefix `[0, t)` chose a different token at `t`
(`steer_f.monitors.branch_recall_at_k`).  Independently sampled continuations
have no siblings, so the quantity does not exist there.  A Phase-0 dump does
have them: `rollout.n=8` means eight rollouts per prompt.

    python scripts/phase1_branch_recall.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --heads checkpoints/mtp_heads.pt \
        --rollouts rollout_data/STEER-F-small/<run>/rollouts.jsonl \
        --calib checkpoints/mtp_calibration.json \
        --kappa 4 --gamma-h 0.85 \
        --phase1-results docs/phase1_results.json \
        --out docs/phase1_recall.json

`A_H` is built with `steer_f.verl_integration.forecast_h_togo` / `compute_a_h`,
the same functions the training patch calls, so this doubles as an offline
exercise of the index alignment that Phase 2 depends on.
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
from steer_f.monitors import branch_recall_at_k  # noqa: E402
from steer_f.mtp_heads import MTPHeads  # noqa: E402
from steer_f.omega_tilde import SteerFConfig  # noqa: E402
from steer_f.validation import evaluate_gate_g1  # noqa: E402
from steer_f.verl_integration import compute_a_h, forecast_h_togo  # noqa: E402


def build_argparser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True)
    p.add_argument("--heads", required=True)
    p.add_argument("--rollouts", required=True,
                   help="JSONL from scripts/phase0_collect_rollouts.py")
    p.add_argument("--calib", default=None, help="checkpoints/mtp_calibration.json")
    p.add_argument("--out", required=True)
    p.add_argument("--phase1-results", default=None,
                   help="docs/phase1_results.json — supplies rho for the joint verdict")
    p.add_argument("--kappa", type=int, default=4)
    p.add_argument("--gamma-h", type=float, default=0.85)
    p.add_argument("--baseline", default="sibling", choices=["sibling", "group"])
    p.add_argument("--top-frac", type=float, default=0.1)
    p.add_argument("--no-restrict-to-support", dest="restrict_to_support",
                   action="store_false",
                   help="score every masked position instead of only those where a "
                        "sibling remains. Reproduces the uninformative variant")
    p.set_defaults(restrict_to_support=True)
    p.add_argument("--max-groups", type=int, default=64,
                   help="prompt groups to score; G1's binomial test wants n_branch >= 30")
    p.add_argument("--min-group-size", type=int, default=2,
                   help="a group of 1 has no siblings and no branch points")
    p.add_argument("--max-response-tokens", type=int, default=1024)
    p.add_argument("--max-prompt-tokens", type=int, default=1024)
    p.add_argument("--chunk-size", type=int, default=2048)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    return p


def load_groups(path, min_size, max_groups):
    """Group rollout records by prompt.  Returns [(prompt, [response, ...])]."""
    by_prompt = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prompt = rec.get("prompt") or rec.get("input")
            response = rec.get("response") or rec.get("output")
            if not prompt or not response:
                continue
            by_prompt[prompt].append(response)

    groups = [(p, r) for p, r in by_prompt.items() if len(r) >= min_size]
    # Largest groups first: they carry the most sibling pairs per forward pass.
    groups.sort(key=lambda kv: -len(kv[1]))
    return groups[:max_groups]


def main(argv=None):
    args = build_argparser().parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    ).to(args.device).eval()

    ckpt = torch.load(args.heads, map_location="cpu", weights_only=False)
    if ckpt.get("untrained"):
        sys.exit("[recall] refusing: --heads is an UNTRAINED (dummy) checkpoint")
    heads = MTPHeads(**ckpt["config"]).to(args.device, dtype=dtype)
    heads.load_state_dict(ckpt["state_dict"])
    heads.eval()
    if args.kappa > ckpt["config"]["num_heads"]:
        sys.exit(f"[recall] kappa={args.kappa} > trained heads K={ckpt['config']['num_heads']}")

    calib = None
    if args.calib and Path(args.calib).exists():
        d = json.loads(Path(args.calib).read_text())
        calib = HeadCalibration(
            scale=torch.tensor(d["scale"]),
            bias=torch.tensor(d["bias"]),
            temperature=torch.tensor(d.get("temperature", [1.0] * len(d["scale"]))),
        )
        print(f"[recall] calibration loaded from {args.calib}")
    else:
        print("[recall] no calibration — distant heads are biased upward (ablation A6)")

    cfg = SteerFConfig(kappa=args.kappa, gamma_h=args.gamma_h, baseline=args.baseline)

    groups = load_groups(args.rollouts, args.min_group_size, args.max_groups)
    if not groups:
        sys.exit(f"[recall] no prompt group of size >= {args.min_group_size} in {args.rollouts}")
    print(f"[recall] {len(groups)} groups, sizes "
          f"{min(len(r) for _, r in groups)}..{max(len(r) for _, r in groups)}")

    all_a_h, all_resp, all_mask, all_uid = [], [], [], []

    for gi, (prompt, responses) in enumerate(groups):
        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=args.max_prompt_tokens)["input_ids"][0]
        resp_ids = [
            tokenizer(r, return_tensors="pt", truncation=True,
                      max_length=args.max_response_tokens,
                      add_special_tokens=False)["input_ids"][0]
            for r in responses
        ]
        resp_ids = [r for r in resp_ids if r.numel() >= 2]
        if len(resp_ids) < args.min_group_size:
            continue

        T = max(r.numel() for r in resp_ids)
        B = len(resp_ids)
        pad_id = tokenizer.pad_token_id
        resp = torch.full((B, T), pad_id, dtype=torch.long)
        mask = torch.zeros((B, T), dtype=torch.long)
        for i, r in enumerate(resp_ids):
            resp[i, : r.numel()] = r
            mask[i, : r.numel()] = 1

        P = prompt_ids.numel()
        ids = torch.cat([prompt_ids.unsqueeze(0).expand(B, -1), resp], dim=1).to(args.device)
        attn = torch.cat([torch.ones((B, P), dtype=torch.long), mask], dim=1).to(args.device)

        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=attn,
                        output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[-1]

        h_vals = forecast_h_togo(
            hidden, model.get_output_embeddings(), heads,
            response_length=T, cfg=cfg, calib=calib, chunk_size=args.chunk_size,
        )
        a_h = compute_a_h(h_vals.cpu().float(), resp, mask, [gi] * B, cfg)

        all_a_h.append(a_h)
        all_resp.append(resp)
        all_mask.append(mask)
        all_uid.extend([gi] * B)
        if (gi + 1) % 8 == 0:
            print(f"[recall] {gi+1}/{len(groups)} groups", flush=True)

    if not all_a_h:
        sys.exit("[recall] every group was dropped — responses too short?")

    # Pad every group to the global T so the top decile is taken over the whole
    # set, exactly as the training loop takes it over a rollout batch.
    T = max(t.shape[1] for t in all_a_h)

    def pad_to(t, fill):
        if t.shape[1] == T:
            return t
        pad = torch.full((t.shape[0], T - t.shape[1]), fill, dtype=t.dtype)
        return torch.cat([t, pad], dim=1)

    a_h = torch.cat([pad_to(t, 0.0) for t in all_a_h])
    resp = torch.cat([pad_to(t, tokenizer.pad_token_id) for t in all_resp])
    mask = torch.cat([pad_to(t, 0) for t in all_mask])

    # Restrict to positions where A_H is defined at all. Outside the sibling
    # support A_H is exactly zero and no branch point can occur, so including
    # those positions makes the statistic measure the support rather than the
    # forecast: with untrained heads it scored 0.4711 against 0.4693 trained,
    # i.e. the test could not tell a warmed-up forecaster from a random one.
    support = sibling_support(resp, mask, all_uid) if args.restrict_to_support else None
    stats = branch_recall_at_k(a_h, resp, mask, all_uid,
                               top_frac=args.top_frac, support=support)
    if support is not None:
        print(f"[recall] sibling support: {int(support.sum())} of "
              f"{int(mask.sum())} masked positions "
              f"({float(support.sum()) / max(1, int(mask.sum())):.1%})")
    print(f"[recall] {json.dumps(stats, indent=2)}")

    rho_mean = float("nan")
    if args.phase1_results and Path(args.phase1_results).exists():
        d = json.loads(Path(args.phase1_results).read_text())
        rho_mean = float(d.get("chosen", {}).get("rho_mean", float("nan")))
        print(f"[recall] rho_mean={rho_mean:.4f} from {args.phase1_results}")

    # The null selection rate is the realised top-decile size, not the nominal
    # top_frac — README: "귀무 선택률은 명목 0.1이 아니라 실제 상위 10분위 크기".
    # branch_recall_at_k already returns it as `baseline` = selected/n_valid.
    selection_rate = stats["baseline"] if stats["n_valid"] else args.top_frac

    gate = evaluate_gate_g1(
        rho_mean=rho_mean,
        recall=stats["recall"],
        n_branch=stats["n_branch"],
        n_hit=stats["n_hit"],
        selection_rate=selection_rate,
    )
    print(gate.summary())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "recall_stats": stats,
        "selection_rate": selection_rate,
        "rho_mean": rho_mean,
        "kappa": args.kappa,
        "gamma_h": args.gamma_h,
        "baseline": args.baseline,
        "n_groups": len(set(all_uid)),
        "restricted_to_support": bool(args.restrict_to_support),
        "gate_passed": gate.passed,
        "gate_summary": gate.summary(),
    }, indent=2))
    print(f"[recall] wrote {args.out}")


if __name__ == "__main__":
    main()
