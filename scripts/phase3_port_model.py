#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Phase 3 — port the MTP heads to another model family.

`MTPHeads` is architecture-agnostic: it consumes the final hidden state and
borrows the body's unembedding, so only `hidden_size` and `vocab_size` change
between Qwen, Llama and EXAONE.  This script does three things:

1. reads the target model's dims and emits a correctly-shaped head checkpoint
   (freshly initialised — head weights are not transferable across families,
   the hidden spaces are unrelated);
2. checks the things that actually differ and can silently break a run —
   tied vs. untied embeddings, vocab/embedding-matrix mismatch, dtype;
3. reports the training-set pass-rate distribution, because plan §4 warns that
   a weak base model produces mostly all-wrong groups, which gives GRPO no
   gradient at all and would make a STEER-F comparison meaningless.

    python scripts/phase3_port_model.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --out checkpoints/mtp_heads_llama3b_init.pt \
        --reference-heads checkpoints/mtp_heads_qwen7b.pt

Per plan §5, hyper-parameters (lambda, kappa, gamma_H, norm) are carried over
from Qwen unchanged — "it just works" is the claim being tested.  Only the
per-head calibration is refit, since it is a property of the head, not of the
method.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from steer_f.mtp_heads import MTPHeads  # noqa: E402


def inspect_model(model):
    """Dims and the embedding facts that break a naive port."""
    cfg = model.config
    lm_head = model.get_output_embeddings()
    input_emb = model.get_input_embeddings()
    return {
        "hidden_size": cfg.hidden_size,
        "config_vocab_size": cfg.vocab_size,
        "lm_head_out": lm_head.weight.shape[0],
        "embedding_rows": input_emb.weight.shape[0],
        "tied_embeddings": bool(getattr(cfg, "tie_word_embeddings", False)),
        "lm_head_has_bias": lm_head.bias is not None,
        "dtype": str(next(model.parameters()).dtype),
        "n_layers": getattr(cfg, "num_hidden_layers", None),
        "architecture": type(model).__name__,
    }


def check_port(info, reference=None):
    """Return warnings for everything that would bite later."""
    problems = []
    if info["lm_head_out"] != info["config_vocab_size"]:
        problems.append(
            f"lm_head outputs {info['lm_head_out']} logits but config.vocab_size is "
            f"{info['config_vocab_size']}. Padded vocabularies are common; STEER-F sizes "
            "the heads from lm_head, so entropies stay correct, but any code indexing by "
            "config.vocab_size will be off."
        )
    if info["lm_head_out"] != info["embedding_rows"]:
        problems.append(
            f"lm_head rows ({info['lm_head_out']}) != input embedding rows "
            f"({info['embedding_rows']}); tie_unembedding=True assumes one shared matrix."
        )
    if info["lm_head_has_bias"]:
        problems.append(
            "lm_head carries a bias. entropy_from_logits handles it, but a fused "
            "linear+CE kernel taking only the weight would silently drop it."
        )
    if reference is not None:
        if reference["hidden_size"] != info["hidden_size"]:
            problems.append(
                f"hidden_size differs from the reference ({reference['hidden_size']} -> "
                f"{info['hidden_size']}): head weights cannot be reused, only the recipe."
            )
        if reference["lm_head_out"] != info["lm_head_out"]:
            problems.append(
                f"vocabulary differs ({reference['lm_head_out']} -> {info['lm_head_out']}): "
                "the reference calibration is invalid here — refit it (plan §5.2)."
            )
    return problems


def group_pass_rate_report(path, uid_column="uid", reward_column="reward"):
    """Distribution of per-prompt pass rates — the plan §5.4 pre-flight check."""
    import pandas as pd

    df = pd.read_parquet(path)
    if uid_column not in df.columns or reward_column not in df.columns:
        return {"error": f"need columns {uid_column!r} and {reward_column!r}; "
                         f"found {list(df.columns)}"}
    rates = df.groupby(uid_column)[reward_column].mean()
    n = len(rates)
    all_wrong = float((rates <= 0).mean())
    all_right = float((rates >= 1).mean())
    return {
        "n_groups": int(n),
        "frac_all_wrong": all_wrong,
        "frac_all_correct": all_right,
        "frac_informative": 1.0 - all_wrong - all_right,
        "mean_pass_rate": float(rates.mean()),
        "verdict": (
            "TOO HARD — over half the groups give no gradient; move to an easier subset "
            "and say so in the report (plan §5.4)"
            if all_wrong > 0.5 else
            "TOO EASY — most groups are saturated; the entropy question barely arises"
            if all_right > 0.5 else "OK"
        ),
    }


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="target model id or path")
    p.add_argument("--out", help="write a freshly initialised head checkpoint here")
    p.add_argument("--reference-heads", help="Qwen checkpoint, for a compatibility diff")
    p.add_argument("--num-heads", type=int, default=None,
                   help="defaults to the reference's K, else 8")
    p.add_argument("--head-hidden", type=int, default=None,
                   help="defaults to the reference's, else 1024")
    p.add_argument("--rollout-parquet", help="run the group pass-rate check on this file")
    p.add_argument("--uid-column", default="uid")
    p.add_argument("--reward-column", default="reward")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--report", help="write the findings as JSON")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    from transformers import AutoModelForCausalLM

    reference = None
    num_heads, head_hidden = args.num_heads or 8, args.head_hidden or 1024
    if args.reference_heads:
        ref = torch.load(args.reference_heads, map_location="cpu")
        reference = {
            "hidden_size": ref["config"]["hidden_size"],
            "lm_head_out": ref["config"]["vocab_size"],
            "model": ref.get("model"),
        }
        num_heads = args.num_heads or ref["config"]["num_heads"]
        head_hidden = args.head_hidden or ref["config"]["head_hidden"]
        print(f"[port] reference: {reference['model']} "
              f"H={reference['hidden_size']} V={reference['lm_head_out']} K={num_heads}")

    print(f"[port] loading {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), trust_remote_code=True
    )
    info = inspect_model(model)
    print("[port] target model:")
    for k, v in info.items():
        print(f"         {k}: {v}")

    problems = check_port(info, reference)
    if problems:
        print("\n[port] ATTENTION:")
        for msg in problems:
            print(f"       - {msg}")
    else:
        print("\n[port] no compatibility issues found")

    report = {"model": args.model, "info": info, "warnings": problems}

    if args.out:
        heads = MTPHeads(
            hidden_size=info["hidden_size"],
            vocab_size=info["lm_head_out"],  # lm_head is the authority, not config
            num_heads=num_heads,
            head_hidden=head_hidden,
            tie_unembedding=True,
            dtype=getattr(torch, args.dtype),
        )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": heads.state_dict(),
                "config": {
                    "hidden_size": info["hidden_size"],
                    "vocab_size": info["lm_head_out"],
                    "num_heads": num_heads,
                    "head_hidden": head_hidden,
                    "tie_unembedding": True,
                },
                "model": args.model,
                "note": "freshly initialised — run phase1_warmup_heads.py before use",
            },
            out_path,
        )
        n_params = sum(p.numel() for p in heads.parameters())
        print(f"\n[port] wrote {out_path} ({n_params/1e6:.1f}M params, untrained)")
        print("[port] next: scripts/phase1_warmup_heads.py, then a reduced phase1_validate "
              "(10 problems / 200 prefixes) to refit calibration only.")
        report["checkpoint"] = str(out_path)

    if args.rollout_parquet:
        stats = group_pass_rate_report(args.rollout_parquet, args.uid_column, args.reward_column)
        print("\n[port] training-set group pass rates:")
        for k, v in stats.items():
            print(f"         {k}: {v}")
        report["group_pass_rates"] = stats

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"[port] wrote {args.report}")


if __name__ == "__main__":
    main()
