#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
# Licensed under the Apache License, Version 2.0
"""Is there anything for measure_forecast_locality.py to measure?

    python3 scripts/check_sibling_support.py validation_data/0.jsonl

WHY THIS EXISTS
    `scripts/measure_forecast_locality.py` loads a 1.5B model and runs a
    forward pass over every rollout -- an hour on CPU.  Everything it reports
    is restricted to the sibling support, and `A_H` is identically zero
    outside it, so a run over rollouts with no support produces nothing after
    an hour.

    Support is a function of TOKEN IDS ALONE (`sibling_support_stats` in
    steer_f/tree_rollout.py reproduces the trainer's mask offline), so the
    same question answers in seconds with no model, no GPU and no torch.
    Run this first.

WHAT THE NUMBER MEANS
    Training measured `support_frac` at .40-.42 on tree rollouts and .003 on
    plain ones -- independent samples share only the prompt, so they diverge
    in the first few tokens and there are no siblings after that.  Validation
    dumps are independent samples.  A low number here is therefore the
    expected reading, not a bug, and it is exactly what decides whether the
    expensive run is worth starting.

    Read `support_frac_by_decile` too: support concentrated in decile 0 means
    the measurement would describe the first few tokens of a response rather
    than the branch points at depths 64/192/384 where the intervention lives.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from steer_f.tree_rollout import sibling_support_stats  # noqa: E402


def load_groups(path, min_size, max_groups):
    """Same grouping as measure_forecast_locality.load_groups."""
    by_prompt = collections.defaultdict(list)
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
    groups = [v for v in by_prompt.values() if len(v) >= min_size]
    groups.sort(key=len, reverse=True)
    return groups[:max_groups]


def to_ids(groups, model, max_response_tokens):
    """Token ids, or whitespace words if no tokenizer is reachable."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except Exception as exc:
        print(f"[support] no tokenizer ({type(exc).__name__}) -- using whitespace "
              f"words.\n[support] words diverge sooner than subwords, so the "
              f"number below is a LOWER bound.")
        vocab: dict = {}
        return [[[vocab.setdefault(w, len(vocab))
                  for w in r.split()[:max_response_tokens]] for r in g]
                for g in groups]
    return [[tok(r, add_special_tokens=False)["input_ids"][:max_response_tokens]
             for r in g] for g in groups]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rollouts", help="JSONL with prompt/response or input/output")
    p.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    p.add_argument("--min-group-size", type=int, default=2)
    p.add_argument("--max-groups", type=int, default=48)
    p.add_argument("--max-response-tokens", type=int, default=1024)
    p.add_argument("--json", default=None, help="also write the stats here")
    args = p.parse_args(argv)

    groups = load_groups(args.rollouts, args.min_group_size, args.max_groups)
    if not groups:
        sys.exit(f"[support] no prompt group of size >= {args.min_group_size} "
                 f"in {args.rollouts}")

    ids = to_ids(groups, args.model, args.max_response_tokens)
    s = sibling_support_stats(ids)
    alive, frac = s["alive_positions"], s["support_frac"]
    est = int(alive * frac)

    print(f"\n  file               {args.rollouts}")
    print(f"  groups             {len(ids)}  (sizes {[len(g) for g in ids][:8]})")
    print(f"  alive positions    {alive:,}")
    print(f"  support_frac       {frac:.4f}"
          f"    <- training: .40-.42 tree, .003 plain")
    print(f"  mean_siblings      {s['mean_siblings']:.3f}")
    print(f"  estimated n        ~{est:,}   (branch_points_only upper bound)")
    print(f"  by decile          "
          + " ".join(f"{d:.3f}" for d in s["support_frac_by_decile"]))

    if est >= 2000:
        print("\n  -> worth running measure_forecast_locality.py.")
    elif est >= 200:
        print("\n  -> thin but usable; treat the correlations as indicative.")
    else:
        print("\n  -> too few positions. The locality run would report noise;")
        print("     it needs rollouts from a TREE run, not independent samples.")
    if s["support_frac_by_decile"][0] > 4 * (frac or 1e-9):
        print("     NOTE: support sits in decile 0 -- this describes the start")
        print("     of a response, not the branch points training intervenes at.")
    print()

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {"rollouts": args.rollouts, "n_groups": len(ids),
             "estimated_n": est, **s}, indent=2))
        print(f"  -> {args.json}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
