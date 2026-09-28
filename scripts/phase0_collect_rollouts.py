#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Phase 0 -> Phase 1 bridge: merge verl's per-step rollout dumps into one JSONL.

`trainer.rollout_data_dir` gives one file per training step,
`<rollout_data_dir>/<step>.jsonl`, whose records are

    {"input": <decoded prompt>, "output": <decoded response>,
     "score": float, "step": int, "acc": bool, "pred": str}

`scripts/phase1_warmup_heads.py` wants one file whose records carry `text`
(or `prompt` + `response`).  README, Phase 0 산출물: "verl의 덤프 형식이 다르면
이 형태로 변환할 것."  This is that conversion, plus the filtering the warm-up
would otherwise have to do blind.

    python scripts/phase0_collect_rollouts.py \
        rollout_data/STEER-F-small/steer_2026.../ \
        --out rollout_data/STEER-F-small/steer_2026.../rollouts.jsonl

Filtering notes
---------------
`--min-steps` drops the earliest steps.  The heads are warmed on the policy's
own output distribution, so rollouts from a policy several updates old are
still on-distribution, but step 1 samples a policy that no longer exists by the
end of the run.  The default keeps everything: with a 100-step run the drift is
small, and the plan's remedy for a weak G1 is *more* data, not cleaner data.

`--dedup` removes exactly-repeated (prompt, response) pairs.  With rollout.n=8
a degenerate prompt can contribute eight identical sequences, which would
otherwise be eight identical gradient contributions to the heads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_argparser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("dumps", help="rollout_data_dir written by verl (contains <step>.jsonl)")
    p.add_argument("--out", required=True, help="destination JSONL")
    p.add_argument("--min-steps", type=int, default=0,
                   help="drop dumps from steps below this (default 0 = keep all)")
    p.add_argument("--max-steps", type=int, default=None,
                   help="drop dumps from steps above this")
    p.add_argument("--min-chars", type=int, default=32,
                   help="drop rollouts whose response is shorter than this")
    p.add_argument("--dedup", action="store_true",
                   help="drop exactly-repeated (prompt, response) pairs")
    p.add_argument("--correct-only", action="store_true",
                   help="keep only rollouts the reward function marked correct")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)

    src = Path(args.dumps)
    files = sorted(src.glob("*.jsonl"), key=lambda p: int(p.stem) if p.stem.isdigit() else -1)
    files = [f for f in files if f.stem.isdigit()]
    if not files:
        sys.exit(f"no <step>.jsonl dumps under {src}")

    kept, seen = 0, set()
    n_read = n_short = n_dup = n_wrong = 0
    steps_used = []

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as out:
        for f in files:
            step = int(f.stem)
            if step < args.min_steps:
                continue
            if args.max_steps is not None and step > args.max_steps:
                continue
            steps_used.append(step)
            for line in f.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                n_read += 1
                prompt = rec.get("input") or rec.get("prompt") or ""
                response = rec.get("output") or rec.get("response") or ""
                if len(response) < args.min_chars:
                    n_short += 1
                    continue
                if args.correct_only and not rec.get("acc", False):
                    n_wrong += 1
                    continue
                if args.dedup:
                    key = hash((prompt, response))
                    if key in seen:
                        n_dup += 1
                        continue
                    seen.add(key)
                out.write(json.dumps({
                    "text": prompt + response,
                    "prompt": prompt,
                    "response": response,
                    "score": rec.get("score"),
                    "acc": rec.get("acc"),
                    "step": step,
                }, ensure_ascii=False) + "\n")
                kept += 1

    if not kept:
        sys.exit("every rollout was filtered out; loosen the filters")

    print(f"[collect] {len(steps_used)} dumps (steps {min(steps_used)}..{max(steps_used)})")
    print(f"[collect] read {n_read}, kept {kept}"
          f"  (dropped: {n_short} short, {n_dup} duplicate, {n_wrong} incorrect)")
    print(f"[collect] wrote {out_path}")


if __name__ == "__main__":
    main()
