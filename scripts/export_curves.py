#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
# Licensed under the Apache License, Version 2.0
"""Training curves from a training log, as one TSV row per step.

    python3 scripts/export_curves.py logs/experiments/train-<run>.log [more logs ...] \
        --out results/steer_metrics/curves/<run>.tsv

The counterparts of STEER's Figure 7 (test accuracy during training, AIME24
avg@32) and Figure 8 (training entropy, actor/entropy), plus the training
reward and response length that read beside them. Validation runs every
test_freq steps, so its columns are empty in between; that is the same
10-step grid the paper selects checkpoints on.

Several logs are read in the order given, a later one overriding a step an
earlier one also printed -- the case of a resumed run, whose first line
re-validates the step it resumed from.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

STEP = re.compile(r"step:(\d+) - (.*)")
KV = re.compile(r"([\w/@.\-]+):(-?[\d.]+(?:[eE][-+]?\d+)?)")

AIME = "val-core/aime_2024_dapo_boxed/acc"
COLUMNS = [
    ("aime24_mean@32", f"{AIME}/mean@32"),
    ("aime24_maj@32", f"{AIME}/maj@32/mean"),
    ("aime24_best@32", f"{AIME}/best@32/mean"),
    ("entropy", "actor/entropy"),
    ("score_mean", "critic/score/mean"),
    ("response_length", "response_length/mean"),
    ("clip_ratio", "response_length/clip_ratio"),
]


def read_steps(paths):
    steps = {}
    for p in paths:
        for line in Path(p).read_text(errors="replace").splitlines():
            m = STEP.search(line)
            if not m:
                continue
            kv = {k: float(v) for k, v in KV.findall(m.group(2))}
            steps.setdefault(int(m.group(1)), {}).update(kv)
    return steps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    steps = read_steps(args.logs)
    if not steps:
        sys.exit(f"[curves] no 'step:N - ...' lines in {args.logs}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        fh.write("step\t" + "\t".join(c for c, _ in COLUMNS) + "\ttrain_acc\n")
        for s in sorted(steps):
            d = steps[s]
            cells = [f"{d[k]:.4f}" if k in d else "" for _, k in COLUMNS]
            sc = d.get("critic/score/mean")
            cells.append(f"{(1 + sc) / 2:.4f}" if sc is not None else "")
            fh.write(f"{s}\t" + "\t".join(cells) + "\n")
    n_val = sum(1 for d in steps.values() if f"{AIME}/mean@32" in d)
    print(f"[curves] {out}: {len(steps)} steps, {n_val} with AIME24 validation, "
          f"last step {max(steps)}")


if __name__ == "__main__":
    main()
