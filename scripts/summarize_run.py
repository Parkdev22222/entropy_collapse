#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Reduce one verl console log to a single TSV row.

verl's console backend prints one `step:N - k:v - k:v ...` line per step and
nothing else machine-readable; the tensorboard event files hold the same
numbers but need a reader and a running server to consult.  Phase 2 produces a
12-run grid, so the queue appends a row per run here and the gate can be judged
from one file.

    python scripts/summarize_run.py logs/phase2-lam0.5-s1.log \
        --lam 0.5 --seed 1 --minutes 187

Columns match the header the queue writes:
    lam  seed  status  final_entropy  branch_gap  mean@16  best@16  steps  minutes

`best@16` is verl's bootstrap best-of-n over the 16 validation samples, i.e.
pass@16 — the primary G2 metric.  `branch_gap` is `steerf/branch_entropy_gap`,
the mechanism check; it is absent by construction on the lambda=0 arm, where
no A_H is computed, and prints as `na`.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

STEP_RE = re.compile(r"step:(\d+) - ")
KV_RE = re.compile(r"([A-Za-z0-9_/@.\-]+):(-?[0-9]+\.?[0-9]*(?:[eE][-+]?\d+)?)")
# verl pretty-prints the validation dict, so a single metric is routinely split
# across lines and wrapped in the console logger's quotes and colour codes:
#
#   (TaskRunner pid=…) "'val-core/math500/acc/best@16/mean': "
#   (TaskRunner pid=…)  'np.float64(0.78591), …
#
# Matching line-by-line therefore finds the key and never the value. The text is
# normalised first (strip ANSI, strip the ray prefix, rejoin the split) and only
# then scanned, with DOTALL so a key and its value may sit on different lines.
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RAY_RE = re.compile(r"\(TaskRunner pid=\d+\)\s*")
VAL_RE = re.compile(
    r"(val-core/[^'\"]+?)'?:\s*[\"']?\s*'?(?:np\.float64\()?(-?[0-9.]+)",
    re.DOTALL,
)


def normalise(text):
    text = ANSI_RE.sub("", text)
    text = RAY_RE.sub("", text)
    return text.replace('"\n', "").replace("'\n", "")


def build_argparser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("log")
    p.add_argument("--lam", default="?")
    p.add_argument("--seed", default="?")
    p.add_argument("--minutes", default="?")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    path = Path(args.log)
    if not path.exists():
        print(f"{args.lam}\t{args.seed}\tnolog\tna\tna\tna\tna\t0\t{args.minutes}")
        return

    text = path.read_text(errors="replace")

    last_step, last_metrics = 0, {}
    for line in text.splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        if step >= last_step:
            last_step = step
            last_metrics = {k: float(v) for k, v in KV_RE.findall(line)}

    vals = {}
    for k, v in VAL_RE.findall(normalise(text)):
        try:
            vals[k.strip()] = float(v)
        except ValueError:
            continue

    def val_like(*needles):
        for k, v in vals.items():
            if all(n in k for n in needles):
                return v
        return None

    mean16 = val_like("acc", "mean@16")
    best16 = val_like("acc", "best@16", "/mean")

    if "Error executing job" in text or "Traceback" in text:
        status = "fail"
    elif last_step == 0:
        status = "nostep"
    else:
        status = "ok"

    def fmt(x):
        return "na" if x is None else f"{x:.4f}"

    print(
        f"{args.lam}\t{args.seed}\t{status}\t"
        f"{fmt(last_metrics.get('actor/entropy'))}\t"
        f"{fmt(last_metrics.get('steerf/branch_entropy_gap'))}\t"
        f"{fmt(mean16)}\t{fmt(best16)}\t{last_step}\t{args.minutes}"
    )


if __name__ == "__main__":
    main()
