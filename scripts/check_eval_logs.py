#!/usr/bin/env python3
"""Does each eval log agree with itself?

    python3 scripts/check_eval_logs.py logs/experiments/eval-*.log

verl prints every benchmark's result twice in the same step line: as accuracy
(val-core/<set>/acc/mean@k) and as reward (val-aux/<set>/reward/mean@k). The
reward is +1 for a correct sample and -1 otherwise, so

    acc == (reward + 1) / 2

holds for every set in a genuine log, to the three decimals the step line
carries. A log whose accuracy was changed after the run breaks it, because
nothing that edits one number knows to edit the other.

Why this exists: on 2026-09-28 three of the five six-benchmark eval logs reached
origin/paper with MATH500, Minerva and GSM8K accuracies lowered by one to four
points, uploaded by mistake from an environment where code had been changed.
Nothing downstream noticed -- collect_results.py reads whatever a log says --
and the signed-vs-uniform contrast read +.0105 from the logs against -.0090 from
the per-problem dump. This identity is what said which side was wrong.

What it cannot see: an edit that changes both numbers consistently. It is a
check that a log is internally coherent, not a proof that it is original; the
original is whatever the run wrote.

It also assumes the reward is exactly +-1 per sample. A reward with a length
penalty or partial credit would break the identity in a genuine log -- if the
reward function changes, this check has to change with it.

Exit status: 0 when every log is coherent, 1 when any is not, 2 on bad input.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

TOL = 0.0015   # both sides are printed to three decimals

STEP_KEY = re.compile(r"val-core/([A-Za-z0-9_]+)/acc/(mean@\d+):(-?[0-9.]+)")


def check(text: str) -> tuple[list[str], list[str]]:
    """-> (sets checked, sets whose accuracy disagrees with their reward)."""
    # \r as well as \n: three of the logs this was written for had every
    # newline turned into a carriage return on the way to the branch.
    text = text.replace("\r", "\n")
    acc: dict[tuple[str, str], float] = {}
    for bench, k, v in STEP_KEY.findall(text):
        acc[(bench, k)] = float(v)            # last occurrence wins
    seen, bad = [], []
    for (bench, k), a in sorted(acc.items()):
        r = re.findall(rf"val-aux/{re.escape(bench)}/reward/{k}:(-?[0-9.]+)", text)
        if not r:
            continue
        seen.append(bench)
        if abs(a - (float(r[-1]) + 1) / 2) > TOL:
            bad.append(f"{bench} acc={a:.3f} reward-implied={(float(r[-1]) + 1) / 2:.3f}")
    return seen, bad


def main(argv: list[str]) -> int:
    paths = [Path(p) for p in argv]
    if not paths:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    worst = 0
    for p in paths:
        if not p.is_file():
            print(f"{p}: no such file", file=sys.stderr)
            return 2
        seen, bad = check(p.read_text(errors="replace"))
        if not seen:
            print(f"{p.name:34s} NO METRICS  (nothing to check -- did the eval finish?)")
            worst = max(worst, 1)
        elif bad:
            print(f"{p.name:34s} MISMATCH    " + "; ".join(bad))
            worst = max(worst, 1)
        else:
            print(f"{p.name:34s} OK          {len(seen)} sets")
    return worst


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
