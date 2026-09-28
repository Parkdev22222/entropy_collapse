#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
# Licensed under the Apache License, Version 2.0
"""Scrape eval logs into paper-table-shaped rows.

Scans the eval console logs that run/run_all_experiments.sh produces, pulls
the val-core accuracy for each benchmark, and writes one TSV row per
(arm, seed) plus a seed-averaged row — the exact numbers that go into the
paper's tables (avg@32 for AIME24/25/AMC23, avg@1 for the rest, avg@4 LCB).

    python scripts/collect_results.py --logs logs/experiments --out results/summary.tsv
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# data_source (as stored in the parquets) -> (paper column, metric suffix)
COLUMNS = [
    ("aime_2024_dapo_boxed", "AIME24",   "mean@32"),
    ("aime_2025_dapo_boxed", "AIME25",   "mean@32"),
    ("amc2023_dapo_boxed",   "AMC23",    "mean@32"),
    ("math500",              "MATH500",  "mean@1"),
    ("minerva_math",         "Minerva",  "mean@1"),
    ("olympiadbench",        "Olympiad", "mean@1"),
    ("gsm8k_test",           "GSM8K*",   "mean@1"),    # extra, not in paper tables
    ("codecontests",         "LCB-v5",   "mean@4"),
]
MATH6 = ["AIME24", "AIME25", "AMC23", "MATH500", "Minerva", "Olympiad"]

# both console formats verl emits: "key:0.123" in step lines and
# "'key': np.float64(0.123)" / "'key': 0.123" in pprint blocks
def _find(text, key):
    pats = [re.escape(key) + r":(-?[0-9.eE+]+)",
            "'" + re.escape(key) + r"'\s*:\s*(?:np\.float64\()?(-?[0-9.eE+]+)"]
    vals = []
    for p in pats:
        vals += [float(v) for v in re.findall(p, text)]
    return vals[-1] if vals else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", default="logs/experiments")
    ap.add_argument("--out", default="results/summary.tsv")
    args = ap.parse_args(argv)

    # eval logs are named eval-<arm>-s<seed>.log by the master script
    rows = {}
    for f in sorted(Path(args.logs).glob("eval-*.log")):
        m = re.match(r"eval-(.+)-s(\d+)\.log$", f.name)
        if not m:
            continue
        arm, seed = m.group(1), m.group(2)
        text = f.read_text(errors="replace")
        row = {}
        for ds, col, suffix in COLUMNS:
            v = _find(text, f"val-core/{ds}/acc/{suffix}")
            if v is not None:
                row[col] = 100.0 * v
        if row:
            rows[(arm, seed)] = row

    if not rows:
        sys.exit(f"[collect] no parsable eval logs under {args.logs}")

    by_arm = defaultdict(list)
    for (arm, seed), row in rows.items():
        by_arm[arm].append((seed, row))

    cols = [c for _, c, _ in COLUMNS]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        fh.write("arm\tseed\t" + "\t".join(cols) + "\tAvg(math6)\n")
        for arm in sorted(by_arm):
            seed_rows = sorted(by_arm[arm])
            for seed, row in seed_rows:
                fh.write(_line(arm, seed, row, cols))
            if len(seed_rows) > 1:  # the number that goes into the paper table
                mean = {c: sum(r[c] for _, r in seed_rows if c in r) / sum(1 for _, r in seed_rows if c in r)
                        for c in cols if any(c in r for _, r in seed_rows)}
                fh.write(_line(arm, "avg", mean, cols))
    print(f"[collect] wrote {out} ({len(rows)} eval logs, {len(by_arm)} arms)")
    print(out.read_text())


def _line(arm, seed, row, cols):
    def fmt(c):
        return f"{row[c]:.1f}" if c in row else "-"
    m6 = [row[c] for c in MATH6 if c in row]
    avg = f"{sum(m6)/len(m6):.1f}" if len(m6) == len(MATH6) else "-"
    return arm + "\t" + str(seed) + "\t" + "\t".join(fmt(c) for c in cols) + "\t" + avg + "\n"


if __name__ == "__main__":
    main()
