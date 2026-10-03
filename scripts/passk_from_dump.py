#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
# Licensed under the Apache License, Version 2.0
"""Unbiased pass@k from a Pass@k evaluation dump.

    python3 scripts/passk_from_dump.py validation_data/steer_metrics/best/r2-s1/passk \
        --label r2-s1 --out results/steer_metrics/best/passk_unbiased.tsv

WHY A SECOND NUMBER. verl prints best@N/mean for the Pass@k pass
(run/eval_steerf.sh PASSK=1), and that is the figure STEER's Figure 6 most
plausibly used, since their code is verl. But best@N/mean is a bootstrap that
draws N of the n samples WITH replacement (verl/trainer/ppo/metric_utils.py,
bootstrap_metric) and averages the max. At N = n = 1024 a problem solved by c
of its 1024 samples scores about 1 - exp(-c), so a problem solved exactly once
reads .63 rather than 1 -- it undercounts precisely the rare solves that
pass@large-k is meant to show. This reports the unbiased estimator of Chen et
al. (2021), pass@k = 1 - C(n-c, k) / C(n, k), averaged over problems, beside
verl's, so the table can carry both and say which is which.

Rows are grouped by (data_source, prompt) and must all carry the same n; a
dump where they do not is refused rather than averaged over unequal draws.
Reading the dump reuses scripts/eval_paired_se.py (read_rows, by_benchmark).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_paired_se import _step_of, by_benchmark, read_rows  # noqa: E402


def pass_at_k(n, c, k):
    """Chen et al. (2021) unbiased estimator, in the numerically stable product form."""
    if k > n:
        raise ValueError(f"k={k} exceeds n={n}")
    if n - c < k:
        return 1.0
    prob_all_wrong = 1.0
    for i in range(n - c + 1, n + 1):
        prob_all_wrong *= 1.0 - k / i
    return 1.0 - prob_all_wrong


def ks_for(n):
    ks, k = [], 1
    while k < n:
        ks.append(k)
        k *= 2
    ks.append(n)
    return ks


def problem_counts(rows, where):
    """[(n, c)] per problem, in first-seen order."""
    order, tally = [], {}
    for r in rows:
        key = r["input"]
        if key not in tally:
            order.append(key)
            tally[key] = [0, 0]
        tally[key][0] += 1
        tally[key][1] += int(r["correct"] > 0)
    ns = {tally[k][0] for k in order}
    if len(ns) != 1:
        sys.exit(f"[passk] REFUSE {where}: problems carry different sample counts {sorted(ns)}")
    return [tuple(tally[k]) for k in order]


def summarise(dump_dir):
    files = sorted(Path(dump_dir).glob("*.jsonl"), key=_step_of)
    if not files:
        sys.exit(f"[passk] no *.jsonl under {dump_dir}")
    out = {}
    for ds, rows in sorted(by_benchmark(read_rows(files[-1])).items()):
        counts = problem_counts(rows, f"{dump_dir}/{ds}")
        n = counts[0][0]
        out[ds] = {"n": n, "problems": len(counts),
                   "pass": {k: sum(pass_at_k(n, c, k) for _, c in counts) / len(counts)
                            for k in ks_for(n)}}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump_dir")
    ap.add_argument("--label", required=True, help="row label, e.g. r2-s1 or base")
    ap.add_argument("--out", required=True, help="TSV holding every label; this label's rows are replaced")
    args = ap.parse_args(argv)

    res = summarise(args.dump_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # one table across labels; a re-run replaces its own rows, never duplicates them
    kept = []
    if out.exists():
        kept = [ln for ln in out.read_text().splitlines()[1:]
                if ln and ln.split("\t", 1)[0] != args.label]
    with out.open("w") as fh:
        fh.write("label\tdata_source\tn\tproblems\tk\tpass@k\n")
        for ln in kept:
            fh.write(ln + "\n")
        for ds, r in res.items():
            for k, v in r["pass"].items():
                fh.write(f"{args.label}\t{ds}\t{r['n']}\t{r['problems']}\t{k}\t{100 * v:.2f}\n")
    for ds, r in res.items():
        pk = r["pass"]
        print(f"[passk] {args.label} {ds}: n={r['n']} problems={r['problems']} "
              + " ".join(f"pass@{k}={100 * pk[k]:.1f}" for k in (1, 256, 512, r['n']) if k in pk))
    print(f"[passk] wrote {args.label} rows to {out}")


if __name__ == "__main__":
    main()
