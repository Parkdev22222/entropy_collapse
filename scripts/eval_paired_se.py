#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
# Licensed under the Apache License, Version 2.0
r"""The paired across-problem error bar the Limitations section says we lack.

    python3 scripts/eval_paired_se.py \
        --val-data validation_data/eval \
        --out  results/paired_se.json \
        --tex  results/numbers-pairedse.tex

WHY THIS EXISTS
    The manuscript's Limitations paragraph reads, verbatim: "the across-problem
    standard error of the *paired* difference, in which per-problem difficulty
    cancels between two arms evaluated on the same questions --- is not
    recoverable from our logs, which record per-benchmark means and standard
    deviations rather than per-problem scores."  That was true of the training
    logs and it is no longer true of the evaluation: with
    `bash run/instrument_phase2.sh --apply` in place, every eval leaves one
    JSONL row per generated sample, carrying the benchmark it came from and
    whether that sample was scored correct.  This script turns those rows into
    the statistic the paragraph names, so the sentence can be replaced by a
    number instead of repeated.

    It does not promise the error bar will be smaller.  How much pairing buys
    is decided by how often the two arms disagree on a problem: if they
    disagree on everything the paired error equals the unpaired one, and if
    they disagree on almost nothing it collapses.  Both errors and the
    disagreement rate are reported side by side, and the paper quotes what
    comes out.

WHICH BENCHMARKS
    The three avg@1 sets with enough problems to carry an error bar --- MATH500
    (500), OlympiadBench (675), Minerva (272), 1447 together.  GSM8K is
    computed and reported but held out of the pooled figure: at avg@1 it sits
    near saturation, where a paired difference measures the ceiling rather than
    the method.  AIME24/25 and AMC23 are excluded for a different reason: they
    are scored avg@32, so a row is one of 32 samples of a problem rather than a
    problem, and a per-problem score would have to be reconstructed by grouping
    replicas --- which the dump gives no key for.  AIME24 is also the set the
    checkpoint was selected on (see --held-out in scripts/analyze_seeds.py).

PAIRING
    Two arms evaluated by run/run_eval_all.sh see the same parquet files in the
    same order through a non-shuffling validation loader, so row i of one arm's
    benchmark segment is the same problem as row i of the other's.  That is an
    assumption, so it is checked rather than trusted: the prompt text of every
    paired row must match, and a single mismatch refuses the pair instead of
    reporting a difference between two different questions.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

# The three that carry the pooled figure, and the one that does not.
POOLED = ("math500", "olympiadbench", "minerva_math")
REPORTED_ONLY = ("gsm8k_test",)

# Contrasts.  These two and no others: they are the only pair in the five-arm
# design that changes exactly one thing (Table 1), and the pre-registration
# names STEER-F - uniform as the contrast whose straddling zero would refute
# the paper.  scripts/analyze_seeds.py emits direction consistency for the
# same two, from the same evaluation.
CONTRASTS = (("signed", "uniform"), ("signed", "permuted"))

# macro stem -> (contrast index or None, key, format).  tests/test_paper.py
# reads this table to decide which slots a run can fill, so a name here is a
# name the manuscript is allowed to ask for.
MACROS = {
    "Pairn":                    (None, "n_problems",   "int"),
    "Pairnseeds":               (None, "n_seeds",      "int"),
    "Pairnobs":                 (None, "n_obs",        "int"),
    "Pairsigneduniform":        (0,    "mean_diff",    "signed"),
    "Pairsigneduniformse":      (0,    "se_paired",    "err"),
    "Pairsigneduniformunpse":   (0,    "se_unpaired",  "err"),
    "Pairsigneduniformt":       (0,    "t",            "t"),
    "Pairsigneduniformdisc":    (0,    "discordance",  "rate"),
    "Pairsignedpermuted":       (1,    "mean_diff",    "signed"),
    "Pairsignedpermutedse":     (1,    "se_paired",    "err"),
    "Pairsignedpermutedunpse":  (1,    "se_unpaired",  "err"),
    "Pairsignedpermutedt":      (1,    "t",            "t"),
    "Pairsignedpermuteddisc":   (1,    "discordance",  "rate"),
}

RUN_DIR = re.compile(r"^(?P<arm>[a-z0-9.-]+)-s(?P<seed>\d+)$")


# ------------------------------------------------------------------ reading
def read_rows(path):
    """One JSONL dump -> [{data_source, input, correct}], in file order."""
    rows = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: {exc}") from None
            if "data_source" not in rec:
                raise SystemExit(
                    f"{path}: no data_source column. The eval ran before "
                    f"`bash run/instrument_phase2.sh --apply` carried it into "
                    f"the dump, so these rows cannot be attributed to a "
                    f"benchmark. Re-run the evaluation.")
            rows.append({
                "data_source": str(rec["data_source"]),
                "input": rec.get("input", ""),
                "correct": score_of(rec),
            })
    return rows


def score_of(rec):
    """1.0 / 0.0 for one sample.

    `acc` is the boolean the benchmark tables are built from
    (val-core/<set>/acc/mean@1 is its mean), so it is preferred. `score` is the
    same decision rendered as +1/-1 by multi_datasets_eval.compute_score_both,
    and is the fallback for a dump written before that key existed.
    """
    if "acc" in rec and rec["acc"] is not None:
        return 1.0 if rec["acc"] in (True, 1, 1.0, "True", "true") else 0.0
    if "score" in rec and rec["score"] is not None:
        return 1.0 if float(rec["score"]) > 0 else 0.0
    raise SystemExit("a dumped row has neither an 'acc' nor a 'score' column")


def by_benchmark(rows):
    """Split a pass into {data_source: [rows]}, keeping file order within each.

    Order is what pairs the two arms, so it is preserved exactly; the rows of
    one benchmark are contiguous in practice but nothing here requires that.
    """
    out = {}
    for r in rows:
        out.setdefault(r["data_source"], []).append(r)
    return out


def load_run(run_dir, passes=("avg1",)):
    """<arm>-s<seed>/ -> {data_source: [rows]} across the named passes."""
    merged = {}
    for tag in passes:
        d = run_dir / tag
        if not d.is_dir():
            continue
        # val_only writes "0.jsonl"; take every step file in numeric order so
        # a dump from a training run (many steps) reads as its last one.
        files = sorted(d.glob("*.jsonl"), key=lambda p: _step_of(p))
        if not files:
            continue
        for ds, rows in by_benchmark(read_rows(files[-1])).items():
            merged.setdefault(ds, []).extend(rows)
    return merged


def _step_of(path):
    m = re.match(r"^(\d+)\.jsonl$", path.name)
    return int(m.group(1)) if m else -1


def discover(root, passes=("avg1",)):
    """validation_data/eval -> {(arm, seed): {data_source: [rows]}}."""
    runs = {}
    for d in sorted(root.iterdir() if root.is_dir() else []):
        m = RUN_DIR.match(d.name)
        if not d.is_dir() or not m:
            continue
        rows = load_run(d, passes)
        if rows:
            runs[(m.group("arm"), int(m.group("seed")))] = rows
    return runs


# --------------------------------------------------------------- statistics
def paired(a_rows, b_rows, where):
    """Per-problem paired difference, or a refusal.

    Returns (diffs, a_vals, b_vals). The prompt check is the point: pairing by
    position is an assumption about the validation loader, and an unchecked
    assumption here would silently difference two different questions.
    """
    if len(a_rows) != len(b_rows):
        raise SystemExit(
            f"{where}: {len(a_rows)} rows vs {len(b_rows)} -- the two arms did "
            f"not see the same problem set, so they cannot be paired.")
    for i, (x, y) in enumerate(zip(a_rows, b_rows)):
        if x["input"] != y["input"]:
            raise SystemExit(
                f"{where}: row {i} is a different problem in the two arms "
                f"(prompts differ). Pairing by position does not hold here; "
                f"refusing to report a difference between two questions.")
    a = [r["correct"] for r in a_rows]
    b = [r["correct"] for r in b_rows]
    return [x - y for x, y in zip(a, b)], a, b


def _mean(xs):
    return sum(xs) / len(xs)


def _sd(xs):
    if len(xs) < 2:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def summarise(diffs, a, b):
    n = len(diffs)
    md = _mean(diffs)
    se_p = _sd(diffs) / math.sqrt(n) if n > 1 else float("nan")
    se_u = (math.sqrt(_sd(a) ** 2 / n + _sd(b) ** 2 / n)
            if n > 1 else float("nan"))
    return {
        "n": n,
        "mean_a": _mean(a),
        "mean_b": _mean(b),
        "mean_diff": md,
        "se_paired": se_p,
        "se_unpaired": se_u,
        # What pairing actually bought, as a fraction of the unpaired error.
        "se_reduction": (1.0 - se_p / se_u) if se_u and se_u == se_u else float("nan"),
        "t": (md / se_p) if se_p else float("nan"),
        # The quantity that decides the line above: how often the two arms
        # disagree on a problem. All agreement -> no paired error at all; all
        # disagreement -> pairing buys nothing.
        "discordance": sum(1 for d in diffs if d != 0) / n,
    }


def analyse(runs, pooled=POOLED, reported=REPORTED_ONLY):
    """{(arm, seed): rows} -> the report."""
    out = {"contrasts": [], "benchmarks": list(pooled) + list(reported)}
    for a_arm, b_arm in CONTRASTS:
        seeds = sorted({s for (arm, s) in runs if arm == a_arm}
                       & {s for (arm, s) in runs if arm == b_arm})
        entry = {"a": a_arm, "b": b_arm, "seeds": seeds,
                 "per_seed": [], "per_benchmark": {}, "pooled": None}
        pool_d, pool_a, pool_b = [], [], []
        per_bench = {}
        for seed in seeds:
            ra, rb = runs[(a_arm, seed)], runs[(b_arm, seed)]
            sd_, sa_, sb_ = [], [], []
            row = {"seed": seed, "benchmarks": {}}
            for ds in list(pooled) + list(reported):
                if ds not in ra or ds not in rb:
                    continue
                d, a, b = paired(ra[ds], rb[ds],
                                 f"{a_arm}/{b_arm} s{seed} {ds}")
                row["benchmarks"][ds] = summarise(d, a, b)
                per_bench.setdefault(ds, ([], [], []))
                per_bench[ds][0].extend(d)
                per_bench[ds][1].extend(a)
                per_bench[ds][2].extend(b)
                if ds in pooled:
                    sd_ += d; sa_ += a; sb_ += b
            if sd_:
                row["pooled"] = summarise(sd_, sa_, sb_)
                pool_d += sd_; pool_a += sa_; pool_b += sb_
            entry["per_seed"].append(row)
        for ds, (d, a, b) in per_bench.items():
            entry["per_benchmark"][ds] = summarise(d, a, b)
        if pool_d:
            entry["pooled"] = summarise(pool_d, pool_a, pool_b)
            # One problem per seed is one observation. Stated, not hidden: the
            # pool treats (seed, problem) pairs as independent, which ignores
            # the clustering by seed. The per-seed rows above are the version
            # that does not.
            entry["pooled"]["n_seeds"] = len(seeds)
            entry["pooled"]["n_problems"] = (len(pool_d) // len(seeds)
                                             if seeds else 0)
        out["contrasts"].append(entry)
    return out


# ------------------------------------------------------------------ output
def fmt(value, how):
    v = float(value)
    if v != v:                       # NaN: no number, so no macro
        return None
    if how == "int":
        return str(int(v))
    if how == "signed":
        return f"{v:+.4f}"
    if how == "err":
        return f"{v:.4f}"
    if how == "t":
        return f"{v:+.2f}"
    return f"{v:.3f}"


def macros(report):
    out = {}
    entries = report.get("contrasts") or []
    for name, (idx, key, how) in MACROS.items():
        if idx is None:
            src = next((e["pooled"] for e in entries if e.get("pooled")), None)
            if src is None:
                continue
            value = {"n_problems": src.get("n_problems"),
                     "n_seeds": src.get("n_seeds"),
                     "n_obs": src.get("n")}[key]
        else:
            if idx >= len(entries) or not entries[idx].get("pooled"):
                continue
            value = entries[idx]["pooled"].get(key)
        if value is None:
            continue                 # stays \PENDING: a red cell, not a lie
        rendered = fmt(value, how)
        if rendered is not None:
            out[name] = rendered
    return out


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val-data", default="validation_data/eval",
                   help="the VAL_DATA_DIR root run/run_eval_all.sh writes to")
    p.add_argument("--passes", default="avg1",
                   help="comma-separated pass tags to read (default: avg1)")
    p.add_argument("--out", default="results/paired_se.json")
    p.add_argument("--tex", default="results/numbers-pairedse.tex")
    args = p.parse_args(argv)

    root = Path(args.val_data)
    runs = discover(root, tuple(t for t in args.passes.split(",") if t))
    if not runs:
        print(f"[paired-se] no per-problem dumps under {root}. Run the "
              f"evaluation with `bash run/instrument_phase2.sh --apply` first "
              f"-- without it verl throws the per-problem scores away.")
        return 1

    print(f"[paired-se] {len(runs)} run(s): "
          + ", ".join(f"{a}-s{s}" for a, s in sorted(runs)))
    report = analyse(runs)

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[paired-se] report -> {dest}")

    values = macros(report)
    lines = ["% generated by scripts/eval_paired_se.py -- do not edit",
             f"% source: {args.val_data}"]
    lines += [f"\\providecommand{{\\{k}}}{{{v}}}" for k, v in sorted(values.items())]
    tex = Path(args.tex)
    tex.parent.mkdir(parents=True, exist_ok=True)
    tex.write_text("\n".join(lines) + "\n")
    print(f"[paired-se] {len(values)} macros -> {tex}")

    for e in report["contrasts"]:
        pooled = e.get("pooled")
        if not pooled:
            print(f"[paired-se] {e['a']} - {e['b']}: no shared seed, nothing to pair")
            continue
        print(f"[paired-se] {e['a']} - {e['b']}  n={pooled['n']} "
              f"({pooled['n_seeds']} seed(s) x {pooled['n_problems']} problems)  "
              f"mean={pooled['mean_diff']:+.4f}  paired SE={pooled['se_paired']:.4f}  "
              f"unpaired SE={pooled['se_unpaired']:.4f}  "
              f"disagreement={pooled['discordance']:.3f}")
    missing = sorted(set(MACROS) - set(values))
    if missing:
        print(f"[paired-se] not computable from this data, left red: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
