#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
# Licensed under the Apache License, Version 2.0
"""Turn the training logs into the paper's main tables.

    python3 scripts/analyze_seeds.py --logs logs/experiments --out results/

Writes
    results/per_seed.tsv     one row per (arm, seed): plateau means
    results/arm_means.tsv    one row per arm: mean +- SE over seeds
    results/contrasts.tsv    paired-by-seed contrasts with t and df
    results/compute_match.tsv  grpo-long read at the wall-clock crossing
    results/tables.tex       the same three, as LaTeX booktabs bodies

THE UNIT IS THE SEED
    The manuscript's earlier statistics paired eight plateau *steps* within one
    run. That measures whether a gap is stable inside one trajectory, which is
    not the question a reviewer asks -- they ask whether it survives re-running.
    So the primary statistic here is the seed-level paired difference: average
    each run over its plateau, pair arms within a seed, and test the differences
    across seeds with df = n_seeds - 1.

    Pairing within a seed is also what makes a mixed-hardware campaign legal:
    any per-seed effect common to all five arms (a different box, a different
    driver) cancels exactly in the difference.

    A contrast's n is min(n_seeds of its two arms), which is why the control
    arms running at three seeds cap their own contrasts at three.

NO SCIPY
    The container has torch and numpy but scipy is not guaranteed. The two-sided
    t p-value is computed from the incomplete beta function via math.lgamma with
    a continued fraction -- the same values scipy.stats.t.sf returns, to ~1e-12.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# metric key in the log  ->  short name used in the tables
METRICS = {
    "val-core/aime_2024_dapo_boxed/acc/mean@32": "acc",
    "val-core/aime_2024_dapo_boxed/acc/maj@32/mean": "maj",
    "actor/entropy": "entropy",
    "response_length/mean": "resp_len",
    "perf/time_per_step": "s_per_step",
    "steerf/branch_corr_frac": "branch_frac",
    "steerf/tw_mean": "tw_mean",
}
DERIVED = {"uplift": lambda r: r["maj"] - r["acc"]}

# Arms in the order the paper reports them. The first is the reference every
# contrast is drawn against.
MAIN_ARMS = ["grpo", "steer", "uniform", "permuted", "signed"]
CONTRASTS = [("signed", "grpo"), ("signed", "steer"), ("signed", "uniform"),
             ("signed", "permuted"), ("steer", "grpo"), ("uniform", "steer")]

STEP_RE = re.compile(r"step:(\d+) - global_seqlen")


def parse_log(path: Path) -> dict[int, dict[str, float]]:
    """step -> {metric: value} for every step line that carries a validation."""
    out: dict[int, dict[str, float]] = {}
    for line in path.read_text(errors="replace").splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        row = {}
        for key, short in METRICS.items():
            hit = re.search(re.escape(key) + r":(-?[0-9.]+)", line)
            if hit:
                row[short] = float(hit.group(1))
        if "acc" in row:            # a validation step, not a plain train step
            out[step] = row
        elif row:                   # keep timing from non-val steps too
            out.setdefault(step, {}).update(row)
    return out


def plateau(steps: dict[int, dict[str, float]], lo: int, hi: int) -> dict[str, float]:
    """Mean of each metric over the validation points inside [lo, hi].

    s_per_step is the exception: it averages over the NON-validation steps in
    the window. Every tenth step also runs AIME24 (~430 s) and saves, so
    averaging cost over validation steps alone reports ~1250 s for an arm whose
    training step costs ~837, and the paper's overhead figure would be wrong.
    """
    rows = [r for s, r in steps.items() if lo <= s <= hi and "acc" in r]
    if not rows:
        return {}
    train_only = [r for s, r in steps.items()
                  if lo <= s <= hi and "acc" not in r and "s_per_step" in r]
    agg = {}
    for short in list(METRICS.values()):
        vals = [r[short] for r in rows if short in r]
        if vals:
            agg[short] = sum(vals) / len(vals)
    if train_only:
        agg["s_per_step"] = sum(r["s_per_step"] for r in train_only) / len(train_only)
        agg["s_per_val_step"] = sum(r["s_per_step"] for r in rows if "s_per_step" in r) \
            / max(1, sum(1 for r in rows if "s_per_step" in r))
    for name, fn in DERIVED.items():
        try:
            agg[name] = fn(agg)
        except KeyError:
            pass
    agg["n_val_points"] = len(rows)
    return agg


# --------------------------------------------------------------- statistics
def _betacf(a: float, b: float, x: float) -> float:
    tiny, eps = 1e-30, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: int) -> float:
    """P(|T_df| >= |t|). Matches scipy.stats.t.sf(|t|, df) * 2."""
    if df <= 0 or not math.isfinite(t):
        return float("nan")
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def paired(diffs: list[float]) -> dict[str, float]:
    n = len(diffs)
    mean = sum(diffs) / n
    if n < 2:
        return {"n": n, "mean": mean, "sd": float("nan"),
                "se": float("nan"), "t": float("nan"), "p": float("nan")}
    sd = statistics.stdev(diffs)
    se = sd / math.sqrt(n)
    t = mean / se if se > 0 else float("inf")
    return {"n": n, "mean": mean, "sd": sd, "se": se, "t": t,
            "p": t_two_sided_p(t, n - 1)}


# ----------------------------------------------------------- compute match
def compute_matched_step(long_steps: dict[int, dict[str, float]],
                         target_seconds: float) -> tuple[int, float] | None:
    """The last step of the long run whose cumulative wall clock fits in budget.

    Rather than assuming a cost ratio, this integrates perf/time_per_step over
    the long run and reports the step where it crosses what the treatment spent.
    """
    total = 0.0
    best = None
    for step in sorted(long_steps):
        total += long_steps[step].get("s_per_step", 0.0)
        if total <= target_seconds:
            best = (step, total)
        else:
            break
    return best


# ------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", default="logs/experiments")
    ap.add_argument("--out", default="results")
    ap.add_argument("--model-tag", default="Qwen2.5-Math-1.5B")
    ap.add_argument("--plateau", default="40:110", help="lo:hi step window")
    ap.add_argument("--steps", type=int, default=110)
    args = ap.parse_args(argv)

    lo, hi = (int(v) for v in args.plateau.split(":"))
    log_dir = Path(args.logs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def run_name(arm: str, seed: int) -> str:
        t = args.model_tag
        return {
            "grpo": f"grpo-{t}-s{seed}",
            "steer": f"steer-{t}-s{seed}",
            "signed": f"steer-f-{t}-s{seed}-tree-rollout",
            "uniform": f"steer-f-{t}-s{seed}-tree-rollout-uniform",
            "permuted": f"steer-f-{t}-s{seed}-tree-rollout-permuted",
            "grpo-long": f"grpo-{t}-s{seed}-long",
        }[arm]

    def find_log(rn: str) -> Path | None:
        # train-<run>.log, or the recovery chain's train-<run>_<tag>.log. The
        # bare "train-<run>*" glob would match sibling arms -- signed is a
        # prefix of permuted (see run/_arms.sh).
        cands = [log_dir / f"train-{rn}.log", *sorted(log_dir.glob(f"train-{rn}_*.log"))]
        live = [c for c in cands if c.is_file()]
        return live[-1] if live else None

    # ------------------------------------------------------------ per seed
    per_seed: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    missing = []
    for arm in MAIN_ARMS:
        for seed in range(1, 6):
            f = find_log(run_name(arm, seed))
            if f is None:
                missing.append(f"{arm} s{seed}")
                continue
            agg = plateau(parse_log(f), lo, hi)
            if agg:
                agg["log"] = f.name
                per_seed[arm][seed] = agg
            else:
                missing.append(f"{arm} s{seed} (no validation point in {lo}-{hi})")

    if not per_seed:
        sys.exit(f"[analyze] no parsable training log under {log_dir}")

    cols = ["acc", "maj", "uplift", "entropy", "resp_len", "s_per_step",
            "s_per_val_step", "branch_frac", "tw_mean", "n_val_points"]

    with (out_dir / "per_seed.tsv").open("w") as fh:
        fh.write("arm\tseed\t" + "\t".join(cols) + "\tlog\n")
        for arm in MAIN_ARMS:
            for seed in sorted(per_seed.get(arm, {})):
                r = per_seed[arm][seed]
                fh.write(f"{arm}\t{seed}\t"
                         + "\t".join(f"{r[c]:.4f}" if c in r else "-" for c in cols)
                         + f"\t{r['log']}\n")

    # ---------------------------------------------------------- arm means
    with (out_dir / "arm_means.tsv").open("w") as fh:
        fh.write("arm\tn_seeds\t" + "\t".join(f"{c}\t{c}_se" for c in cols[:-1]) + "\n")
        for arm in MAIN_ARMS:
            rows = list(per_seed.get(arm, {}).values())
            if not rows:
                continue
            cells = [arm, str(len(rows))]
            for c in cols[:-1]:
                v = [r[c] for r in rows if c in r]
                if not v:
                    cells += ["-", "-"]
                    continue
                m = sum(v) / len(v)
                se = statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else float("nan")
                cells += [f"{m:.4f}", ("-" if math.isnan(se) else f"{se:.4f}")]
            fh.write("\t".join(cells) + "\n")

    # ---------------------------------------------------------- contrasts
    contrast_rows = []
    with (out_dir / "contrasts.tsv").open("w") as fh:
        fh.write("contrast\tmetric\tn_seeds\tmean_diff\tsd\tse\tt\tp\tseeds\n")
        for a, b in CONTRASTS:
            shared = sorted(set(per_seed.get(a, {})) & set(per_seed.get(b, {})))
            if not shared:
                continue
            for metric in ("acc", "maj", "uplift", "entropy"):
                d = [per_seed[a][s][metric] - per_seed[b][s][metric]
                     for s in shared
                     if metric in per_seed[a][s] and metric in per_seed[b][s]]
                if not d:
                    continue
                st = paired(d)
                contrast_rows.append((f"{a} - {b}", metric, st))
                fh.write(f"{a} - {b}\t{metric}\t{st['n']}\t{st['mean']:+.4f}\t"
                         f"{st['sd']:.4f}\t{st['se']:.4f}\t{st['t']:+.2f}\t"
                         f"{st['p']:.4f}\t{','.join(map(str, shared))}\n")

    # ------------------------------------------------------- compute match
    with (out_dir / "compute_match.tsv").open("w") as fh:
        fh.write("seed\tsteerf_steps\tsteerf_seconds\tgrpo_long_step\t"
                 "grpo_long_seconds\tsteerf_acc\tgrpo_long_acc\tdiff\n")
        for seed in sorted(per_seed.get("signed", {})):
            f_long = find_log(run_name("grpo-long", seed))
            f_sig = find_log(run_name("signed", seed))
            if f_long is None or f_sig is None:
                continue
            sig_steps = parse_log(f_sig)
            budget = sum(r.get("s_per_step", 0.0)
                         for s, r in sig_steps.items() if s <= args.steps)
            long_steps = parse_log(f_long)
            hit = compute_matched_step(long_steps, budget)
            if hit is None:
                continue
            step, spent = hit
            # the last validation at or before the matched step
            val = [s for s in sorted(long_steps) if s <= step and "acc" in long_steps[s]]
            if not val:
                continue
            g_acc = long_steps[val[-1]]["acc"]
            s_acc = per_seed["signed"][seed]["acc"]
            fh.write(f"{seed}\t{args.steps}\t{budget:.0f}\t{val[-1]}\t{spent:.0f}\t"
                     f"{s_acc:.4f}\t{g_acc:.4f}\t{s_acc - g_acc:+.4f}\n")

    # ------------------------------------------------------------- LaTeX
    with (out_dir / "tables.tex").open("w") as fh:
        fh.write("% generated by scripts/analyze_seeds.py -- do not edit\n")
        fh.write("% Table: arm means over seeds (plateau steps "
                 f"{lo}-{hi}, unit = seed)\n")
        for arm in MAIN_ARMS:
            rows = list(per_seed.get(arm, {}).values())
            if not rows:
                continue
            def ms(c):
                v = [r[c] for r in rows if c in r]
                if not v:
                    return "--"
                m = sum(v) / len(v)
                if len(v) < 2:
                    return f"{m:.4f}"
                return f"{m:.4f} $\\pm$ {statistics.stdev(v)/math.sqrt(len(v)):.4f}"
            fh.write(f"{arm} & {len(rows)} & {ms('acc')} & {ms('maj')} & "
                     f"{ms('uplift')} & {ms('entropy')} \\\\\n")
        fh.write("%\n% Table: paired-by-seed contrasts\n")
        for name, metric, st in contrast_rows:
            if metric != "acc":
                continue
            fh.write(f"{name} & {st['n']} & {st['mean']:+.4f} & "
                     f"{st['t']:+.2f} & {st['p']:.3f} \\\\\n")

    print(f"[analyze] wrote {out_dir}/per_seed.tsv, arm_means.tsv, contrasts.tsv, "
          "compute_match.tsv, tables.tex")
    for arm in MAIN_ARMS:
        print(f"  {arm:9s} {len(per_seed.get(arm, {}))} seed(s)")
    if missing:
        print(f"  MISSING ({len(missing)}): {', '.join(missing[:12])}"
              + (" ..." if len(missing) > 12 else ""))
    print()
    print((out_dir / "contrasts.tsv").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
