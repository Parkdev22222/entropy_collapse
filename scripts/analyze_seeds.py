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
import subprocess
import tempfile
import sys
from collections import defaultdict
from pathlib import Path

# metric key in the log  ->  short name used in the tables
METRICS = {
    "val-core/aime_2024_dapo_boxed/acc/mean@32": "acc",
    "val-core/aime_2024_dapo_boxed/acc/maj@32/mean": "maj",
    # The per-problem spread of a single arm's accuracy. Every statement in the
    # manuscript about "one standard error on 30 problems" is derived from it,
    # and those statements had drifted into three different hand-typed values.
    "val-aux/aime_2024_dapo_boxed/acc/std@32": "std32",
    "actor/entropy": "entropy",
    "response_length/mean": "resp_len",
    "perf/time_per_step": "s_per_step",
    "steerf/branch_corr_frac": "branch_frac",
    "steerf/tw_mean": "tw_mean",
}
DERIVED = {"uplift": lambda r: r["maj"] - r["acc"]}

# Arms in the order the paper reports them. The first is the reference every
# contrast is drawn against.
# Section 12.8's benchmarks, in the order collect_results.py writes them.
MATH6 = ["AIME24", "AIME25", "AMC23", "MATH500", "Minerva", "Olympiad"]

MAIN_ARMS = ["grpo", "steer", "uniform", "permuted", "signed"]

# The follow-up ablations of Table 10, and the compute-matched control. They
# are seed 1 only and are not part of the seed-level statistics, so they live
# outside MAIN_ARMS -- but the manuscript has a slot for each, and until this
# table existed nothing emitted them: the runs could all finish and Table 10
# would stay red. The value is the macro stem, because the paper names the
# lambda arms by role (lamlo/lamhi/lamzero) rather than by value.
FOLLOWUP_ARMS = {
    "lam0.1":       "lamlo",
    "lam0.5":       "lamhi",
    "lam0-tree":    "lamzero",
    # The forecaster's own control: same tree, same lambda, H_togo read
    # from the policy's realised entropy instead of the MTP heads.
    "oracle":       "oracle",
    "xclip-signed": "xclipsigned",
    "xclip-steer":  "xclipsteer",
    "rloo-signed":  "rloosigned",
    "rloo-steer":   "rloosteer",
    "opo-signed":   "oposigned",
    "opo-steer":    "oposteer",
    # Stock STEER at the released token_weight_min=0.8 rather than our 0.7.
    "wmin-steer":   "wminsteer",
}
CONTRASTS = [("signed", "grpo"), ("signed", "steer"), ("signed", "uniform"),
             ("signed", "permuted"), ("steer", "grpo"), ("uniform", "steer")]

STEP_RE = re.compile(r"step:(\d+) - global_seqlen")


def extract_from_git(ref: str, log_dir: str, dest: Path) -> int:
    """Copy every train-*.log at <ref>:<log_dir> into dest. Returns the count.

    The training logs live on the `paper` branch and the tooling lives here, so
    without this the script only runs on a pod that happens to have both.
    """
    names = subprocess.run(["git", "ls-tree", "-r", "--name-only", ref, log_dir],
                           capture_output=True, text=True).stdout.split()
    n = 0
    for name in names:
        if not (name.endswith(".log") and Path(name).name.startswith("train-")):
            continue
        blob = subprocess.run(["git", "show", f"{ref}:{name}"],
                              capture_output=True)
        if blob.returncode:
            continue
        (dest / Path(name).name).write_bytes(blob.stdout)
        n += 1
    return n


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


OFFLOAD_RE = re.compile(r"'param_offload':\s*(True|False)")


def offloaded(path: Path) -> str:
    """Did this run put params and the optimizer on the CPU?

    It matters for reading a table, not for the method: OFFLOAD=1 is how the
    queues recover from a CUDA OOM without touching ppo_micro_batch_size_per_gpu
    (which IS the treatment -- it is STEER's min-max pool). But the AdamW update
    then runs on CPU fp32, so such a seed is not bit-identical to its neighbours
    even though it is the same algorithm. A reader should be able to see which
    seeds that was, rather than find it in a queue log months later.
    """
    hits = OFFLOAD_RE.findall(path.read_text(errors="replace"))
    if not hits:
        return "?"
    return "cpu" if hits[-1] == "True" else "gpu"


def direction_consistency(tsv: Path, a: str, b: str) -> tuple[int, int] | None:
    """On how many of the six benchmarks does arm `a` beat arm `b`?

    Section 12.8's headline is not any single cell -- a 30-question benchmark
    cannot separate a method from a lucky draw -- but whether the ordering seen
    on AIME24 survives elsewhere. The count is the statistic, and the paper
    reports it "whatever it returns".

    Reads the table scripts/collect_results.py already produces, rather than
    parsing the evaluation logs a second time; a second parser is a second
    thing to drift.
    """
    if not tsv.is_file():
        return None
    rows: dict[str, dict[str, float]] = {}
    lines = tsv.read_text().splitlines()
    if not lines:
        return None
    head = lines[0].split("\t")
    for line in lines[1:]:
        cell = line.split("\t")
        if len(cell) != len(head):
            continue
        # the averaged row when there is one, else the single seed
        row = {h: c for h, c in zip(head, cell)}
        arm = row.get("arm", "")
        if arm in rows and row.get("seed") != "avg":
            continue
        vals = {}
        for bench in MATH6:
            try:
                vals[bench] = float(row.get(bench, "-"))
            except ValueError:
                pass
        if len(vals) == len(MATH6):
            rows[arm] = vals
    if a not in rows or b not in rows:
        return None
    return sum(1 for x in MATH6 if rows[a][x] > rows[b][x]), len(MATH6)


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
    ap.add_argument("--unbalanced", dest="balanced", action="store_false",
                    help="average each arm over every seed it has, even when "
                         "the arms have different seeds (the tables then stop "
                         "agreeing with the contrasts)")
    ap.add_argument("--long-steps", type=int, default=200,
                    help="final step of the compute-matched grpo-long control")
    ap.add_argument("--macro-prefix", default="",
                    help="prepend this to every macro name, so a backbone's run "
                         "can be analysed with the same code and merged into one "
                         "numbers.tex (Table 12 names them Bqwenbig, Bllama, "
                         "Bmistral)")
    ap.add_argument("--eval-table", default="results/summary.tsv",
                    help="the six-benchmark table scripts/collect_results.py writes")
    ap.add_argument("--git-ref", default=None,
                    help="read the logs out of this ref instead of the working "
                         "tree (they live on `paper`)")
    ap.add_argument("--tex-macros", default=None,
                    help="also emit \\newcommand macros the manuscript can \\input "
                         "(default: <out>/numbers.tex)")
    args = ap.parse_args(argv)

    lo, hi = (int(v) for v in args.plateau.split(":"))
    tmp = None
    if args.git_ref:
        tmp = tempfile.TemporaryDirectory()
        n = extract_from_git(args.git_ref, args.logs, Path(tmp.name))
        print(f"[analyze] {n} training log(s) from {args.git_ref}:{args.logs}")
        log_dir = Path(tmp.name)
    else:
        log_dir = Path(args.logs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def steps_for_arm(arm: str) -> int:
        # grpo-long is the compute-matched control and runs past the crossing;
        # run/_arms.sh:218 owns the same exception for the queues.
        return args.long_steps if arm == "grpo-long" else args.steps

    def run_name(arm: str, seed: int) -> str:
        t = args.model_tag
        return {
            "grpo": f"grpo-{t}-s{seed}",
            "steer": f"steer-{t}-s{seed}",
            "signed": f"steer-f-{t}-s{seed}-tree-rollout",
            "uniform": f"steer-f-{t}-s{seed}-tree-rollout-uniform",
            "permuted": f"steer-f-{t}-s{seed}-tree-rollout-permuted",
            "grpo-long": f"grpo-{t}-s{seed}-long",
            # run/_arms.sh:run_name_for owns these; kept in step with it.
            "lam0.1": f"steer-f-{t}-s{seed}-tree-rollout-lam0.1",
            "lam0.5": f"steer-f-{t}-s{seed}-tree-rollout-lam0.5",
            "lam0-tree": f"steer-f-{t}-s{seed}-tree-rollout-lam0",
            "oracle": f"steer-f-{t}-s{seed}-tree-rollout-oracle",
            "xclip-signed": f"steer-f-{t}-s{seed}-tree-rollout-xclip",
            "xclip-steer": f"steer-{t}-s{seed}-xclip",
            "rloo-signed": f"steer-f-{t}-s{seed}-tree-rollout-rloo",
            "rloo-steer": f"steer-{t}-s{seed}-rloo",
            "opo-signed": f"steer-f-{t}-s{seed}-tree-rollout-opo",
            "opo-steer": f"steer-{t}-s{seed}-opo",
            "wmin-steer": f"steer-{t}-s{seed}-wmin08",
        }[arm]

    def reached(path: Path, want: int) -> bool:
        return any(s >= want for s in parse_log(path))

    def find_log(rn: str, want: int) -> Path | None:
        """The run this arm/seed is, when more than one log carries the name.

        Seed 1 has two logs for three arms: the original, and a `_0905`
        re-launch made after the huggingface-hub break of 2026-09-08 killed a
        chain mid-flight. Both completed. Picking whichever reads better is
        cherry-picking, so the rule is fixed and independent of the outcome:

          1. among logs that reached the final step, take the BARE name --
             the original run;
          2. if the original never finished and a re-launch did, that
             re-launch IS the run;
          3. if none finished, report the bare one and let plateau() say how
             little is there.

        The re-launches are a second draw of the same configuration, and they
        are reported as that, in the within-run stability appendix.

        This used to return the LAST candidate, which silently preferred
        `_0905` over the original: the signed arm's plateau accuracy read
        .1392 instead of .1495 and the headline contrast shrank by two thirds,
        with nothing anywhere saying a different run had been substituted.
        """
        bare = log_dir / f"train-{rn}.log"
        tagged = sorted(log_dir.glob(f"train-{rn}_*.log"))
        # A bare "train-<run>*" glob would match sibling arms -- signed is a
        # prefix of permuted (see run/_arms.sh).
        cands = [c for c in [bare, *tagged] if c.is_file()]
        if not cands:
            return None
        done = [c for c in cands if reached(c, want)]
        if done:
            return bare if bare in done else done[0]
        return bare if bare.is_file() else cands[0]

    # ------------------------------------------------------------ per seed
    per_seed: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    # The same runs kept step by step. The seed-level statistics collapse each
    # run to one number, which is right for the question the paper asks, but
    # the within-run appendix needs the eight points back.
    per_step: dict[str, dict[int, dict[int, dict[str, float]]]] = defaultdict(dict)
    missing = []
    for arm in MAIN_ARMS:
        for seed in range(1, 6):
            f = find_log(run_name(arm, seed), steps_for_arm(arm))
            if f is None:
                missing.append(f"{arm} s{seed}")
                continue
            steps = parse_log(f)
            agg = plateau(steps, lo, hi)
            if agg:
                agg["log"] = f.name
                agg["stack"] = offloaded(f)
                per_seed[arm][seed] = agg
                per_step[arm][seed] = {k: v for k, v in steps.items()
                                       if lo <= k <= hi and "acc" in v}
            else:
                missing.append(f"{arm} s{seed} (no validation point in {lo}-{hi})")

    # The follow-up arms, at the one seed they were run at. They carry no error
    # bar by design (Table 10's caption says so), so they are plateau means and
    # nothing more.
    followups: dict[str, dict[str, float]] = {}
    for arm, stem in FOLLOWUP_ARMS.items():
        f = find_log(run_name(arm, 1), args.steps)
        if f is None:
            continue
        agg = plateau(parse_log(f), lo, hi)
        if agg:
            followups[stem] = agg

    # ------------------------------------------------- within-run stability
    # A paired t over the validation points of ONE run. It asks whether a gap
    # holds along a single trajectory, which is a different and narrower
    # question than whether it survives re-running -- the points are
    # consecutive validations of one run and are not independent, so df = 7 is
    # generous. The manuscript reports it in an appendix, explicitly not as an
    # estimate of run-to-run uncertainty, and only because earlier drafts
    # quoted it. The seed to use is the lowest one both arms have, so the
    # choice does not move as the campaign fills in.
    within: dict[str, dict[str, float]] = {}
    within_seed = None
    for a, b in CONTRASTS:
        shared = sorted(set(per_step.get(a, {})) & set(per_step.get(b, {})))
        if not shared:
            continue
        seed = shared[0]
        within_seed = seed if within_seed is None else min(within_seed, seed)
        sa, sb = per_step[a][seed], per_step[b][seed]
        pts = sorted(set(sa) & set(sb))
        if len(pts) < 2:
            continue
        for metric in ("acc", "maj"):
            d = [sa[t][metric] - sb[t][metric] for t in pts
                 if metric in sa[t] and metric in sb[t]]
            if len(d) >= 2:
                within[f"{a}{b}{metric}"] = paired(d)

    if not per_seed:
        sys.exit(f"[analyze] no parsable training log under {log_dir}")

    # Seeds every main arm has. The arm means and the contrasts must be drawn
    # from the SAME set, or a reader who subtracts two rows of Table 1 gets a
    # different number from the contrast table and has no way to know why. With
    # the campaign mid-flight that is a live hazard: GRPO finished seed 2 while
    # every other arm is still on seed 1, so an unbalanced mean would move GRPO
    # alone and shrink the headline gap by a third for no reason a reviewer
    # could see. The extra seeds are not discarded -- per_seed.tsv keeps them,
    # and they are what the manuscript quotes for run-to-run spread.
    common = set.intersection(*[set(per_seed.get(a, {})) for a in MAIN_ARMS]) \
        if all(per_seed.get(a) for a in MAIN_ARMS) else set()
    if args.balanced and common:
        def seeds_of(arm):
            return sorted(common)
    else:
        def seeds_of(arm):
            return sorted(per_seed.get(arm, {}))

    cols = ["acc", "maj", "uplift", "entropy", "resp_len", "s_per_step",
            "s_per_val_step", "branch_frac", "tw_mean", "n_val_points"]

    with (out_dir / "per_seed.tsv").open("w") as fh:
        fh.write("arm\tseed\t" + "\t".join(cols) + "\tstack\tlog\n")
        for arm in MAIN_ARMS:
            for seed in sorted(per_seed.get(arm, {})):
                r = per_seed[arm][seed]
                fh.write(f"{arm}\t{seed}\t"
                         + "\t".join(f"{r[c]:.4f}" if c in r else "-" for c in cols)
                         + f"\t{r.get('stack', '?')}\t{r['log']}\n")

    # ---------------------------------------------------------- arm means
    with (out_dir / "arm_means.tsv").open("w") as fh:
        fh.write("arm\tn_seeds\t" + "\t".join(f"{c}\t{c}_se" for c in cols[:-1]) + "\n")
        for arm in MAIN_ARMS:
            rows = [per_seed[arm][s_] for s_ in seeds_of(arm) if s_ in per_seed.get(arm, {})]
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
    # Table 11's cells, kept so the macro block below can emit them. The TSV
    # was written since this script existed, but nothing carried the numbers
    # into the manuscript, so the table would have stayed red however long the
    # long GRPO run went.
    match: dict[str, float] = {}
    with (out_dir / "compute_match.tsv").open("w") as fh:
        fh.write("seed\tsteerf_steps\tsteerf_seconds\tgrpo_long_step\t"
                 "grpo_long_seconds\tsteerf_acc\tgrpo_long_acc\tdiff\n")
        for seed in sorted(per_seed.get("signed", {})):
            f_long = find_log(run_name("grpo-long", seed), args.long_steps)
            f_sig = find_log(run_name("signed", seed), args.steps)
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
            if not match:          # the manuscript's table is one seed wide
                g_budget = sum(r.get("s_per_step", 0.0)
                               for st, r in long_steps.items() if st <= args.long_steps)
                match = {"Matchstep": val[-1], "Longsteps": args.long_steps,
                         "Rgrpolongacc": g_acc, "Wsigned": budget / 3600.0,
                         "Wgrpolong": spent / 3600.0,
                         "Wgrpo": g_budget * args.steps / max(1, args.long_steps) / 3600.0}

    # ------------------------------------------------------------- LaTeX
    with (out_dir / "tables.tex").open("w") as fh:
        fh.write("% generated by scripts/analyze_seeds.py -- do not edit\n")
        fh.write("% Table: arm means over seeds (plateau steps "
                 f"{lo}-{hi}, unit = seed)\n")
        for arm in MAIN_ARMS:
            rows = [per_seed[arm][s_] for s_ in seeds_of(arm) if s_ in per_seed.get(arm, {})]
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

    # ------------------------------------------------------------- macros
    # The manuscript \input{}s these, so a finished campaign turns into a
    # finished paper with one pdflatex and no retyping. Every macro that has no
    # data yet renders as a visible placeholder rather than a plausible number:
    # a paper must never contain a figure nobody measured, and "I will replace
    # it later" is how a predicted number reaches a submission.
    # A backbone run is the same five-arm analysis under another model tag, so
    # Table 12's cells are R<arm>acc and C signed-vs-<arm> acc with a prefix.
    # One invocation per backbone, each writing its own macro file:
    #
    #   analyze_seeds.py --model-tag Qwen2.5-Math-7B --macro-prefix Bqwenbig \
    #       --tex-macros results/numbers-qwenbig.tex
    #
    # and the manuscript \input{}s them all. Nothing about the statistics
    # changes; only the names do.
    macro_path = Path(args.tex_macros) if args.tex_macros else out_dir / "numbers.tex"
    letters = str.maketrans("", "", "0123456789-_.")

    def mac(name: str, value, fmt: str = "{:.4f}") -> str:
        nm = (args.macro_prefix + name).translate(letters)
        # ".1495", not "0.1495" -- the table style the manuscript already uses.
        if fmt in ("{:.4f}", "{:+.4f}") and isinstance(value, (int, float)) \
                and value == value and abs(value) < 1:
            fmt = fmt.replace(":", ":").replace("{:", "{:")  # keep the sign flag
            txt = ("{:+.4f}" if "+" in fmt else "{:.4f}").format(value)
            txt = txt.replace("0.", ".", 1) if txt.lstrip("+-").startswith("0.") else txt
            return "\\providecommand{\\%s}{%s}\n" % (nm, txt)
        # \providecommand, not \newcommand: the per-contrast n macro is emitted
        # once per metric, and a duplicate \newcommand is a LaTeX error rather
        # than a no-op.
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return "\\providecommand{\\%s}{\\PENDING}\n" % nm
        return "\\providecommand{\\%s}{%s}\n" % (nm, fmt.format(value))

    with macro_path.open("w") as fh:
        fh.write("% generated by scripts/analyze_seeds.py -- do not edit\n")
        fh.write("% \\PENDING marks a number no run has produced yet.\n")
        fh.write("\\providecommand{\\PENDING}{\\textbf{??}}\n")
        # Across-problem precision of ONE measurement on AIME24, averaged over
        # every converged-window validation we have. Emitted rather than typed:
        # the manuscript quoted .027 in one place and .029 in two others for
        # the same quantity, and the measured value is neither.
        allstd = [r["std32"] for a in MAIN_ARMS
                  for r in per_seed.get(a, {}).values() if "std32" in r]
        fh.write(mac("STDaime", (sum(allstd) / len(allstd)) if allstd else None))
        fh.write(mac("SEaime", (sum(allstd) / len(allstd) / math.sqrt(30))
                     if allstd else None))
        fh.write(mac("Nseeds", len(per_seed.get("signed", {})), "{:d}"))
        fh.write(mac("Plateaulo", lo, "{:d}"))
        fh.write(mac("Plateauhi", hi, "{:d}"))
        for arm in MAIN_ARMS:
            rows = [per_seed[arm][s_] for s_ in seeds_of(arm) if s_ in per_seed.get(arm, {})]
            fh.write(mac(f"N{arm}", len(rows), "{:d}"))
            # how many that arm actually has, balanced or not
            fh.write(mac(f"Nall{arm}", len(per_seed.get(arm, {})), "{:d}"))
            # Run-to-run spread of ONE arm across ALL its seeds. This is the
            # only direct measurement of how much a seed moves a number, and
            # the manuscript compares the headline gap against it -- a gap the
            # size of the spread is not yet a result.
            for c in ("acc", "maj"):
                v = [r[c] for r in per_seed.get(arm, {}).values() if c in r]
                fh.write(mac(f"S{arm}{c}", (max(v) - min(v)) if len(v) > 1 else None))
                fh.write(mac(f"Lo{arm}{c}", min(v) if len(v) > 1 else None))
                fh.write(mac(f"Hi{arm}{c}", max(v) if len(v) > 1 else None))
            for c in ("acc", "maj", "uplift", "entropy"):
                v = [r[c] for r in rows if c in r]
                fh.write(mac(f"R{arm}{c}", (sum(v) / len(v)) if v else None))
                fh.write(mac(f"E{arm}{c}",
                             (statistics.stdev(v) / math.sqrt(len(v))) if len(v) > 1 else None))
            v = [r["resp_len"] for r in rows if "resp_len" in r]
            fh.write(mac(f"R{arm}len", (sum(v) / len(v)) if v else None, "{:.0f}"))
            v = [r["s_per_step"] for r in rows if "s_per_step" in r]
            fh.write(mac(f"R{arm}cost", (sum(v) / len(v)) if v else None, "{:.0f}"))
        # Table 12's bottom line: on how many backbones does STEER-F - GRPO
        # keep its sign? Each backbone writes its own macro file with a prefix,
        # so count the ones already on disk plus this run. Reported as a count
        # out of four because the rows are not on a common scale and must not
        # be averaged -- docs/preregistration_5seed.md fixes that in advance.
        if not args.macro_prefix:
            signs = []
            for f in sorted(out_dir.glob("numbers-*.tex")) + [macro_path]:
                if not f.is_file():
                    continue
                for m in re.finditer(r"\\providecommand\{\\[A-Za-z]*Csignedgrpoacc\}"
                                     r"\{([^}]*)\}", f.read_text()):
                    v = m.group(1).replace("PENDING", "")
                    try:
                        signs.append(float(v.lstrip("+")) > 0)
                    except ValueError:
                        pass
            fh.write(mac("Bsigncount", sum(signs) if signs else None, "{:d}"))

        # Section 12.8. Absent evaluation table -> PENDING, not a guess.
        ev = Path(args.eval_table)
        for name, other in (("Dirconsistency", "uniform"),
                            ("Dirconsistencyperm", "permuted")):
            hit = direction_consistency(ev, "signed", other)
            fh.write(mac(name, hit[0] if hit else None, "{:d}"))

        # Table 11. Wall clocks are hours, which is how the caption reads them.
        for k in ("Matchstep", "Longsteps"):
            fh.write(mac(k, match.get(k), "{:d}") if k in match else mac(k, None))
        fh.write(mac("Rgrpolongacc", match.get("Rgrpolongacc")))
        for k in ("Wsigned", "Wgrpo", "Wgrpolong"):
            fh.write(mac(k, match.get(k), "{:.1f}") if k in match else mac(k, None))
        for stem, agg in sorted(followups.items()):
            # uplift too: the lambda=0 tree arm's uplift is the number that
            # settles whether the structural gain belongs to the sampler.
            for c in ("acc", "maj", "uplift"):
                fh.write(mac(f"R{stem}{c}", agg.get(c)))
        fh.write(mac("Wseed", within_seed, "{:d}") if within_seed
                 else mac("Wseed", None))
        for key, st in sorted(within.items()):
            fh.write(mac(f"W{key}", st["mean"], "{:+.4f}"))
            fh.write(mac(f"WT{key}", st["t"], "{:+.2f}"))
            fh.write(mac(f"WN{key}", st["n"], "{:d}"))
        seen = set()
        for name, metric, st in contrast_rows:
            a, b = name.split(" - ")
            key = f"{a}{b}{metric}"
            if key in seen:
                continue
            seen.add(key)
            fh.write(mac(f"C{key}", st["mean"], "{:+.4f}"))
            fh.write(mac(f"T{key}", st["t"], "{:+.2f}"))
            fh.write(mac(f"P{key}", st["p"], "{:.3f}"))
            fh.write(mac(f"N{a}{b}", st["n"], "{:d}"))

    print(f"[analyze] wrote {out_dir}/per_seed.tsv, arm_means.tsv, contrasts.tsv, "
          f"compute_match.tsv, tables.tex, {macro_path.name}")
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
