#!/usr/bin/env python3
"""Sibling spread of A_H per branch point, STEER-F against its two controls.

    python3 scripts/sibling_spread.py --git-ref origin/paper

Implements docs/sibling_spread_analysis.md exactly; that file fixed the quantity,
the window, the seeds and the verdict rule before this script was written. The
analysis is exploratory and post hoc with respect to the pre-registration.

Writes results/sibling_spread.tsv (one row per run) and
results/numbers-spread.tex (the manuscript's \\num{} macros).
"""
from __future__ import annotations

import argparse
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_seeds import t_two_sided_p  # noqa: E402

ARMS = ["signed", "uniform", "permuted"]
CONTROLS = {"uniform": "Uniform", "permuted": "Permuted"}
SEEDS = [1, 3, 4]
LO, HI = 40, 110
GROUP = 512 * 8                      # responses per optimization step
KEYS = {"a": "steerf/a_h_std", "nb": "steerf/n_branch_points",
        "len": "response_length/mean"}
STEP_RE = re.compile(r"step:(\d+) - ")


def names() -> set[str]:
    """Every macro this script can emit. tests/test_paper.py reads this."""
    out = {"SpreadNseeds", "SpreadVerdict"}
    for c in CONTROLS.values():
        out |= {f"SpreadRel{c}", f"SpreadPos{c}", f"SpreadT{c}", f"SpreadP{c}",
                f"SpreadLo{c}", f"SpreadHi{c}"}
    return out


def read_log(ref: str | None, log_dir: str, name: str) -> str:
    if ref:
        return subprocess.run(["git", "show", f"{ref}:{log_dir}/{name}"],
                              capture_output=True, check=True).stdout.decode("utf-8", "replace")
    return (Path(log_dir) / name).read_text(errors="replace")


def per_step(text: str) -> dict[int, float]:
    """step -> S_t, the RMS of A_H per branch point (last line wins per step).

    Amendment 1 of the plan: a_h_abs_mean prints as 0.000, so the RMS is built
    from a_h_std, whose square times the token count is the sum of A_H^2 (A_H
    has mean zero over the batch because it sums to zero per sibling set).
    """
    out: dict[int, float] = {}
    for line in text.replace("\r", "\n").split("\n"):
        m = STEP_RE.search(line)
        if not m:
            continue
        vals = {}
        for k, key in KEYS.items():
            hit = re.search(re.escape(key) + r":(-?[0-9.eE+-]+)", line)
            if hit:
                vals[k] = float(hit.group(1))
        if len(vals) == len(KEYS) and vals["nb"] > 0:
            out[int(m.group(1))] = vals["a"] * math.sqrt(GROUP * vals["len"] / vals["nb"])
    return out


def window_mean(series: dict[int, float]) -> tuple[float, int]:
    vals = [v for s, v in series.items() if LO <= s <= HI]
    return (sum(vals) / len(vals), len(vals)) if vals else (float("nan"), 0)


def logs_from_per_seed(path: Path) -> dict[tuple[str, int], str]:
    rows = path.read_text().splitlines()
    head = rows[0].split("\t")
    out = {}
    for r in rows[1:]:
        rec = dict(zip(head, r.split("\t")))
        if rec["arm"] in ARMS and int(rec["seed"]) in SEEDS:
            out[(rec["arm"], int(rec["seed"]))] = rec["log"]
    return out


RESOLUTION = 0.05      # amendment 1: per-step quantization of a_h_std


def verdict(stats: dict[str, dict]) -> str:
    means = [s["mean"] for s in stats.values()]
    if any(abs(m) < RESOLUTION for m in means):
        return "inconclusive"
    pos_ok = all(s["pos"] >= 2 for s in stats.values())
    if all(m > 0 for m in means) and pos_ok:
        return "consistent with widening"
    if all(m < 0 for m in means):
        return "against"
    return "inconclusive"


def analyse(git_ref, log_dir, per_seed_tsv):
    logs = logs_from_per_seed(per_seed_tsv)
    runs = {}
    for (arm, seed), name in sorted(logs.items()):
        s, n = window_mean(per_step(read_log(git_ref, log_dir, name)))
        runs[(arm, seed)] = {"S": s, "n_steps": n, "log": name}
    stats = {}
    for ctrl in CONTROLS:
        rel = [runs[("signed", sd)]["S"] / runs[(ctrl, sd)]["S"] - 1
               for sd in SEEDS if ("signed", sd) in runs and (ctrl, sd) in runs]
        n = len(rel)
        mean = sum(rel) / n
        sd = statistics.stdev(rel) if n > 1 else float("nan")
        t = mean / (sd / math.sqrt(n)) if n > 1 and sd > 0 else float("nan")
        stats[ctrl] = {"rel": rel, "n": n, "mean": mean, "t": t,
                       "p": t_two_sided_p(t, n - 1) if n > 1 else float("nan"),
                       "pos": sum(1 for x in rel if x > 0),
                       "lo": min(rel), "hi": max(rel)}
    return runs, stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--git-ref", default="origin/paper")
    p.add_argument("--log-dir", default="logs/experiments")
    p.add_argument("--per-seed", default="results/per_seed.tsv")
    p.add_argument("--out", default="results")
    args = p.parse_args(argv)

    runs, stats = analyse(args.git_ref or None, args.log_dir, Path(args.per_seed))
    out = Path(args.out)
    with open(out / "sibling_spread.tsv", "w") as fh:
        fh.write("arm\tseed\tspread\tn_steps\tlog\n")
        for (arm, seed), r in sorted(runs.items()):
            fh.write(f"{arm}\t{seed}\t{r['S']:.6f}\t{r['n_steps']}\t{r['log']}\n")

    def pct(x):
        return f"{100 * x:+.1f}"
    v = verdict(stats)
    lines = ["% generated by scripts/sibling_spread.py -- do not edit",
             "% plan fixed beforehand in docs/sibling_spread_analysis.md",
             f"\\providecommand{{\\SpreadNseeds}}{{{len(SEEDS)}}}",
             f"\\providecommand{{\\SpreadVerdict}}{{{v}}}"]
    for ctrl, c in CONTROLS.items():
        s = stats[ctrl]
        lines += [f"\\providecommand{{\\SpreadRel{c}}}{{{pct(s['mean'])}}}",
                  f"\\providecommand{{\\SpreadPos{c}}}{{{s['pos']}}}",
                  f"\\providecommand{{\\SpreadT{c}}}{{{s['t']:+.2f}}}",
                  f"\\providecommand{{\\SpreadP{c}}}{{{s['p']:.3f}}}",
                  f"\\providecommand{{\\SpreadLo{c}}}{{{pct(s['lo'])}}}",
                  f"\\providecommand{{\\SpreadHi{c}}}{{{pct(s['hi'])}}}"]
    (out / "numbers-spread.tex").write_text("\n".join(lines) + "\n")

    for (arm, seed), r in sorted(runs.items()):
        print(f"[spread] {arm:9s} s{seed}  S={r['S']:.5f}  ({r['n_steps']} steps)")
    for ctrl, s in stats.items():
        print(f"[spread] STEER-F vs {ctrl}: per seed "
              + ", ".join(pct(x) + "%" for x in s["rel"])
              + f"; mean {pct(s['mean'])}%, t={s['t']:+.2f}, p={s['p']:.3f}, "
              f"positive {s['pos']}/{s['n']}")
    print(f"[spread] verdict (rule fixed in advance): {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
