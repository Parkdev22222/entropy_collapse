#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
# Licensed under the Apache License, Version 2.0
"""Regenerate the seed-1 results table from the training logs.

    python3 scripts/seed1_table.py --git-ref origin/paper --out docs/results_seed1.md

WHY THIS EXISTS
    The seed-1 numbers have been recomputed by hand in three separate sessions.
    Each time the same two traps came back: the majority-vote key is
    ``acc/maj@32/mean`` (not ``maj@32/mean@32``), and three of the logs contain
    more than one launch, so a naive pass mixes a dead run's early steps into
    the plateau. This script settles both once, and writes a document instead
    of a chat message.

    It also carries the one row nobody can recompute. The GRPO seed-1 training
    log is not in any branch; its numbers exist only because they were pasted
    into a session. They live in docs/seed1_grpo_transcript.json and are always
    labelled as transcript-sourced. The moment a real log appears at
    logs/experiments/train-grpo-<tag>-s1.log this script prefers it and the
    label changes by itself.

WHAT THE STATISTICS MEAN -- READ THIS BEFORE QUOTING A t
    The contrasts here pair the eight plateau *steps of one run*. That asks
    whether a gap is stable inside one trajectory. It does NOT ask whether the
    gap survives re-running, and the eight points are consecutive validations of
    the same run, so they are not independent and df = 7 is generous.
    docs/preregistration_5seed.md fixes the primary unit as the SEED; with one
    seed there is no test. Use scripts/analyze_seeds.py once the campaign lands.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_seeds import (extract_from_git, paired, parse_log,  # noqa: E402
                           plateau)

# Seed-1 arms, in the order the document reports them. GRPO comes first because
# every contrast is drawn against it.
#
# The last two log names do not follow run/_arms.sh's run_name_for(): they
# predate it. Keeping the legacy names here is the whole reason this table is
# not just a call to analyze_seeds.py, whose find_log() derives names from the
# current convention.
ARMS = [
    ("grpo",     "GRPO",     "train-grpo-{tag}-s1.log",                          "-",     "plain", "none (`vanilla`)"),
    ("steer",    "STEER",    "train-steer-{tag}-s1.log",                         "0",     "plain", "local Omega only"),
    ("uniform",  "uniform",  "train-steer-f-{tag}-s1-tree-rollout-uniform.log",  ".25",   "tree",  "uniform decay on the support"),
    ("permuted", "permuted", "train-steer-f-{tag}-s1-tree-rollout-permuted.log", ".25",   "tree",  "A_H shuffled among siblings"),
    ("signed",   "STEER-F",  "train-steer-f-{tag}-s1-tree-rollout.log",          ".25",   "tree",  "**A_H**"),
    ("rank",     "rank",     "train-math-{tag}-s1.log",                          ".25",   "plain", "A_H, rank mapping"),
]
# Arms that did not run at the campaign's topology/length. Rendered as footnote
# markers in the table so nobody averages across them by accident.
CAVEAT = {
    "rank": "one GPU, tp=1, 140 steps -- every other arm ran on two GPUs, tp=2, 110 steps",
}
STEP0_RE = re.compile(r"step:0 - .*?val-core/aime_2024_dapo_boxed/acc/mean@32:([0-9.]+)")


def step_zero_acc(path: Path) -> float | None:
    """The pre-training validation, which parse_log skips.

    parse_log keys on ``step:N - global_seqlen`` and step 0 has no training
    step, so it never appears there. The number matters: it is how we know two
    arms started from the same checkpoint.
    """
    hits = STEP0_RE.findall(path.read_text(errors="replace"))
    return float(hits[-1]) if hits else None


def per_step(path: Path, lo: int, hi: int) -> dict[int, dict[str, float]]:
    return {s: r for s, r in parse_log(path).items()
            if lo <= s <= hi and "acc" in r}


def fmt(v, spec="{:.4f}", strip_zero=True):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "--"
    txt = spec.format(v)
    if strip_zero and txt.lstrip("+-").startswith("0."):
        txt = txt.replace("0.", ".", 1)
    return txt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", default="logs/experiments")
    ap.add_argument("--git-ref", default=None,
                    help="read the logs out of this ref instead of the working tree "
                         "(the training logs live on `paper`)")
    ap.add_argument("--out", default="docs/results_seed1.md")
    ap.add_argument("--transcript", default="docs/seed1_grpo_transcript.json")
    ap.add_argument("--model-tag", default="Qwen2.5-Math-1.5B")
    ap.add_argument("--plateau", default="40:110")
    args = ap.parse_args(argv)

    lo, hi = (int(v) for v in args.plateau.split(":"))
    tmp = None
    if args.git_ref:
        tmp = tempfile.TemporaryDirectory()
        n = extract_from_git(args.git_ref, args.logs, Path(tmp.name))
        print(f"[seed1] {n} training log(s) from {args.git_ref}:{args.logs}")
        log_dir = Path(tmp.name)
    else:
        log_dir = Path(args.logs)

    transcript = {}
    tpath = Path(args.transcript)
    if tpath.is_file():
        transcript = json.loads(tpath.read_text())

    rows, steps_of, missing = {}, {}, []
    for key, _label, pat, *_ in ARMS:
        f = log_dir / pat.format(tag=args.model_tag)
        if f.is_file():
            agg = plateau(parse_log(f), lo, hi)
            if agg:
                agg["step0"] = step_zero_acc(f)
                agg["source"] = f.name
                rows[key] = agg
                steps_of[key] = {s: r for s, r in per_step(f, lo, hi).items()}
                continue
        if key == transcript.get("arm") and transcript.get("seed") == 1:
            val = {int(s): v for s, v in transcript["val"].items()}
            win = [v for s, v in val.items() if lo <= s <= hi]
            agg = {
                "acc": sum(v["acc"] for v in win) / len(win),
                "maj": sum(v["maj"] for v in win) / len(win),
                "n_val_points": len(win),
                "step0": val.get(0, {}).get("acc"),
                "source": f"{tpath.name} -- NOT IN GIT",
            }
            agg["uplift"] = agg["maj"] - agg["acc"]
            agg.update({k: v for k, v in transcript.get("plateau_only", {}).items()
                        if k in ("entropy", "resp_len", "s_per_step")})
            rows[key] = agg
            steps_of[key] = {s: v for s, v in val.items() if lo <= s <= hi}
            continue
        missing.append(key)

    if not rows:
        print(f"[seed1] no seed-1 log under {log_dir} -- nothing to write")
        return 0

    ref = "grpo"
    contrasts = []
    if ref in steps_of:
        for key, label, *_ in ARMS:
            if key == ref or key not in steps_of:
                continue
            shared = sorted(set(steps_of[key]) & set(steps_of[ref]))
            if len(shared) < 2:
                continue
            cell = {"arm": key, "label": label, "n": len(shared)}
            for metric in ("acc", "maj", "uplift"):
                def get(d, s, m=metric):
                    r = d[s]
                    return r["maj"] - r["acc"] if m == "uplift" else r[m]
                cell[metric] = paired([get(steps_of[key], s) - get(steps_of[ref], s)
                                       for s in shared])
            contrasts.append(cell)
        contrasts.sort(key=lambda c: -c["acc"]["mean"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w = out.open("w")
    w.write("<!-- generated by scripts/seed1_table.py -- do not edit by hand -->\n")
    w.write("# Seed-1 results (AIME 2024)\n\n")
    w.write(f"Plateau window: steps {lo}-{hi}. Model: {args.model_tag}. "
            "Regenerate with\n\n")
    w.write("```bash\npython3 scripts/seed1_table.py --git-ref origin/paper\n```\n\n")
    w.write("> **One seed.** Every t below pairs the plateau *steps of one run*, so it\n"
            "> measures whether a gap is stable inside one trajectory -- not whether it\n"
            "> survives re-running, which is the question a reviewer asks. The eight\n"
            "> points are consecutive validations of the same run and are not\n"
            "> independent. `docs/preregistration_5seed.md` fixes the primary unit as the\n"
            "> seed; at n=1 there is no seed-level test. **Do not quote these as the\n"
            "> paper's significance.**\n\n")

    w.write("## Plateau means\n\n")
    w.write("| arm | lam | rollout | signal | acc@32 | maj@32 | uplift | entropy | resp len | step-0 acc | source |\n")
    w.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
    for key, label, _pat, lam, roll, sig in ARMS:
        if key not in rows:
            w.write(f"| **{label}** | {lam} | {roll} | {sig} | "
                    + " | ".join(["MISSING"] * 6) + " | -- |\n")
            continue
        r = rows[key]
        mark = "[^%s]" % key if key in CAVEAT else ""
        w.write(f"| **{label}**{mark} | {lam} | {roll} | {sig} | "
                f"{fmt(r.get('acc'))} | {fmt(r.get('maj'))} | {fmt(r.get('uplift'))} | "
                f"{fmt(r.get('entropy'))} | {fmt(r.get('resp_len'), '{:.0f}', False)} | "
                f"{fmt(r.get('step0'))} | `{r['source']}` |\n")
    w.write("\n")
    for key, why in CAVEAT.items():
        if key in rows:
            w.write(f"[^{key}]: {why}.\n")
    w.write("\n")

    if contrasts:
        w.write(f"## Paired against {ref.upper()} (same plateau steps)\n\n")
        w.write("| contrast | n | d acc | t | d maj | t | d uplift | t |\n")
        w.write("|---|---|---|---|---|---|---|---|\n")
        for c in contrasts:
            w.write(f"| {c['label']} - {ref.upper()} | {c['n']} | "
                    + " | ".join(
                        f"{fmt(c[m]['mean'], '{:+.4f}')} | {c[m]['t']:+.2f}"
                        for m in ("acc", "maj", "uplift"))
                    + " |\n")
        w.write("\n")

    w.write("## What this does and does not show\n\n")
    w.write("* The ladder is `GRPO ~ STEER ~ uniform << STEER-F`. The three controls sit\n"
            "  at zero and only the treatment moves. Shuffling A_H among siblings\n"
            "  (permuted) removes most of the gain, which is what separates the signal\n"
            "  from the apparatus that carries it.\n")
    w.write("* Most of the gain is not per-sample accuracy: uplift (maj - acc) carries\n"
            "  the larger share of the majority-vote difference.\n")
    w.write("* Entropy is not the channel. The two arms whose plateau entropy is\n"
            "  indistinguishable (STEER and STEER-F) are the two furthest apart in\n"
            "  accuracy, and the arm with the highest entropy is not the best arm.\n")
    w.write("* **Not established.** One seed, one 30-problem benchmark. The only\n"
            "  repeat we have of any arm (STEER-F run twice, common window 40-90)\n"
            "  differed by .0088, against a headline gap of .0145 -- the same order of\n"
            "  magnitude. The five-seed campaign is what decides this.\n\n")

    if missing:
        w.write("## Missing\n\n")
        for key in missing:
            w.write(f"* `{key}` -- no log at `{args.logs}/"
                    f"{dict((a[0], a[2]) for a in ARMS)[key].format(tag=args.model_tag)}`\n")
        w.write("\n")
    w.write("Also not in the table: the lambda=0 + tree arm "
            f"(`train-{args.model_tag}-s1-tree-rollout.log`) has a single validation "
            "point at step 110 (acc .1430), too few for a plateau. The H100 "
            "`lam0-tree` follow-up fills that cell.\n")
    w.close()

    print(f"[seed1] wrote {out}")
    for key, label, *_ in ARMS:
        r = rows.get(key)
        print(f"  {label:9s} " + ("MISSING" if not r else
              f"acc {fmt(r.get('acc'))}  maj {fmt(r.get('maj'))}  "
              f"n={r.get('n_val_points')}  {r['source']}"))
    if tmp:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
