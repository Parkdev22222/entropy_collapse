#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
# Licensed under the Apache License, Version 2.0
"""Turn the two-root, 200-step runs into the manuscript's \\num{} macros.

    python3 scripts/emit_r2_numbers.py --git-ref origin/paper \
        --out results/numbers-r2.tex

WHY THIS EXISTS
    Section "Against the reported numbers" sets our two 200-step runs
    (steerf-r2-1p5b-s1-200, -s2-200) beside the base method's own Table 12,
    under the protocol that table states: the AIME24-best checkpoint of each
    run, averaged over two runs, scored on six benchmarks.  Every number in that
    table comes from here or from docs/paper_reference.tsv; none is typed.

WHERE EACH CELL COMES FROM
    AIME24 is the training-time validation value at the selected step, read
    from the training log -- the value the selection itself used (decision
    recorded 2026-10-06).  The other five benchmarks were never validated
    during training and come from the six-benchmark evaluation of that same
    checkpoint (eval-r2-s<seed>-step<step>.log).  The reference rows are the
    base method's published 1.5B rows, transcribed once into
    docs/paper_reference.tsv.

WHAT IT REFUSES
    A training-log AIME24 value is used only if
      * its accuracy agrees with its own reward, acc == (reward + 1) / 2, to
        the three decimals the step line prints (same rule as
        check_eval_logs.py), and
      * it sits on the 1/960 grid an average over 30 problems x 32 samples
        must sit on, and agrees with the trainer's own "New best ... at step"
        line when there is one.
    On 2026-10-05 the published copy of the seed-2 log carried 0.181 at step
    190 against a reward of -0.662 (0.169) and a "New best ... 0.1688" line;
    these checks are what keep such a value out of the table.

    An evaluation log is used only if check_eval_logs.check() finds it
    coherent and its model path names this run's checkpoint at this step.

    Anything refused or missing is left out, so the manuscript prints its red
    placeholder for it rather than a number.  The two-run mean of a cell needs
    both runs.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from check_eval_logs import check as check_eval_text  # noqa: E402
from collect_results import _find_any  # noqa: E402

SEEDS = {1: "One", 2: "Two"}
RUN = "steerf-r2-1p5b-s{seed}-200"
LAST_STEP = 200
AIME_KEY = "val-core/aime_2024_dapo_boxed/acc/mean@32"
AIME_REWARD = "val-aux/aime_2024_dapo_boxed/reward/mean@32"
AIME_N = 960            # 30 problems x 32 samples
TOL = 0.0015            # both sides of the identity printed to three decimals

# data_source, macro stem, metric suffix, column in docs/paper_reference.tsv
BENCH = [
    ("aime_2024_dapo_boxed", "Aime", "mean@32", "AIME24"),
    ("aime_2025_dapo_boxed", "Aimefive", "mean@32", "AIME25"),
    ("amc2023_dapo_boxed", "Amc", "mean@32", "AMC23"),
    ("math500", "Mathfive", "mean@1", "MATH500"),
    ("minerva_math", "Minerva", "mean@1", "Minerva"),
    ("olympiadbench", "Oly", "mean@1", "Olympiad"),
]
REF_ROWS = {"Base": "Base", "GRPO": "Grpo", "STEER": "Steer"}
REF_TABLE, REF_MODEL = "T12", "Qwen2.5-Math-1.5B"


def names() -> set[str]:
    """Every macro this script can emit. tests/test_paper.py reads this."""
    out = {f"ProtostepBest{who}" for who in SEEDS.values()}
    cols = [b[1] for b in BENCH] + ["Avg"]
    for row in REF_ROWS.values():
        out |= {f"Protoref{row}{c}" for c in cols}
    for point in ("best", "last"):
        for who in list(SEEDS.values()) + ["Mean"]:
            out |= {f"Proto{point}{who}{c}" for c in cols}
    out |= {"ProtocmpSteerBelow", "ProtocmpSteerNotbelow", "ProtocmpGrpoAbove"}
    return out


# ---------------------------------------------------------------- reading logs
def read_log(name: str, git_ref: str | None, log_dir: str) -> str | None:
    if git_ref:
        r = subprocess.run(["git", "show", f"{git_ref}:{log_dir}/{name}"],
                           capture_output=True)
        return r.stdout.decode(errors="replace") if r.returncode == 0 else None
    p = Path(log_dir) / name
    return p.read_text(errors="replace") if p.is_file() else None


def best_step(train_text: str) -> int | None:
    """The step the trainer last announced as a new best."""
    hits = re.findall(r"New best \S+: [0-9.]+ at step (\d+)", train_text)
    return int(hits[-1]) if hits else None


def step_line(text: str, step: int) -> str | None:
    lines = [l for l in text.replace("\r", "\n").split("\n")
             if re.search(rf"step:{step} - global_seqlen", l)]
    return lines[-1] if lines else None


def aime_from_training(train_text: str, step: int) -> tuple[int | None, str]:
    """-> (k, reason): the AIME24 value at `step` as k/960, or None and why."""
    line = step_line(train_text, step)
    if line is None:
        return None, f"no step-{step} line"
    a = re.search(re.escape(AIME_KEY) + r":(-?[0-9.]+)", line)
    r = re.search(re.escape(AIME_REWARD) + r":(-?[0-9.]+)", line)
    if not a or not r:
        return None, f"step {step} carries no AIME24 validation"
    acc, rew = float(a.group(1)), float(r.group(1))
    if abs(acc - (rew + 1) / 2) > TOL:
        return None, (f"step {step}: acc {acc:.3f} disagrees with its reward "
                      f"({(rew + 1) / 2:.3f})")
    k = round(acc * AIME_N)
    if abs(acc - k / AIME_N) > 0.0005 + 1e-9:
        return None, f"step {step}: {acc:.3f} is not on the 1/{AIME_N} grid"
    tracker = re.findall(rf"New best \S+: ([0-9.]+) at step {step}\b",
                         train_text)
    if tracker and abs(float(tracker[-1]) - k / AIME_N) > 0.00005 + 1e-9:
        return None, (f"step {step}: step line {acc:.3f} disagrees with the "
                      f"trainer's New best {tracker[-1]}")
    return k, ""


def eval_values(eval_text: str, run: str, step: int) -> tuple[dict, str]:
    """-> ({data_source: fraction}, reason-if-refused)."""
    seen, bad = check_eval_text(eval_text)
    if bad:
        return {}, "incoherent: " + "; ".join(bad)
    if not seen:
        return {}, "no metrics (the evaluation did not finish)"
    paths = set(re.findall(r"actor_rollout_ref\.model\.path=(\S+)", eval_text))
    want = f"{run}/global_step_{step}/"
    if not paths or not all(want in p for p in paths):
        return {}, f"model path {sorted(paths)} is not {want}"
    out = {}
    for ds, _, suffix, _ in BENCH:
        v, _ = _find_any(eval_text, ds, suffix)
        if v is not None:
            out[ds] = v
    return out, ""


def load_reference(path: Path) -> dict[str, dict[str, float]]:
    """method -> {column: percent} for the 1.5B table."""
    rows, header = {}, None
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cells = line.split("\t")
        if header is None:
            header = cells
            continue
        rec = dict(zip(header, cells))
        if rec.get("table") == REF_TABLE and rec.get("model") == REF_MODEL:
            rows[rec["method"]] = {c: float(rec[c]) for c in header[3:]}
    return rows


# ---------------------------------------------------------------- assembling
def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def collect(git_ref, log_dir, ref_path):
    values, notes = {}, []

    ref = load_reference(ref_path)
    for method, stem in REF_ROWS.items():
        if method not in ref:
            notes.append(f"reference row {method} missing from {ref_path}")
            continue
        for _, b, _, col in BENCH:
            values[f"Protoref{stem}{b}"] = f"{ref[method][col]:.1f}"
        values[f"Protoref{stem}Avg"] = f"{ref[method]['Avg']:.1f}"

    # per point, per seed: {data_source: fraction}; AIME24 also kept as k
    cells = {"best": {}, "last": {}}
    aime_k = {"best": {}, "last": {}}
    for seed, who in SEEDS.items():
        run = RUN.format(seed=seed)
        train = read_log(f"train-{run}.log", git_ref, log_dir)
        if train is None:
            notes.append(f"{run}: no training log")
            continue
        steps = {"best": best_step(train), "last": LAST_STEP}
        if steps["best"] is None:
            notes.append(f"{run}: no New best line")
        else:
            values[f"ProtostepBest{who}"] = str(steps["best"])
        for point, step in steps.items():
            if step is None:
                continue
            row = {}
            k, why = aime_from_training(train, step)
            if k is None:
                notes.append(f"{run} AIME24 refused: {why}")
            else:
                row["aime_2024_dapo_boxed"] = k / AIME_N
                aime_k[point][seed] = k
            ev = read_log(f"eval-r2-s{seed}-step{step}.log", git_ref, log_dir)
            if ev is None:
                notes.append(f"{run} step {step}: no evaluation log")
            else:
                got, why = eval_values(ev, run, step)
                if why:
                    notes.append(f"{run} step {step} evaluation refused: {why}")
                for ds, v in got.items():
                    if ds != "aime_2024_dapo_boxed":   # AIME24 is the training value
                        row[ds] = v
            cells[point][seed] = row

    for point, per_seed in cells.items():
        for seed, who in SEEDS.items():
            row = per_seed.get(seed, {})
            for ds, b, _, _ in BENCH:
                if ds in row:
                    values[f"Proto{point}{who}{b}"] = pct(row[ds])
            if all(ds in row for ds, *_ in BENCH):
                values[f"Proto{point}{who}Avg"] = pct(
                    sum(row[ds] for ds, *_ in BENCH) / len(BENCH))
        both = [per_seed.get(s, {}) for s in SEEDS]
        mean = {}
        for ds, b, _, _ in BENCH:
            if all(ds in r for r in both):
                if ds == "aime_2024_dapo_boxed":
                    # exact: average the counts, not two rounded fractions
                    mean[ds] = sum(aime_k[point][s] for s in SEEDS) / (
                        AIME_N * len(SEEDS))
                else:
                    mean[ds] = sum(r[ds] for r in both) / len(both)
                values[f"Proto{point}Mean{b}"] = pct(mean[ds])
        if len(mean) == len(BENCH):
            values[f"Proto{point}MeanAvg"] = pct(sum(mean.values()) / len(BENCH))
    values.update(direction_counts(values))
    return values, notes


def direction_counts(values: dict) -> dict:
    """How the selected-checkpoint two-run mean sits against the reported rows.

    Compared at the printed precision (one decimal), because the reported rows
    have no more; a printed tie counts as neither above nor below.  Emitted only
    when every cell it compares is present, so the prose cannot state a count
    the table does not show.
    """
    cells = [(b, col) for _, b, _, col in BENCH]
    need = [f"ProtobestMean{b}" for b, _ in cells]
    need += [f"Protoref{r}{b}" for r in ("Steer", "Grpo") for b, _ in cells]
    if not all(k in values for k in need):
        return {}
    ours = {b: float(values[f"ProtobestMean{b}"]) for b, _ in cells}
    steer = {b: float(values[f"ProtorefSteer{b}"]) for b, _ in cells}
    grpo = {b: float(values[f"ProtorefGrpo{b}"]) for b, _ in cells}
    notbelow = [col for b, col in cells if ours[b] >= steer[b]]
    return {
        "ProtocmpSteerBelow": str(sum(ours[b] < steer[b] for b, _ in cells)),
        "ProtocmpSteerNotbelow": " and ".join(notbelow) if notbelow else "none",
        "ProtocmpGrpoAbove": str(sum(ours[b] > grpo[b] for b, _ in cells)),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--git-ref", default="origin/paper",
                   help="branch holding the logs; '' reads --log-dir from disk")
    p.add_argument("--log-dir", default="logs/experiments")
    p.add_argument("--reference", default="docs/paper_reference.tsv")
    p.add_argument("--out", default="results/numbers-r2.tex")
    args = p.parse_args(argv)

    values, notes = collect(args.git_ref or None, args.log_dir,
                            Path(args.reference))
    unknown = set(values) - names()
    assert not unknown, f"emitting names not declared in names(): {unknown}"

    lines = ["% generated by scripts/emit_r2_numbers.py -- do not edit",
             f"% source: {args.git_ref or args.log_dir}, {args.reference}"]
    lines += [f"\\providecommand{{\\{k}}}{{{v}}}" for k, v in sorted(values.items())]
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n")

    print(f"[r2-macros] {len(values)} macros -> {dest}")
    for n in notes:
        print(f"[r2-macros] {n}")
    missing = sorted(names() - set(values))
    if missing:
        print(f"[r2-macros] left red ({len(missing)}): {' '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
