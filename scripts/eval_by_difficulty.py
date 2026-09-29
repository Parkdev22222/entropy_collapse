#!/usr/bin/env python3
"""Does STEER-F's difference from the other arms grow with problem difficulty?

    python3 scripts/eval_by_difficulty.py --val-data validation_data/eval
    python3 scripts/eval_by_difficulty.py --levels math500_levels.jsonl   # offline

EXPLORATORY. The hypothesis came from looking at the six-benchmark table --
STEER-F led on AIME24 and OlympiadBench and trailed on AMC23, MATH500 and
Minerva -- so testing it on the same evaluation is looking twice at one set of
data. What comes out goes into the manuscript as an exploratory analysis beside
the registered six-benchmark result, never in place of it. A claim needs the
confirmatory step: the hypothesis and the rule below written down first, then a
problem set nobody has looked at.

FIXED BEFORE THIS WAS RUN (see the commit that added it):
  difficulty   MATH's own `level`, 1 to 5, from HuggingFaceH4/MATH-500. Not a
               difficulty derived from these models' scores: sorting problems by
               how the arms did and then comparing the arms builds in regression
               to the mean, which manufactures exactly the trend being looked for.
  statistic    per-problem paired difference (STEER-F minus the other arm)
               regressed on level; the slope and its t.
  contrasts    STEER-F against GRPO (the registered primary contrast), uniform
               and permuted (the two eval_paired_se.CONTRASTS), each on the seeds
               the two arms share.
  go on to B   when at least two of the three contrasts have slope > 0 and
               t >= 2. Otherwise stop here and report this as exploratory only.

Reading, grouping and pairing are eval_paired_se.py's (discover, per_problem,
paired) -- imported, not rewritten, so the two cannot disagree about which rows
are the same problem.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_paired_se as eps  # noqa: E402

BENCH = "math500"
CONTRASTS = (("signed", "grpo"),) + tuple(eps.CONTRASTS)
MIN_MATCH = 0.98
GO_B_T = 2.0
GO_B_COUNT = 2


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def load_levels(path: str | None) -> list[tuple[str, int]]:
    """[(normalised problem text, level)]."""
    if path:
        out = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out.append((_norm(r["problem"]), int(r["level"])))
        return out
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("no --levels file and the `datasets` package is not installed; "
                         "pass --levels <jsonl with problem, level>") from None
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [(_norm(r["problem"]), int(r["level"])) for r in ds]


def attach_levels(rows: list[dict], levels: list[tuple[str, int]], where: str) -> list[int | None]:
    """The level of each dumped problem, found by its text inside the prompt.

    The dump's `input` is the chat-templated prompt, so the original problem is a
    substring of it. The longest matching problem wins, in case one problem's
    text happens to contain a shorter one's.
    """
    out, hit = [], 0
    ordered = sorted(levels, key=lambda pl: -len(pl[0]))
    for r in rows:
        text = _norm(r["input"])
        lv = next((lvl for prob, lvl in ordered if prob and prob in text), None)
        hit += lv is not None
        out.append(lv)
    rate = hit / len(rows) if rows else 0.0
    if rate < MIN_MATCH:
        raise SystemExit(f"{where}: only {hit}/{len(rows)} problems ({rate:.1%}) matched a "
                         f"MATH level; refusing below {MIN_MATCH:.0%} -- a per-level "
                         f"table over a guessed subset is not the table it claims to be")
    return out


def slope(xs: list[float], ys: list[float]) -> dict:
    """OLS of ys on xs: slope, its standard error and t."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if n < 3 or sxx == 0:
        return {"slope": float("nan"), "se": float("nan"), "t": float("nan"), "n": n}
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    s2 = sum((y - a - b * x) ** 2 for x, y in zip(xs, ys)) / (n - 2)
    se = math.sqrt(s2 / sxx)
    return {"slope": b, "se": se, "t": b / se if se else float("nan"), "n": n}


def analyse(runs: dict, levels: list[tuple[str, int]]) -> dict:
    report = {"benchmark": BENCH, "contrasts": [],
              "rule": f"go on to B when >= {GO_B_COUNT} of {len(CONTRASTS)} contrasts "
                      f"have slope > 0 and t >= {GO_B_T}"}
    for a, b in CONTRASTS:
        seeds = sorted(s for (arm, s) in runs if arm == a and BENCH in runs[(arm, s)]
                       and (b, s) in runs and BENCH in runs[(b, s)])
        entry = {"a": a, "b": b, "seeds": seeds}
        if not seeds:
            entry["note"] = "no shared seed"
            report["contrasts"].append(entry)
            continue
        xs, ys = [], []
        for s in seeds:
            where = f"{a}-s{s} vs {b}-s{s} {BENCH}"
            ar, br = runs[(a, s)][BENCH], runs[(b, s)][BENCH]
            diffs, _, _ = eps.paired(ar, br, where)
            lv = attach_levels(ar, levels, where)
            for d, l in zip(diffs, lv):
                if l is not None:
                    xs.append(float(l)); ys.append(d)
        per = {}
        for l in sorted(set(xs)):
            ds = [y for x, y in zip(xs, ys) if x == l]
            sd = eps._sd(ds)
            per[str(int(l))] = {"n": len(ds), "mean_diff": sum(ds) / len(ds),
                                "se": sd / math.sqrt(len(ds)) if len(ds) > 1 else float("nan")}
        entry.update({"per_level": per, "trend": slope(xs, ys)})
        report["contrasts"].append(entry)
    report["go_to_B"] = decide(report)
    return report


def decide(report: dict) -> bool:
    """The rule fixed before the run: enough contrasts with a positive, clear slope."""
    hits = sum(1 for c in report["contrasts"]
               if "trend" in c and c["trend"]["slope"] > 0 and c["trend"]["t"] >= GO_B_T)
    return hits >= GO_B_COUNT


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val-data", default="validation_data/eval")
    p.add_argument("--passes", default="avg1")
    p.add_argument("--levels", default=None,
                   help="JSONL of {problem, level}; default: HuggingFaceH4/MATH-500 via datasets")
    p.add_argument("--out", default="results/difficulty.json")
    args = p.parse_args(argv)

    runs = eps.discover(Path(args.val_data), tuple(t for t in args.passes.split(",") if t))
    if not runs:
        print(f"[difficulty] no per-problem dumps under {args.val_data}")
        return 1
    report = analyse(runs, load_levels(args.levels))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"[difficulty] EXPLORATORY -- {BENCH}, paired within seed, MATH level 1-5")
    for c in report["contrasts"]:
        if "trend" not in c:
            print(f"  {c['a']} - {c['b']}: {c.get('note')}")
            continue
        tr = c["trend"]
        levels = "  ".join(f"L{k}:{v['mean_diff']:+.3f}(n={v['n']})"
                           for k, v in c["per_level"].items())
        print(f"  {c['a']} - {c['b']}  seeds={c['seeds']}  slope={tr['slope']:+.4f}/level  "
              f"t={tr['t']:+.2f}\n      {levels}")
    print(f"[difficulty] rule: {report['rule']}")
    print(f"[difficulty] go on to B (confirmatory): {'YES' if report['go_to_B'] else 'NO'}")
    print(f"[difficulty] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
