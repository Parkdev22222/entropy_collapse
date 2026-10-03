#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
# Licensed under the Apache License, Version 2.0
r"""Are the CORRECT traces of one arm more varied than another's?

    # lexical diversity, computed here, no GPU, no network
    python3 scripts/trace_diversity.py auto \
        --arm signed=validation_data/eval/signed-s1/avg32 \
        --arm grpo=validation_data/eval/grpo-s1/avg32 \
        --contrast signed:grpo --k 4 --out results/trace_diversity_auto.json

    # strategy-level diversity: blind tasks for a judge, then score its labels
    python3 scripts/trace_diversity.py export-judge --arm ... --k 4 --m 8 \
        --out-dir judge/
    python3 scripts/trace_diversity.py import-judge --tasks judge/tasks.jsonl \
        --key judge/key.jsonl --labels judge/labels.jsonl \
        --contrast signed:grpo --k 4 --out results/trace_diversity_judge.json

WHAT IS COMPARED
    Only correct samples, and only problems on which every arm produced at
    least k of them. On each such problem every arm contributes exactly k
    correct traces, drawn at random (and re-drawn --draws times), so an arm
    that simply gets a problem right more often is not scored as more diverse
    for having more traces to choose from. The plan this implements, with its
    decision rule fixed before any result, is
    docs/preregistration_trace_diversity.md.

THE CONFOUNDS HANDLED HERE
    count    k traces per arm per problem, never "all correct ones".
    length   lexical metrics are computed twice: on the full traces, and with
             every trace in a draw cut to the shortest one in that draw
             (across all arms), because a longer text has more distinct
             n-grams for no reason that has anything to do with reasoning.
             The truncated figure is the one the pre-registration reads.
    pairing  arms are matched on (data_source, prompt text); a problem that
             is not in every arm is dropped and counted, never guessed at.

METRICS (per arm, per problem, averaged over draws)
    pair_dist   mean over the k(k-1)/2 pairs of 1 - Jaccard(4-gram sets)
    distinct    unique 4-grams / all 4-grams over the k traces
    vendi       exp(entropy of the eigenvalues of K/k), K the 4-gram Jaccard
                similarity matrix: the effective number of distinct traces,
                1 when all k are identical and k when they share nothing
    strategies  (import-judge only) the expected number of distinct solution
                strategies among k correct traces, from blinded judge labels,
                computed exactly by rarefaction rather than by sampling

    Lexical metrics cannot tell a new idea from a paraphrase. That is why the
    pre-registered primary metric is the judge's, and these are secondary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_paired_se import _step_of, score_of  # noqa: E402

NGRAM = 4
TOKEN = re.compile(r"\w+|[^\w\s]")
LEXICAL = ("pair_dist", "distinct", "vendi")


# ------------------------------------------------------------------ reading
def dump_file(path):
    """An arm path -> the one JSONL to read: the file itself, or the
    highest-step file in the directory (val_only writes 0.jsonl)."""
    p = Path(path)
    if p.is_file():
        return p
    files = sorted(p.glob("*.jsonl"), key=_step_of)
    if not files:
        raise SystemExit(f"{p}: no *.jsonl dump here. Point --arm at the pass "
                         f"directory (e.g. .../<arm>-s<seed>/avg32).")
    return files[-1]


def load_arm(path):
    """-> {(data_source, prompt): [(output, correct), ...]} in file order."""
    f = dump_file(path)
    out = {}
    with f.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{f}:{lineno}: {exc}") from None
            if "data_source" not in rec:
                raise SystemExit(f"{f}: no data_source column; rows cannot be "
                                 f"attributed to a benchmark. Re-run the evaluation.")
            if "output" not in rec:
                raise SystemExit(f"{f}:{lineno}: no 'output' column, so there "
                                 f"is no trace to measure.")
            key = (str(rec["data_source"]), rec.get("input", ""))
            out.setdefault(key, []).append((rec["output"], score_of(rec) > 0.5))
    return out


def parse_arms(specs):
    arms = {}
    for s in specs:
        name, sep, path = s.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"--arm wants NAME=PATH, got {s!r}")
        if name in arms:
            raise SystemExit(f"--arm {name} given twice")
        arms[name] = load_arm(path)
    if len(arms) < 2:
        raise SystemExit("at least two --arm are needed to compare anything")
    return arms


def parse_contrasts(specs, arms):
    out = []
    for s in specs:
        a, sep, b = s.partition(":")
        if not sep or a not in arms or b not in arms or a == b:
            raise SystemExit(f"--contrast {s!r}: want A:B with A, B two --arm names")
        out.append((a, b))
    if not out:
        raise SystemExit("name at least one --contrast A:B (the difference is A - B)")
    return out


def eligible(arms, k, sets=None):
    """Problems present in every arm with >= k correct traces in every arm.

    Returns (keys, report) where report counts what was dropped and why, so a
    small n is visible as a small n rather than discovered later."""
    names = list(arms)
    common = set(arms[names[0]])
    for n in names[1:]:
        common &= set(arms[n])
    union = set().union(*(set(a) for a in arms.values()))
    if sets:
        common = {key for key in common if key[0] in sets}
    keys, short = [], 0
    for key in sorted(common):
        if all(sum(c for _, c in arms[n][key]) >= k for n in names):
            keys.append(key)
        else:
            short += 1
    by_set = Counter(key[0] for key in keys)
    return keys, {"problems_in_any_arm": len(union),
                  "problems_in_every_arm": len(common),
                  "dropped_fewer_than_k_correct": short,
                  "eligible": len(keys),
                  "eligible_by_set": dict(sorted(by_set.items()))}


# ------------------------------------------------------------------ metrics
def tokens(text):
    return TOKEN.findall(text)


def ngrams(toks, n=NGRAM):
    if len(toks) < n:
        return [tuple(toks)] if toks else []
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def vendi(sim):
    """Effective number of distinct items from a PSD similarity matrix with a
    unit diagonal (Friedman & Dieng, 2023)."""
    k = sim.shape[0]
    lam = np.clip(np.linalg.eigvalsh(sim / k), 0.0, None)
    lam = lam[lam > 1e-12]
    return float(math.exp(-np.sum(lam * np.log(lam))))


def lexical(token_lists):
    """k token lists -> {pair_dist, distinct, vendi}."""
    grams = [ngrams(t) for t in token_lists]
    sets = [set(g) for g in grams]
    k = len(sets)
    sim = np.eye(k)
    dists = []
    for i in range(k):
        for j in range(i + 1, k):
            s = jaccard(sets[i], sets[j])
            sim[i, j] = sim[j, i] = s
            dists.append(1.0 - s)
    total = sum(len(g) for g in grams)
    uniq = len(set().union(*sets)) if sets else 0
    return {"pair_dist": float(np.mean(dists)) if dists else 0.0,
            "distinct": uniq / total if total else 0.0,
            "vendi": vendi(sim)}


def stable_rng(seed, *parts):
    h = hashlib.sha256(("|".join([str(seed)] + [str(p) for p in parts])).encode())
    return random.Random(int.from_bytes(h.digest()[:8], "big"))


def score_problem(arms, key, k, draws, seed):
    """-> {arm: {metric[_trunc]: mean over draws, len_full, len_trunc}}."""
    names = list(arms)
    correct = {n: [tokens(o) for o, c in arms[n][key] if c] for n in names}
    acc = {n: Counter() for n in names}
    for d in range(draws):
        rng = stable_rng(seed, key[0], key[1], d)
        pick = {n: rng.sample(correct[n], k) for n in names}
        cut = min(len(t) for n in names for t in pick[n])
        for n in names:
            full = lexical(pick[n])
            trunc = lexical([t[:cut] for t in pick[n]])
            for m in LEXICAL:
                acc[n][m] += full[m]
                acc[n][m + "_trunc"] += trunc[m]
            acc[n]["len_full"] += float(np.mean([len(t) for t in pick[n]]))
            acc[n]["len_trunc"] += cut
    return {n: {m: v / draws for m, v in acc[n].items()} for n in names}


# ------------------------------------------------------------------ statistics
def bootstrap_ci(diffs, n_boot=2000, seed=0, level=0.95):
    x = np.asarray(diffs, dtype=float)
    if x.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def contrast(per_problem, keys, a, b, metric):
    """Paired A - B over problems, with a problem-level bootstrap CI."""
    diffs = [per_problem[key][a][metric] - per_problem[key][b][metric] for key in keys]
    lo, hi = bootstrap_ci(diffs)
    by_set = {}
    for key, d in zip(keys, diffs):
        by_set.setdefault(key[0], []).append(d)
    return {"n": len(diffs),
            "mean_a": float(np.mean([per_problem[k_][a][metric] for k_ in keys])) if keys else None,
            "mean_b": float(np.mean([per_problem[k_][b][metric] for k_ in keys])) if keys else None,
            "mean_diff": float(np.mean(diffs)) if diffs else None,
            "ci95": [lo, hi],
            "excludes_zero": (lo is not None and (lo > 0 or hi < 0)),
            "n_pos": sum(d > 0 for d in diffs),
            "n_neg": sum(d < 0 for d in diffs),
            "by_set": {s: {"n": len(v), "mean_diff": float(np.mean(v))}
                       for s, v in sorted(by_set.items())}}


def rarefied_strategies(labels, k):
    """Expected number of distinct labels among k drawn without replacement
    from `labels` -- exact, so no sampling noise enters the primary metric."""
    m = len(labels)
    if m < k:
        raise ValueError(f"{m} labelled traces, cannot rarefy to k={k}")
    total = math.comb(m, k)
    return sum(1.0 - math.comb(m - c, k) / total for c in Counter(labels).values())


# ------------------------------------------------------------------ commands
def cmd_auto(args):
    arms = parse_arms(args.arm)
    pairs = parse_contrasts(args.contrast, arms)
    keys, report = eligible(arms, args.k, args.sets)
    per_problem = {key: score_problem(arms, key, args.k, args.draws, args.seed)
                   for key in keys}
    metrics = [m + "_trunc" for m in LEXICAL] + list(LEXICAL) + ["len_full", "len_trunc"]
    result = {"command": "auto", "k": args.k, "draws": args.draws, "seed": args.seed,
              "ngram": NGRAM, "filter": report,
              "contrasts": {f"{a}-{b}": {m: contrast(per_problem, keys, a, b, m)
                                         for m in metrics} for a, b in pairs}}
    emit(result, args.out)
    print_table(result, metrics)


def cmd_export(args):
    arms = parse_arms(args.arm)
    if args.m < args.k:
        raise SystemExit("--m must be at least --k: the judge has to see at least "
                         "k traces of each arm to rarefy to k")
    keys, report = eligible(arms, args.k, args.sets)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks, key_rows = [], []
    for i, key in enumerate(keys, 1):
        tid = f"P{i:04d}"
        rng = stable_rng(args.seed, "export", key[0], key[1])
        pool = []
        for name, rows in arms.items():
            corr = [o for o, c in rows[key] if c]
            pool += [(name, o) for o in rng.sample(corr, min(args.m, len(corr)))]
        rng.shuffle(pool)
        traces = []
        for j, (name, text) in enumerate(pool, 1):
            trace_id = f"T{j:02d}"
            traces.append({"id": trace_id, "text": text})
            key_rows.append({"task_id": tid, "id": trace_id, "arm": name})
        tasks.append({"task_id": tid, "data_source": key[0], "problem": key[1],
                      "traces": traces})
    write_jsonl(out / "tasks.jsonl", tasks)
    write_jsonl(out / "key.jsonl", key_rows)
    print(json.dumps(report, indent=1))
    print(f"wrote {len(tasks)} tasks to {out / 'tasks.jsonl'}; the arm key is "
          f"{out / 'key.jsonl'} -- never give it to the judge.")


def cmd_import(args):
    tasks = {t["task_id"]: t for t in read_jsonl(args.tasks)}
    key = {}
    for r in read_jsonl(args.key):
        key.setdefault(r["task_id"], {})[r["id"]] = r["arm"]
    labels = {r["task_id"]: r["clusters"] for r in read_jsonl(args.labels)}
    arm_names = sorted({a for t in key.values() for a in t.values()})
    pairs = parse_contrasts(args.contrast, dict.fromkeys(arm_names))
    per_problem, keys, missing = {}, [], []
    for tid, task in sorted(tasks.items()):
        if tid not in labels:
            missing.append(tid)
            continue
        got = labels[tid]
        ids = {t["id"] for t in task["traces"]}
        if set(got) != ids:
            raise SystemExit(f"{tid}: labels cover {sorted(got)} but the task has "
                             f"{sorted(ids)}; every trace needs exactly one label")
        per_arm = {}
        for trace_id, arm in key[tid].items():
            per_arm.setdefault(arm, []).append(str(got[trace_id]))
        k_ = (task["data_source"], task["problem"])
        per_problem[k_] = {arm: {"strategies": rarefied_strategies(v, args.k)}
                           for arm, v in per_arm.items()}
        keys.append(k_)
    if missing:
        raise SystemExit(f"{len(missing)} task(s) have no labels (first: {missing[0]}). "
                         f"A judged subset chosen after the fact is not the "
                         f"pre-registered sample; label every task or none.")
    result = {"command": "import-judge", "k": args.k, "n_tasks": len(keys),
              "contrasts": {f"{a}-{b}": {"strategies": contrast(per_problem, keys, a, b,
                                                                "strategies")}
                            for a, b in pairs}}
    emit(result, args.out)
    print_table(result, ["strategies"])


# ------------------------------------------------------------------ io
def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def emit(result, out):
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {out}")


def print_table(result, metrics):
    if "filter" in result:
        print("filter:", json.dumps(result["filter"]))
    for name, block in result["contrasts"].items():
        print(f"\n{name}  (A - B, paired over problems, 95% bootstrap CI)")
        for m in metrics:
            c = block[m]
            if not c["n"]:
                print(f"  {m:16s} n=0")
                continue
            lo, hi = c["ci95"]
            flag = "*" if c["excludes_zero"] else " "
            print(f"  {m:16s} n={c['n']:4d}  A={c['mean_a']:.4f}  B={c['mean_b']:.4f}  "
                  f"diff={c['mean_diff']:+.4f}  [{lo:+.4f}, {hi:+.4f}]{flag}  "
                  f"+{c['n_pos']}/-{c['n_neg']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, contrast=True):
        sp.add_argument("--arm", action="append", default=[], metavar="NAME=PATH")
        sp.add_argument("--k", type=int, default=4)
        sp.add_argument("--sets", nargs="*", default=None,
                        help="data_source names to keep (default: all shared)")
        sp.add_argument("--seed", type=int, default=0)
        if contrast:
            sp.add_argument("--contrast", action="append", default=[], metavar="A:B")

    a = sub.add_parser("auto", help="lexical diversity of correct traces")
    common(a)
    a.add_argument("--draws", type=int, default=50)
    a.add_argument("--out")
    a.set_defaults(fn=cmd_auto)

    e = sub.add_parser("export-judge", help="write blinded tasks for a judge")
    common(e, contrast=False)
    e.add_argument("--m", type=int, default=8, help="correct traces per arm per task")
    e.add_argument("--out-dir", required=True)
    e.set_defaults(fn=cmd_export)

    i = sub.add_parser("import-judge", help="score judge labels")
    i.add_argument("--tasks", required=True)
    i.add_argument("--key", required=True)
    i.add_argument("--labels", required=True)
    i.add_argument("--contrast", action="append", default=[], metavar="A:B")
    i.add_argument("--k", type=int, default=4)
    i.add_argument("--out")
    i.set_defaults(fn=cmd_import)

    args = p.parse_args(argv)
    if getattr(args, "k", 2) < 2:
        raise SystemExit("--k must be at least 2: diversity needs a pair")
    args.fn(args)


if __name__ == "__main__":
    main()
