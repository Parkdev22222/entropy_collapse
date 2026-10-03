# Copyright 2026 STEER-F authors
# Licensed under the Apache License, Version 2.0
"""Diversity of correct traces, and the three ways the comparison can lie.

The metrics are a few lines of set arithmetic. What is worth testing is that
the comparison does not reward the wrong thing: an arm that is right more
often (count), an arm that writes longer (length), and a judged subset picked
after the fact. Each of those produces a confident-looking difference if
nobody checks.
"""
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import trace_diversity as mod  # noqa: E402

WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu "
         "xi omicron pi rho sigma tau upsilon phi chi psi omega").split()


def text(seed, n=60):
    """A deterministic trace of n distinct-ish words."""
    rng = np.random.default_rng(seed)
    return " ".join(f"{WORDS[i % len(WORDS)]}{i // len(WORDS)}"
                    for i in rng.permutation(n * 3)[:n])


def write_dump(path, spec):
    """spec: {(ds, prompt): [(output, correct), ...]} -> 0.jsonl"""
    path.mkdir(parents=True, exist_ok=True)
    with (path / "0.jsonl").open("w") as fh:
        for (ds, prompt), rows in spec.items():
            for out, ok in rows:
                fh.write(json.dumps({"input": prompt, "output": out, "data_source": ds,
                                     "acc": ok, "score": 1.0 if ok else -1.0,
                                     "step": 0}) + "\n")
    return path


# ------------------------------------------------------------------ metrics
def test_vendi_bounds():
    assert mod.vendi(np.ones((4, 4))) == pytest.approx(1.0)
    assert mod.vendi(np.eye(4)) == pytest.approx(4.0)


def test_lexical_identical_and_disjoint():
    same = mod.lexical([mod.tokens(text(1))] * 4)
    assert same["pair_dist"] == 0.0 and same["vendi"] == pytest.approx(1.0)
    disjoint = mod.lexical([[f"w{i}_{j}" for j in range(20)] for i in range(4)])
    assert disjoint["pair_dist"] == 1.0
    assert disjoint["vendi"] == pytest.approx(4.0)
    assert disjoint["distinct"] == 1.0


@pytest.mark.parametrize("labels,k", [
    (list("AAAABBBB"), 4), (list("ABCDEFGH"), 4), (list("AAAAAAAB"), 2),
    (list("AABBCD"), 3),
])
def test_rarefaction_matches_brute_force(labels, k):
    brute = np.mean([len(set(c)) for c in combinations(labels, k)])
    assert mod.rarefied_strategies(labels, k) == pytest.approx(brute)


# ------------------------------------------------------------------ the comparison
def two_arms(tmp_path, diverse, repetitive, n_problems=12):
    spec_d, spec_r = {}, {}
    for p in range(n_problems):
        key = ("amc23", f"problem {p}")
        spec_d[key] = diverse(p)
        spec_r[key] = repetitive(p)
    return (write_dump(tmp_path / "div", spec_d), write_dump(tmp_path / "rep", spec_r))


def run_auto(tmp_path, a, b, k=4, extra=()):
    out = tmp_path / "res.json"
    mod.main(["auto", "--arm", f"div={a}", "--arm", f"rep={b}",
              "--contrast", "div:rep", "--k", str(k), "--draws", "10",
              "--out", str(out), *extra])
    return json.loads(out.read_text())


def test_diverse_arm_scores_higher(tmp_path):
    a, b = two_arms(tmp_path,
                    lambda p: [(text(100 * p + j), True) for j in range(6)],
                    lambda p: [(text(100 * p), True) for _ in range(6)])
    c = run_auto(tmp_path, a, b)["contrasts"]["div-rep"]
    for m in ("pair_dist_trunc", "vendi_trunc", "distinct_trunc"):
        assert c[m]["mean_diff"] > 0 and c[m]["excludes_zero"], m
        assert c[m]["n"] == 12


def test_more_correct_answers_is_not_more_diversity(tmp_path):
    """Both arms repeat ONE trace per problem; one is right 30 times, one 4.
    k traces per arm means the frequent solver is not scored as diverse."""
    a, b = two_arms(tmp_path,
                    lambda p: [(text(p), True)] * 30 + [(text(p + 50), False)] * 2,
                    lambda p: [(text(p), True)] * 4 + [(text(p + 50), False)] * 28)
    c = run_auto(tmp_path, a, b)["contrasts"]["div-rep"]
    for m in ("pair_dist_trunc", "vendi_trunc"):
        assert c[m]["mean_diff"] == pytest.approx(0.0)


def test_length_alone_is_not_more_diversity(tmp_path):
    """Same traces, one arm appends a long tail of fresh tokens. The full
    metric rewards the tail; the truncated one -- the registered one -- must
    not."""
    def tail(p, j):
        return " " + " ".join(f"pad{p}_{j}_{i}" for i in range(200))
    a, b = two_arms(tmp_path,
                    lambda p: [(text(p) + tail(p, j), True) for j in range(5)],
                    lambda p: [(text(p), True) for _ in range(5)])
    c = run_auto(tmp_path, a, b)["contrasts"]["div-rep"]
    assert c["pair_dist"]["mean_diff"] > 0.5
    assert c["pair_dist_trunc"]["mean_diff"] == pytest.approx(0.0)
    assert c["len_full"]["mean_diff"] > 100


def test_wrong_traces_are_ignored(tmp_path):
    a, b = two_arms(tmp_path,
                    lambda p: [(text(p), True)] * 4 + [(text(900 + j), False) for j in range(8)],
                    lambda p: [(text(p), True)] * 4)
    c = run_auto(tmp_path, a, b)["contrasts"]["div-rep"]
    assert c["vendi_trunc"]["mean_diff"] == pytest.approx(0.0)


def test_filter_reports_what_it_dropped(tmp_path):
    spec_a = {("amc23", "p1"): [(text(1), True)] * 4,
              ("amc23", "p2"): [(text(2), True)] * 4,
              ("amc23", "only-a"): [(text(3), True)] * 4}
    spec_b = {("amc23", "p1"): [(text(1), True)] * 4,
              ("amc23", "p2"): [(text(2), True)] * 3 + [(text(9), False)]}
    a = write_dump(tmp_path / "a", spec_a)
    b = write_dump(tmp_path / "b", spec_b)
    keys, rep = mod.eligible({"a": mod.load_arm(a), "b": mod.load_arm(b)}, k=4)
    assert keys == [("amc23", "p1")]
    assert rep == {"problems_in_any_arm": 3, "problems_in_every_arm": 2,
                   "dropped_fewer_than_k_correct": 1, "eligible": 1,
                   "eligible_by_set": {"amc23": 1}}


def test_refuses_dump_without_benchmark_column(tmp_path):
    d = tmp_path / "old"
    d.mkdir()
    (d / "0.jsonl").write_text(json.dumps({"input": "x", "output": "y", "acc": True}) + "\n")
    with pytest.raises(SystemExit, match="data_source"):
        mod.load_arm(d)


def test_reads_the_last_step_file(tmp_path):
    d = tmp_path / "run"
    write_dump(d, {("amc23", "p"): [("old", True)]})
    (d / "0.jsonl").rename(d / "10.jsonl")
    write_dump(d, {("amc23", "p"): [("new", True)]})
    (d / "0.jsonl").rename(d / "200.jsonl")
    assert mod.load_arm(d)[("amc23", "p")] == [("new", True)]


# ------------------------------------------------------------------ judge round trip
def test_judge_round_trip_is_blind_and_exact(tmp_path):
    a, b = two_arms(tmp_path,
                    lambda p: [(f"A-strategy-{j} p{p}", True) for j in range(8)],
                    lambda p: [(f"B-same p{p} copy{j}", True) for j in range(8)],
                    n_problems=5)
    jd = tmp_path / "judge"
    mod.main(["export-judge", "--arm", f"div={a}", "--arm", f"rep={b}",
              "--k", "4", "--m", "6", "--out-dir", str(jd)])
    tasks = mod.read_jsonl(jd / "tasks.jsonl")
    assert len(tasks) == 5
    blob = json.dumps(tasks)
    assert "div" not in blob.replace("data_source", "") and '"rep"' not in blob
    assert all(len(t["traces"]) == 12 for t in tasks)

    # A perfect judge: every div trace its own strategy, all rep traces one.
    labels = []
    for t in tasks:
        cl = {}
        for tr in t["traces"]:
            cl[tr["id"]] = (tr["text"].split()[0] if tr["text"].startswith("A-")
                            else "B")
        labels.append({"task_id": t["task_id"], "clusters": cl})
    mod.write_jsonl(jd / "labels.jsonl", labels)
    out = tmp_path / "j.json"
    mod.main(["import-judge", "--tasks", str(jd / "tasks.jsonl"),
              "--key", str(jd / "key.jsonl"), "--labels", str(jd / "labels.jsonl"),
              "--contrast", "div:rep", "--k", "4", "--out", str(out)])
    c = json.loads(out.read_text())["contrasts"]["div-rep"]["strategies"]
    assert c["mean_a"] == pytest.approx(4.0)
    assert c["mean_b"] == pytest.approx(1.0)
    assert c["n"] == 5


def test_judge_subset_is_refused(tmp_path):
    a, b = two_arms(tmp_path,
                    lambda p: [(f"x{j} p{p}", True) for j in range(4)],
                    lambda p: [(f"y{j} p{p}", True) for j in range(4)], n_problems=3)
    jd = tmp_path / "judge"
    mod.main(["export-judge", "--arm", f"a={a}", "--arm", f"b={b}",
              "--k", "4", "--m", "4", "--out-dir", str(jd)])
    tasks = mod.read_jsonl(jd / "tasks.jsonl")
    one = tasks[0]
    mod.write_jsonl(jd / "labels.jsonl", [{"task_id": one["task_id"],
                    "clusters": {t["id"]: "s" for t in one["traces"]}}])
    with pytest.raises(SystemExit, match="no labels"):
        mod.main(["import-judge", "--tasks", str(jd / "tasks.jsonl"),
                  "--key", str(jd / "key.jsonl"), "--labels", str(jd / "labels.jsonl"),
                  "--contrast", "a:b", "--k", "4"])


def test_judge_missing_trace_label_is_refused(tmp_path):
    a, b = two_arms(tmp_path,
                    lambda p: [(f"x{j} p{p}", True) for j in range(4)],
                    lambda p: [(f"y{j} p{p}", True) for j in range(4)], n_problems=1)
    jd = tmp_path / "judge"
    mod.main(["export-judge", "--arm", f"a={a}", "--arm", f"b={b}",
              "--k", "4", "--m", "4", "--out-dir", str(jd)])
    t = mod.read_jsonl(jd / "tasks.jsonl")[0]
    cl = {tr["id"]: "s" for tr in t["traces"][:-1]}
    mod.write_jsonl(jd / "labels.jsonl", [{"task_id": t["task_id"], "clusters": cl}])
    with pytest.raises(SystemExit, match="exactly one label"):
        mod.main(["import-judge", "--tasks", str(jd / "tasks.jsonl"),
                  "--key", str(jd / "key.jsonl"), "--labels", str(jd / "labels.jsonl"),
                  "--contrast", "a:b", "--k", "4"])


def test_bootstrap_ci_brackets_the_mean():
    lo, hi = mod.bootstrap_ci([0.1, 0.2, 0.3, 0.15, 0.25])
    assert lo < 0.2 < hi
    assert mod.bootstrap_ci([]) == (None, None)


def test_k_below_two_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="at least 2"):
        mod.main(["auto", "--arm", "a=x", "--arm", "b=y", "--contrast", "a:b", "--k", "1"])
