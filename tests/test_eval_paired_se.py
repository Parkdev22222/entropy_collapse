# Copyright 2026 STEER-F authors
# Licensed under the Apache License, Version 2.0
"""The paired across-problem error bar, and the two ways it can lie.

The statistic itself is arithmetic and would be dull to test alone. What is
worth testing is the refusals: a dump with no benchmark column (the evaluation
ran before `instrument_phase2.sh --apply`) and a dump whose rows do not line up
between arms (pairing by position does not hold). Both produce a plausible
number if nobody checks, and a plausible number in an error bar is worse than
no number -- which is the sentence the manuscript currently prints instead.
"""
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_paired_se as mod  # noqa: E402


# ------------------------------------------------------------------ fixtures
def write_run(root, arm, seed, spec, tag="avg1", key="acc"):
    """spec: {data_source: [(prompt, correct), ...]} -> one avg1 dump."""
    d = root / f"{arm}-s{seed}" / tag
    d.mkdir(parents=True, exist_ok=True)
    lines = []
    for ds, rows in spec.items():
        for prompt, correct in rows:
            rec = {"input": prompt, "output": "...", "step": 0,
                   "data_source": ds}
            if key == "acc":
                rec["acc"] = bool(correct)
                rec["score"] = 1.0 if correct else -1.0
            else:
                rec["score"] = 1.0 if correct else -1.0
            lines.append(json.dumps(rec))
    (d / "0.jsonl").write_text("\n".join(lines) + "\n")
    return d


def spec_from(bits, prefix="p"):
    """{data_source: [0/1, ...]} -> the fixture shape, with stable prompts."""
    return {ds: [(f"{prefix}{ds}{i}", v) for i, v in enumerate(vals)]
            for ds, vals in bits.items()}


@pytest.fixture()
def two_arms(tmp_path):
    """signed beats uniform on two problems out of twelve, agrees elsewhere."""
    a = {"math500":       [1, 1, 1, 0, 0, 0],
         "minerva_math":  [1, 1, 0],
         "olympiadbench": [1, 0, 0],
         "gsm8k_test":    [1, 1, 1, 1]}
    b = {"math500":       [1, 0, 1, 0, 0, 0],
         "minerva_math":  [1, 1, 0],
         "olympiadbench": [1, 0, 0],
         "gsm8k_test":    [1, 1, 1, 0]}
    write_run(tmp_path, "signed", 3, spec_from(a))
    write_run(tmp_path, "uniform", 3, spec_from(b))
    return tmp_path


# ------------------------------------------------------------------ the good case
def test_the_three_avg1_sets_carry_the_pooled_figure(two_arms):
    runs = mod.discover(two_arms)
    assert set(runs) == {("signed", 3), ("uniform", 3)}
    rep = mod.analyse(runs)
    su = rep["contrasts"][0]
    assert (su["a"], su["b"]) == ("signed", "uniform")
    # 6 + 3 + 3 = 12 problems pooled. GSM8K's four are reported, not pooled.
    assert su["pooled"]["n"] == 12
    assert su["pooled"]["n_problems"] == 12
    assert su["pooled"]["n_seeds"] == 1
    assert set(su["per_benchmark"]) == {"math500", "minerva_math",
                                        "olympiadbench", "gsm8k_test"}
    assert su["per_benchmark"]["gsm8k_test"]["n"] == 4


def test_the_pooled_difference_is_the_per_problem_mean(two_arms):
    su = mod.analyse(mod.discover(two_arms))["contrasts"][0]
    # one problem flipped, of twelve
    assert su["pooled"]["mean_diff"] == pytest.approx(1 / 12)
    assert su["pooled"]["discordance"] == pytest.approx(1 / 12)
    # GSM8K's flip is in its own row and nowhere near the pooled one
    assert su["per_benchmark"]["gsm8k_test"]["mean_diff"] == pytest.approx(0.25)


def test_a_contrast_with_no_shared_seed_reports_nothing(tmp_path):
    write_run(tmp_path, "signed", 3, spec_from({"math500": [1, 0]}))
    write_run(tmp_path, "uniform", 3, spec_from({"math500": [1, 1]}))
    rep = mod.analyse(mod.discover(tmp_path))
    perm = rep["contrasts"][1]
    assert (perm["a"], perm["b"]) == ("signed", "permuted")
    assert perm["seeds"] == [] and perm["pooled"] is None
    # ... and its macros stay red rather than borrowing the other contrast's
    names = mod.macros(rep)
    assert "Pairsigneduniform" in names
    assert not [n for n in names if n.startswith("Pairsignedpermuted")]


# ------------------------------------------------------- pairing is not a gift
def test_pairing_shrinks_the_error_only_when_the_arms_agree(tmp_path):
    """The claim the script refuses to make in advance.

    Agreeing arms: the paired error collapses toward zero while the unpaired
    one does not move. Disagreeing arms: the paired error is the LARGER of the
    two. Both are computed the same way; which one happens is data, which is
    why the manuscript quotes the measurement instead of promising a shrink.
    """
    n = 40
    agree_a = [i % 2 for i in range(n)]
    agree_b = list(agree_a)
    agree_b[0] = 1 - agree_b[0]                      # differ on one problem
    write_run(tmp_path / "agree", "signed", 1, spec_from({"math500": agree_a}))
    write_run(tmp_path / "agree", "uniform", 1, spec_from({"math500": agree_b}))
    s = mod.analyse(mod.discover(tmp_path / "agree"))["contrasts"][0]["pooled"]
    assert s["se_paired"] < s["se_unpaired"]
    assert s["se_reduction"] > 0.7

    opp_a = [i % 2 for i in range(n)]
    opp_b = [1 - v for v in opp_a]                   # differ on every problem
    write_run(tmp_path / "opp", "signed", 1, spec_from({"math500": opp_a}))
    write_run(tmp_path / "opp", "uniform", 1, spec_from({"math500": opp_b}))
    o = mod.analyse(mod.discover(tmp_path / "opp"))["contrasts"][0]["pooled"]
    assert o["discordance"] == 1.0
    assert o["se_paired"] > o["se_unpaired"]
    assert o["se_reduction"] < 0


# ------------------------------------------------------------------ refusals
def test_a_dump_without_the_benchmark_column_is_refused(tmp_path):
    """The failure the third instrument_phase2.sh change exists to prevent.

    One pass is one flat file with four benchmarks concatenated. Attributing
    its rows by position and parquet lengths would produce a number, and the
    number would be a mixture of MATH500 and Minerva.
    """
    d = tmp_path / "signed-s3" / "avg1"
    d.mkdir(parents=True)
    (d / "0.jsonl").write_text(json.dumps(
        {"input": "q", "output": "a", "score": 1.0, "acc": True}) + "\n")
    with pytest.raises(SystemExit) as exc:
        mod.discover(tmp_path)
    assert "instrument_phase2.sh" in str(exc.value)


def test_rows_that_are_different_problems_are_refused(tmp_path):
    write_run(tmp_path, "signed", 3,
              {"math500": [("apples", 1), ("pears", 0)]})
    write_run(tmp_path, "uniform", 3,
              {"math500": [("apples", 1), ("plums", 1)]})
    with pytest.raises(SystemExit) as exc:
        mod.analyse(mod.discover(tmp_path))
    assert "different problem" in str(exc.value)


def test_a_different_problem_count_is_refused(tmp_path):
    write_run(tmp_path, "signed", 3, spec_from({"math500": [1, 0, 1]}))
    write_run(tmp_path, "uniform", 3, spec_from({"math500": [1, 0]}))
    with pytest.raises(SystemExit) as exc:
        mod.analyse(mod.discover(tmp_path))
    assert "cannot be paired" in str(exc.value)


# ------------------------------------------------------------------ scoring
def test_acc_is_preferred_and_score_is_the_fallback():
    assert mod.score_of({"acc": True, "score": -1.0}) == 1.0
    assert mod.score_of({"acc": False, "score": 1.0}) == 0.0
    assert mod.score_of({"score": 1.0}) == 1.0
    assert mod.score_of({"score": -1.0}) == 0.0
    with pytest.raises(SystemExit):
        mod.score_of({"input": "q"})


def test_a_dump_written_with_scores_only_still_pairs(tmp_path):
    write_run(tmp_path, "signed", 3, spec_from({"math500": [1, 1]}), key="score")
    write_run(tmp_path, "uniform", 3, spec_from({"math500": [1, 0]}), key="score")
    su = mod.analyse(mod.discover(tmp_path))["contrasts"][0]
    assert su["pooled"]["mean_diff"] == pytest.approx(0.5)


# ------------------------------------------------------------------ emission
def test_every_macro_in_the_table_is_written_when_the_data_is_there(tmp_path):
    for arm, bits in (("signed",   [1, 1, 1, 0, 0, 0, 1, 0]),
                      ("uniform",  [1, 0, 1, 0, 0, 0, 1, 0]),
                      ("permuted", [1, 1, 0, 0, 0, 0, 1, 0])):
        for seed in (3, 4):
            write_run(tmp_path, arm, seed, spec_from({
                "math500": bits, "minerva_math": bits[:4],
                "olympiadbench": bits[:4]}))
    rep = mod.analyse(mod.discover(tmp_path))
    names = mod.macros(rep)
    assert set(names) == set(mod.MACROS), sorted(set(mod.MACROS) - set(names))
    assert names["Pairnseeds"] == "2"
    assert names["Pairn"] == "16"      # 8 + 4 + 4 problems
    assert names["Pairnobs"] == "32"   # two seeds of them
    assert names["Pairsigneduniform"].startswith("+")
    assert float(names["Pairsigneduniformse"]) > 0


def test_a_nan_never_reaches_the_page():
    """One problem cannot carry a standard error; the slot stays red."""
    s = mod.summarise([1.0], [1.0], [0.0])
    assert math.isnan(s["se_paired"])
    assert mod.fmt(s["se_paired"], "err") is None
