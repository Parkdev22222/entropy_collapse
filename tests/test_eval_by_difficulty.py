"""The difficulty analysis: MATH levels, a paired slope, and a rule fixed first.

Exploratory by construction -- the hypothesis came from the table it is tested
on -- so what these pin is that the analysis cannot quietly become something
else: levels come from MATH, not from the arms' own scores; a table over a
partly-matched subset is refused; and the go/no-go for the confirmatory step is
the rule written down before any result, not a reading chosen afterwards.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_by_difficulty as D  # noqa: E402


def dump(root, arm, seed, correct):
    """correct: list of 0/1, one per problem p0.. (level = i % 5 + 1)."""
    d = root / f"{arm}-s{seed}" / "avg1"
    d.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps({"input": f"<|user|> Solve: problem number {i} here <|assistant|>",
                        "output": "...", "data_source": "math500", "acc": bool(c)})
            for i, c in enumerate(correct)]
    (d / "0.jsonl").write_text("\n".join(rows) + "\n")


def levels_file(tmp_path, n):
    f = tmp_path / "levels.jsonl"
    f.write_text("\n".join(json.dumps({"problem": f"Solve: problem number {i} here",
                                       "level": i % 5 + 1}) for i in range(n)) + "\n")
    return str(f)


def test_a_gap_that_grows_with_level_gives_a_positive_slope(tmp_path):
    n = 100
    # signed wins exactly on the level-4 and level-5 problems; uniform on none
    signed = [1 if (i % 5 + 1) >= 4 else 0 for i in range(n)]
    uniform = [0] * n
    dump(tmp_path / "v", "signed", 4, signed)
    dump(tmp_path / "v", "uniform", 4, uniform)
    rep = D.analyse(D.eps.discover(tmp_path / "v"), D.load_levels(levels_file(tmp_path, n)))
    c = next(c for c in rep["contrasts"] if c["b"] == "uniform")
    assert c["per_level"]["1"]["mean_diff"] == 0 and c["per_level"]["5"]["mean_diff"] == 1
    assert c["trend"]["slope"] > 0 and c["trend"]["t"] > 2
    # grpo and permuted have no dumps: reported as such, not guessed
    assert next(c for c in rep["contrasts"] if c["b"] == "grpo")["note"] == "no shared seed"


def test_a_partly_matched_problem_set_is_refused(tmp_path):
    dump(tmp_path / "v", "signed", 4, [1] * 50)
    dump(tmp_path / "v", "uniform", 4, [0] * 50)
    with pytest.raises(SystemExit, match="matched a MATH level"):
        D.analyse(D.eps.discover(tmp_path / "v"), D.load_levels(levels_file(tmp_path, 40)))


def test_the_go_on_rule_is_the_one_fixed_in_advance():
    def c(slope, t):
        return {"trend": {"slope": slope, "t": t}}
    assert D.decide({"contrasts": [c(.01, 2.5), c(.02, 2.0), c(-.01, -3)]}) is True
    assert D.decide({"contrasts": [c(.01, 2.5), c(.02, 1.9), c(.03, 1.0)]}) is False
    assert D.decide({"contrasts": [c(-.01, 3.0), c(-.02, 2.5), c(.01, 2.1)]}) is False


def test_it_pairs_with_eval_paired_se_rather_than_its_own_parser():
    src = (ROOT / "scripts" / "eval_by_difficulty.py").read_text()
    assert "eps.discover(" in src and "eps.paired(" in src
    assert "json.loads(line)" in src.split("def load_levels")[1].split("def ")[0]
    assert "def read_rows" not in src
