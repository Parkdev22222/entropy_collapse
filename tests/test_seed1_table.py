"""The seed-1 table has to come out of the logs, not out of a session.

Three things have gone wrong every time these numbers were recomputed by hand:
the majority-vote key was guessed as maj@32/mean@32 (it is acc/maj@32/mean),
a log with two launches had its dead first launch averaged into the plateau,
and the one arm whose log is not in git quietly acquired the same standing as
the five that are. These tests pin all three.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_seeds import parse_log, plateau          # noqa: E402
from seed1_table import extract_from_git, step_zero_acc  # noqa: E402

TAG = "Qwen2.5-Math-1.5B"
# The values this repository's committed logs produce over steps 40-110. They
# are the table; if a parser change moves one of them, the table moved too.
EXPECTED = {
    "train-steer-%s-s1.log" % TAG: (0.1330, 0.1984),
    "train-steer-f-%s-s1-tree-rollout.log" % TAG: (0.1495, 0.2349),
    "train-steer-f-%s-s1-tree-rollout-uniform.log" % TAG: (0.1313, 0.1953),
    "train-steer-f-%s-s1-tree-rollout-permuted.log" % TAG: (0.1370, 0.2094),
    "train-math-%s-s1.log" % TAG: (0.1404, 0.2161),
}


def has_ref(ref="origin/paper"):
    return subprocess.run(["git", "rev-parse", "--verify", "-q", ref],
                          cwd=str(ROOT), capture_output=True).returncode == 0


@pytest.fixture(scope="module")
def logs(tmp_path_factory):
    if not has_ref():
        pytest.skip("origin/paper is not fetched; the training logs live there")
    d = tmp_path_factory.mktemp("logs")
    cwd = Path.cwd()
    import os
    os.chdir(ROOT)
    try:
        n = extract_from_git("origin/paper", "logs/experiments", d)
    finally:
        os.chdir(cwd)
    if not n:
        pytest.skip("no training logs at origin/paper:logs/experiments")
    return d


@pytest.mark.parametrize("name,expect", sorted(EXPECTED.items()))
def test_plateau_matches_the_published_table(logs, name, expect):
    f = logs / name
    if not f.is_file():
        pytest.skip(f"{name} is not on origin/paper")
    agg = plateau(parse_log(f), 40, 110)
    assert agg["n_val_points"] == 8, "the plateau window must hold 8 validations"
    assert round(agg["acc"], 4) == expect[0]
    assert round(agg["maj"], 4) == expect[1]


def test_a_restarted_log_is_not_double_counted(logs):
    """The signed log holds two launches: a dead 1-9 and the real 0-118.

    A pass that appended rows instead of keying on the step would report ~20
    validations below step 110 and a plateau contaminated by the dead run.
    parse_log sees 11 of the 12 -- step 0 has no training step and so no
    global_seqlen, which is why step_zero_acc() exists.
    """
    f = logs / ("train-steer-f-%s-s1-tree-rollout.log" % TAG)
    if not f.is_file():
        pytest.skip("signed log is not on origin/paper")
    raw = f.read_text(errors="replace")
    assert raw.count("val-core/aime_2024_dapo_boxed/acc/mean@32") > 12, \
        "this fixture is only interesting because the file has more than one launch"
    steps = parse_log(f)
    vals = [s for s, r in steps.items() if s <= 110 and "acc" in r]
    assert len(vals) == 11, f"expected 11 validations in 1..110, got {len(vals)}"
    assert step_zero_acc(f) is not None, "and step 0 is the twelfth"


def test_step_zero_is_read_even_though_parse_log_skips_it(logs):
    """Step 0 has no training step, so it never carries global_seqlen."""
    f = logs / ("train-steer-f-%s-s1-tree-rollout.log" % TAG)
    if not f.is_file():
        pytest.skip("signed log is not on origin/paper")
    assert 0 not in parse_log(f)
    assert step_zero_acc(f) == pytest.approx(0.039)


def logs_without_grpo(logs, tmp_path):
    """A copy of the log set with the GRPO arm removed.

    These tests used to rely on origin/paper simply not having that log. It
    landed on 2026-09-15 (d6e603f), and the tests broke -- a test that asserts
    about what a remote does NOT contain is a test with an expiry date. Build
    the condition instead.
    """
    d = tmp_path / "no-grpo"
    d.mkdir(exist_ok=True)
    for f in logs.glob("train-*.log"):
        if f.name.startswith("train-grpo-"):
            continue
        (d / f.name).write_bytes(f.read_bytes())
    return d

def run_table(tmp_path, log_dir, transcript=None):
    out = tmp_path / "out.md"
    cmd = [sys.executable, str(ROOT / "scripts" / "seed1_table.py"),
           "--logs", str(log_dir), "--out", str(out),
           "--transcript", str(transcript) if transcript else str(tmp_path / "none.json")]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    return p, out


def test_a_missing_arm_is_marked_not_invented(tmp_path, logs):
    """No GRPO log and no transcript: the row says MISSING and the exit is 0."""
    p, out = run_table(tmp_path, logs_without_grpo(logs, tmp_path))
    assert p.returncode == 0, p.stderr
    body = out.read_text()
    assert "| **GRPO** | - | plain | none (`vanilla`) | MISSING" in body
    assert "no log at" in body


def test_the_transcript_row_is_labelled_as_not_in_git(tmp_path, logs):
    p, out = run_table(tmp_path, logs_without_grpo(logs, tmp_path),
                       ROOT / "docs" / "seed1_grpo_transcript.json")
    assert p.returncode == 0, p.stderr
    body = out.read_text()
    row = [l for l in body.splitlines() if l.startswith("| **GRPO**")][0]
    assert "NOT IN GIT" in row, "a number nobody can recompute must say so in the table"
    assert "| .1350 |" in row and "| .1962 |" in row


def test_a_real_log_wins_over_the_transcript(tmp_path, logs):
    """Drop a GRPO log in place; the transcript must stop being consulted."""
    src = logs / ("train-steer-%s-s1.log" % TAG)
    if not src.is_file():
        pytest.skip("no log to stand in for grpo")
    d = tmp_path / "logs"
    d.mkdir()
    for f in logs.glob("train-*.log"):
        (d / f.name).write_bytes(f.read_bytes())
    (d / ("train-grpo-%s-s1.log" % TAG)).write_bytes(src.read_bytes())
    p, out = run_table(tmp_path, d, ROOT / "docs" / "seed1_grpo_transcript.json")
    assert p.returncode == 0, p.stderr
    row = [l for l in out.read_text().splitlines() if l.startswith("| **GRPO**")][0]
    assert "NOT IN GIT" not in row
    assert "train-grpo-%s-s1.log" % TAG in row


def test_the_transcript_file_is_well_formed():
    d = json.loads((ROOT / "docs" / "seed1_grpo_transcript.json").read_text())
    assert d["arm"] == "grpo" and d["seed"] == 1
    assert d["config"]["loss_mode"] == "vanilla", \
        "a GRPO baseline that is not loss_mode=vanilla is not a baseline"
    steps = sorted(int(s) for s in d["val"])
    assert steps == list(range(0, 111, 10))
    win = [d["val"][str(s)] for s in steps if 40 <= s <= 110]
    assert round(sum(v["acc"] for v in win) / len(win), 4) == 0.1350
    assert round(sum(v["maj"] for v in win) / len(win), 4) == 0.1962


def test_no_empty_box_crash(tmp_path):
    """The first thing anyone runs is this, on a box with no logs."""
    empty = tmp_path / "empty"
    empty.mkdir()
    p, _ = run_table(tmp_path, empty)
    assert p.returncode == 0, p.stderr
    assert "nothing to write" in p.stdout


def test_the_grpo_log_is_now_on_the_branch_and_matches_the_transcript(logs):
    """d6e603f published it. The transcript said .1350/.1962; so does the log.

    This is the check that made the transcript worth keeping: a number nobody
    could recompute turned out to be right when the source finally arrived.
    """
    f = logs / ("train-grpo-%s-s1.log" % TAG)
    if not f.is_file():
        pytest.skip("the GRPO seed-1 log is not on origin/paper (yet)")
    agg = plateau(parse_log(f), 40, 110)
    assert agg["n_val_points"] == 8
    assert round(agg["acc"], 4) == 0.1350
    assert round(agg["maj"], 4) == 0.1962
    d = json.loads((ROOT / "docs" / "seed1_grpo_transcript.json").read_text())
    win = [d["val"][str(s)] for s in range(40, 111, 10)]
    assert round(sum(v["acc"] for v in win) / len(win), 4) == round(agg["acc"], 4)
