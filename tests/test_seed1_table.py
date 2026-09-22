"""The seed-1 table has to come out of the logs, not out of a session.

Three things have gone wrong every time these numbers were recomputed by hand:
the majority-vote key was guessed as maj@32/mean@32 (it is acc/maj@32/mean),
a log with two launches had its dead first launch averaged into the plateau,
and the one arm whose log is not in git quietly acquired the same standing as
the five that are. These tests pin all three.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_seeds import parse_log, plateau          # noqa: E402
from seed1_table import step_zero_acc  # noqa: E402
from analyze_seeds import extract_from_git  # noqa: E402

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


# ------------------------------------------- which run an arm/seed resolves to
def test_a_recovery_log_does_not_silently_replace_the_original(tmp_path):
    """seed 1 has two logs for three arms; the choice moved the headline.

    analyze_seeds.find_log used to return the LAST candidate, so
    train-<run>_0905.log won over train-<run>.log. The signed arm's plateau
    accuracy then read .1392 instead of .1495 and STEER-F - GRPO shrank from
    +.0145 to +.0056, with nothing in any output saying a different run had
    been substituted. The rule is now fixed and outcome-independent: among logs
    that finished, the bare name wins.
    """
    out = tmp_path / "out"
    logs = tmp_path / "logs"
    logs.mkdir()

    def write(name, acc):
        body = []
        for step in range(1, 111):
            line = f"step:{step} - global_seqlen: 1"
            if step % 10 == 0:
                line += (f" - val-core/aime_2024_dapo_boxed/acc/mean@32:{acc}"
                         f" - val-core/aime_2024_dapo_boxed/acc/maj@32/mean:{acc}")
            body.append(line)
        (logs / name).write_text("\n".join(body) + "\n")

    run = f"steer-f-{TAG}-s1-tree-rollout"
    write(f"train-{run}.log", "0.500")        # the original
    write(f"train-{run}_0905.log", "0.100")   # the re-launch
    write(f"train-grpo-{TAG}-s1.log", "0.400")

    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_seeds.py"),
                        "--logs", str(logs), "--out", str(out)],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    rows = (out / "per_seed.tsv").read_text().splitlines()
    signed = [l for l in rows if l.startswith("signed\t")][0].split("\t")
    assert signed[2].startswith("0.5000"), f"took the re-launch: {signed}"
    assert f"train-{run}.log" in signed[-1]


def test_a_relaunch_is_used_when_the_original_never_finished(tmp_path):
    """Rule 2: an original that died is not a run, and the re-launch is."""
    out = tmp_path / "out"
    logs = tmp_path / "logs"
    logs.mkdir()
    run = f"steer-f-{TAG}-s1-tree-rollout"
    (logs / f"train-{run}.log").write_text("step:3 - global_seqlen: 1\n")
    body = []
    for step in range(1, 111):
        line = f"step:{step} - global_seqlen: 1"
        if step % 10 == 0:
            line += (" - val-core/aime_2024_dapo_boxed/acc/mean@32:0.200"
                     " - val-core/aime_2024_dapo_boxed/acc/maj@32/mean:0.300")
        body.append(line)
    (logs / f"train-{run}_0905.log").write_text("\n".join(body) + "\n")

    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_seeds.py"),
                        "--logs", str(logs), "--out", str(out)],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    signed = [l for l in (out / "per_seed.tsv").read_text().splitlines()
              if l.startswith("signed\t")][0]
    assert "_0905.log" in signed, signed


def test_arm_means_use_only_the_seeds_every_arm_has():
    """Table 1 and Table 2 must be drawn from the same runs.

    GRPO finished seed 2 while the rest are on seed 1; averaging arms over
    different seed sets makes a reader who subtracts two rows of the means
    table get a different number from the contrast table.

    The manuscript now reports the unbalanced means -- the balanced set had
    shrunk to one seed per arm, which is a table with no error bar on it -- and
    says in the caption that its rows are not meant to be subtracted. Both
    modes stay reachable, which is what this pins.
    """
    body = (ROOT / "scripts" / "analyze_seeds.py").read_text()
    assert "set.intersection" in body
    assert "--unbalanced" in body
    assert '"--balanced"' in body, "the balanced mode lost its flag"


def test_the_within_run_appendix_gets_its_own_seed_and_stays_on_it(tmp_path):
    """Appendix D reports one run's eight points, and must name which run.

    The campaign is still filling in, so the seed it uses has to be chosen by a
    rule rather than by whatever happens to be present -- otherwise the
    appendix silently changes run between drafts. The rule is the lowest seed
    both arms have.
    """
    out = tmp_path / "out"
    logs = tmp_path / "logs"
    logs.mkdir()

    def write(run, per_step):
        body = []
        for step in range(1, 111):
            line = f"step:{step} - global_seqlen: 1"
            if step % 10 == 0:
                a = per_step(step)
                line += (f" - val-core/aime_2024_dapo_boxed/acc/mean@32:{a:.3f}"
                         f" - val-core/aime_2024_dapo_boxed/acc/maj@32/mean:{a:.3f}")
            body.append(line)
        (logs / f"train-{run}.log").write_text("\n".join(body) + "\n")

    for seed, base in ((1, 0.10), (2, 0.30)):
        write(f"grpo-{TAG}-s{seed}", lambda s, b=base: b)
        write(f"steer-f-{TAG}-s{seed}-tree-rollout", lambda s, b=base: b + 0.02)

    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_seeds.py"),
                        "--logs", str(logs), "--out", str(out)],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    macros = dict(re.findall(r"\\providecommand\{\\([A-Za-z]+)\}\{([^}]*)\}",
                             (out / "numbers.tex").read_text()))
    assert macros["Wseed"] == "1", "the lowest shared seed, not the newest"
    assert macros["Wsignedgrpoacc"] == "+.0200"
    assert macros["WNsignedgrpoacc"] == "8", "eight converged-window points"
    # constant difference -> zero spread -> an infinite t, not a crash
    assert "WTsignedgrpoacc" in macros


def test_appendix_d_has_no_pending_slot_left():
    """The eight slots the logs can already answer."""
    numbers = (ROOT / "results" / "numbers.tex").read_text()
    macros = dict(re.findall(r"\\providecommand\{\\([A-Za-z]+)\}\{([^}]*)\}", numbers))
    for arm in ("grpo", "steer", "uniform", "permuted"):
        for pre in ("W", "WT"):
            k = f"{pre}signed{arm}acc"
            assert k in macros, f"{k} is not emitted"
            assert macros[k] != "\\PENDING", f"{k} is still pending"
