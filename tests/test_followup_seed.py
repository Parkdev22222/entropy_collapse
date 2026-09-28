"""A follow-up arm is found at whatever seed it ran at, not at seed 1.

scripts/analyze_seeds.py has always built a run name from (arm, seed), but the
follow-up loop of Table 10 passed a literal 1. The oracle control -- the
head-free arm that replaces the MTP forecaster with the policy's own realised
entropy -- was run at seed 4 on purpose, so that it shares a seed with signed
s4 and uniform s4 in the paired six-benchmark evaluation. Under the old loop
that finished run left Roracleacc, Roraclemaj and Roracleuplift at \\PENDING,
which is exactly what an experiment that was never launched looks like.

These tests pin the replacement rule and pin that it stays independent of the
outcome: the LOWEST seed that finished, never the one that reads better.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = "Qwen2.5-Math-1.5B"


def write_log(path, acc, maj=None, last=110, sec=None, gpus=None):
    """A log with a validation point every tenth step up to `last`."""
    maj = acc if maj is None else maj
    body = []
    if gpus is not None:
        body.append("{'n_gpus_per_node': %d}" % gpus)
    for step in range(1, last + 1):
        line = f"step:{step} - global_seqlen: 1"
        if sec is not None:
            line += f" - perf/time_per_step:{sec}"
        if step % 10 == 0:
            line += (f" - val-core/aime_2024_dapo_boxed/acc/mean@32:{acc}"
                     f" - val-core/aime_2024_dapo_boxed/acc/maj@32/mean:{maj}")
        body.append(line)
    path.write_text("\n".join(body) + "\n")


def run(tmp_path, logs):
    # analyze_seeds.py exits if it can find no main-arm log at all, and none of
    # these fixtures is about that, so every one of them gets a GRPO seed 1.
    write_log(logs / f"train-grpo-{TAG}-s1.log", "0.140")
    out = tmp_path / "out"
    mac = tmp_path / "macros.tex"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_seeds.py"),
                        "--logs", str(logs), "--out", str(out),
                        "--tex-macros", str(mac)],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    txt = mac.read_text()
    out = {m.group(1): m.group(2) for m in
           re.finditer(r"\\providecommand\{\\([A-Za-z]+)\}\{([^}]*)\}", txt)}
    out["__stdout__"] = r.stdout
    return out


def test_a_followup_at_seed_four_is_found_and_says_so(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    write_log(logs / f"train-steer-f-{TAG}-s4-tree-rollout-oracle.log", "0.200")
    m = run(tmp_path, logs)
    assert m["Roracleacc"] == ".2000", m["Roracleacc"]
    assert m["Roracleseed"] == "4", m["Roracleseed"]


def test_the_lowest_finished_seed_wins_not_the_better_one(tmp_path):
    """Two finished draws: the rule picks by seed index, never by accuracy."""
    logs = tmp_path / "logs"
    logs.mkdir()
    write_log(logs / f"train-steer-f-{TAG}-s1-tree-rollout-oracle.log", "0.100")
    write_log(logs / f"train-steer-f-{TAG}-s4-tree-rollout-oracle.log", "0.500")
    m = run(tmp_path, logs)
    assert m["Roracleseed"] == "1", m["Roracleseed"]
    assert m["Roracleacc"] == ".1000", m["Roracleacc"]


def test_an_unfinished_lower_seed_is_not_the_run(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    write_log(logs / f"train-steer-f-{TAG}-s1-tree-rollout-oracle.log",
              "0.100", last=30)
    write_log(logs / f"train-steer-f-{TAG}-s4-tree-rollout-oracle.log", "0.500")
    m = run(tmp_path, logs)
    assert m["Roracleseed"] == "4", m["Roracleseed"]


def test_the_contrast_is_against_signed_at_the_same_seed(tmp_path):
    """Table 10 has no error bars, so the comparison is the same-seed main arm.

    signed s1 is present and reads higher than signed s4 here: if the contrast
    took the across-seed mean, or seed 1, the number would be different.
    """
    logs = tmp_path / "logs"
    logs.mkdir()
    write_log(logs / f"train-steer-f-{TAG}-s1-tree-rollout.log",
              "0.300", "0.400", gpus=4)
    write_log(logs / f"train-steer-f-{TAG}-s4-tree-rollout.log",
              "0.200", "0.250", gpus=4)
    write_log(logs / f"train-steer-f-{TAG}-s4-tree-rollout-oracle.log",
              "0.150", "0.200", gpus=4)
    m = run(tmp_path, logs)
    assert m["Roracleseed"] == "4"
    assert m["Fsignedoracleacc"] == "+.0500", m["Fsignedoracleacc"]
    assert m["Fsignedoraclemaj"] == "+.0500", m["Fsignedoraclemaj"]
    # steps 40,50,...,110 -- the registered eight-point window.
    assert m["Fnsignedoracle"] == "8", m["Fnsignedoracle"]


def test_no_signed_run_at_that_seed_leaves_the_contrast_pending(tmp_path):
    """A macro is never invented: without signed at the seed there is no value."""
    logs = tmp_path / "logs"
    logs.mkdir()
    write_log(logs / f"train-steer-f-{TAG}-s4-tree-rollout-oracle.log", "0.150")
    m = run(tmp_path, logs)
    assert "Fsignedoracleacc" not in m or m["Fsignedoracleacc"] == "\\PENDING"


def test_the_cost_saving_is_measured_against_the_same_seed(tmp_path):
    """The oracle arm's point is that it removes the MTP forward pass.

    Section 12.6 times the arms against each other across seeds; the number
    this arm is for is the same-seed one, and it is emitted rather than typed.
    """
    logs = tmp_path / "logs"
    logs.mkdir()
    write_log(logs / f"train-steer-f-{TAG}-s4-tree-rollout.log",
              "0.200", sec="500.0", gpus=4)
    write_log(logs / f"train-steer-f-{TAG}-s4-tree-rollout-oracle.log",
              "0.150", sec="400.0", gpus=4)
    m = run(tmp_path, logs)
    assert m["Roraclecost"] == "400", m["Roraclecost"]
    assert m["Fsignedoraclecost"] == "+100", m["Fsignedoraclecost"]
    assert m["Fsignedoraclesaving"] == "20.0", m["Fsignedoraclesaving"]


def test_a_contrast_across_two_boxes_is_not_reported(tmp_path):
    """A seed index names a machine for the main arms, not for these.

    The lambda sweep ran at seed 1 on the four-GPU box while STEER-F seed 1 ran
    on the two-GPU one. Differencing those step times reported the machine as
    the ablation -- a sixty-eight percent "saving" from changing a damping
    coefficient -- and the ACCURACY difference is worse, because the machine is
    worth about the size of these contrasts and the larger box scores higher,
    so the confound flatters the follow-up. A first version of this gate
    covered the step time only. Without a matching topology there is no
    contrast at all; the row itself is still reported, with its box.
    """
    logs = tmp_path / "logs"
    logs.mkdir()
    write_log(logs / f"train-steer-f-{TAG}-s4-tree-rollout.log",
              "0.200", sec="500.0", gpus=2)
    write_log(logs / f"train-steer-f-{TAG}-s4-tree-rollout-oracle.log",
              "0.150", sec="400.0", gpus=4)
    m = run(tmp_path, logs)
    # The row survives and says which box it was.
    assert m["Roracleacc"] == ".1500", m["Roracleacc"]
    assert m["Roraclegpus"] == "4", m["Roraclegpus"]
    # The contrast does not.
    for k in ("Fsignedoracleacc", "Fsignedoraclemaj",
              "Fsignedoraclecost", "Fsignedoraclesaving"):
        assert k not in m or m[k] == "\\PENDING", (k, m.get(k))


def test_a_log_whose_dump_disagrees_with_its_name_is_excluded(tmp_path):
    """R{stem}seed is printed in the caption, so the seed has to be the run's.

    The main loop has checked the trainer's config dump against the file name
    since a seed-2 run arrived as train-grpo-<tag>-s4.log. The follow-up loop
    read the dump only for the GPU count, so a misfiled log would have had its
    file name printed as the seed of the row.
    """
    logs = tmp_path / "logs"
    logs.mkdir()
    f = logs / f"train-steer-f-{TAG}-s4-tree-rollout-oracle.log"
    write_log(f, "0.150")
    f.write_text("{'seed': 2, 'n_gpus_per_node': 4}\n" + f.read_text())
    m = run(tmp_path, logs)
    assert "Roracleseed" not in m or m["Roracleseed"] == "\\PENDING"
    assert "EXCLUDED" in m["__stdout__"], m["__stdout__"]


def test_the_search_moves_on_from_a_mislabelled_candidate(tmp_path):
    """Excluded, not substituted -- and the arm is not lost either."""
    logs = tmp_path / "logs"
    logs.mkdir()
    bad = logs / f"train-steer-f-{TAG}-s1-tree-rollout-oracle.log"
    write_log(bad, "0.100")
    bad.write_text("{'seed': 2, 'n_gpus_per_node': 4}\n" + bad.read_text())
    write_log(logs / f"train-steer-f-{TAG}-s4-tree-rollout-oracle.log", "0.500")
    m = run(tmp_path, logs)
    assert m["Roracleseed"] == "4", m["Roracleseed"]
    assert m["Roracleacc"] == ".5000", m["Roracleacc"]
