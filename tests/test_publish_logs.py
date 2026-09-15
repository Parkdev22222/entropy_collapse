"""The logs ARE the results, and publishing them has three ways to go wrong.

Every number in the manuscript is recomputed from these files; a box reclaimed
with an uncommitted log takes a run's evidence with it, which already happened
to the GRPO seed-1 arm. But the obvious `git add logs && git commit` is wrong
on a training box in three specific ways, and these tests pin each:

  it must not check the training tree out to another branch (the trainer reads
  run/ and steer_f/ off that tree), it must not freeze a log that is still
  being appended to, and it must never stage validation_data/ -- ~150 MB per
  run, and not gitignored.
"""
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# NOT ROOT/run/publish_logs.sh: the script resolves its repository root from
# its own location, so invoking the real path with a fixture LOG_DIR publishes
# into the REAL repository. That happened on 2026-09-15 and put three synthetic
# logs on origin/paper. Every test drives the copy inside its fixture pod.
SCRIPT = "run/publish_logs.sh"
TAG = "Qwen2.5-Math-1.5B"


def sh(cmd, cwd, **env):
    e = dict(os.environ)
    e.update({k: str(v) for k, v in env.items()})
    return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True,
                          text=True, cwd=str(cwd), env=e)


def make_log(d, run, last_step, tail=""):
    f = d / f"train-{run}.log"
    f.write_text("".join(f"step:{i} - global_seqlen: 1\n"
                         for i in range(1, last_step + 1)) + tail)
    return f


@pytest.fixture
def box(tmp_path):
    """A bare 'origin', a clone standing in for the pod, and some logs."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-b", "paper", str(origin)], check=True,
                   capture_output=True)
    pod = tmp_path / "pod"
    subprocess.run(["git", "clone", str(origin), str(pod)], check=True, capture_output=True)
    for k, v in (("user.email", "t@t"), ("user.name", "t")):
        sh(["git", "config", k, v], pod)
    # A symlinked run/ would resolve `run/..` back to the real repo under
    # `pwd -P`; copy so the fixture pod is a self-contained checkout.
    shutil.copytree(ROOT / "run", pod / "run")
    logs = pod / "logs" / "experiments"
    logs.mkdir(parents=True)
    (pod / "README.md").write_text("seed\n")
    sh(["git", "add", "README.md"], pod)
    sh(["git", "commit", "-m", "init"], pod)
    sh(["git", "push", "-u", "origin", "paper"], pod)
    return pod, origin, logs


def run_publish(pod, logs, *args, **env):
    # invoked the way every queue invokes it, from inside the fixture checkout
    return sh(["bash", SCRIPT, *args], pod, LOG_DIR=str(logs), **env)


# --------------------------------------------------------------- what it picks
def test_a_dry_run_changes_nothing(box):
    pod, origin, logs = box
    make_log(logs, f"grpo-{TAG}-s2", 110)
    p = run_publish(pod, logs)
    assert p.returncode == 0, p.stderr
    assert "dry run" in p.stdout
    assert not (Path(pod).parent / "entropy_logs_paper").exists()
    assert sh(["git", "ls-tree", "-r", "--name-only", "paper"], origin).stdout.strip() \
        == "README.md"


def test_an_unfinished_log_is_marked_but_still_published(box):
    """A crashed run's log is the evidence for why it crashed."""
    pod, _, logs = box
    make_log(logs, f"steer-{TAG}-s2", 47, "torch.OutOfMemoryError: CUDA out of memory\n")
    p = run_publish(pod, logs)
    assert "1 of them unfinished" in p.stdout
    assert f"~ train-steer-{TAG}-s2.log" in p.stdout


def test_done_only_leaves_the_unfinished_out_and_says_zero(box):
    pod, _, logs = box
    make_log(logs, f"steer-{TAG}-s2", 47)
    make_log(logs, f"grpo-{TAG}-s2", 110)
    p = run_publish(pod, logs, "--done-only")
    assert "1 to publish, 0 of them unfinished" in p.stdout
    assert f"train-steer-{TAG}-s2.log" not in p.stdout


def test_queue_logs_are_opt_in(box):
    pod, _, logs = box
    make_log(logs, f"grpo-{TAG}-s2", 110)
    (logs / "campaign.log").write_text("queue driver\n")
    assert "campaign.log" not in run_publish(pod, logs).stdout
    assert "campaign.log" in run_publish(pod, logs, "--queue-logs").stdout


def test_a_live_run_is_skipped(box):
    """Committing a log that is still being appended freezes a partial record."""
    pod, _, logs = box
    run = f"steer-{TAG}-s2"
    make_log(logs, run, 47)
    fake = subprocess.Popen(
        ["bash", "-c",
         f'exec -a "python3 -m verl.trainer.main_ppo trainer.experiment_name={run}" sleep 30'])
    try:
        time.sleep(0.4)
        p = run_publish(pod, logs)
        assert "1 skipped as live" in p.stdout
        assert "a trainer is writing these right now" in p.stdout
        p = run_publish(pod, logs, "--force")
        assert "0 skipped as live" in p.stdout
    finally:
        fake.kill(); fake.wait()


def test_a_big_log_is_flagged(box):
    pod, _, logs = box
    f = make_log(logs, f"grpo-{TAG}-s2", 110)
    with f.open("a") as fh:
        fh.write("x" * (21 * 1024 * 1024))
    assert "WARN:" in run_publish(pod, logs).stdout


# ------------------------------------------------------------------ publishing
def test_push_lands_the_logs_on_the_branch(box):
    pod, origin, logs = box
    make_log(logs, f"grpo-{TAG}-s2", 110)
    make_log(logs, f"steer-{TAG}-s2", 110)
    p = run_publish(pod, logs, "--push")
    assert p.returncode == 0, p.stdout + p.stderr
    tree = sh(["git", "ls-tree", "-r", "--name-only", "paper"], origin).stdout
    assert f"logs/experiments/train-grpo-{TAG}-s2.log" in tree
    assert f"logs/experiments/train-steer-{TAG}-s2.log" in tree


def test_the_training_tree_never_changes_branch(box):
    """The trainer reads run/ and steer_f/ off this tree while it runs."""
    pod, _, logs = box
    before = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"], pod).stdout.strip()
    make_log(logs, f"grpo-{TAG}-s2", 110)
    run_publish(pod, logs, "--push")
    assert sh(["git", "rev-parse", "--abbrev-ref", "HEAD"], pod).stdout.strip() == before


def test_validation_data_is_never_staged(box):
    """~150 MB per run, not gitignored, and permanent once committed."""
    pod, origin, logs = box
    make_log(logs, f"grpo-{TAG}-s2", 110)
    vd = Path(pod) / "validation_data"
    vd.mkdir()
    (vd / "step_10.jsonl").write_text('{"score": 1}\n' * 100)
    run_publish(pod, logs, "--push")
    tree = sh(["git", "ls-tree", "-r", "--name-only", "paper"], origin).stdout
    assert "validation_data" not in tree


def test_a_second_box_rebases_onto_the_first(box, tmp_path):
    """A100 and H100 publish to one branch; the file sets are disjoint."""
    pod, origin, logs = box
    make_log(logs, f"grpo-{TAG}-s2", 110)
    assert run_publish(pod, logs, "--push").returncode == 0

    other = tmp_path / "pod2"
    subprocess.run(["git", "clone", str(origin), str(other)], check=True, capture_output=True)
    for k, v in (("user.email", "t@t"), ("user.name", "t")):
        sh(["git", "config", k, v], other)
    shutil.copytree(ROOT / "run", other / "run")
    # the clone already carries the first box's logs/ directory
    logs2 = other / "logs" / "experiments"
    logs2.mkdir(parents=True, exist_ok=True)
    make_log(logs2, f"steer-f-{TAG}-s1-tree-rollout-lam0", 110)
    p = sh(["bash", SCRIPT, "--push"], other, LOG_DIR=str(logs2))
    assert p.returncode == 0, p.stdout + p.stderr
    tree = sh(["git", "ls-tree", "-r", "--name-only", "paper"], origin).stdout
    assert f"logs/experiments/train-grpo-{TAG}-s2.log" in tree, "the first box's log survived"
    assert f"logs/experiments/train-steer-f-{TAG}-s1-tree-rollout-lam0.log" in tree


def test_publishing_twice_is_a_no_op(box):
    pod, _, logs = box
    make_log(logs, f"grpo-{TAG}-s2", 110)
    run_publish(pod, logs, "--push")
    p = run_publish(pod, logs, "--push")
    assert "already up to date" in p.stdout


def test_an_empty_box_exits_clean(box):
    pod, _, logs = box
    p = run_publish(pod, logs, "--push")
    assert p.returncode == 0
    assert "nothing to publish" in p.stdout


def test_a_log_dir_outside_the_checkout_is_refused(box, tmp_path):
    """The failure that put three fixture logs on origin/paper.

    The script resolves its repository root from its own path, so an absolute
    invocation plus a LOG_DIR elsewhere publishes those files into whatever
    repository the script lives in.
    """
    pod, origin, _ = box
    foreign = tmp_path / "elsewhere"
    foreign.mkdir()
    make_log(foreign, f"grpo-{TAG}-s2", 110)
    p = sh(["bash", SCRIPT, "--push"], pod, LOG_DIR=str(foreign))
    assert p.returncode == 1
    assert "LOG_DIR is outside this checkout" in p.stderr
    tree = sh(["git", "ls-tree", "-r", "--name-only", "paper"], origin).stdout
    assert "train-grpo" not in tree

    p = sh(["bash", SCRIPT], pod, LOG_DIR=str(foreign), ALLOW_FOREIGN_LOGS="1")
    assert p.returncode == 0, "the escape hatch still exists for fixtures"
