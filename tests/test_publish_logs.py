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
    # A training box is never sitting on the log branch: it is on whatever
    # branch the code came from. That is what makes the worktree path the
    # normal one, and the ROOT path (already on `paper`) the exception.
    sh(["git", "checkout", "-q", "-b", "work"], pod)
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


# ------------------------------------------------- recovering from itself
def wt_of(pod):
    return Path(pod).parent / "entropy_logs_paper"


def test_an_interrupted_run_does_not_wedge_the_next_one(box):
    """The H100 failure of 2026-09-15.

    A run that dies between `git add` and `git commit` leaves the worktree's
    index dirty, and every later run then died at the rebase with "Your index
    contains uncommitted changes" -- permanently, because nothing cleaned up.
    """
    pod, origin, logs = box
    make_log(logs, f"grpo-{TAG}-s2", 110)
    run_publish(pod, logs, "--push")          # creates the worktree
    wt = wt_of(pod)
    assert wt.is_dir()

    # exactly what an interrupted run leaves behind
    (wt / "logs" / "experiments" / f"train-steer-{TAG}-s2.log").write_text("half\n")
    sh(["git", "add", "logs/experiments"], wt)
    assert sh(["git", "diff", "--cached", "--quiet"], wt).returncode != 0

    make_log(logs, f"steer-{TAG}-s2", 110)
    p = run_publish(pod, logs, "--push")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "REFUSE" not in p.stderr
    tree = sh(["git", "ls-tree", "-r", "--name-only", "paper"], origin).stdout
    assert f"logs/experiments/train-steer-{TAG}-s2.log" in tree
    # and the real log went up, not the half-written stand-in
    blob = sh(["git", "show", f"paper:logs/experiments/train-steer-{TAG}-s2.log"],
              origin).stdout
    assert "half" not in blob and "step:110" in blob


def test_a_commit_that_was_never_pushed_survives_the_cleanup(box):
    """reset --hard HEAD keeps commits; only the index and worktree go back."""
    pod, origin, logs = box
    make_log(logs, f"grpo-{TAG}-s2", 110)
    run_publish(pod, logs, "--push")
    wt = wt_of(pod)

    # a commit made but not pushed, plus leftover staging on top
    (wt / "logs" / "experiments" / f"train-steer-{TAG}-s3.log").write_text("step:110 - x\n")
    sh(["git", "add", "logs/experiments"], wt)
    sh(["git", "commit", "-m", "an unpushed commit"], wt)
    (wt / "logs" / "experiments" / "stray.log").write_text("leftover\n")
    sh(["git", "add", "logs/experiments"], wt)

    make_log(logs, f"steer-{TAG}-s4", 110)
    p = run_publish(pod, logs, "--push")
    assert p.returncode == 0, p.stdout + p.stderr
    tree = sh(["git", "ls-tree", "-r", "--name-only", "paper"], origin).stdout
    assert f"logs/experiments/train-steer-{TAG}-s3.log" in tree, \
        "the unpushed commit must reach the remote, not be discarded"
    assert "stray.log" not in tree, "the leftover staging must not"


def test_it_never_resets_the_users_own_tree(box):
    """The WORKTREE == ROOT branch is the user's checkout, not scratch space."""
    pod, _, logs = box
    make_log(logs, f"grpo-{TAG}-s2", 110)
    sh(["git", "checkout", "-q", "paper"], pod)   # take the ROOT path
    precious = Path(pod) / "README.md"
    precious.write_text("work in progress I have not committed\n")
    p = run_publish(pod, logs, "--push")
    assert p.returncode == 1
    assert "will not reset it" in p.stderr
    assert precious.read_text() == "work in progress I have not committed\n"


def test_the_refusal_names_the_recovery_command(box):
    """A stuck operator must not have to improvise on the results branch."""
    pod, _, logs = box
    make_log(logs, f"grpo-{TAG}-s2", 110)
    run_publish(pod, logs, "--push")
    wt = wt_of(pod)
    # a genuine conflict: the same path with different content on both sides
    sh(["git", "checkout", "-q", "-b", "side"], wt)
    (wt / "logs" / "experiments" / f"train-grpo-{TAG}-s2.log").write_text("theirs\n")
    sh(["git", "add", "-A"], wt)
    sh(["git", "commit", "-m", "diverge"], wt)
    body = (ROOT / "run" / "publish_logs.sh").read_text()
    assert "reset --hard HEAD, then re-run this." in body
    assert "pull --rebase origin ${BRANCH} &&" in body


def test_a_dead_stub_cannot_overwrite_a_finished_run(box, tmp_path):
    """The H100 publish of 2026-09-15.

    Both boxes had a train-grpo-<tag>-s2.log: one a finished 110-step run, the
    other an 8 KB carcass of a start that died in seconds. The second publisher
    overwrote the first, and the branch tip -- which is what every analysis
    reads -- silently became the stub.
    """
    pod, origin, logs = box
    make_log(logs, f"grpo-{TAG}-s2", 110)
    assert run_publish(pod, logs, "--push").returncode == 0

    other = tmp_path / "pod2"
    subprocess.run(["git", "clone", str(origin), str(other)], check=True, capture_output=True)
    for k, v in (("user.email", "t@t"), ("user.name", "t")):
        sh(["git", "config", k, v], other)
    shutil.copytree(ROOT / "run", other / "run")
    sh(["git", "checkout", "-q", "-b", "work"], other)
    logs2 = other / "logs" / "experiments"
    logs2.mkdir(parents=True, exist_ok=True)
    make_log(logs2, f"grpo-{TAG}-s2", 3)          # the carcass
    make_log(logs2, f"steer-{TAG}-s1-xclip", 110)  # and something genuinely new

    p = sh(["bash", SCRIPT, "--push"], other, LOG_DIR=str(logs2))
    assert p.returncode == 0, p.stdout + p.stderr
    assert "NOT overwritten" in p.stdout
    assert f"train-grpo-{TAG}-s2.log ours=3 theirs=110" in p.stdout

    kept = sh(["git", "show", f"paper:logs/experiments/train-grpo-{TAG}-s2.log"],
              origin).stdout
    assert "step:110" in kept, "the finished run must survive"
    tree = sh(["git", "ls-tree", "-r", "--name-only", "paper"], origin).stdout
    assert f"logs/experiments/train-steer-{TAG}-s1-xclip.log" in tree, \
        "the genuinely new log still goes up"


def test_force_overwrites_a_longer_log_when_asked(box, tmp_path):
    pod, origin, logs = box
    make_log(logs, f"grpo-{TAG}-s2", 110)
    run_publish(pod, logs, "--push")
    make_log(logs, f"grpo-{TAG}-s2", 3)
    p = run_publish(pod, logs, "--push", "--force")
    assert p.returncode == 0, p.stdout + p.stderr
    kept = sh(["git", "show", f"paper:logs/experiments/train-grpo-{TAG}-s2.log"],
              origin).stdout
    assert "step:110" not in kept


# ------------------------------------------------ learning from the other box
def peer_box(origin, tmp_path, name="pod2"):
    other = tmp_path / name
    subprocess.run(["git", "clone", str(origin), str(other)], check=True, capture_output=True)
    for k, v in (("user.email", "t@t"), ("user.name", "t")):
        sh(["git", "config", k, v], other)
    shutil.copytree(ROOT / "run", other / "run")
    sh(["git", "checkout", "-q", "-b", "work"], other)
    # A real second box has been running for weeks on its own branch and has
    # only its OWN logs on disk; a fresh clone of the log branch would already
    # carry the peer's, which is not the situation --pull exists for.
    logs = other / "logs" / "experiments"
    if logs.is_dir():
        shutil.rmtree(logs)
    logs.mkdir(parents=True)
    return other, logs


def test_pull_teaches_a_box_what_the_other_finished(box, tmp_path):
    """Two boxes sharing a campaign disagree about what is left.

    Each queue reads logs/experiments on its own box, so on 2026-09-15 the
    A100 listed all ten follow-up arms as remaining while the H100 had already
    finished two. Starting that queue would have re-run 29 GPU-hours.
    """
    pod, origin, logs = box
    make_log(logs, f"steer-f-{TAG}-s1-tree-rollout-lam0", 110)
    assert run_publish(pod, logs, "--push").returncode == 0

    other, logs2 = peer_box(origin, tmp_path)
    assert not (logs2 / f"train-steer-f-{TAG}-s1-tree-rollout-lam0.log").is_file()
    p = sh(["bash", SCRIPT, "--pull"], other, LOG_DIR=str(logs2))
    assert p.returncode == 0, p.stdout + p.stderr
    got = logs2 / f"train-steer-f-{TAG}-s1-tree-rollout-lam0.log"
    assert got.is_file(), p.stdout
    assert "step:110" in got.read_text()


def test_pull_never_overwrites_a_run_this_box_got_further_on(box, tmp_path):
    pod, origin, logs = box
    make_log(logs, f"grpo-{TAG}-s2", 20)
    run_publish(pod, logs, "--push")

    other, logs2 = peer_box(origin, tmp_path)
    make_log(logs2, f"grpo-{TAG}-s2", 110)          # this box is ahead
    p = sh(["bash", SCRIPT, "--pull"], other, LOG_DIR=str(logs2))
    assert p.returncode == 0, p.stderr
    assert "left 1 alone" in p.stdout
    assert "step:110" in (logs2 / f"train-grpo-{TAG}-s2.log").read_text()


def test_pull_does_not_touch_a_log_a_trainer_is_writing(box, tmp_path):
    pod, origin, logs = box
    run = f"steer-{TAG}-s2"
    make_log(logs, run, 110)
    run_publish(pod, logs, "--push")

    other, logs2 = peer_box(origin, tmp_path)
    make_log(logs2, run, 5)                          # ours is behind, but live
    fake = subprocess.Popen(
        ["bash", "-c",
         f'exec -a "python3 -m verl.trainer.main_ppo trainer.experiment_name={run}" sleep 30'])
    try:
        time.sleep(0.4)
        p = sh(["bash", SCRIPT, "--pull"], other, LOG_DIR=str(logs2))
        assert p.returncode == 0, p.stderr
        assert "step:110" not in (logs2 / f"train-{run}.log").read_text()
    finally:
        fake.kill(); fake.wait()


def test_a_finished_arm_is_not_called_live_by_a_sibling(box):
    """The prefix trap, in the function publish_logs.sh asks.

    Run names are prefixes of each other: steer-f-<tag>-s1-tree-rollout is a
    prefix of every tree arm, and ...-lam0 of ...-lam0.1. A substring test
    called a finished arm live whenever a sibling was training, which is why
    the lam0 log never reached the branch.
    """
    pod, _, logs = box
    base = f"steer-f-{TAG}-s1-tree-rollout"
    make_log(logs, base, 110)
    make_log(logs, f"{base}-lam0", 110)
    fake = subprocess.Popen(
        ["bash", "-c",
         f'exec -a "python3 -m verl.trainer.main_ppo '
         f'trainer.experiment_name={base}-lam0.1" sleep 30'])
    try:
        time.sleep(0.4)
        p = run_publish(pod, logs)
        assert "0 skipped as live" in p.stdout, p.stdout
        assert f"train-{base}.log" in p.stdout
        assert f"train-{base}-lam0.log" in p.stdout
    finally:
        fake.kill(); fake.wait()


def test_a_failed_fetch_stops_at_the_real_error(tmp_path):
    """2026-09-21: every fetch retry printed 'Disk quota exceeded', the script
    carried on, and the user saw 'You have unstaged changes' from the rebase
    three steps later -- whose printed advice is `reset --hard HEAD`, the very
    command that had just failed for the same reason. A circle."""
    src = (ROOT / "run" / "publish_logs.sh").read_text()
    assert 'fetched=0' in src
    assert 'could not fetch origin/${BRANCH} after 4 tries' in src
    assert 'Disk quota exceeded' in src


def test_the_scratch_reset_does_not_swallow_its_own_failure():
    """reset --hard WRITES files, so it is the first casualty of a full disk.
    `2>/dev/null || true` made that invisible."""
    src = (ROOT / "run" / "publish_logs.sh").read_text()
    assert 'git -C "${WORKTREE}" reset -q --hard HEAD 2>/dev/null || true' not in src
    assert 'could not reset the scratch worktree' in src


def test_the_users_own_tree_is_still_never_reset():
    """Regression guard from task 27: when WORKTREE == ROOT the checkout is the
    user's real tree and must be refused, not cleaned."""
    src = (ROOT / "run" / "publish_logs.sh").read_text()
    assert 'this script will not reset it' in src


# ----------------------------------- a copy of the same run, degraded (09-21)
def blob_bytes(origin, path):
    """The blob as stored. `text=True` would translate \r to \n on the way
    back, which is exactly the difference these tests are about."""
    return subprocess.run(["git", "show", f"paper:{path}"],
                          capture_output=True, cwd=str(origin)).stdout


def _second_box(tmp_path, origin):
    other = tmp_path / "pod_dmg"
    subprocess.run(["git", "clone", str(origin), str(other)], check=True,
                   capture_output=True)
    for k, v in (("user.email", "t@t"), ("user.name", "t")):
        sh(["git", "config", k, v], other)
    shutil.copytree(ROOT / "run", other / "run")
    sh(["git", "checkout", "-q", "-b", "work"], other)
    logs2 = other / "logs" / "experiments"
    logs2.mkdir(parents=True, exist_ok=True)
    return other, logs2


def test_a_copy_whose_newlines_became_carriage_returns_is_refused(box, tmp_path):
    """2026-09-21, and the step test could not see it.

    A push replaced train-grpo-<tag>-s4.log with a copy of itself whose every
    newline had become a carriage return: 4057 LF and 133 CR became 0 LF and
    4194 CR. Both copies reached step 110, so `theirs > mine` was false and the
    good one was overwritten. The data survived -- Python's splitlines() cuts
    on \\r too -- but these logs are the ledger every manuscript number is
    recomputed from, and grep, wc and awk all see one enormous line.
    """
    pod, origin, logs = box
    make_log(logs, f"grpo-{TAG}-s4", 110)
    assert run_publish(pod, logs, "--push").returncode == 0

    other, logs2 = _second_box(tmp_path, origin)
    good = (logs / f"train-grpo-{TAG}-s4.log").read_text()
    (logs2 / f"train-grpo-{TAG}-s4.log").write_text(good.replace("\n", "\r"))

    p = sh(["bash", SCRIPT, "--push"], other, LOG_DIR=str(logs2))
    assert p.returncode == 0, p.stdout + p.stderr
    assert "line endings" in p.stdout, p.stdout
    assert "git show origin/paper" in p.stdout, "no recovery command offered"

    kept = blob_bytes(origin, f"logs/experiments/train-grpo-{TAG}-s4.log")
    assert kept.count(b"\n") > 100, "the intact copy was overwritten anyway"


def test_a_truncated_copy_of_the_same_run_is_refused(box, tmp_path):
    """Logs are append-only, so a later copy of a run cannot be smaller. One
    that is has been cut short in transit, whatever step it still mentions."""
    pod, origin, logs = box
    make_log(logs, f"grpo-{TAG}-s4", 110)
    assert run_publish(pod, logs, "--push").returncode == 0

    other, logs2 = _second_box(tmp_path, origin)
    good = (logs / f"train-grpo-{TAG}-s4.log").read_text()
    # Same final step, half the bytes: the tail survived, the body did not.
    (logs2 / f"train-grpo-{TAG}-s4.log").write_text(good[len(good) // 2:])

    p = sh(["bash", SCRIPT, "--push"], other, LOG_DIR=str(logs2))
    assert "truncated" in p.stdout, p.stdout
    kept = blob_bytes(origin, f"logs/experiments/train-grpo-{TAG}-s4.log")
    assert len(kept) >= len(good) - 2, "the intact copy was overwritten anyway"


def test_a_log_that_genuinely_grew_still_publishes(box, tmp_path):
    """The guard must not stop the normal case: a run that continued."""
    pod, origin, logs = box
    make_log(logs, f"grpo-{TAG}-s4", 40)
    assert run_publish(pod, logs, "--push").returncode == 0

    other, logs2 = _second_box(tmp_path, origin)
    make_log(logs2, f"grpo-{TAG}-s4", 110)

    p = sh(["bash", SCRIPT, "--push"], other, LOG_DIR=str(logs2))
    assert p.returncode == 0, p.stdout + p.stderr
    kept = sh(["git", "show", f"paper:logs/experiments/train-grpo-{TAG}-s4.log"],
              origin).stdout
    assert "step:110" in kept, "a longer run must overwrite a shorter one"


def test_force_still_overwrites_a_degraded_copy(box, tmp_path):
    pod, origin, logs = box
    make_log(logs, f"grpo-{TAG}-s4", 110)
    assert run_publish(pod, logs, "--push").returncode == 0

    other, logs2 = _second_box(tmp_path, origin)
    good = (logs / f"train-grpo-{TAG}-s4.log").read_text()
    (logs2 / f"train-grpo-{TAG}-s4.log").write_text(good.replace("\n", "\r"))

    p = sh(["bash", SCRIPT, "--push", "--force"], other, LOG_DIR=str(logs2))
    assert p.returncode == 0, p.stdout + p.stderr
    kept = blob_bytes(origin, f"logs/experiments/train-grpo-{TAG}-s4.log")
    assert kept.count(b"\n") < 10, "--force must still overwrite"


# --------------------- the same run, with its recorded numbers changed (09-22)
def test_a_copy_whose_metric_lines_were_edited_is_refused(box, tmp_path):
    """2026-09-22, and every other check here passed.

    A push to train-grpo-<tag>-s4.log altered the accuracy in twelve metric
    lines -- steps 60 through 110, converged-window mean .1560 -> .1375 --
    while the global_seqlen fingerprints on those same lines stayed identical,
    so it was one run's lines with the numbers replaced. Same run, same final
    step, more bytes, real newlines: the step test, the line-ending test and
    the truncation test all had nothing to say.

    A log is append-only, so the committed copy must be a prefix of the
    incoming one. Changed content is not new data.
    """
    pod, origin, logs = box
    name = f"train-grpo-{TAG}-s4.log"
    make_log(logs, f"grpo-{TAG}-s4", 110)
    # give the step lines something to edit
    good = (logs / name).read_text().replace(
        "step:70 - global_seqlen: 1", "step:70 - global_seqlen: 1 - acc:0.168")
    (logs / name).write_text(good)
    assert run_publish(pod, logs, "--push").returncode == 0

    other, logs2 = _second_box(tmp_path, origin)
    (logs2 / name).write_text(good.replace("acc:0.168", "acc:0.138"))

    p = sh(["bash", SCRIPT, "--push"], other, LOG_DIR=str(logs2))
    assert p.returncode == 0, p.stdout + p.stderr
    assert "rewritten" in p.stdout, p.stdout
    assert "70" in p.stdout, f"the changed step is not named: {p.stdout}"

    kept = blob_bytes(origin, f"logs/experiments/{name}")
    assert b"acc:0.168" in kept, "the recorded numbers were overwritten anyway"
    assert b"acc:0.138" not in kept


def test_force_still_overwrites_an_edited_copy(box, tmp_path):
    """A line-ending repair trips the same test, and should: rewriting a
    committed log is something a person confirms, not something that happens
    on the way past."""
    pod, origin, logs = box
    name = f"train-grpo-{TAG}-s4.log"
    make_log(logs, f"grpo-{TAG}-s4", 110, tail="acc:0.168\n")
    assert run_publish(pod, logs, "--push").returncode == 0

    other, logs2 = _second_box(tmp_path, origin)
    good = (logs / name).read_text()
    (logs2 / name).write_text(good.replace("acc:0.168", "acc:0.138"))

    p = sh(["bash", SCRIPT, "--push", "--force"], other, LOG_DIR=str(logs2))
    assert p.returncode == 0, p.stdout + p.stderr
    kept = blob_bytes(origin, f"logs/experiments/{name}")
    assert b"acc:0.138" in kept, "--force did not overwrite"


def test_an_append_after_a_restart_still_publishes(box, tmp_path):
    """The guard must not stop a relaunch. A restart appends a second launch,
    so step numbers repeat -- which is why the test is a byte prefix and not a
    per-step comparison."""
    pod, origin, logs = box
    name = f"train-grpo-{TAG}-s4.log"
    make_log(logs, f"grpo-{TAG}-s4", 40)
    assert run_publish(pod, logs, "--push").returncode == 0

    other, logs2 = _second_box(tmp_path, origin)
    first = (logs / name).read_text()
    relaunch = "".join(f"step:{i} - global_seqlen: 1\n" for i in range(1, 111))
    (logs2 / name).write_text(first + relaunch)

    p = sh(["bash", SCRIPT, "--push"], other, LOG_DIR=str(logs2))
    assert "rewritten" not in p.stdout, p.stdout
    kept = blob_bytes(origin, f"logs/experiments/{name}")
    assert len(kept) > len(first), "the relaunch was refused"
