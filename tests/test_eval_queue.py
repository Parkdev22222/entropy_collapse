"""The evaluation queue: name the runs, and do not take the GPUs from a trainer.

Two failures this pins, both silent.

The queue built an ``arms x seeds`` cross product, which assumes every arm
finished the same seeds.  A campaign split across boxes does not leave a
rectangle, and there was no way to ask for one arm at one seed and another at a
different one -- the shape a half-lost campaign actually has.

And it was the one queue with no busy guard.  The three training queues have
refused to start on a busy box since September; this one is the one most likely
to be run by hand *while* a campaign is mid-flight, because checkpoints only
become interesting once runs finish.  Starting there takes the GPUs out from
under the trainer, which for a tree arm is up to sixteen hours of work.
"""
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BANNER = "### run_steerf.sh  run_name={run}  seed={seed}\n"
STEP = "step:{n} - global_seqlen/min:1.0 - actor/entropy:0.1\n"


def train_log(log_dir: Path, run: str, last_step: int) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    body = "".join(STEP.format(n=n) for n in range(1, last_step + 1))
    (log_dir / f"train-{run}.log").write_text(body)


def queue(tmp_path: Path, env: dict | None = None, dry: str = "1"):
    e = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": "/tmp",
        "DRY": dry,
        "LOG_DIR": str(tmp_path / "logs"),
        "CKPT_ROOT": str(tmp_path / "ckpt"),
        **(env or {}),
    }
    return subprocess.run(["bash", str(ROOT / "run" / "run_eval_all.sh")],
                          capture_output=True, text=True, cwd=str(ROOT), env=e)


def finished_seed1(tmp_path: Path) -> None:
    """All five arms of seed 1, each at the final step."""
    logs = tmp_path / "logs"
    for run in ("grpo-Qwen2.5-Math-1.5B-s1",
                "steer-Qwen2.5-Math-1.5B-s1",
                "steer-f-Qwen2.5-Math-1.5B-s1-tree-rollout",
                "steer-f-Qwen2.5-Math-1.5B-s1-tree-rollout-uniform",
                "steer-f-Qwen2.5-Math-1.5B-s1-tree-rollout-permuted"):
        train_log(logs, run, 110)


# --- naming the runs --------------------------------------------------------

def test_named_pairs_queue_in_order(tmp_path):
    finished_seed1(tmp_path)
    p = queue(tmp_path, {"EVAL_RUNS": "grpo:1 signed:1"})
    queued = [ln.split()[1] for ln in p.stdout.splitlines() if ln.startswith("  QUEUE")]
    assert queued == ["grpo", "signed"], p.stdout


def test_pairs_need_not_share_a_seed(tmp_path):
    """The shape the cross product could not express."""
    logs = tmp_path / "logs"
    train_log(logs, "grpo-Qwen2.5-Math-1.5B-s1", 110)
    train_log(logs, "steer-Qwen2.5-Math-1.5B-s2", 110)
    p = queue(tmp_path, {"EVAL_RUNS": "grpo:1 steer:2"})
    assert "2 eval(s) queued" in p.stdout, p.stdout


def test_commas_group_the_printout(tmp_path):
    finished_seed1(tmp_path)
    p = queue(tmp_path, {"EVAL_RUNS": "grpo:1 steer:1, uniform:1 permuted:1"})
    assert "-- group 1" in p.stdout and "-- group 2" in p.stdout, p.stdout
    assert "4 eval(s) queued" in p.stdout, p.stdout


def test_named_pairs_replace_the_cross_product(tmp_path):
    """SEEDS must not smuggle extra runs in beside EVAL_RUNS."""
    finished_seed1(tmp_path)
    p = queue(tmp_path, {"EVAL_RUNS": "grpo:1", "SEEDS": "1 2 3 4 5"})
    assert "1 eval(s) queued" in p.stdout, p.stdout


def test_a_malformed_pair_is_fatal(tmp_path):
    """Not silently dropped: a typo that shrinks the queue is the whole hazard."""
    finished_seed1(tmp_path)
    p = queue(tmp_path, {"EVAL_RUNS": "grpo:1 steer"})
    assert p.returncode == 2
    assert "is not <arm>:<seed>" in p.stderr, p.stderr


def test_an_unfinished_run_is_skipped_with_its_reason(tmp_path):
    logs = tmp_path / "logs"
    train_log(logs, "grpo-Qwen2.5-Math-1.5B-s1", 110)
    train_log(logs, "steer-Qwen2.5-Math-1.5B-s1", 40)      # died early
    p = queue(tmp_path, {"EVAL_RUNS": "grpo:1 steer:1"})
    assert "1 eval(s) queued" in p.stdout, p.stdout
    assert "never reached step 110" in p.stdout, p.stdout


# --- not taking the box from a trainer --------------------------------------

def busy_marker(tmp_path: Path):
    """A live process whose command line a custom BUSY_RE matches."""
    marker = tmp_path / "pretend_trainer_xyzzy"
    marker.write_text("#!/bin/sh\nsleep 60\n")
    marker.chmod(0o755)
    return subprocess.Popen(["/bin/sh", str(marker)])


def test_the_queue_refuses_while_a_trainer_holds_the_box(tmp_path):
    finished_seed1(tmp_path)
    proc = busy_marker(tmp_path)
    try:
        time.sleep(0.3)
        p = queue(tmp_path, {"EVAL_RUNS": "grpo:1",
                             "BUSY_RE": "[p]retend_trainer_xyzzy"}, dry="0")
        assert p.returncode == 2, (p.stdout, p.stderr)
        assert "training process is already running" in p.stderr, p.stderr
        assert "WAIT=1" in p.stderr, p.stderr
    finally:
        proc.kill()
        proc.wait()


def test_a_dry_run_still_works_on_a_busy_box(tmp_path):
    """Planning while a campaign runs is the normal case, not a mistake."""
    finished_seed1(tmp_path)
    proc = busy_marker(tmp_path)
    try:
        time.sleep(0.3)
        p = queue(tmp_path, {"EVAL_RUNS": "grpo:1",
                             "BUSY_RE": "[p]retend_trainer_xyzzy"})
        assert p.returncode == 0, (p.stdout, p.stderr)
        assert "1 eval(s) queued" in p.stdout, p.stdout
    finally:
        proc.kill()
        proc.wait()


def test_an_idle_box_gets_past_the_busy_guard(tmp_path):
    """The guard must not be the thing that stops every evaluation."""
    finished_seed1(tmp_path)
    p = queue(tmp_path, {"EVAL_RUNS": "grpo:1",
                         "BUSY_RE": "[n]othing_matches_this_xyzzy"}, dry="0")
    assert "training process is already running" not in p.stderr, p.stderr


# --- the checkpoint has to hold weights -------------------------------------

def test_checkpoint_selection_asks_for_weights():
    """fsdp_checkpoint_manager writes huggingface/ with config and tokenizer
    always and the weights only when hf_model is in save_contents, so the
    directory existing says nothing. Regression detector for the eval path,
    which tested exactly that until 2026-09-23."""
    body = (ROOT / "run" / "run_eval_all.sh").read_text()
    picker = body[body.index("resolve_ckpt ()"):body.index("# ------------------------------------------------------------------- run")]
    assert "hf_weights_present" in picker
    assert '[ -d "${CKPT_ROOT}/${cand}/global_step_${d}/actor/huggingface" ]' not in picker
