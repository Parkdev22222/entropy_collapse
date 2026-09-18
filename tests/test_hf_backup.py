"""A backup must never read a checkpoint the trainer is still writing.

hf_backup.sh's guard used to be a name test: LIVE_TAG defaulted to _0905, which
only run_0905_chain.sh ever appends, so every campaign run walked past it. The
upload would then have been "verified" too -- verification compares local size
against Hub size, and a truncated file matches its own truncated upload. These
tests pin the replacement, which asks the box rather than the name.
"""
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run" / "hf_backup.sh"


def build_box(tmp_path, runs=("done-run", "live-run"), steps=(100, 110)):
    """A fake STEER_ROOT: the real run/ scripts, fake checkpoints, fake logs."""
    box = tmp_path / "box"
    (box / "logs" / "experiments").mkdir(parents=True)
    (box / "run").symlink_to(ROOT / "run")
    for run in runs:
        for st in steps:
            actor = box / "checkpoints" / "STEER-F" / run / f"global_step_{st}" / "actor"
            (actor / "huggingface").mkdir(parents=True)
            (actor / "huggingface" / "model.safetensors").write_text("weights")
            (actor / "optim_world_size_2_rank_0.pt").write_text("optimizer")
            (actor / "extra_state_world_size_2_rank_0.pt").write_text("extra")
    # done-run finished three days ago and its log says so; live-run is fresh.
    old = time.time() - 3 * 86400
    for path in (box / "checkpoints" / "STEER-F" / "done-run").rglob("*"):
        os.utime(path, (old, old))
    os.utime(box / "checkpoints" / "STEER-F" / "done-run", (old, old))
    (box / "logs" / "experiments" / "train-done-run.log").write_text(
        "step:110 - global_seqlen: 1\n")
    return box


def stub_hf(tmp_path):
    """A fake `hf` that records its arguments instead of talking to the Hub."""
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    cli = d / "hf"
    cli.write_text('#!/bin/sh\necho "$*" >> "$STUB_LOG"\nexit 0\n')
    cli.chmod(0o755)
    return cli


def run(box, arg, tmp_path, **env):
    e = dict(os.environ)
    e.update({"STEER_ROOT": str(box), "STUB_LOG": str(tmp_path / "hf.log")})
    e.update({k: str(v) for k, v in env.items()})
    e.setdefault("HF_CLI", str(stub_hf(tmp_path)))
    e.setdefault("REPO", "fake/repo")
    return subprocess.run(["bash", str(SCRIPT), arg], capture_output=True,
                          text=True, env=e, cwd=str(ROOT))


def calls(tmp_path):
    f = tmp_path / "hf.log"
    return f.read_text().splitlines() if f.is_file() else []


# ----------------------------------------------------------------- the guard
def test_a_freshly_written_run_is_refused(tmp_path):
    box = build_box(tmp_path)
    p = run(box, "live-run", tmp_path)
    assert p.returncode == 1
    assert "REFUSE" in p.stdout and "last 30 min" in p.stdout
    assert calls(tmp_path) == [], "nothing may be uploaded once the guard fires"


def test_the_refusal_names_a_run_that_is_safe_instead(tmp_path):
    box = build_box(tmp_path)
    p = run(box, "live-run", tmp_path)
    assert "safe to back up instead" in p.stdout
    assert "done-run" in p.stdout.split("safe to back up instead")[1]


def test_a_quiet_finished_run_is_uploaded(tmp_path):
    box = build_box(tmp_path)
    p = run(box, "done-run", tmp_path)
    got = calls(tmp_path)
    assert len(got) == 2, f"expected one upload per step, got {got}"
    assert "done-run/global_step_100" in got[0]
    assert "WARN" not in p.stdout, "a finished run should not warn"


def test_a_trainer_holding_the_name_is_refused_even_when_the_files_are_old(tmp_path):
    """The mtime window is not the only test: a live process wins outright."""
    box = build_box(tmp_path)
    # argv[0] is what pgrep -f reads, so this looks exactly like a trainer.
    fake = subprocess.Popen(
        ["bash", "-c",
         'exec -a "python3 -m verl.trainer.main_ppo '
         'trainer.experiment_name=done-run" sleep 30'])
    try:
        time.sleep(0.4)
        p = run(box, "done-run", tmp_path)
        assert p.returncode == 1, p.stdout
        assert "is training it right now" in p.stdout
        assert calls(tmp_path) == []
    finally:
        fake.kill()
        fake.wait()


def test_force_overrides_but_says_so(tmp_path):
    box = build_box(tmp_path)
    p = run(box, "live-run", tmp_path, FORCE=1)
    assert "WARN" in p.stdout and "FORCE=1" in p.stdout
    assert len(calls(tmp_path)) == 2


def test_the_legacy_name_test_still_works(tmp_path):
    box = build_box(tmp_path, runs=("done-run",))
    p = run(box, "done-run", tmp_path, LIVE_TAG="done")
    assert p.returncode == 1
    assert "LIVE_TAG=done" in p.stdout


def test_an_unfinished_run_warns_but_proceeds(tmp_path):
    """A crashed run's checkpoints are still worth keeping -- warn, don't refuse."""
    box = build_box(tmp_path, runs=("done-run", "crashed"))
    old = time.time() - 3 * 86400
    for path in (box / "checkpoints" / "STEER-F" / "crashed").rglob("*"):
        os.utime(path, (old, old))
    os.utime(box / "checkpoints" / "STEER-F" / "crashed", (old, old))
    p = run(box, "crashed", tmp_path)
    assert "WARN" in p.stdout and "reaching step 110" in p.stdout
    assert len(calls(tmp_path)) == 2, "it still uploads"


# ---------------------------------------------------------------- delete/prune
def test_a_failed_verification_keeps_the_local_copy(tmp_path):
    """huggingface_hub is not installed here, so verify() fails -- as it should."""
    box = build_box(tmp_path)
    p = run(box, "done-run", tmp_path, DELETE=1)
    assert "verification FAILED" in p.stdout
    assert (box / "checkpoints" / "STEER-F" / "done-run" / "global_step_110").is_dir()


def test_prune_keeps_huggingface_and_drops_the_rest(tmp_path):
    box = build_box(tmp_path)
    p = run(box, "done-run", tmp_path, PRUNE=1)
    assert p.returncode == 0, p.stdout
    actor = box / "checkpoints" / "STEER-F" / "done-run" / "global_step_110" / "actor"
    assert (actor / "huggingface" / "model.safetensors").is_file()
    assert not (actor / "optim_world_size_2_rank_0.pt").exists()
    assert not (actor / "extra_state_world_size_2_rank_0.pt").exists()
    assert calls(tmp_path) == [], "prune must not touch the network"


def test_prune_refuses_a_live_run(tmp_path):
    box = build_box(tmp_path)
    p = run(box, "live-run", tmp_path, PRUNE=1)
    assert p.returncode == 1
    actor = box / "checkpoints" / "STEER-F" / "live-run" / "global_step_110" / "actor"
    assert (actor / "optim_world_size_2_rank_0.pt").is_file()


def test_prune_skips_a_step_with_no_hf_copy(tmp_path):
    """Dropping the shards where huggingface/ is missing would leave nothing."""
    box = build_box(tmp_path)
    actor = box / "checkpoints" / "STEER-F" / "done-run" / "global_step_100" / "actor"
    shutil.rmtree(actor / "huggingface")
    # rmtree just touched actor/, which is exactly what the freshness guard
    # watches -- age it back or the guard (correctly) refuses the whole run.
    old = time.time() - 3 * 86400
    os.utime(actor, (old, old))
    p = run(box, "done-run", tmp_path, PRUNE=1)
    assert "no actor/huggingface" in p.stdout
    assert (actor / "optim_world_size_2_rank_0.pt").is_file()


def test_prune_needs_no_repo(tmp_path):
    box = build_box(tmp_path)
    e = dict(os.environ)
    e.update({"STEER_ROOT": str(box), "PRUNE": "1"})
    e.pop("REPO", None)
    p = subprocess.run(["bash", str(SCRIPT), "done-run"], capture_output=True,
                       text=True, env=e, cwd=str(ROOT))
    assert p.returncode == 0, p.stderr


# --------------------------------------------------------------------- basics
def test_no_argument_lists_the_runs(tmp_path):
    box = build_box(tmp_path)
    e = dict(os.environ)
    e["STEER_ROOT"] = str(box)
    p = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True,
                       env=e, cwd=str(ROOT))
    assert p.returncode == 2
    assert "usage:" in p.stdout and "done-run" in p.stdout


def test_an_unknown_run_is_a_clear_error(tmp_path):
    box = build_box(tmp_path)
    p = run(box, "no-such-run", tmp_path)
    assert p.returncode == 1
    assert "FATAL: no" in p.stdout


def test_root_comes_from_the_script_not_a_pinned_path(tmp_path):
    """The repo moved off /workspace on 2026-09-18 and this line did not.

    run_backbones.sh:255 calls this with DELETE=1 after every finished run, so
    a pinned absolute root is not merely a missing upload. If the old checkout
    still exists the upload reads ITS checkpoints and deletes those; if it does
    not, the queue never frees anything and fills the disk it is guarding.
    """
    src = (ROOT / "run" / "hf_backup.sh").read_text()
    assert "STEER_ROOT:-/workspace/entropy_collapse" not in src
    assert 'STEER_ROOT=${STEER_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}' in src


def test_an_explicit_root_still_wins(tmp_path):
    """Operators pass STEER_ROOT when the checkpoints live off the checkout."""
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "checkpoints" / "STEER-F").mkdir(parents=True)
    p = subprocess.run(["bash", str(ROOT / "run" / "hf_backup.sh")],
                       capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
                            "STEER_ROOT": str(elsewhere)})
    assert "FATAL" not in p.stdout + p.stderr
    assert p.returncode == 2          # usage, having found the root fine
