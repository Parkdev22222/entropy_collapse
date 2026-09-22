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


def test_prune_does_not_make_the_run_look_live(tmp_path):
    """2026-09-21: prune deletes inside global_step_N/actor, which bumps that
    directory's mtime -- and the liveness guard is `find -maxdepth 3 -mmin
    -FRESH_MIN`, which matches actor/ exactly. So pruning made the next half
    hour of this same script REFUSE the run it had just pruned, blaming its own
    deletions on a trainer. Three runs hit this at once on the A100 box.
    """
    ckpt = tmp_path / "checkpoints" / "STEER-F" / "arun" / "global_step_10" / "actor"
    (ckpt / "huggingface").mkdir(parents=True)
    (ckpt / "huggingface" / "model.safetensors").write_bytes(b"w")
    (ckpt / "optim_world_size_2_rank_0.pt").write_bytes(b"o" * 4096)
    # The guard scans everything within maxdepth 3 of the run directory, so
    # backdate the whole tree -- on a real box those timestamps come from the
    # trainer's last write, and prune touches only actor/.
    old = 10_000_000          # well outside any freshness window
    run_dir = tmp_path / "checkpoints" / "STEER-F" / "arun"
    for path in sorted(run_dir.rglob("*"), reverse=True):
        os.utime(path, (old, old))
    os.utime(run_dir, (old, old))

    # A finished log, or the completion guard below refuses before we get to
    # test the timestamp at all.
    logs = tmp_path / "logs" / "experiments"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "train-arun.log").write_text(
        "".join(f"step:{i} - global_seqlen: 100\n" for i in range(1, 111)))
    (tmp_path / "run").symlink_to(ROOT / "run")

    p = subprocess.run(["bash", str(ROOT / "run" / "hf_backup.sh"), "arun"],
                       capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
                            "STEER_ROOT": str(tmp_path), "PRUNE": "1"})
    assert p.returncode == 0, p.stdout + p.stderr
    assert not (ckpt / "optim_world_size_2_rank_0.pt").exists(), "prune did nothing"
    assert (ckpt / "huggingface" / "model.safetensors").exists(), "prune ate the eval copy"
    assert abs(ckpt.stat().st_mtime - old) < 2, (
        "prune left a fresh mtime, so the liveness guard will refuse this run")


def _ckpt(tmp_path, run, step, *, with_optim=True, backdate=True):
    d = tmp_path / "checkpoints" / "STEER-F" / run / f"global_step_{step}" / "actor"
    (d / "huggingface").mkdir(parents=True)
    (d / "huggingface" / "model.safetensors").write_bytes(b"w")
    if with_optim:
        (d / "optim_world_size_2_rank_0.pt").write_bytes(b"o" * 4096)
    if backdate:
        old = 10_000_000
        root = tmp_path / "checkpoints" / "STEER-F" / run
        for path in sorted(root.rglob("*"), reverse=True):
            os.utime(path, (old, old))
        os.utime(root, (old, old))
    return d


def _log(tmp_path, run, last_step):
    d = tmp_path / "logs" / "experiments"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"train-{run}.log").write_text(
        "".join(f"step:{i} - global_seqlen: 100\n" for i in range(1, last_step + 1)))


def _prune(tmp_path, run, **env):
    # hf_backup.sh:71 sources ${STEER_ROOT}/run/_arms.sh, and train_log_done
    # lives there. Without it the fixture would exercise a path the real box
    # never takes.
    link = tmp_path / "run"
    if not link.exists():
        link.symlink_to(ROOT / "run")
    e = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
         "STEER_ROOT": str(tmp_path), "PRUNE": "1"}
    e.update(env)
    return subprocess.run(["bash", str(ROOT / "run" / "hf_backup.sh"), run],
                          capture_output=True, text=True, env=e)


def test_prune_refuses_an_unfinished_run(tmp_path):
    """2026-09-21, and it cost eighteen hours.

    live_guard only WARNS when a run never reached STEPS, which is right for an
    upload -- a crashed run's checkpoints are worth keeping and uploading takes
    nothing away. Prune is the opposite: what it deletes is precisely what lets
    an unfinished run resume. A 40/110 run was pruned by a loop over every run,
    and step 40 stopped being a resume point.
    """
    d = _ckpt(tmp_path, "arun", 40)
    _log(tmp_path, "arun", 40)                      # 40 of 110
    p = _prune(tmp_path, "arun")
    assert p.returncode == 1, p.stdout + p.stderr
    assert "never reached step 110" in p.stdout + p.stderr
    assert (d / "optim_world_size_2_rank_0.pt").exists(), "prune destroyed the resume point"


def test_prune_still_works_on_a_finished_run(tmp_path):
    d = _ckpt(tmp_path, "brun", 110)
    _log(tmp_path, "brun", 110)
    p = _prune(tmp_path, "brun")
    assert p.returncode == 0, p.stdout + p.stderr
    assert not (d / "optim_world_size_2_rank_0.pt").exists()
    assert (d / "huggingface" / "model.safetensors").exists()


def test_force_still_lets_you_prune_an_unfinished_run(tmp_path):
    d = _ckpt(tmp_path, "crun", 40)
    _log(tmp_path, "crun", 40)
    p = _prune(tmp_path, "crun", FORCE="1")
    assert p.returncode == 0, p.stdout + p.stderr
    assert not (d / "optim_world_size_2_rank_0.pt").exists()


def test_prune_refuses_when_it_cannot_tell(tmp_path):
    """No _arms.sh means no train_log_done, and a `command -v` guard would skip
    the check entirely -- on the one path whose job is deletion. Not knowing is
    a reason to stop."""
    d = _ckpt(tmp_path, "drun", 40)
    _log(tmp_path, "drun", 40)
    p = subprocess.run(["bash", str(ROOT / "run" / "hf_backup.sh"), "drun"],
                       capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
                            "STEER_ROOT": str(tmp_path), "PRUNE": "1"})
    assert p.returncode == 1, p.stdout + p.stderr
    assert "cannot tell" in p.stdout + p.stderr
    assert (d / "optim_world_size_2_rank_0.pt").exists()


# ------------------------------- a huggingface/ that holds no model at all
def _weightless(box, run, step):
    """Strip the weights but leave the directory, config and tokenizer.

    That is what verl produces when `hf_model` is not in save_contents:
    fsdp_checkpoint_manager.py:228 writes huggingface/ with the tokenizer and
    config "no matter whether huggingface model is requested to be saved or
    not", and :263 writes the weights only when it is. Four of the five runs
    pushed on 2026-09-22 recorded ['model','optimizer','extra'] and are exactly
    this shape, which every `-d` test in this repo accepted.
    """
    hf = box / "checkpoints" / "STEER-F" / run / f"global_step_{step}" / "actor" / "huggingface"
    (hf / "model.safetensors").unlink()
    (hf / "config.json").write_text("{}")
    (hf / "tokenizer.json").write_text("{}")
    old = time.time() - 3 * 86400
    root = box / "checkpoints" / "STEER-F" / run
    for path in sorted(root.rglob("*"), reverse=True):
        os.utime(path, (old, old))
    os.utime(root, (old, old))


def test_a_weightless_checkpoint_is_not_uploaded(tmp_path):
    """It would reach the Hub and then verify, because both sides are config."""
    box = build_box(tmp_path, runs=("done-run",), steps=(110,))
    _weightless(box, "done-run", 110)
    p = run(box, "done-run", tmp_path)
    assert "no weights" in p.stdout or "holds no weights" in p.stdout, p.stdout
    assert calls(tmp_path) == [], \
        f"the stub CLI was called for a weightless checkpoint: {calls(tmp_path)}"


def test_prune_refuses_a_weightless_checkpoint(tmp_path):
    """The dangerous half: the shards beside it are the only copy.

    PRUNE keeps huggingface/ and deletes everything else in actor/. On a run
    that saved without `hf_model` that deletes the weights and keeps a config.
    """
    box = build_box(tmp_path, runs=("done-run",), steps=(110,))
    actor = box / "checkpoints" / "STEER-F" / "done-run" / "global_step_110" / "actor"
    (actor / "model_world_size_2_rank_0.pt").write_text("the only weights")
    _weightless(box, "done-run", 110)

    p = run(box, "done-run", tmp_path, PRUNE=1)
    assert "REFUSE" in p.stdout, p.stdout
    assert (actor / "model_world_size_2_rank_0.pt").is_file(), \
        "PRUNE deleted the only copy of the weights"
    assert (actor / "optim_world_size_2_rank_0.pt").is_file()


def test_a_real_checkpoint_is_still_uploaded(tmp_path):
    """Regression guard: the weight test must not reject a healthy run."""
    box = build_box(tmp_path, runs=("done-run",), steps=(110,))
    p = run(box, "done-run", tmp_path)
    # The exit code is not the signal here: verification imports
    # huggingface_hub, which this container does not have, so it fails after a
    # successful upload. What this test guards is that the upload happened at
    # all -- the same thing test_a_quiet_finished_run_is_uploaded asserts.
    got = calls(tmp_path)
    assert got, "a real checkpoint stopped being uploaded"
    assert "done-run/global_step_110" in got[0]
    assert "no weights" not in p.stdout
