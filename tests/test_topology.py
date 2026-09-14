"""One box, one topology -- and a status view that agrees with the queues.

run_uniform_ablation.sh hardcodes N_GPUS=2 and EXPORTS it without sourcing
_gpu_defaults.sh, so on a four-card box the six follow-up arms that go through
it took two cards while the three that go through run_steerf.sh took four --
inside one table, with xclip-signed and xclip-steer, a directly compared pair,
landing on different topologies. These tests pin the fix.
"""
import subprocess
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def arms(script: str, env: dict | None = None, cwd: Path | None = None):
    """Source _arms.sh and run a snippet against it."""
    return subprocess.run(
        ["bash", "-c", f'. "{ROOT}/run/_arms.sh"\n{script}'],
        capture_output=True, text=True, cwd=str(cwd or ROOT),
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": "/tmp", **(env or {})})


def fake_torch(tmp_path: Path, count: int) -> dict:
    """A python3 earlier on PATH that reports `count` CUDA devices."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "python3").write_text(textwrap.dedent(f"""\
        #!/bin/sh
        # only answers the device-count probe; anything else is not our business
        case "$*" in *device_count*) echo {count} ;; *) exit 1 ;; esac
    """))
    (bin_dir / "python3").chmod(0o755)
    return {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": "/tmp"}


def topology(tmp_path: Path, count: int, **over):
    env = fake_torch(tmp_path, count)
    env.update(over)
    p = arms('gpu_topology; echo "${N_GPUS} ${TP_SIZE}"', env=env)
    return p.stdout.strip().splitlines()[-1]


def test_four_cards_use_all_four(tmp_path):
    """The H100. Before this, six of nine arms silently took two."""
    assert topology(tmp_path, 4) == "4 4"


def test_two_cards(tmp_path):
    """The A100, where the old hardcoded 2/2 happened to be right. Unchanged."""
    assert topology(tmp_path, 2) == "2 2"


def test_eight_cards_keep_tp_four(tmp_path):
    """TP is the largest of 4, 2, 1 that divides -- the paper's own rule."""
    assert topology(tmp_path, 8) == "8 4"


def test_awkward_count_falls_back_to_tp_one(tmp_path):
    assert topology(tmp_path, 3) == "3 1"


def test_no_gpu_counts_as_one_never_as_unset(tmp_path):
    """An empty N_GPUS reaches the launchers' 8-GPU fallback and dies in verl's
    resource-pool check, so zero must become one rather than stay empty."""
    assert topology(tmp_path, 0) == "1 1"


def test_explicit_environment_wins(tmp_path):
    """The launch command stays the authority -- that is how the H100 is told
    to use four cards even before anyone trusts the detection."""
    assert topology(tmp_path, 2, N_GPUS="4", TP_SIZE="4") == "4 4"


def test_guard_refuses_a_tp_that_does_not_divide(tmp_path):
    env = fake_torch(tmp_path, 4)
    env.update({"N_GPUS": "4", "TP_SIZE": "3"})
    p = arms("topology_guard", env=env)
    assert p.returncode != 0
    assert "REFUSE" in p.stdout + p.stderr


def test_guard_passes_and_announces(tmp_path):
    p = arms("topology_guard", env=fake_torch(tmp_path, 4))
    assert p.returncode == 0
    assert "N_GPUS=4 TP_SIZE=4" in p.stdout


# --------------------------------------------------------------- run_status.sh
def write_log(d: Path, run: str, steps: int, *, gpus: str | None = None) -> Path:
    lines = [f"step:{i} - global_seqlen:1 - perf/time_per_step:1489.3"
             for i in range(1, steps + 1)]
    if gpus:
        lines.append(f" seed / gpus    1 / {gpus}")
    f = d / f"train-{run}.log"
    f.write_text("\n".join(lines) + "\n")
    return f


def status(log_dir: Path, **over):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": "/tmp",
           "LOG_DIR": str(log_dir), "SEEDS": "1", **over}
    return subprocess.run(["bash", str(ROOT / "run" / "run_status.sh")],
                          capture_output=True, text=True, cwd=str(ROOT), env=env)


def test_status_survives_an_empty_box(tmp_path):
    """The first thing anyone types on a fresh pod. It must not blow up when
    there are no logs, no GPUs and nothing running."""
    p = status(tmp_path)
    assert p.returncode == 0, p.stderr
    assert "is anything training?" in p.stdout


def test_status_separates_done_running_and_never_started(tmp_path):
    write_log(tmp_path, "steer-f-Qwen2.5-Math-1.5B-s1-tree-rollout-lam0", 110)
    write_log(tmp_path, "steer-f-Qwen2.5-Math-1.5B-s1-tree-rollout-lam0.1", 37)
    (tmp_path / "train-steer-f-Qwen2.5-Math-1.5B-s1-tree-rollout-lam0.5.log"
     ).write_text("Traceback ...\nModuleNotFoundError: No module named 'word2number'\n")
    out = status(tmp_path, ARMS="lam0-tree lam0.1 lam0.5").stdout
    assert "110/110" in out and "done" in out
    assert "37/110" in out
    assert "NEVER STARTED" in out
    # a run with no step 1 is a startup failure, not a training crash
    assert "STARTUP FAILURE" in out


def test_status_flags_mixed_topologies(tmp_path):
    """The finding itself: two cards for one arm, four for the next."""
    write_log(tmp_path, "steer-f-Qwen2.5-Math-1.5B-s1-tree-rollout-lam0", 5,
              gpus="2 (tp=2)")
    write_log(tmp_path, "steer-Qwen2.5-Math-1.5B-s1-xclip", 5, gpus="4 (tp=4)")
    out = status(tmp_path, ARMS="lam0-tree xclip-steer").stdout
    assert "more than one topology" in out


def test_status_is_quiet_when_topology_agrees(tmp_path):
    write_log(tmp_path, "steer-f-Qwen2.5-Math-1.5B-s1-tree-rollout-lam0", 5,
              gpus="4 (tp=4)")
    write_log(tmp_path, "steer-Qwen2.5-Math-1.5B-s1-xclip", 5, gpus="4 (tp=4)")
    out = status(tmp_path, ARMS="lam0-tree xclip-steer").stdout
    assert "more than one topology" not in out
    assert "4 (tp=4)" in out
