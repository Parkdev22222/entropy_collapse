"""The gate has to know what the queues actually open.

Four runs have now been lost to the same shape: something every arm needs, that
no gate looked at, failing after the queue had already said OK.

    2026-09-13  opentelemetry -> ray      ray.init() timed out
    2026-09-13  flash_attn ABI            worker init, 1-2 minutes in
    2026-09-14  word2number               step-0 validation, minutes in
    2026-09-23  tensorboard               trainer init, before step 1

The last one is what these tests pin.  Every arm sets
``trainer.logger=['console','tensorboard']`` and verl opens the backend in
Tracking before step 1, so a box without it loses one run per queue pass while
the training log reads as if that single arm had failed.

The second half matters as much as the first.  run_steerf.sh:236 hardcodes
``trainer.logger="['console','wandb']"`` and wandb has never had to be
installed, because the queue appends _arms.sh's override after it and hydra
keeps the last value.  A check that unioned every ``trainer.logger=`` in run/
would demand wandb -- and ``pip install wandb`` is the command that pulled
opentelemetry 1.26 -> 1.44 on 2026-09-13 and broke vllm.  So a launcher default
must stay a report and never become a pip command.
"""
import importlib.util
import subprocess
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TB = ("torch.utils.tensorboard", "SummaryWriter", "tensorboard")


def check_deps():
    spec = importlib.util.spec_from_file_location(
        "_check_deps_under_test", ROOT / "run" / "_check_deps.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tree(tmp_path: Path, files: dict[str, str]) -> Path:
    (tmp_path / "run").mkdir(exist_ok=True)
    for name, body in files.items():
        (tmp_path / "run" / name).write_text(body)
    return tmp_path


# --- what the checker requires ---------------------------------------------

def test_the_repo_requires_tensorboard():
    """The whole point. Before this the list named 18 modules and not this one."""
    required, _ = check_deps().logger_spec(ROOT)
    assert TB in required


def test_it_checks_the_import_verl_takes_not_the_pip_name():
    """`import tensorboard` can succeed against a partial install.

    torch.utils.tensorboard is what raises -- the word2number.w2n lesson.
    """
    required, _ = check_deps().logger_spec(ROOT)
    assert [e for e in required if e[0] == "tensorboard"] == []


def test_the_repo_does_not_require_wandb():
    """run_steerf.sh:236 names wandb. Installing it is the 09-13 accident."""
    required, _ = check_deps().logger_spec(ROOT)
    assert all(e[2] != "wandb" for e in required)


def test_tensorboard_reaches_the_need_list():
    """End to end: the pip name the box is told to install."""
    p = subprocess.run(["python3", str(ROOT / "run" / "_check_deps.py")],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert "tensorboard" in p.stdout.split()


# --- derived, not hardcoded -------------------------------------------------

def test_changing_the_queue_override_changes_what_is_required(tmp_path):
    """Swap the backend _arms.sh passes and the requirement follows it.

    This is the property that keeps the fix from rotting: the list lives in the
    line the queues actually apply, not in a copy kept in the checker.
    """
    root = tree(tmp_path, {"_arms.sh": "\"trainer.logger=['console','wandb']\" \\\n"})
    required, _ = check_deps().logger_spec(root)
    assert [e[2] for e in required] == ["wandb"]
    assert TB not in required


def test_console_asks_for_nothing(tmp_path):
    root = tree(tmp_path, {"_arms.sh": "\"trainer.logger=['console']\" \\\n"})
    required, reported = check_deps().logger_spec(root)
    assert required == [] and reported == []


def test_an_unknown_backend_is_reported_not_installed(tmp_path):
    root = tree(tmp_path, {"_arms.sh": "\"trainer.logger=['console','mlflow']\" \\\n"})
    required, reported = check_deps().logger_spec(root)
    assert required == []
    assert any("mlflow" in name for _, name, _ in reported)


# --- the 09-13 regression detector ------------------------------------------

def test_a_launcher_default_is_reported_and_never_required(tmp_path):
    """The queue's override wins, so the launcher's wandb must not be installed."""
    root = tree(tmp_path, {
        "_arms.sh": "\"trainer.logger=['console','tensorboard']\" \\\n",
        "run_steerf.sh": "    trainer.logger=\"['console','wandb']\" \\\n",
    })
    required, reported = check_deps().logger_spec(root)
    assert required == [TB]
    assert any(where.endswith("run_steerf.sh:1") and name == "wandb"
               for where, name, _ in reported)


def test_the_quoted_spelling_is_seen(tmp_path):
    """run_steerf.sh quotes the whole value; the queues do not. Both parse."""
    root = tree(tmp_path, {"_arms.sh": "trainer.logger=\"['console','wandb']\"\n"})
    required, _ = check_deps().logger_spec(root)
    assert [e[2] for e in required] == ["wandb"]


# --- the gate itself --------------------------------------------------------

def stub_python(tmp_path: Path, tensorboard: bool) -> dict:
    """A python3 that answers each of env_preflight's probes in turn."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    tb = "exit 0" if tensorboard else \
        "echo \"ModuleNotFoundError: No module named 'tensorboard'\" >&2; exit 1"
    (bin_dir / "python3").write_text(textwrap.dedent(f"""\
        #!/bin/sh
        case "$*" in
          *_check_steer_f*)          exit 0 ;;
          *reward_score*)            exit 0 ;;
          *torch.utils.tensorboard*) {tb} ;;
          *ray.init*)                exit 0 ;;
          *flash_attn*)              exit 0 ;;
          *)                         exit 0 ;;
        esac
    """))
    (bin_dir / "python3").chmod(0o755)
    return {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": "/tmp"}


def preflight(tmp_path: Path, tensorboard: bool):
    return subprocess.run(
        ["bash", "-c", f'. "{ROOT}/run/_arms.sh"\nenv_preflight "{ROOT}"'],
        capture_output=True, text=True, cwd=str(ROOT),
        env=stub_python(tmp_path, tensorboard))


def test_the_gate_refuses_without_the_logger(tmp_path):
    p = preflight(tmp_path, tensorboard=False)
    assert p.returncode == 1
    assert "logger backend" in p.stdout


def test_the_refusal_names_the_command(tmp_path):
    """The point of the message: the reader should not have to work it out."""
    p = preflight(tmp_path, tensorboard=False)
    assert "pip install tensorboard" in p.stdout
    assert "bash run/setup_env.sh" in p.stdout


def test_the_gate_passes_with_the_logger(tmp_path):
    p = preflight(tmp_path, tensorboard=True)
    assert p.returncode == 0
    assert "logger backend" not in p.stdout


def test_setup_env_verifies_it_too():
    """Section 6 lists the imports that have broken before. This is one now."""
    body = (ROOT / "run" / "setup_env.sh").read_text()
    assert "from torch.utils.tensorboard import SummaryWriter" in body
