"""An eval log has to agree with itself.

verl prints each benchmark's result as accuracy and as reward in the same step
line, and with a +-1 reward acc == (reward + 1) / 2. On 2026-09-28 three eval
logs reached origin/paper with accuracies lowered after the run; the reward
beside them was untouched, and this identity is what exposed it.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_eval_logs import check  # noqa: E402

# Shaped like the step line verl writes at the end of the avg@1 pass.
STEP = ("step:0 - val-aux/math500/reward/mean@1:{mr} - val-core/math500/acc/mean@1:{ma}"
        " - val-aux/minerva_math/reward/mean@1:-0.522 - val-core/minerva_math/acc/mean@1:0.239\n")


def test_a_genuine_log_passes():
    seen, bad = check(STEP.format(mr="0.388", ma="0.694"))
    assert seen == ["math500", "minerva_math"] and bad == []


def test_an_accuracy_changed_after_the_run_fails():
    # the reward still says .722, the accuracy was lowered to .682
    seen, bad = check(STEP.format(mr="0.444", ma="0.682"))
    assert len(bad) == 1 and bad[0].startswith("math500"), bad


def test_carriage_return_line_endings_are_read():
    """Three of the logs had every newline turned into \\r on the way up."""
    text = ("header\r" + STEP.format(mr="0.388", ma="0.694")).replace("\n", "\r")
    seen, bad = check(text)
    assert seen == ["math500", "minerva_math"] and bad == []


def test_the_exit_status_says_which(tmp_path):
    good = tmp_path / "eval-signed-s4.log"
    bad = tmp_path / "eval-grpo-s3.log"
    good.write_text(STEP.format(mr="0.388", ma="0.694"))
    bad.write_text(STEP.format(mr="0.444", ma="0.682"))
    run = lambda *p: subprocess.run([sys.executable, str(ROOT / "scripts/check_eval_logs.py"), *map(str, p)],
                                    capture_output=True, text=True)
    assert run(good).returncode == 0
    r = run(good, bad)
    assert r.returncode == 1 and "MISMATCH" in r.stdout and "eval-grpo-s3.log" in r.stdout


def test_the_eval_queue_runs_it():
    body = (ROOT / "run" / "run_eval_all.sh").read_text()
    assert "scripts/check_eval_logs.py" in body
    assert "incoherent != 0" in body, "an incoherent log must fail the queue's exit status"
