"""The monitor named `branch_entropy` does not split on branch points.

`steer_f/monitors.py:branch_token_entropy` selects a FIXED top decile of `A_H`
(`top_frac=0.1`, never overridden at its call site in
`verl/trainer/ppo/core_algos.py`). The positions the correction reaches are
`branch_corr_frac`, measured at ~.012 across the tree arms -- an eighth as many.
`A_H` is exactly zero wherever a rollout has become its own only sibling, so the
remainder of that decile is `torch.topk`'s tie-break among zeros, and that
tie-break takes a contiguous index slab rather than a random sample.

A draft of the manuscript read `branch_entropy_gap` as a branch-point effect. It
is mostly a positional one. `run/instrument_campaign.sh --apply` adds keys that
split on `a_h != 0` instead -- the predicate the correction itself uses -- as new
keys, leaving `branch_entropy` untouched so finished seeds stay comparable.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run" / "instrument_campaign.sh"
DONOR = "origin/paper"

torch = pytest.importorskip("torch")


def _donor(path):
    r = subprocess.run(["git", "show", f"{DONOR}:{path}"], cwd=ROOT,
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"{DONOR} is not in this checkout")
    return r.stdout


def _pod(tmp_path, monitors_src):
    """A tree the script can patch: two launchers and one steer_f package."""
    (tmp_path / "run").mkdir()
    (tmp_path / "steer_f").mkdir()
    shutil.copy(SCRIPT, tmp_path / "run" / SCRIPT.name)
    (tmp_path / "run" / "run_steerf.sh").write_text(
        "#!/bin/bash\n++trainer.save_best_only=False \\\n")
    (tmp_path / "run" / "run_grpo.sh").write_text(
        "#!/bin/bash\nARGS=(\n  ++trainer.save_best_only=False\n)\n")
    (tmp_path / "steer_f" / "__init__.py").write_text("")
    (tmp_path / "steer_f" / "monitors.py").write_text(monitors_src)
    return tmp_path


def _run(pod, mode):
    env = dict(os.environ, PYTHONPATH=str(pod))
    return subprocess.run(["bash", f"run/{SCRIPT.name}", mode], cwd=pod,
                          env=env, capture_output=True, text=True)


# ------------------------------------------------- the defect, as arithmetic
def test_the_top_decile_is_mostly_not_the_support():
    """The regression detector. If this ever fails the metric became honest."""
    n = 10_000
    a = torch.zeros(n)
    idx = torch.randperm(n)[:120]          # 1.2%, matching branch_corr_frac
    a[idx[:60]] = torch.rand(60) * 0.5
    a[idx[60:]] = -torch.rand(60) * 0.5

    k = int(0.1 * n)                        # top_frac=0.1, as called
    sel = torch.zeros(n, dtype=torch.bool)
    sel[torch.topk(a, k).indices] = True

    zeros_in_bucket = int(((a == 0) & sel).sum())
    assert zeros_in_bucket / k > 0.9, (
        "the top decile is supposed to be dominated by tie-broken zeros; "
        f"only {zeros_in_bucket}/{k} were")

    # and the tie-break is not a random sample of those zeros: a random one
    # would sit near the middle of the index range.
    picked = torch.nonzero((a == 0) & sel).flatten().float()
    assert abs(float(picked.mean()) - n / 2) > n / 20, (
        "the tie-break looked unbiased here; the positional confound this test "
        "pins may have gone away")


# ------------------------------------------------------------ the applier
def test_check_then_apply_then_check_is_idempotent(tmp_path):
    pod = _pod(tmp_path, _donor("steer_f/monitors.py"))

    first = _run(pod, "--check")
    assert "branch_token_entropy splits on the top decile only" in first.stdout

    applied = _run(pod, "--apply")
    assert "support-restricted entropy" in applied.stdout
    assert "Nothing to do" in applied.stdout, applied.stdout

    again = _run(pod, "--check")
    assert "support-restricted entropy already logged" in again.stdout
    assert "[needed] branch_token_entropy" not in again.stdout


def test_branch_entropy_is_left_alone(tmp_path):
    """Finished seeds logged it; redefining it would make them incomparable."""
    src = _donor("steer_f/monitors.py")
    pod = _pod(tmp_path, src)
    _run(pod, "--apply")
    after = (pod / "steer_f" / "monitors.py").read_text()
    for key in ('"steerf/branch_entropy"', '"steerf/nonbranch_entropy"',
                '"steerf/branch_entropy_gap"', "top_frac: float = 0.1"):
        assert key in after, f"{key} was disturbed"


def test_the_other_steer_f_lineage_is_skipped_not_patched(tmp_path):
    """This branch carries a different steer_f; patching it changes nothing.

    run/bootstrap_pod.sh takes steer_f from the donor even where HEAD tracks the
    same path, so a patch applied to HEAD's copy would be invisible to training.
    """
    src = (ROOT / "steer_f" / "monitors.py").read_text()
    assert "_top_k_selection" not in src, "the two lineages have converged"
    pod = _pod(tmp_path, src)

    out = _run(pod, "--check")
    assert "the other lineage" in out.stdout
    assert (pod / "steer_f" / "monitors.py").read_text() == src
