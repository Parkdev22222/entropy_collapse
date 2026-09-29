"""verl/trainer/ppo/ckpt_policy.py: keep 50/100/last and the best, nothing else."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ckpt_policy", ROOT / "verl/trainer/ppo/ckpt_policy.py")
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)


def simulate(curve: dict, final=110, every=50, freq=10):
    """Run the policy over a validation curve; return the steps left on disk."""
    disk, best_v, best_s = set(), float("-inf"), None
    for step in range(freq, final + 1, freq):
        p = cp.plan(step, final, curve.get(step), best_v, best_s, every)
        if p["save"]:
            disk.add(step)
        disk -= set(p["delete"])
        best_v, best_s = p["best_value"], p["best_step"]
    return disk, best_s


def test_best_in_the_middle_is_kept_with_50_100_110():
    curve = {s: 0.10 for s in range(10, 111, 10)}
    curve[70] = 0.20
    disk, best = simulate(curve)
    assert best == 70
    assert disk == {50, 70, 100, 110}


def test_superseded_bests_are_deleted_but_never_a_protected_step():
    curve = {10: .1, 20: .2, 30: .3, 40: .35, 50: .4, 60: .1, 70: .1,
             80: .45, 90: .1, 100: .5, 110: .1}
    disk, best = simulate(curve)
    assert best == 100
    assert disk == {50, 100, 110}            # 20/30/40/80 were bests once, then removed


def test_best_on_the_last_step():
    curve = {s: s / 1000 for s in range(10, 111, 10)}
    disk, best = simulate(curve)
    assert best == 110 and disk == {50, 100, 110}


def test_steps_without_validation_or_nan_never_become_best():
    p = cp.plan(20, 110, None, float("-inf"), None, 50)
    assert not p["save"] and not p["new_best"]
    p = cp.plan(20, 110, float("nan"), float("-inf"), None, 50)
    assert not p["save"]


def test_long_run_keeps_every_50():
    disk, _ = simulate({s: 0.0 for s in range(10, 201, 10)}, final=200)
    assert {50, 100, 150, 200} <= disk


@pytest.mark.parametrize("keep,save,test", [(0, 10, 10), (55, 10, 10), (50, 10, 20), (50, 0, 10)])
def test_validate_rejects_settings_that_would_miss_a_point(keep, save, test):
    with pytest.raises(ValueError):
        cp.validate(keep, save, test)


def test_validate_accepts_the_campaign_settings():
    cp.validate(50, 10, 10)


def test_trainer_and_launchers_are_wired_to_the_policy():
    rt = (ROOT / "verl/trainer/ppo/ray_trainer.py").read_text()
    body = rt[rt.index("def _save_checkpoint"):rt.index("def _load_checkpoint")]
    assert "ckpt_policy.plan(" in body and "ckpt_policy.validate(" in body
    # superseded bests are removed only after the new checkpoint and its json
    assert body.index("json.dump(best_checkpoint_info") < body.index("for old_step in superseded")
    assert "keep_every is set but save_after" in body
    assert "++trainer.keep_every=${KEEP_EVERY}" in (ROOT / "run/run_steerf.sh").read_text()
    assert "++trainer.keep_every=${KEEP_EVERY}" in (ROOT / "run/run_grpo.sh").read_text()
