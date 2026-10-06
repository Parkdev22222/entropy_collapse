"""scripts/sibling_spread.py: the quantity and the verdict rule of the plan."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sibling_spread", ROOT / "scripts" / "sibling_spread.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def line(step, a, nb, length):
    return (f"(TaskRunner pid=1) step:{step} - steerf/a_h_std:{a} - "
            f"steerf/n_branch_points:{nb} - response_length/mean:{length} - x:1\n")


def test_spread_is_rms_ah_per_branch_point():
    # 4096 responses x 1000 tokens, 4096 branch points: RMS per point = std * sqrt(1000)
    s = mod.per_step(line(50, 0.01, 4096, 1000))
    assert abs(s[50] - 0.01 * 1000 ** 0.5) < 1e-9


def test_window_and_last_line_wins():
    text = line(39, 1, 1, 1) + line(40, 0.002, 4096, 1000) + line(40, 0.004, 4096, 1000) \
        + line(111, 9, 1, 1)
    m, n = mod.window_mean(mod.per_step(text))
    assert n == 1 and abs(m - 0.004 * 1000 ** 0.5) < 1e-9


def test_verdict_rule():
    st = lambda mean, pos: {"mean": mean, "pos": pos}
    assert mod.verdict({"u": st(.1, 3), "p": st(.05, 2)}) == "consistent with widening"
    assert mod.verdict({"u": st(.1, 3), "p": st(.05, 1)}) == "inconclusive"
    assert mod.verdict({"u": st(-.1, 0), "p": st(-.05, 1)}) == "against"
    assert mod.verdict({"u": st(.1, 3), "p": st(-.06, 1)}) == "inconclusive"
    # amendment 1: under 5% is not distinguishable, whatever the signs
    assert mod.verdict({"u": st(.1, 3), "p": st(.04, 3)}) == "inconclusive"
    assert mod.verdict({"u": st(-.1, 0), "p": st(-.03, 0)}) == "inconclusive"
