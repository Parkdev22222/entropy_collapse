"""CPU tests for the manipulation check's statistics (scripts/manip_check.py)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import manip_check as M  # noqa: E402


def test_hi_lo_follows_entropy_to_go_not_token_order():
    togo = {"p": [[1.0, 1.0], [5.0, 5.0]]}        # token b is the richer sibling
    base = {"p": [-1.0, -2.0]}
    model = {"p": [-1.1, -1.5]}                    # a moved .1, b moved .5
    [r] = M.point_stats(togo, base, model)
    assert abs(r["D"] - (0.5 - 0.1)) < 1e-9
    assert abs(r["S"] - (0.5 - (-0.1))) < 1e-9
    assert r["gap"] == 4.0


def test_ties_and_missing_points_are_dropped():
    togo = {"tie": [[2.0], [2.0]], "gone": [[1.0], [3.0]], "ok": [[3.0], [1.0]]}
    base = {"tie": [0, 0], "ok": [0, 0]}
    model = {"tie": [0, 0], "ok": [0.2, 0.1], "gone": [0, 0]}
    rows = M.point_stats(togo, base, model)
    assert [r["id"] for r in rows] == ["ok"]


def test_verdict_needs_both_contrasts_positive_on_every_seed():
    assert M.verdict({"signed-uniform": [.1, .2, .3],
                      "signed-permuted": [.1, .1, .1]}) == "passed"
    assert M.verdict({"signed-uniform": [.1, -.2, .3],
                      "signed-permuted": [.1, .1, .1]}) == "not passed"
    assert M.verdict({"signed-uniform": [.1, .2, .3],
                      "signed-permuted": []}) == "not passed"


def _write(probe, label, lp, step=None):
    (probe / "scores").mkdir(parents=True, exist_ok=True)
    name = "base" if label == "base" else label.replace(":", "-s")
    (probe / "scores" / f"{name}.json").write_text(
        json.dumps({"label": label, "model": "x", "step": step, "lp": lp}))


def test_analyze_pairs_within_seed(tmp_path):
    togo = {f"p{i}": [[3.0], [1.0]] for i in range(4)}   # token a richer
    (tmp_path / "togo.json").write_text(json.dumps(togo))
    base = {k: [0.0, 0.0] for k in togo}
    _write(tmp_path, "base", base)
    for seed in (1, 3, 4):
        # signed moves the richer sibling more; the controls move both equally
        _write(tmp_path, f"signed:{seed}", {k: [0.3, 0.1] for k in togo}, 100)
        _write(tmp_path, f"uniform:{seed}", {k: [0.2, 0.2] for k in togo}, 90)
        _write(tmp_path, f"permuted:{seed}", {k: [0.1, 0.1] for k in togo}, 80)
    res = M.analyze(tmp_path)
    assert res["verdict"] == "passed"
    c = res["contrasts"]["signed-uniform:all:D"]
    assert c["n"] == 3 and abs(c["mean"] - 0.2) < 1e-9
    assert res["per_model"]["signed:1"]["step"] == 100
