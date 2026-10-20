"""CPU tests for the two-channel decomposition (scripts/channel_decomp.py)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import channel_decomp as C  # noqa: E402


def _write(out, samples, under, sums, model=None):
    d = out / "ent"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{samples}__{under}.json").write_text(json.dumps(
        {"samples": samples, "under": under, "model": model or under,
         "samples_model": samples, "sums": sums}))


def test_terms_add_up_exactly_in_both_orderings():
    E = {("a", "a"): [100.0, 80.0], ("a", "b"): [70.0, 60.0],
         ("b", "a"): [90.0, 50.0], ("b", "b"): [60.0, 40.0]}
    t = C.interval_terms(E, "a", "b")
    for i in range(2):
        assert abs(t["L"][i] + t["V"][i] - t["dH"][i]) < 1e-9
        assert abs(t["L_b_states"][i] + (E[("b", "a")][i] - E[("a", "a")][i])
                   - t["dH"][i]) < 1e-9
    # prompt 0: dH -40; b-states L -30, V -10; a-states L -30, V -10
    assert t["dH"][0] == -40 and t["L"][0] == -30 and t["V"][0] == -10


def test_pure_visitation_change_has_no_local_term(tmp_path):
    """b visits low-entropy states more, but entropy at every state is unchanged:
    a and b score any response identically, so L = 0 and V = dH."""
    n = 40
    _write(tmp_path, "base", "base", [[100.0] * 4] * n)
    _write(tmp_path, "base", "grpo100", [[100.0] * 4] * n)     # same conditionals
    _write(tmp_path, "grpo100", "base", [[60.0, 62.0, 58.0, 60.0]] * n)
    _write(tmp_path, "grpo100", "grpo100", [[60.0, 62.0, 58.0, 60.0]] * n)
    res = C.analyze(tmp_path, n_boot=500)
    r = res["intervals"]["base->grpo100"]
    assert abs(r["L"]["mean"]) < 1e-9 and abs(r["V"]["mean"] + 40) < 1e-9
    assert abs(r["V_share"]["mean"] - 1.0) < 1e-9
    assert res["q1_verdict"] == "visitation share established"


def test_pure_local_change_is_not_called_visitation(tmp_path):
    """Same responses visited, every conditional sharper: V = 0."""
    n = 40
    _write(tmp_path, "base", "base", [[100.0, 104.0, 96.0, 100.0]] * n)
    _write(tmp_path, "base", "grpo100", [[70.0, 74.0, 66.0, 70.0]] * n)
    _write(tmp_path, "grpo100", "base", [[100.0, 104.0, 96.0, 100.0]] * n)
    _write(tmp_path, "grpo100", "grpo100", [[70.0, 74.0, 66.0, 70.0]] * n)
    res = C.analyze(tmp_path, n_boot=500)
    r = res["intervals"]["base->grpo100"]
    assert abs(r["V"]["mean"]) < 1e-9 and abs(r["L"]["mean"] + 30) < 1e-9
    assert res["q1_verdict"] == "not established"


def test_arm_contrast_is_paired_on_prompts(tmp_path):
    n = 40
    for arm, (loc, vis) in {"grpo100": (-30.0, -10.0), "perm100": (-10.0, -10.0)}.items():
        # base samples: S_base = 100, S_arm = 100 + loc ; arm samples: S_base = 100 + vis
        _write(tmp_path, "base", "base", [[100.0 + (i % 3)] * 4 for i in range(n)])
        _write(tmp_path, "base", arm, [[100.0 + loc + (i % 3)] * 4 for i in range(n)])
        _write(tmp_path, arm, "base", [[100.0 + vis + (i % 3)] * 4 for i in range(n)])
        _write(tmp_path, arm, arm, [[100.0 + vis + loc + (i % 3)] * 4 for i in range(n)])
    res = C.analyze(tmp_path, n_boot=500)
    c = res["contrasts"]["perm100-grpo100"]
    assert abs(c["L"]["mean"] - 20.0) < 1e-9 and abs(c["V"]["mean"]) < 1e-9
    assert c["consistent_with_local"] is True


def test_macros_name_every_registered_quantity(tmp_path):
    n = 40
    for arm, (loc, vis) in {"grpo100": (-30.0, -10.0), "perm100": (-10.0, -12.0),
                            "signed100": (-20.0, -11.0)}.items():
        _write(tmp_path, "base", "base", [[100.0 + (i % 3)] * 4 for i in range(n)])
        _write(tmp_path, "base", arm, [[100.0 + loc + (i % 3)] * 4 for i in range(n)])
        _write(tmp_path, arm, "base", [[100.0 + vis + (i % 3)] * 4 for i in range(n)])
        _write(tmp_path, arm, arm, [[100.0 + vis + loc + (i % 3)] * 4 for i in range(n)])
    res = C.analyze(tmp_path, n_boot=200)
    res["exploratory_robustness"] = {"x": {"a": {"V_share": 0.81}, "b": {"V_share": 0.86}}}
    out = C.emit(res)
    assert {k for k in C.MACROS if not k.startswith(("DecRob", "DecSec"))} <= set(out)
    assert out["DecGrpoDH"] == "-40.0" and out["DecGrpoV"] == "-10.0"
    assert out["DecGrpoShare"] == "0.25" and out["DecNprompts"] == "40"
    assert out["DecPermdL"] == "20.0" and out["DecPermdV"] == "-2.0"
    assert (out["DecRobustMin"], out["DecRobustMax"]) == ("0.81", "0.86")


def test_macros_leave_missing_intervals_pending(tmp_path):
    n = 10
    _write(tmp_path, "base", "base", [[100.0] * 4] * n)
    _write(tmp_path, "base", "grpo100", [[90.0] * 4] * n)
    _write(tmp_path, "grpo100", "base", [[80.0] * 4] * n)
    _write(tmp_path, "grpo100", "grpo100", [[70.0] * 4] * n)
    out = C.emit(C.analyze(tmp_path, n_boot=100))
    assert "DecGrpoDH" in out and "DecPermDH" not in out and "DecRobustMin" not in out
