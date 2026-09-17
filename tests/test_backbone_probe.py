"""A backbone row costs ~140 GPU-hours; the probe decides whether it can exist.

2026-09-17: meta-llama/Llama-3.2-3B, the BASE checkpoint, scored .020 on
MATH500 and .024 on grade-school arithmetic under this protocol. Nothing in the
stack raised: the reward function is fine (multi_datasets_eval.py:327 scores
`dapo_correct OR qwen_correct`, and -0.96 is exactly 0.02*1 + 0.98*(-1)), the
run would have started, and 110 steps later the table would have carried three
numbers from a model that never learned -- because at a 2% solve rate .98^8 of
the GRPO groups come back all-wrong and carry no advantage at all.

These tests pin the two halves of the fix: the probe can refuse, and the number
that rejected the base checkpoint reaches the manuscript through a generator
rather than a hand-typed literal.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.emit_probe_numbers import MACROS, emit  # noqa: E402
from scripts.phase3_port_model import gate_verdict   # noqa: E402


# ----------------------------------------------------------------- the gate
def test_gate_is_silent_until_asked():
    """The command shipped as a diagnostic. Callers that do not opt in keep
    their exit code, or this change breaks every existing invocation."""
    refuse, explain = gate_verdict(all_wrong=0.9, informative=0.1,
                                   min_informative=None)
    assert refuse is False
    assert explain is True        # still says so, just does not fail


def test_gate_refuses_below_the_threshold():
    refuse, explain = gate_verdict(all_wrong=0.9, informative=0.10,
                                   min_informative=0.20)
    assert refuse and explain


def test_gate_passes_at_or_above_the_threshold():
    for informative in (0.20, 0.35):
        refuse, _ = gate_verdict(all_wrong=0.4, informative=informative,
                                 min_informative=0.20)
        assert not refuse, informative


def test_advice_is_to_change_the_backbone_not_the_data():
    """The original message told the reader to move to an easier subset. That
    would give each backbone a different training set, and Section 11.1 says
    every backbone trains on DAPO-Math-17k -- the cross-backbone comparison is
    the only thing those rows are for."""
    src = (ROOT / "scripts" / "phase3_port_model.py").read_text()
    assert "백본을 바꾸는 것" in src
    assert "난이도 하위 서브셋으로 조정" not in src


def test_probe_records_which_grader_it_used():
    """This tool grades with phase1_validate.answers_match, the training reward
    is compute_score_both. Same word, different number: without the field in
    the record, a later reader compares a pass rate to a reward rate."""
    src = (ROOT / "scripts" / "phase3_port_model.py").read_text()
    assert '"grader": "phase1_validate.answers_match (NOT the training reward)"' in src


# ------------------------------------------------------------- the macros
DOC = {"llama_base_eval": {"avg_at_1": {"math500": 0.020, "gsm8k_test": 0.024}}}


def test_emits_the_measurement_that_rejected_the_base_checkpoint():
    out = emit(DOC)
    assert out["Bllamabaseacc"] == "0.020"
    assert out["Bllamabasegsm"] == "0.024"


def test_an_unmeasured_probe_stays_red_rather_than_guessed():
    out = emit({})
    assert out == {}, "a missing measurement must leave the cell PENDING"


def test_passrate_ratio_is_derived_not_typed():
    out = emit(DOC, {"informative_frac": 0.40}, {"informative_frac": 0.10})
    assert out["Bqweninform"] == "0.400"
    assert out["Bllamainform"] == "0.100"
    assert out["Bllamainformratio"] == "0.25"


def test_one_passrate_file_alone_gives_no_ratio():
    out = emit(DOC, {"informative_frac": 0.40}, None)
    assert "Bqweninform" in out and "Bllamainformratio" not in out


def test_checked_in_record_produces_the_macros_the_paper_asks_for(tmp_path):
    """End to end on the real record, so a malformed edit to it fails here."""
    dest = tmp_path / "numbers-probe.tex"
    rc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "emit_probe_numbers.py"),
         "--json", str(ROOT / "docs" / "backbone_probe.json"),
         "--passrate-ref", str(tmp_path / "absent.json"),
         "--passrate-new", str(tmp_path / "absent.json"),
         "--out", str(dest)],
        capture_output=True, text=True, cwd=str(ROOT))
    assert rc.returncode == 0, rc.stdout + rc.stderr
    body = dest.read_text()
    for name in MACROS:
        assert f"\\providecommand{{\\{name}}}" in body, name


def test_the_record_says_where_its_numbers_came_from():
    """The GRPO seed-1 precedent: a number whose provenance is a chat window
    has to be labelled as one, or it is indistinguishable from a logged run."""
    doc = json.loads((ROOT / "docs" / "backbone_probe.json").read_text())
    assert "source" in doc["llama_base_eval"]
    assert doc["llama_base_eval"]["checkpoint_kind"] == "base"
