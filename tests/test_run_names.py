# Copyright 2026 STEER-F authors
# Licensed under the Apache License, Version 2.0
"""`train_log_done` must not confuse one arm's log for another's.

The run names are prefixes of each other by construction:

    signed    steer-f-Qwen2.5-Math-1.5B-s2-tree-rollout
    permuted  steer-f-Qwen2.5-Math-1.5B-s2-tree-rollout-permuted
    uniform   steer-f-Qwen2.5-Math-1.5B-s2-tree-rollout-uniform

so a `train-<run>*.log` glob marks signed "already done" the moment permuted
finishes -- and run_campaign.sh runs permuted first. The queue is built once
per invocation, so a running campaign survives it; the next restart drops the
treatment arm from the queue with no error printed anywhere. That is the kind
of bug a run does not reveal, which is why it is pinned here rather than left
to a dry run.

The recovery chain's train-<run>_0905.log must still count as the base run
being finished -- that is the one suffix the matcher is allowed to accept.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ARMS_SH = REPO / "run" / "_arms.sh"
STEPS = 110
DONE_LINE = "step:110 - global_seqlen/min:2388866.000 - actor/entropy:0.121\n"

CAMPAIGN = ["grpo", "steer", "permuted", "signed", "uniform"]
FOLLOWUPS = ["lam0.1", "lam0.5", "lam0-tree", "xclip-signed", "xclip-steer",
             "rloo-signed", "rloo-steer", "opo-signed", "opo-steer",
             "wmin-steer"]


def _bash(script: str) -> str:
    out = subprocess.run(["bash", "-c", f'. "{ARMS_SH}"\n{script}'],
                         capture_output=True, text=True, check=True)
    return out.stdout


def run_name(arm: str, seed: int) -> str:
    return _bash(f'run_name_for "{arm}" {seed}').strip()


def done_flags(log_dir: Path, names: list[str]) -> dict[str, bool]:
    """One bash call for the whole list -- 350 subprocesses is a slow test."""
    body = "\n".join(
        f'train_log_done "{log_dir}" "{n}" {STEPS} && echo "1 {n}" || echo "0 {n}"'
        for n in names)
    flags = {}
    for line in _bash(body).splitlines():
        got, name = line.split(" ", 1)
        flags[name] = got == "1"
    return flags


@pytest.fixture(scope="module")
def all_names() -> list[str]:
    names = [run_name(a, s) for s in (1, 2) for a in CAMPAIGN]
    names += [run_name(a, 1) for a in FOLLOWUPS]
    assert len(set(names)) == len(names), "two arms resolved to the same run name"
    return names


def test_one_finished_run_marks_only_itself(tmp_path, all_names):
    """The general form: no completed log may mark any other run done."""
    for finished in all_names:
        log_dir = tmp_path / finished.replace("/", "_")
        log_dir.mkdir()
        (log_dir / f"train-{finished}.log").write_text(DONE_LINE)
        flags = done_flags(log_dir, all_names)
        wrong = sorted(n for n, d in flags.items() if d and n != finished)
        assert not wrong, (
            f"train-{finished}.log wrongly marked these runs finished: {wrong}")
        assert flags[finished], f"train-{finished}.log did not mark its own run finished"


def test_recovery_tag_counts_as_the_base_run(tmp_path):
    """train-<run>_0905.log is the recovery chain's name for the same arm."""
    signed = run_name("signed", 1)
    (tmp_path / f"train-{signed}_0905.log").write_text(DONE_LINE)
    assert done_flags(tmp_path, [signed])[signed]


def test_permuted_does_not_finish_signed(tmp_path):
    """The exact case that would have dropped the treatment arm from seed 2."""
    signed, permuted = run_name("signed", 2), run_name("permuted", 2)
    (tmp_path / f"train-{permuted}.log").write_text(DONE_LINE)
    flags = done_flags(tmp_path, [signed, permuted])
    assert flags[permuted]
    assert not flags[signed]


def test_an_unfinished_log_is_not_done(tmp_path):
    """A run that died at import has a log but never reached the final step."""
    signed = run_name("signed", 2)
    (tmp_path / f"train-{signed}.log").write_text(
        "ImportError: huggingface-hub>=0.34.0,<1.0 is required\n")
    assert not done_flags(tmp_path, [signed])[signed]


# --------------------------------------------------------------------------
# The model the run trains must match the model its NAME claims.
#
# 2026-09-13: run_grpo.sh and run_uniform_ablation.sh default to 1.5B but
# run_steerf.sh defaults to Qwen2.5-Math-7B, and the campaign's steer arm is the
# one arm that calls run_steerf.sh directly. The pod where the campaign started
# had that file edited to 1.5B by hand, uncommitted; a second pod cloned the
# committed tree and its steer arm resolved model.path=Qwen/Qwen2.5-Math-7B
# under experiment_name=steer-Qwen2.5-Math-1.5B-s5. Nothing errors: it trains
# the wrong network and files the numbers under the right-looking name.
def _arms(env=None):
    """Source run/_arms.sh and return MODEL_TAG, MODEL_PATH and model_guard's rc."""
    script = (
        'set -u; . run/_arms.sh; '
        'printf "%s\\n%s\\n" "${MODEL_TAG}" "${MODEL_PATH}"; '
        'model_guard >/dev/null 2>&1; printf "%s\\n" "$?"'
    )
    e = dict(os.environ)
    e.pop("MODEL_PATH", None)
    e.pop("MODEL_TAG", None)
    if env:
        e.update(env)
    out = subprocess.run(["bash", "-c", script], cwd=REPO, env=e,
                         capture_output=True, text=True).stdout.split("\n")
    return out[0], out[1], out[2]


def test_model_path_defaults_to_the_tag_in_the_run_names():
    tag, path, rc = _arms()
    assert path.rsplit("/", 1)[-1] == tag, f"{path} does not end in {tag}"
    assert rc == "0", "model_guard rejected its own default"


def test_model_guard_refuses_a_model_the_run_names_do_not_claim():
    _, _, rc = _arms({"MODEL_PATH": "Qwen/Qwen2.5-Math-7B"})
    assert rc != "0", "a 7B model under 1.5B run names must be refused"


def test_every_queue_exports_model_path_before_launching():
    # Whichever launcher a queue calls, none of their own defaults may be
    # reachable -- run_steerf.sh's is a different model.
    for q in ("run_campaign.sh", "run_followups.sh", "run_0905_chain.sh"):
        text = (REPO / "run" / q).read_text()
        assert "MODEL_PATH" in text, f"{q} never mentions MODEL_PATH"
        assert "model_guard" in text, f"{q} does not call model_guard"


def test_the_analyser_and_the_queue_agree_on_every_run_name():
    """Two tables name the same runs; they drift silently.

    run/_arms.sh:run_name_for is what the queue launches under, and
    scripts/analyze_seeds.py has its own copy to find the logs afterwards.
    Adding the `oracle` arm to one and not the other made the analyser raise
    KeyError on every invocation -- which the seed-1 table caught only because
    it shells out to the analyser. A disagreement in the other direction is
    worse: the analyser would look for a log the queue never writes and report
    the arm as missing forever.
    """
    import ast
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "scripts" / "analyze_seeds.py").read_text()
    body = re.search(r"def run_name\(arm: str, seed: int\) -> str:\n"
                     r"\s+t = args\.model_tag\n\s+return (\{.*?\n\s+\})\[arm\]",
                     src, re.S)
    assert body, "analyze_seeds.run_name no longer has a literal table"
    arms = [k.value for k in ast.parse(body.group(1), mode="eval").body.keys]

    for arm in arms:
        theirs = run_name(arm, 1)
        assert theirs, f"run/_arms.sh does not know the arm {arm!r}"
        # The python side interpolates the same tag; compare the whole string.
        mine = re.search(rf'"{re.escape(arm)}": f"([^"]+)"', body.group(1))
        assert mine, arm
        expected = mine.group(1).replace("{t}", "Qwen2.5-Math-1.5B").replace("{seed}", "1")
        assert theirs == expected, f"{arm}: bash {theirs!r} vs python {expected!r}"


def test_the_released_damping_arm_is_stock_steer_at_their_value():
    """2026-09-21. We measure STEER below GRPO by -.0101 over three seeds,
    which the base method's own published results contradict.

    Reading the released repository against ours found the weighting function
    identical line for line -- f_x, advantage/old_prob, the symmetric abs(),
    the exponential map and its 0.02 floor, the signature defaults -- and one
    configuration difference: run/run_exp.sh there sets token_weight_min=0.8
    where run_steerf.sh:201 defaults to 0.7. Ours therefore attenuates the
    extreme-|Omega| tokens by up to 30% against their 20%.

    This arm must be stock STEER (lambda = 0) at their value and nothing else:
    a lambda that slipped through would make it a STEER-F arm wearing the name,
    and the table would answer a different question than it asks.
    """
    import re
    src = (REPO / "run" / "run_followups.sh").read_text()
    m = re.search(r"wmin-steer\)\s*echo\s+'?\"([^\"]+)\"", src)
    assert m, "run_followups.sh has no wmin-steer spec"
    spec = set(m.group(1).split())
    assert spec == {"plain", "STEERF_LAM=0", "TOKEN_WEIGHT_MIN=0.8"}, spec


def test_the_launcher_actually_reads_token_weight_min():
    """The spec is only worth anything if run_steerf.sh honours the variable;
    otherwise the arm trains at 0.7 under a name that says 0.8."""
    out = subprocess.run(["git", "show", "origin/paper:run/run_steerf.sh"],
                         capture_output=True, text=True, cwd=str(REPO)).stdout
    if not out:
        pytest.skip("origin/paper is not fetched in this checkout")
    assert "token_weight_min=${TOKEN_WEIGHT_MIN:-0.7}" in out
