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

import subprocess
from pathlib import Path

import pytest

ARMS_SH = Path(__file__).resolve().parents[1] / "run" / "_arms.sh"
STEPS = 110
DONE_LINE = "step:110 - global_seqlen/min:2388866.000 - actor/entropy:0.121\n"

CAMPAIGN = ["grpo", "steer", "permuted", "signed", "uniform"]
FOLLOWUPS = ["lam0.1", "lam0.5", "lam0-tree", "xclip-signed", "xclip-steer",
             "rloo-signed", "rloo-steer", "opo-signed", "opo-steer"]


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
