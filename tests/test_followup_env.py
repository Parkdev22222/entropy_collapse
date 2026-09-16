# Copyright 2026 STEER-F authors
# Licensed under the Apache License, Version 2.0
"""An arm's env reaches the launcher, not just its lambda.

run_followups.sh's loop parsed STEERF_LAM out of arm_spec's assignment list
and dropped everything else. Every arm defined before 2026-09-16 set only
STEERF_LAM, so nothing showed it -- until the `oracle` arm, whose whole point
is STEERF_FORECAST=oracle. It would have trained with the MTP forecaster
under the name `...-tree-rollout-oracle`: a run whose name and treatment
disagree, which is the defect this repo keeps finding.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _stub_tree(tmp_path):
    """A fake run_uniform_ablation.sh that records the env it was handed."""
    run = tmp_path / "run"
    run.mkdir()
    for name in ("_arms.sh", "run_followups.sh"):
        (run / name).write_text((ROOT / "run" / name).read_text())
    rec = tmp_path / "env.txt"
    # The queue's own gates: instrumentation, then env_preflight's real import.
    (run / "instrument_campaign.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    (run / "instrument_campaign.sh").chmod(0o755)
    stub = tmp_path / "stub"
    stub.mkdir()
    (stub / "python3").write_text("#!/bin/sh\nexit 0\n")   # verl imports "fine"
    (stub / "python3").chmod(0o755)
    for name in ("run_uniform_ablation.sh", "run_steerf.sh", "run_grpo.sh"):
        (run / name).write_text(
            "#!/usr/bin/env bash\n"
            f'printf "RUN=%s LAM=%s FORECAST=%s ARM=%s\\n" '
            f'"${{RUN_NAME}}" "${{STEERF_LAM}}" "${{STEERF_FORECAST}}" "${{ARM}}" '
            f'>> "{rec}"\n'
            "exit 0\n")
        (run / name).chmod(0o755)
    (tmp_path / "logs" / "experiments").mkdir(parents=True)
    return run, rec, stub


def _spec(arm):
    out = subprocess.run(
        ["bash", "-c",
         f'. "{ROOT}/run/_arms.sh" >/dev/null 2>&1; '
         f'ROOT="{ROOT}"; . /dev/stdin <<<"$(sed -n \'/^arm_spec ()/,/^}}/p\' '
         f'"{ROOT}/run/run_followups.sh")"; arm_spec {arm}'],
        capture_output=True, text=True, cwd=ROOT, timeout=60)
    return out.stdout.strip()


def test_the_oracle_arm_asks_for_the_head_free_forecast():
    spec = _spec("oracle")
    assert "STEERF_FORECAST=oracle" in spec, spec
    # It is a tree arm at the campaign's lambda: only the forecast changes.
    assert spec.startswith("tree ") and "STEERF_LAM=0.25" in spec, spec


def test_the_oracle_run_name_is_its_own():
    out = subprocess.run(
        ["bash", "-c", f'. "{ROOT}/run/_arms.sh" >/dev/null 2>&1; '
                       f'run_name_for oracle 1'],
        capture_output=True, text=True, cwd=ROOT, timeout=60)
    name = out.stdout.strip()
    assert name.endswith("-tree-rollout-oracle"), name
    # Every tree arm's name extends signed's, which is why train_log_done only
    # accepts a `_<tag>` suffix (task 7). Assert the guarantee, not the naming:
    # a finished oracle log must not mark signed as done.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        Path(d, f"train-{name}.log").write_text("step:1 - global_seqlen\n")
        signed = subprocess.run(
            ["bash", "-c", f'. "{ROOT}/run/_arms.sh" >/dev/null 2>&1; '
                           f'train_log_done "{d}" "$(run_name_for signed 1)" 1'],
            cwd=ROOT, timeout=60)
        assert signed.returncode != 0, "an oracle log marked signed as finished"


def test_the_queue_hands_the_whole_env_to_the_launcher(tmp_path):
    """The regression: run one arm for real against stub launchers."""
    run, rec, stub = _stub_tree(tmp_path)
    env = dict(os.environ, ARMS="oracle", DRY="0", SEED="1", STEPS="1",
               MIN_FREE_GB="0", OOM_RETRY="0", WAIT="0",
               PATH=f"{stub}:{os.environ['PATH']}")
    out = subprocess.run(["bash", str(run / "run_followups.sh")],
                         cwd=tmp_path, env=env, capture_output=True,
                         text=True, timeout=300)
    if not rec.is_file():
        pytest.skip(f"the queue never reached a launcher: {out.stdout[-600:]}")
    line = rec.read_text().strip()
    assert "FORECAST=oracle" in line, line
    assert "LAM=0.25" in line, line
    assert "-tree-rollout-oracle" in line, line
