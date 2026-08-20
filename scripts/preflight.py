#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
# Licensed under the Apache License, Version 2.0
"""Fail fast, loudly and with the fix, before queueing days of GPU time.

run/run_all_experiments.sh runs this before anything else. Every check here
corresponds to a failure that has actually been hit: base `pip install -e .`
does not install the vLLM extra (-> ModuleNotFoundError: msgspec inside the
Ray TaskRunner, ~50s into every stage), the qwen math eval toolkit imports
word2number at module load (-> validation crash), and a GPU-count mismatch
kills the resource pool after Ray is already up.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys

FAILS: list[str] = []
WARNS: list[str] = []


def need(mod: str, fix: str, why: str) -> None:
    if importlib.util.find_spec(mod) is None:
        FAILS.append(f"{mod}: {why}\n    fix: {fix}")


def want(mod: str, fix: str, why: str) -> None:
    if importlib.util.find_spec(mod) is None:
        WARNS.append(f"{mod}: {why}\n    fix: {fix}")


def main() -> int:
    # --- hard requirements -------------------------------------------------
    need("verl", "pip install -e .  (from the repo root)",
         "the training framework itself")
    need("steer_f", "pip install -e .  (setup.py packages steer_f alongside verl)",
         "Ray workers import it by absolute name; PYTHONPATH does not reach "
         "reused raylets")
    need("ray", "pip install -e .", "verl runs everything through Ray")
    need("vllm", 'pip install "vllm==0.8.5.post1"',
         "rollout backend; NOT part of base `pip install -e .` (it is the "
         "[vllm] extra)")
    need("msgspec", 'pip install "vllm==0.8.5.post1"  (msgspec ships with it)',
         "verl.utils.vllm_utils imports it at module level — this is the "
         "50-second crash")
    need("tensordict", "pip install 'tensordict<=0.6.2'", "verl data plumbing")
    need("word2number", "pip install word2number",
         "qwen_math_eval_toolkit/parser.py imports it at module load; every "
         "math500/aime validation dies without it")
    need("math_verify", "pip install math-verify", "answer verification for eval")

    # --- soft requirements -------------------------------------------------
    want("flash_attn", "pip install flash-attn --no-build-isolation "
         "(use the prebuilt wheel matching your torch/cuda/python)",
         "training runs but much slower / may OOM at paper batch sizes")
    want("tensorboard", "pip install tensorboard==2.18.0",
         "scripts/select_best_checkpoint.py reads tb events for the paper's "
         "checkpoint-selection rule")
    want("wandb", "pip install wandb (or keep trainer.logger console+tensorboard)",
         "run scripts default to the wandb logger")

    # --- GPUs ----------------------------------------------------------------
    n_want = int(os.environ.get("N_GPUS", "8"))
    tp = int(os.environ.get("TP_SIZE", "4"))
    try:
        import torch
        n_have = torch.cuda.device_count()
        if n_have < n_want:
            FAILS.append(
                f"GPUs: found {n_have}, but N_GPUS={n_want} (paper setting is 8).\n"
                f"    fix: run with N_GPUS={max(n_have,1)} TP_SIZE={1 if n_have < tp else tp} "
                "— every script reads these envs. Expect far longer wall-clock "
                "than the paper's 8xH20 budget."
            )
        elif n_want % tp != 0:
            FAILS.append(f"N_GPUS={n_want} not divisible by TP_SIZE={tp}")
    except Exception as e:  # torch missing is already fatal via verl
        FAILS.append(f"torch/cuda probe failed: {e}")

    # --- datasets ------------------------------------------------------------
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for f in ("DAPO-Math-17k.parquet", "aime24.parquet", "math500.parquet"):
        if not os.path.exists(os.path.join(root, "datasets", f)):
            FAILS.append(f"datasets/{f} missing — incomplete checkout?")

    for w in WARNS:
        print(f"[preflight] WARN  {w}")
    for f in FAILS:
        print(f"[preflight] FAIL  {f}")
    if FAILS:
        print(f"\n[preflight] {len(FAILS)} blocking problem(s). Nothing was queued.")
        return 1
    print(f"[preflight] OK ({len(WARNS)} warnings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
