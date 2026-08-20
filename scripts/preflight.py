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
import subprocess
import sys

FAILS: list[str] = []
WARNS: list[str] = []


def importable(mod: str, timeout: int = 120):
    """Actually import `mod` in a subprocess; return None on success else the error.

    find_spec() only proves a file is on disk. The failures that cost the most
    time here are *installed but broken*: a flash-attn wheel built against a
    different torch C++ ABI imports fine as a spec and dies with `undefined
    symbol: _ZN3c105Error...` the moment a Ray worker touches it, ~90 seconds
    into a stage. A subprocess also survives the abort/segfault cases.
    """
    r = subprocess.run([sys.executable, "-c", f"import {mod}"],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode == 0:
        return None
    tail = (r.stderr.strip().splitlines() or ["(no stderr)"])[-1]
    return tail


def need(mod: str, fix: str, why: str) -> None:
    if importlib.util.find_spec(mod) is None:
        FAILS.append(f"{mod}: {why}\n    fix: {fix}")


def want(mod: str, fix: str, why: str) -> None:
    if importlib.util.find_spec(mod) is None:
        WARNS.append(f"{mod}: {why}\n    fix: {fix}")


def _flash_attn_hint() -> str:
    """Name the exact wheel for this interpreter's torch build."""
    try:
        import torch
        ver, cuda = torch.__version__.split("+")[0], (torch.version.cuda or "").replace(".", "")
        abi = "TRUE" if torch.compiled_with_cxx11_abi() else "FALSE"
        mm = ".".join(ver.split(".")[:2])
        py = f"cp{sys.version_info.major}{sys.version_info.minor}"
        whl = (f"flash_attn-2.7.4.post1+cu{cuda[:2]}torch{mm}cxx11abi{abi}"
               f"-{py}-{py}-linux_x86_64.whl")
        return (
            f"    detected: torch {torch.__version__}, cxx11abi={abi}, python {py}\n"
            f"    fix: pip uninstall -y flash-attn && pip install \\\n"
            f"           https://github.com/Dao-AILab/flash-attention/releases/download/"
            f"v2.7.4.post1/{whl}\n"
            "    (no matching wheel / offline? build against the installed torch instead:\n"
            "     MAX_JOBS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation "
            "— 30-90 min)"
        )
    except Exception:
        return ("    fix: install the flash-attn 2.7.4.post1 wheel matching your "
                "torch/cuda/python/cxx11-abi from the Dao-AILab releases page")


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
    want("tensorboard", "pip install tensorboard==2.18.0",
         "scripts/select_best_checkpoint.py reads tb events for the paper's "
         "checkpoint-selection rule")
    want("wandb", "pip install wandb (or keep trainer.logger console+tensorboard)",
         "run scripts default to the wandb logger")

    # --- ray <-> opentelemetry conflict --------------------------------------
    # Newer Ray (>= ~2.47) imports opentelemetry's prometheus exporter inside
    # the dashboard agent; if vllm pinned an older opentelemetry-semconv, the
    # agent crash-loops, the raylet never registers, and ray.init dies with
    # "The current node timed out during startup". Reproduce the agent's
    # import here, in-process, so the failure is a one-line fix instead of a
    # 30-second timeout with the real error buried in dashboard.log.
    # find_spec on a dotted path RAISES ModuleNotFoundError when a parent
    # package is absent (e.g. ray 2.46, which predates ray._private.telemetry
    # entirely) — absence of the module means absence of the conflict.
    try:
        _otel_spec = importlib.util.find_spec(
            "ray._private.telemetry.open_telemetry_metric_recorder")
    except ModuleNotFoundError:
        _otel_spec = None
    if _otel_spec is not None:
        try:
            importlib.import_module(
                "ray._private.telemetry.open_telemetry_metric_recorder")
        except Exception as e:
            FAILS.append(
                "ray dashboard agent cannot import its telemetry stack "
                f"({type(e).__name__}: {e})." + "\n"
                '    fix: pip install "ray[default]==2.46.0" && ray stop --force' + "\n"
                "    (2.46 predates the opentelemetry-prometheus dependency and is "
                "the combination proven with vllm 0.8.5.post1 / torch 2.6)"
            )

    # --- installed-but-broken imports ----------------------------------------
    # verl/workers/actor/dp_actor.py imports flash_attn.bert_padding at module
    # level whenever CUDA is available, so it is a HARD requirement regardless
    # of use_remove_padding — and the usual failure is a wheel/torch C++ ABI
    # mismatch, which only a real import reveals.
    if importlib.util.find_spec("flash_attn") is None:
        FAILS.append(
            "flash_attn: not installed, but verl's dp_actor imports\n"
            "    flash_attn.bert_padding unconditionally on CUDA.\n"
            + _flash_attn_hint()
        )
    else:
        err = importable("flash_attn")
        if err:
            FAILS.append(
                f"flash_attn: installed but fails to import — {err}\n"
                "    ('undefined symbol: _ZN3c105Error...' means the wheel's C++ ABI\n"
                "     does not match this torch build.)\n"
                + _flash_attn_hint()
            )

    for mod, fix in (("msgspec", 'pip install "vllm==0.8.5.post1"'),
                     ("word2number", "pip install word2number"),
                     ("math_verify", "pip install math-verify")):
        if importlib.util.find_spec(mod) is not None:
            err = importable(mod, timeout=60)
            if err:
                FAILS.append(f"{mod}: installed but fails to import — {err}\n    fix: {fix}")

    # --- GPUs ----------------------------------------------------------------
    # run_all_experiments.sh exports N_GPUS/TP_SIZE after detecting them; when
    # preflight is run standalone, fall back to the detected count rather than
    # the paper's 8, so a small node does not produce a spurious failure.
    try:
        import torch
        n_have = torch.cuda.device_count()
        n_want = int(os.environ.get("N_GPUS") or n_have or 0)
        tp = int(os.environ.get("TP_SIZE") or (4 if n_want >= 4 and n_want % 4 == 0
                                               else 2 if n_want >= 2 and n_want % 2 == 0
                                               else 1))
        if n_have == 0:
            FAILS.append(
                "GPUs: torch sees no CUDA device — training cannot run here.\n"
                "    fix: run this on the GPU node (or check CUDA_VISIBLE_DEVICES)."
            )
        elif n_want > n_have:
            FAILS.append(
                f"GPUs: N_GPUS={n_want} requested but only {n_have} visible.\n"
                f"    fix: N_GPUS={n_have} TP_SIZE={tp if n_have % tp == 0 else 1} "
                "(or unset both — run_all_experiments.sh auto-detects)."
            )
        elif n_want % tp != 0:
            FAILS.append(f"N_GPUS={n_want} is not divisible by TP_SIZE={tp}")
        elif n_have < 8:
            WARNS.append(
                f"GPUs: {n_have} visible; the paper used 8xH20. The optimisation is "
                "unchanged (same global batch, more gradient accumulation) but "
                "wall-clock scales ~inversely and 7B/14B full fine-tuning will "
                "likely OOM.\n    fix (if it OOMs): start with "
                "MATH_MODELS=Qwen/Qwen2.5-Math-1.5B SEEDS=1"
            )
    except Exception as e:  # torch missing is already fatal via verl
        FAILS.append(f"torch/cuda probe failed: {e}")

    # --- VRAM vs the models actually queued ----------------------------------
    # FSDP on one GPU cannot shard, so full fine-tuning needs roughly
    # 15 bytes/param (bf16 weights + bf16 grads + fp32 AdamW master/m/v) resident
    # before activations, and the hybrid engine also hands vLLM its share.
    PARAMS_B = {"1.5B": 1.54, "3B": 3.09, "7B": 7.62, "14B": 14.7}
    try:
        import torch
        if torch.cuda.device_count() > 0:
            vram = torch.cuda.get_device_properties(0).total_memory / 2**30
            n = torch.cuda.device_count()
            offload = os.environ.get("OFFLOAD", "0") == "1"
            models = (os.environ.get("MATH_MODELS", "") + " "
                      + os.environ.get("CODE_MODELS", "")).split() or ["Qwen/Qwen2.5-Math-7B"]
            for m in models:
                size = next((k for k in PARAMS_B if k in m), None)
                if size is None:
                    continue
                p = PARAMS_B[size] * 1e9
                bytes_per_param = 6 if offload else 15  # offload sends AdamW to CPU
                static = p * bytes_per_param / 2**30 / max(n, 1)
                # activations: bf16 logits [mbs*seq, V] + chunked entropy scratch.
                # micro_batch_size_per_gpu is fixed at 8 by the method (STEER's
                # min-max is per micro-batch), so this term does not shrink.
                act = (8 * 3250 * 152064 * 2 + 2 * 2048 * 152064 * 4) / 2**30
                if static + act > vram * 0.92:
                    hint = ("" if offload else
                            "\n    try: OFFLOAD=1 (AdamW state to CPU, much slower)")
                    WARNS.append(
                        f"VRAM: {m} peaks at ~{static + act:.0f} GiB per GPU "
                        f"({static:.0f} static + {act:.0f} activations) across "
                        f"{n} GPU(s) of {vram:.0f} GiB — expect OOM."
                        + hint +
                        "\n    also try: GPU_MEM_UTIL=0.4 (less reserved for vLLM)"
                    )
    except Exception:
        pass

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
