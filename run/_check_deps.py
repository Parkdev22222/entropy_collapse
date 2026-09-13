#!/usr/bin/env python3
"""Verify pure-Python deps by a real symbol, not by import success.

A successful ``import X`` does not mean X is installed.  A directory without
``__init__.py`` becomes a namespace package and imports fine, and this repo
ships a ``datasets/`` folder for its parquet files while every launcher puts
the repo root on ``PYTHONPATH``.  With HuggingFace ``datasets`` absent that
folder *becomes* the ``datasets`` module, and training dies deep into startup
with

    AttributeError: module 'datasets' has no attribute 'load_dataset'

Import success is therefore the wrong test.  Each entry below names a symbol
the real package must expose.  Prints the pip names that need installing to
stdout (one line, shell-quoted); diagnostics go to stderr so the caller can
capture just the list.
"""
import importlib
import sys

SPEC = [
    # (module, symbol that must exist, pip name)
    ("datasets",              "load_dataset",   "datasets"),
    ("hydra",                 "main",           "hydra-core"),
    ("omegaconf",             "OmegaConf",      "omegaconf"),
    ("pyarrow",               "__version__",    "pyarrow>=19.0.0"),
    ("pandas",                "read_parquet",   "pandas"),
    ("numpy",                 "ndarray",        "numpy"),
    ("accelerate",            "Accelerator",    "accelerate"),
    ("peft",                  "get_peft_model", "peft"),
    ("codetiming",            "Timer",          "codetiming"),
    ("dill",                  "dumps",          "dill"),
    ("pylatexenc",            "__version__",    "pylatexenc"),
    ("torchdata",             "__version__",    "torchdata"),
    ("math_verify",           "parse",          "math_verify"),
    ("latex2sympy2_extended", "__name__",       "latex2sympy2_extended"),
    ("tensordict",            "TensorDict",     "tensordict<=0.6.2"),
]

# Present-is-fine, absent-is-fine: reported, never turned into a pip command.
#
# wandb was in SPEC above until 2026-09-13, when the `pip install wandb` this
# file asked for pulled opentelemetry 1.26 -> 1.44 and protobuf 4.25 -> 7.36 and
# broke vllm 0.8.4's declared pins on a freshly built H100 box. Nothing needs
# it: every arm any queue launches logs to console+tensorboard --
# run_grpo.sh:207, run_uniform_ablation.sh:170 for the tree arms, and
# _arms.sh:102 (steer_plain_args) for the plain ones. run_steerf.sh's own
# wandb default at :236 is reached by no queue in this repo.
OPTIONAL = [
    ("wandb", "init", "wandb", "어느 큐도 쓰지 않습니다 (전부 console+tensorboard)"),
]

GREEN, YELLOW, RED, OFF = "\033[32m", "\033[33m", "\033[31m", "\033[0m"


def main():
    need = []
    for mod, attr, pkg in SPEC:
        try:
            m = importlib.import_module(mod)
        except Exception as exc:
            print(f"  {YELLOW}WARN{OFF}  {mod} 없음 ({type(exc).__name__})", file=sys.stderr)
            need.append(pkg)
            continue
        if hasattr(m, attr):
            print(f"  {GREEN}OK{OFF}    {mod}", file=sys.stderr)
            continue
        where = getattr(m, "__file__", None) or getattr(m, "__path__", "?")
        print(f"  {RED}FAIL{OFF}  {mod}: .{attr} 가 없습니다 — 진짜 패키지가 아닙니다", file=sys.stderr)
        print(f"        {mod} -> {where}", file=sys.stderr)
        print(f"        레포의 {mod}/ 디렉터리가 PYTHONPATH 를 통해 가리고 있습니다.", file=sys.stderr)
        need.append(pkg)

    for mod, attr, _pkg, why in OPTIONAL:
        try:
            m = importlib.import_module(mod)
        except Exception:
            print(f"  {GREEN}OK{OFF}    {mod} 없음 — 정상 ({why})", file=sys.stderr)
            continue
        state = "" if hasattr(m, attr) else " (심볼 없음)"
        print(f"  {GREEN}OK{OFF}    {mod} 설치돼 있음{state} — {why}", file=sys.stderr)

    print(" ".join(f'"{p}"' if any(c in p for c in "<>=") else p for p in need))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
