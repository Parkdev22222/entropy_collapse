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
import pathlib
import re
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
    # The reward scorer. verl imports these lazily, the first time it scores a
    # generation -- which is step-0 validation, several minutes into a run and
    # long after every gate has said OK. On 2026-09-14 three H100 arms died
    # there in a row on a missing word2number:
    #   verl/utils/reward_score/qwen_math_eval_toolkit/parser.py:7
    #     from word2number import w2n
    # word2number.w2n, not word2number: the package's __init__ is empty and
    # binds no submodule, so `hasattr(word2number, "w2n")` is False on a
    # perfectly good install. verl imports the submodule
    # (`from word2number import w2n`), so that is what to check.
    ("word2number.w2n",       "word_to_num",    "word2number"),
    ("sympy",                 "simplify",       "sympy"),
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
    ("wandb", "init", "wandb", "어느 큐도 쓰지 않습니다 (전부 console+tensorboard, 아래 참조)"),
]

# The logger backend is a dependency like any other, and it is the one this file
# has been able to name since 2026-09-13 without ever checking it: the OPTIONAL
# note above drops wandb *because* every arm logs to console+tensorboard, and
# nothing verified that tensorboard was installed.  On 2026-09-23 a fresh H100
# box that got its GPU stack from plain pip -- never `bash run/setup_env.sh` --
# lost its first run to exactly that.  verl opens the backend in Tracking at
# trainer init, before step 1, so the training log reads as if that arm failed
# and the queue moved on to the next one with every gate still saying OK.
#
# Checked by the import path verl actually takes, not by the pip name.
# `torch.utils.tensorboard` is what raises when the tensorboard distribution is
# absent; `import tensorboard` on its own can succeed against a partial install.
# Same lesson as word2number.w2n above.
LOGGER_BACKENDS = {
    "console":     None,                                        # verl prints it itself
    "tensorboard": ("torch.utils.tensorboard", "SummaryWriter", "tensorboard"),
    "wandb":       ("wandb", "init", "wandb"),
}


def logger_spec(root):
    """What the queues ask the trainer to log to, read from the queues.

    Returns (required, reported).  `required` is SPEC-shaped and joins the list
    above; `reported` is every other place in run/ that names a backend, printed
    but never turned into a pip command.

    The split is not tidiness, it is the 09-13 accident.  run_steerf.sh:236
    hardcodes `trainer.logger="['console','wandb']"`, and wandb has never had to
    be installed because the queue appends _arms.sh's steer_plain_args override
    after it and hydra keeps the last value.  A check that unioned every
    trainer.logger= it could find would demand wandb, `setup_env.sh` would
    install it, and that pip run is what pulled opentelemetry 1.26 -> 1.44 and
    broke vllm 0.8.4 on a freshly built box.  So only the override this repo
    owns and every queue applies decides what gets installed.  Change
    _arms.sh:steer_plain_args and this follows; a launcher default only ever
    gets mentioned.
    """
    def backends(path):
        try:
            text = (root / path).read_text()
        except OSError:
            return []
        out = []
        for lineno, line in enumerate(text.splitlines(), 1):
            # run_steerf.sh:236 and run_steerf_linear.sh:124 quote the whole
            # value (trainer.logger="['console','wandb']"); the queues do not.
            # Both spellings have to be seen or the report misses the launcher
            # defaults, which is the half of this that matters.
            for body in re.findall(r"""trainer\.logger=["']?\[([^\]]*)\]""", line):
                names = [n.strip().strip("'\"") for n in body.split(",")]
                out.append((lineno, [n for n in names if n]))
        return out

    required, reported = [], []
    for lineno, names in backends("run/_arms.sh"):
        for name in names:
            entry = LOGGER_BACKENDS.get(name, ())
            if entry is None:                       # console: nothing to install
                continue
            if entry == ():
                reported.append((f"run/_arms.sh:{lineno}", name, "모르는 백엔드"))
                continue
            if entry not in required:
                required.append(entry)
    wanted = {e[0] for e in required}
    for path in sorted(p.name for p in (root / "run").glob("run_*.sh")):
        for lineno, names in backends(f"run/{path}"):
            for name in names:
                entry = LOGGER_BACKENDS.get(name, ())
                if entry is None or (entry and entry[0] in wanted):
                    continue
                why = ("필수 아님 — 큐가 이 값을 덮거나 이 스크립트를 실행하지 "
                       "않습니다") if entry else "모르는 백엔드"
                reported.append((f"run/{path}:{lineno}", name, why))
    return required, reported

GREEN, YELLOW, RED, OFF = "\033[32m", "\033[33m", "\033[31m", "\033[0m"


def main():
    need = []
    root = pathlib.Path(__file__).resolve().parents[1]
    logger_required, logger_reported = logger_spec(root)
    for mod, attr, pkg in list(SPEC) + logger_required:
        try:
            m = importlib.import_module(mod)
        except Exception as exc:
            print(f"  {YELLOW}WARN{OFF}  {mod} 없음 ({type(exc).__name__})", file=sys.stderr)
            need.append(pkg)
            continue
        if hasattr(m, attr):
            print(f"  {GREEN}OK{OFF}    {mod}", file=sys.stderr)
            continue
        where = str(getattr(m, "__file__", None) or getattr(m, "__path__", "?"))
        # Only accuse the repo when the module actually resolves inside it.
        # This message used to be unconditional, and told a box with a correct
        # word2number in dist-packages that the repo was shadowing it -- while
        # printing the dist-packages path one line above, contradicting itself.
        shadowed = where.startswith(str(pathlib.Path(__file__).resolve().parents[1]))
        print(f"  {RED}FAIL{OFF}  {mod}: .{attr} 가 없습니다", file=sys.stderr)
        print(f"        {mod} -> {where}", file=sys.stderr)
        if shadowed:
            print(f"        레포의 {mod.split('.')[0]}/ 디렉터리가 PYTHONPATH 를 통해 "
                  f"진짜 패키지를 가리고 있습니다.", file=sys.stderr)
        else:
            print(f"        레포 밖에서 온 모듈입니다 — 설치가 덜 됐거나, 이 검사가 "
                  f"기대하는 심볼 이름이 틀렸습니다.", file=sys.stderr)
        need.append(pkg)

    for mod, attr, _pkg, why in OPTIONAL:
        try:
            m = importlib.import_module(mod)
        except Exception:
            print(f"  {GREEN}OK{OFF}    {mod} 없음 — 정상 ({why})", file=sys.stderr)
            continue
        state = "" if hasattr(m, attr) else " (심볼 없음)"
        print(f"  {GREEN}OK{OFF}    {mod} 설치돼 있음{state} — {why}", file=sys.stderr)

    # Named in run/ but not required.  Printed so that a launcher default which
    # stops being overridden -- or a backend nobody here knows about -- is
    # visible before it costs a run, without ever reaching `need`.
    for where, name, why in logger_reported:
        print(f"  {YELLOW}NOTE{OFF}  {where}: logger '{name}' — {why}", file=sys.stderr)

    print(" ".join(f'"{p}"' if any(c in p for c in "<>=") else p for p in need))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
