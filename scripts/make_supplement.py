#!/usr/bin/env python3
"""Build the anonymous ARR supplementary archive: logs and result tables, no code.

    python3 scripts/make_supplement.py --git-ref origin/paper \
        --out paper/arr_supplement.zip

WHAT GOES IN
    * every training and evaluation log the paper's numbers are computed from,
      logs/experiments/{train,eval}-*.log on the given git ref -- including the
      runs the paper says failed or were excluded, since it says those are
      released too;
    * the result tables those logs produce (results/*.tsv, paired_se.json);
    * a README saying what each part is.

WHAT STAYS OUT
    * all code (the authors' decision for this submission);
    * orchestration logs (paper_h100.log, driver-*, warmup-*, ...): they record
      uploads to a personal model hub account and are not inputs to any number;
    * git history, which carries names and session links.

ANONYMITY
    Every file passes through redact() and then check(). check() fails the
    build if any identifying pattern survives, so a new log that mentions an
    account cannot slip into the archive silently. The archive's timestamps are
    fixed so the zip metadata carries no date of its own.
"""
from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
import zipfile
from pathlib import Path

LOG_DIR = "logs/experiments"
LOG_RE = re.compile(r"^(train|eval)-.*\.log$")
RESULT_FILES = ["per_seed.tsv", "arm_means.tsv", "contrasts.tsv",
                "compute_match.tsv", "summary.tsv", "paired_se.json",
                "sibling_spread.tsv"]

# Model-hub owners that are public organisations, not people: their paths are
# kept (they name the models and datasets the paper uses).
PUBLIC_HUB_OWNERS = {"Qwen", "meta-llama", "mistralai", "deepseek-ai",
                     "BytedTsinghua-SIA", "HuggingFaceH4", "openai", "google",
                     "datasets", "api", "docs", "models", "spaces"}

# Strings that identify the authors' accounts. Kept here, in the one place the
# build checks, rather than scattered across the repository.
IDENTITY = ["Parkdev22222", "parkdev", "jeonghobag", "DSDSh"]

REDACTIONS: list[tuple[re.Pattern, object]] = [
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[email]"),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"), "[token]"),
    (re.compile(r"\b(?:wandb_|WANDB_API_KEY=)[A-Za-z0-9]{20,}\b"), "[token]"),
    (re.compile(r"github\.com/[A-Za-z0-9_.-]+"), "github.com/[anon]"),
    (re.compile(r"huggingface\.co/([A-Za-z0-9_.-]+)"),
     lambda m: m.group(0) if m.group(1) in PUBLIC_HUB_OWNERS
     else "huggingface.co/[anon]"),
    (re.compile(r"/home/(?!user\b)[A-Za-z0-9_.-]+"), "/home/[anon]"),
    (re.compile(r"/Users/[A-Za-z0-9_.-]+"), "/Users/[anon]"),
    (re.compile(r"[A-Za-z0-9.-]*\.runpod\.(?:net|io)(?::\d+)?"), "[host]"),
    (re.compile(r"claude\.ai/[^\s\"']*"), "[link]"),
] + [(re.compile(re.escape(s), re.I), "[anon]") for s in IDENTITY]

# What must not appear in the archive at all, after redaction.
FORBIDDEN = [re.compile(re.escape(s), re.I) for s in IDENTITY] + [
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"claude\.ai/"),
    re.compile(r"Claude-Session"),
]

README = """\
Supplementary material (anonymous submission)
==============================================

logs/   Training and evaluation logs of every run the paper reports, as written
        by the trainer (verl). Training logs print one line per optimization
        step with the AIME24 validation metrics every 10 steps; evaluation logs
        hold the six-benchmark re-evaluations. Runs that failed or were
        excluded from the statistics are included, as the paper states.
        File names encode the arm and the data seed, e.g.
          train-steer-f-Qwen2.5-Math-1.5B-s3-tree-rollout-uniform.log
            = the UNIFORM arm, seed 3.
          train-grpo-...-s4.log = GRPO, seed 4; train-steer-...-s4.log = STEER.
          train-steer-f-...-s4-tree-rollout.log = STEER-F, seed 4.
          train-steerf-r2-1p5b-s{1,2}-200.log = the two 200-step runs of
          Section 6.5 / Appendix F; eval-r2-* are their evaluations.
results/  Per-seed window means and the paired contrasts computed from the logs
        (per_seed.tsv, arm_means.tsv, contrasts.tsv, compute_match.tsv), the
        six-benchmark summary (summary.tsv) and the paired across-problem
        errors (paired_se.json).

Identifying strings (accounts, e-mail addresses, hosts, tokens) have been
replaced by [anon], [email], [host] or [token]. Code is not included in this
submission.
"""


def git_ls(ref: str, path: str) -> list[str]:
    out = subprocess.run(["git", "ls-tree", "-r", "--name-only", ref, path],
                         capture_output=True, text=True, check=True).stdout
    return [p for p in out.splitlines() if p]


def git_show(ref: str, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{ref}:{path}"],
                          capture_output=True, check=True).stdout


def redact(text: str) -> tuple[str, int]:
    """-> (text with identifying strings replaced, number of replacements)."""
    total = 0
    for pat, rep in REDACTIONS:
        def sub(m, rep=rep):
            nonlocal total
            new = rep(m) if callable(rep) else rep
            total += new != m.group(0)       # a kept public path is not a redaction
            return new
        text = pat.sub(sub, text)
    return text, total


def check(name: str, text: str) -> list[str]:
    """Patterns that survived redaction, as 'file: match' strings."""
    return [f"{name}: {m.group(0)!r}" for pat in FORBIDDEN
            for m in pat.finditer(text)]


def collect(git_ref: str, results_dir: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in git_ls(git_ref, LOG_DIR):
        name = path.rsplit("/", 1)[-1]
        if "/." in path or not LOG_RE.match(name):
            continue
        files[f"logs/{name}"] = git_show(git_ref, path).decode("utf-8", "replace")
    for name in RESULT_FILES:
        p = results_dir / name
        if p.is_file():
            files[f"results/{name}"] = p.read_text()
    files["README.txt"] = README
    return files


def build(files: dict[str, str], out: Path) -> tuple[int, list[str]]:
    redacted, leaks = 0, []
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):
            text, n = redact(files[name])
            redacted += n
            leaks += check(name, text)
            info = zipfile.ZipInfo(f"supplement/{name}", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, text)
    if not leaks:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(buf.getvalue())
    return redacted, leaks


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--git-ref", default="origin/paper")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out", default="paper/arr_supplement.zip")
    args = p.parse_args(argv)

    files = collect(args.git_ref, Path(args.results_dir))
    redacted, leaks = build(files, Path(args.out))
    if leaks:
        print("[supplement] REFUSED: identifying strings survived redaction:")
        for l in leaks[:20]:
            print("  ", l)
        return 1
    logs = sum(1 for f in files if f.startswith("logs/"))
    size = Path(args.out).stat().st_size
    print(f"[supplement] {logs} logs, {len(files) - logs - 1} result files, "
          f"{redacted} redactions -> {args.out} ({size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
