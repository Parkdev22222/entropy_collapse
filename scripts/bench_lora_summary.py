#!/usr/bin/env python3
"""Per-step timing of the LoRA speed benchmark, one row per case.

    python3 scripts/bench_lora_summary.py --steps 6 \
        --case full_tp4=logs/bench/train-a.log \
        --case lora_1gpu_x4=logs/bench/train-g0.log,logs/bench/train-g1.log,...

The question the benchmark answers is throughput per BOX, not per run: a case
that runs four one-GPU trainers side by side finishes four steps in the time
one of them takes. So the headline column is box_steps_per_hour = runs * 3600 /
median step seconds, and `speedup` compares it with the baseline case.

Step 1 carries vLLM warm-up and CUDA graph capture and the last step can carry
the final save, so both are left out of the medians. A run that did not reach
--steps is not averaged in: the case is FAILED and names the log, because a
throughput that quietly counts a run which died of OOM is the wrong number.
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

KEYS = {
    "step_s": "timing_s/step",
    "gen_s": "timing_s/gen",
    "old_log_prob_s": "timing_s/old_log_prob",
    "update_actor_s": "timing_s/update_actor",
    "mem_gb": "perf/max_memory_allocated_gb",
}
STEP_RE = re.compile(r"\bstep:(\d+)\b")


def parse_log(path: Path) -> list[dict]:
    """One dict per logged training step: {'step': n, <KEYS>: float}."""
    text = Path(path).read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    rows = {}
    for line in text.splitlines():
        m = STEP_RE.search(line)
        if not m or "timing_s/step" not in line:
            continue
        row = {"step": int(m.group(1))}
        for name, key in KEYS.items():
            v = re.search(re.escape(key) + r":([-0-9.eE+]+)", line)
            if v:
                row[name] = float(v.group(1))
        rows[row["step"]] = row
    return [rows[k] for k in sorted(rows)]


def summarise_case(case: str, logs: list[Path], expected_steps: int) -> dict:
    out = {"case": case, "runs": len(logs), "status": "ok"}
    short = []
    pooled = {name: [] for name in KEYS}
    for log in logs:
        rows = parse_log(log) if Path(log).exists() else []
        if len(rows) < expected_steps:
            short.append(f"{Path(log).name} ({len(rows)}/{expected_steps} steps)")
            continue
        for r in rows[1:expected_steps - 1] if expected_steps > 2 else rows:
            for name in KEYS:
                if name in r:
                    pooled[name].append(r[name])
    if short:
        out["status"] = "FAILED: " + ", ".join(short)
        return out
    for name, vals in pooled.items():
        out[name] = statistics.median(vals) if vals else float("nan")
    out["mem_gb"] = max(pooled["mem_gb"]) if pooled["mem_gb"] else float("nan")
    step = out["step_s"]
    out["box_steps_per_hour"] = len(logs) * 3600.0 / step if step and step == step else float("nan")
    return out


def with_speedup(rows: list[dict], baseline: str) -> list[dict]:
    base = next((r for r in rows if r["case"] == baseline and r["status"] == "ok"), None)
    for r in rows:
        if base and r["status"] == "ok":
            r["speedup"] = r["box_steps_per_hour"] / base["box_steps_per_hour"]
    return rows


COLS = ["case", "runs", "status", "step_s", "gen_s", "old_log_prob_s",
        "update_actor_s", "mem_gb", "box_steps_per_hour", "speedup"]


def _fmt(v) -> str:
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", action="append", required=True,
                   help="name=log[,log...]  (repeat per case, in the order to print)")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--baseline", default="full_tp4")
    p.add_argument("--out", default=None, help="also write the table as TSV here")
    args = p.parse_args(argv)

    rows = []
    for spec in args.case:
        name, _, logs = spec.partition("=")
        rows.append(summarise_case(name, [Path(x) for x in logs.split(",") if x], args.steps))
    rows = with_speedup(rows, args.baseline)

    lines = ["\t".join(COLS)] + ["\t".join(_fmt(r.get(c, "")) for c in COLS) for r in rows]
    print("\n".join(lines))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(lines) + "\n")
    return 0 if all(r["status"] == "ok" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
