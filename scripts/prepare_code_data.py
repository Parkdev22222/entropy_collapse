#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
# Licensed under the Apache License, Version 2.0
"""Build the code-generation parquets for the paper's coding track.

Paper (arXiv:2510.10150v4, §6.1 + Appendix E.1, Table 9):

  * training   : ArcherCodeR — 6,753 code-generation tasks sourced from
                 DeepCoder / CodeContests / CodeForces, with regenerated test
                 cases. Public: https://huggingface.co/datasets/Fate-Zero/ArcherCodeR-Dataset
  * evaluation : LiveCodeBench v5, 279 problems, avg@4.
  * models     : Qwen2.5-Coder-3B / 7B / 14B (Table 5 rows), same GRPO recipe
                 as math (512/32/8, n=8, lr 1e-6, prompt 1024 / response 3072).

The paper's code-EDITING track (internal 51k corpus + Zeta) is NOT
reproducible — the training corpus is in-house — and is deliberately out of
scope here.

Output schema matches the math parquets this repo ships, so the verl
pipeline needs no changes:

    data_source   str    — must route to a code reward in
                           verl/utils/reward_score/__init__.py; we use
                           "codecontests" (-> prime_code, APPS-style tests,
                           or sandbox_fusion when a sandbox URL is given)
    prompt        list   — chat messages [{"role": "user", "content": ...}]
    ability       str    — "code"
    reward_model  dict   — {"style": "rule", "ground_truth": json.dumps(
                            {"inputs": [...], "outputs": [...],
                             "fn_name": optional})}
    extra_info    dict   — {"index": int, "split": ...}

Run this ON THE GPU BOX (needs HF access):

    python scripts/prepare_code_data.py --archer --out datasets/
    python scripts/prepare_code_data.py --lcb    --out datasets/

Then sanity-check before training — the two assertions this script prints at
the end (row counts 6753 / 279, and a decoded test-case sample) must match
Table 9. This file was written without HF access, so the source-column
detection is defensive: if a dataset revision renames fields, the loud
failure modes below are the point.
"""

from __future__ import annotations

import argparse
import base64
import json
import pickle
import sys
import zlib
from pathlib import Path


def _chat(problem_text: str) -> list:
    # Zero-shot, no extra prompt (paper Appendix E.2: "All evaluations are
    # zero-shot with no additional prompts"). The ```python fence is what
    # prime_code's extractor expects the model to emit.
    return [{"role": "user", "content": problem_text}]


def _std_row(idx, split, question, tests_json, data_source="codecontests"):
    return {
        "data_source": data_source,
        "prompt": _chat(question),
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": tests_json},
        "extra_info": {"index": idx, "split": split},
    }


# ----------------------------------------------------------------------
# ArcherCodeR (training)
# ----------------------------------------------------------------------
def build_archer(out_dir: Path) -> None:
    from datasets import load_dataset

    ds = load_dataset("Fate-Zero/ArcherCodeR-Dataset", split="train")
    print(f"[archer] loaded {len(ds)} rows; columns = {ds.column_names}")

    rows = []
    for i, ex in enumerate(ds):
        # ArcherCodeR is released in verl schema (the Archer project trains
        # with verl). Prefer its own fields; fall back defensively.
        if "prompt" in ex and "reward_model" in ex:
            row = {
                "data_source": "codecontests",  # force the prime_code route
                "prompt": ex["prompt"],
                "ability": "code",
                "reward_model": ex["reward_model"],
                "extra_info": ex.get("extra_info", {"index": i, "split": "train"}),
            }
            gt = row["reward_model"].get("ground_truth")
            tc = json.loads(gt) if isinstance(gt, str) else gt
            if not (isinstance(tc, dict) and "inputs" in tc and "outputs" in tc):
                raise SystemExit(
                    f"[archer] row {i}: ground_truth is not APPS-style "
                    f"{{inputs, outputs}} — inspect the dataset revision. Got keys: "
                    f"{list(tc)[:8] if isinstance(tc, dict) else type(tc)}"
                )
        else:
            q = ex.get("question") or ex.get("problem") or ex.get("prompt")
            t = ex.get("tests") or ex.get("test_cases") or ex.get("input_output")
            if q is None or t is None:
                raise SystemExit(
                    f"[archer] row {i}: unrecognised schema {list(ex)[:10]} — "
                    "update the field mapping in this script."
                )
            row = _std_row(i, "train", q, t if isinstance(t, str) else json.dumps(t))
        rows.append(row)

    _write(rows, out_dir / "archercoder.parquet")
    assert len(rows) == 6753, (
        f"[archer] expected 6753 tasks (paper Table 9), got {len(rows)} — "
        "dataset revision mismatch; note it in the report if you proceed."
    )


# ----------------------------------------------------------------------
# LiveCodeBench v5 (evaluation)
# ----------------------------------------------------------------------
def _lcb_decode_private(blob: str):
    """LCB private_test_cases: base64 -> zlib -> pickle -> json."""
    try:
        return json.loads(blob)
    except Exception:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(blob.encode()))))


def build_lcb(out_dir: Path) -> None:
    from datasets import load_dataset

    # release_v5 = contest window the paper evaluates (279 problems).
    ds = load_dataset(
        "livecodebench/code_generation_lite", split="test",
        version_tag="release_v5", trust_remote_code=True,
    )
    print(f"[lcb] loaded {len(ds)} rows; columns = {ds.column_names}")

    rows = []
    for i, ex in enumerate(ds):
        public = json.loads(ex["public_test_cases"])
        private = _lcb_decode_private(ex["private_test_cases"])
        meta = json.loads(ex.get("metadata") or "{}")
        fn_name = meta.get("func_name")

        tests = {"inputs": [], "outputs": []}
        if fn_name:
            tests["fn_name"] = fn_name
        for t in list(public) + list(private):
            tests["inputs"].append(t["input"])
            tests["outputs"].append(t["output"])

        question = ex["question_content"]
        if ex.get("starter_code"):
            question += (
                "\n\nYou will use the following starter code to write the solution "
                "and enclose your code within delimiters.\n```python\n"
                + ex["starter_code"] + "\n```"
            )
        else:
            question += (
                "\n\nRead the inputs from stdin, solve the problem and write the "
                "answer to stdout. Enclose your code within delimiters.\n"
                "```python\n# YOUR CODE HERE\n```"
            )
        rows.append(_std_row(i, "test", question, json.dumps(tests)))

    _write(rows, out_dir / "livecodebench_v5.parquet")
    assert len(rows) == 279, (
        f"[lcb] expected 279 problems (paper Table 9) from release_v5, got "
        f"{len(rows)} — check the version_tag / dataset revision."
    )


def _write(rows, path: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    sample = json.loads(df.iloc[0]["reward_model"]["ground_truth"]) \
        if isinstance(df.iloc[0]["reward_model"], dict) else None
    print(f"[write] {path}: {len(df)} rows")
    if sample:
        print(f"[write] first ground_truth: {len(sample.get('inputs', []))} tests, "
              f"fn_name={sample.get('fn_name')}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archer", action="store_true", help="build the ArcherCodeR training parquet")
    ap.add_argument("--lcb", action="store_true", help="build the LiveCodeBench v5 eval parquet")
    ap.add_argument("--out", default="datasets", help="output directory")
    args = ap.parse_args(argv)
    if not (args.archer or args.lcb):
        ap.error("pass --archer and/or --lcb")
    out = Path(args.out)
    if args.archer:
        build_archer(out)
    if args.lcb:
        build_lcb(out)


if __name__ == "__main__":
    sys.exit(main())
