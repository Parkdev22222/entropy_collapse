#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
# Licensed under the Apache License, Version 2.0
"""Write one copy of each problem from a replicated eval parquet.

    python3 scripts/dedupe_replicas.py datasets/aime24.parquet datasets/passk/aime24.parquet

The avg@32 parquets (aime24, aime25, amc23) carry every problem 32 times, so the
avg@32 pass can run at val_kwargs.n=1 (run/eval_steerf.sh, as upstream's
eval.sh does). The Pass@k pass runs at n=1024, and on those files that is
32 x 1024 = 32,768 samples per problem -- 32 times the generation the figure
needs. This keeps the first row of each prompt, in file order, and refuses a
file whose problems are not all replicated the same number of times: an
uneven file is not the kind this was written for, and guessing which copy is
"the" problem there would be a silent choice.

Exit status: 0 on success, 2 on a file it refuses.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def prompt_key(row):
    return json.dumps(row["prompt"], sort_keys=True, default=str)


def dedupe(src, dst, expect=None):
    table = pq.read_table(src)
    keys = [prompt_key(r) for r in table.select(["prompt"]).to_pylist()]
    counts = Counter(keys)
    copies = set(counts.values())
    if len(copies) != 1:
        sys.exit(f"[dedupe] REFUSE {src}: problems are replicated unevenly "
                 f"({dict(Counter(counts.values()))} problems per copy count)")
    seen, keep = set(), []
    for i, k in enumerate(keys):
        if k not in seen:
            seen.add(k)
            keep.append(i)
    if expect is not None and len(keep) != expect:
        sys.exit(f"[dedupe] REFUSE {src}: {len(keep)} unique problems, expected {expect}")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.take(keep), dst)
    print(f"[dedupe] {src}: {table.num_rows} rows = {len(keep)} problems x {copies.pop()} "
          f"-> {dst} ({len(keep)} rows)")
    return len(keep)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--expect", type=int, default=None,
                    help="refuse unless this many unique problems remain")
    args = ap.parse_args(argv)
    dedupe(args.src, args.dst, args.expect)


if __name__ == "__main__":
    main()
