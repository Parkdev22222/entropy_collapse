#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Recompute a `phase1_sibling_spread.py` report from its saved JSON.

`phase1_sibling_spread.py` writes every per-cut-point, per-branch record it
measured, and `summarise()` reads only those records -- no model, no sampling,
no `args`.  So when the analysis changes, the report can be regenerated from a
run that already happened instead of spending the GPU hours again:

    python scripts/resummarise_sibling_spread.py \
        docs/phase1_sibling_spread_Qwen2.5-Math-1.5B.json

That is what this exists for right now.  Runs made before the tail statistics
were added report Part A as a single median-based `sibling / position ratio`,
which cannot distinguish

    world 1  every branch point has near-equal siblings
    world 2  most are equal but a small tail is wildly unequal

-- both give the same ratio, and they lead to different papers.  Re-running
this over the saved records prints the within-cut-point spread quantiles and
the exceedance fractions that separate them.

`--write` updates the JSON's `summary` block in place (records untouched) so
the file carries the current analysis; the default only prints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.phase1_sibling_spread import render_report, summarise  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("json_path", help="output of scripts/phase1_sibling_spread.py")
    p.add_argument("--write", action="store_true",
                   help="also overwrite the file's `summary` block with the new one")
    args = p.parse_args(argv)

    path = Path(args.json_path)
    if not path.exists():
        raise SystemExit(f"[resummarise] no such file: {path}")
    data = json.loads(path.read_text())

    records = data.get("records")
    if not records:
        raise SystemExit(
            f"[resummarise] {path} has no `records` block. Only runs that saved "
            "their per-cut-point records can be re-analysed; rerun "
            "phase1_sibling_spread.py to produce one."
        )

    # summarise() takes an `args` parameter it does not read; None keeps that
    # explicit rather than fabricating a stand-in whose fields might start
    # mattering silently later.
    summary = summarise(records, None)

    n_skipped = data.get("skipped_no_branch")
    print(f"[resummarise] {path}")
    print(f"[resummarise] {len(records)} cut points"
          + (f", {n_skipped} skipped at measurement time" if n_skipped is not None else ""))
    src = data.get("args") or {}
    if src:
        print(f"[resummarise] source run: model={src.get('model')} "
              f"K_mc={src.get('n_continuations')} min_group={src.get('min_group')} "
              f"embed_model={src.get('embed_model')!r}")
        if not src.get("embed_model"):
            print("[resummarise] WARNING: that run had no --embed-model, so branch "
                  "diversity fell back to first-divergence depth. Every continuation "
                  "inside a branch shares its first token, so the fallback floors at "
                  "depth 1 and UNDERSTATES Part A -- it can manufacture a 'siblings "
                  "are uniform' reading that is an artefact.")
    print()
    print(render_report(summary))

    if args.write:
        data["summary"] = summary
        path.write_text(json.dumps(data, indent=2))
        print(f"\n[resummarise] updated `summary` in {path} (records untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
