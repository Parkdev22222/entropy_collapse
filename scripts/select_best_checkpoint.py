#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
# Licensed under the Apache License, Version 2.0
"""Pick the checkpoint the paper's selection rule points at.

Paper (App. E.2): "All methods save a checkpoint every 10 steps, and the
checkpoint achieving the highest AIME24 accuracy is selected for test."

Reads the run's tensorboard event files (pure-python parser, no tensorboard
install needed), restricts to steps that actually have a saved checkpoint on
disk, takes the argmax of the metric, and prints the HF-model path that
run/eval_steerf.sh expects.

    python scripts/select_best_checkpoint.py \
        --project STEER-F --run <run_name> \
        [--metric val-core/aime_2024_dapo_boxed/acc/mean@32] \
        [--ckpt-root checkpoints] [--tb-root tensorboard_log]
"""
from __future__ import annotations

import argparse
import glob
import os
import struct
import sys


def _varint(b, i):
    r = s = 0
    while True:
        x = b[i]; i += 1
        r |= (x & 0x7F) << s; s += 7
        if not x & 0x80:
            return r, i


def _fields(b):
    i, n = 0, len(b)
    while i < n:
        k, i = _varint(b, i)
        fn, wt = k >> 3, k & 7
        if wt == 0:
            v, i = _varint(b, i); yield fn, 0, v
        elif wt == 1:
            yield fn, 1, b[i:i + 8]; i += 8
        elif wt == 2:
            ln, i = _varint(b, i); yield fn, 2, b[i:i + ln]; i += ln
        elif wt == 5:
            yield fn, 5, b[i:i + 4]; i += 4
        else:
            raise ValueError(f"wire type {wt}")


def read_scalars(path, wanted_tag):
    """{step: value} for one tag from one tfevents file."""
    out = {}
    data = open(path, "rb").read()
    i = 0
    while i + 12 <= len(data):
        ln = struct.unpack("<Q", data[i:i + 8])[0]
        i += 12
        rec = data[i:i + ln]
        i += ln + 4
        if len(rec) < ln:
            break
        step, vals = None, {}
        for fn, wt, v in _fields(rec):
            if fn == 2 and wt == 0:
                step = v
            elif fn == 5 and wt == 2:  # summary
                for sfn, swt, sv in _fields(v):
                    if sfn == 1 and swt == 2:
                        tag = simple = None
                        for vfn, vwt, vv in _fields(sv):
                            if vfn == 1 and vwt == 2:
                                tag = vv.decode("utf8", "replace")
                            elif vfn == 2 and vwt == 5:
                                simple = struct.unpack("<f", vv)[0]
                        if tag == wanted_tag and simple is not None:
                            vals[tag] = simple
        if vals and step is not None:
            out[step] = vals[wanted_tag]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--metric", default="val-core/aime_2024_dapo_boxed/acc/mean@32")
    ap.add_argument("--ckpt-root", default="checkpoints")
    ap.add_argument("--tb-root", default="tensorboard_log")
    ap.add_argument("--quiet", action="store_true", help="print only the path")
    args = ap.parse_args(argv)

    tb_dir = os.path.join(args.tb_root, args.project, args.run)
    events = sorted(glob.glob(os.path.join(tb_dir, "events.out.tfevents.*")))
    if not events:
        sys.exit(f"[select] no tfevents under {tb_dir} — was the run started with "
                 "trainer.logger including 'tensorboard'?")

    series = {}
    for ev in events:  # later files win on step collisions (restarts)
        series.update(read_scalars(ev, args.metric))
    if not series:
        sys.exit(f"[select] metric {args.metric!r} absent from {tb_dir} — check "
                 "test_freq / val_kwargs, or pass --metric explicitly.")

    ckpt_dir = os.path.join(args.ckpt_root, args.project, args.run)
    saved = {}
    for p in glob.glob(os.path.join(ckpt_dir, "global_step_*")):
        try:
            saved[int(os.path.basename(p).split("_")[-1])] = p
        except ValueError:
            pass
    if not saved:
        sys.exit(f"[select] no global_step_* under {ckpt_dir}")

    scored = {s: v for s, v in series.items() if s in saved}
    if not scored:
        sys.exit(f"[select] no overlap between validated steps {sorted(series)[:8]}... "
                 f"and saved steps {sorted(saved)[:8]}... — check save_freq vs test_freq.")

    best = max(scored, key=scored.get)
    hf = os.path.join(saved[best], "actor", "huggingface")
    if not os.path.isdir(hf):
        # save_contents=['hf_model'] layouts differ across verl minors; take
        # whatever huggingface dir exists under the step.
        cands = glob.glob(os.path.join(saved[best], "**", "huggingface"), recursive=True)
        if not cands:
            sys.exit(f"[select] no huggingface/ dir under {saved[best]}")
        hf = cands[0]

    if not args.quiet:
        top = sorted(scored.items(), key=lambda kv: -kv[1])[:5]
        print(f"[select] {args.metric} over {len(scored)} checkpointed steps; top-5: "
              + ", ".join(f"step {s}={v:.4f}" for s, v in top), file=sys.stderr)
        print(f"[select] best step {best} ({scored[best]:.4f})", file=sys.stderr)
    print(hf)


if __name__ == "__main__":
    main()
