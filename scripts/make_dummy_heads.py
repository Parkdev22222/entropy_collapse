#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Write an untrained MTP-head checkpoint, for plumbing tests only.

`docs/experiment_log.md` ("Phase 2 미해결 리스크" #1) records that the forecast
pass — `_steerf_unembedding()` and the extra no-grad forward it drives — has
never been executed under FSDP.  That path is only reachable with lambda > 0
and a loadable head checkpoint, which Phase 1 does not produce until much
later.  This script makes the cheap half of that pair available immediately so
the integration can fail in three minutes rather than at the top of Phase 2.

The heads are `zero_init_output=True` and residual, i.e. every head starts as a
copy of the +1 predictor.  The resulting `A_H` is therefore near-degenerate and
carries no forecasting signal whatsoever.

    NOT AN EXPERIMENT.  A run using these heads measures whether the code runs,
    never whether STEER-F works.  Phase 2 must load Phase 1's real checkpoint.

    python scripts/make_dummy_heads.py --model <hf id or path> \
        --out checkpoints/mtp_heads_smoke.pt --num-heads 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from steer_f.mtp_heads import MTPHeads  # noqa: E402


def build_argparser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--out", required=True)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--head-hidden", type=int, default=1024)
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)

    # Read the shapes from config.json rather than instantiating the model:
    # this runs while a training job may still hold the GPU.
    cfg_path = Path(args.model) / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
    else:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True).to_dict()

    hidden_size = cfg["hidden_size"]
    vocab_size = cfg["vocab_size"]
    dtype = getattr(torch, args.dtype)

    heads = MTPHeads(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_heads=args.num_heads,
        head_hidden=args.head_hidden,
        tie_unembedding=True,
        dtype=dtype,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": heads.state_dict(),
            "config": {
                "hidden_size": hidden_size,
                "vocab_size": vocab_size,
                "num_heads": args.num_heads,
                "head_hidden": args.head_hidden,
                "tie_unembedding": True,
            },
            "model": args.model,
            "untrained": True,  # tripwire: real checkpoints do not carry this
        },
        out,
    )
    print(f"[dummy-heads] wrote UNTRAINED heads to {out} "
          f"(H={hidden_size}, V={vocab_size}, K={args.num_heads})")
    print("[dummy-heads] plumbing test only — not a STEER-F result")


if __name__ == "__main__":
    main()
