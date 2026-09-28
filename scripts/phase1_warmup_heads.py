#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Phase 1 — warm up the MTP heads with the policy frozen.

Trains heads 1..K by cross-entropy against the tokens actually observed at
offsets +1..+K in rollout text.  The policy never updates, so this is cheap:
one forward per batch with `torch.no_grad`, and gradients only through the
small head MLPs.

Input is a JSONL of rollouts, one object per line with a `text` field (or
`prompt` + `response`).  Produce it from a Phase-0 run — verl writes rollouts
to `trainer.rollout_data_dir` — or generate more with
`scripts/phase1_sample_rollouts.py`.

    python scripts/phase1_warmup_heads.py \
        --model Qwen/Qwen2.5-Math-7B \
        --rollouts rollout_data/STEER/run/rollouts.jsonl \
        --out checkpoints/mtp_heads_qwen7b.pt \
        --num-heads 8 --epochs 1

Plan §3.1 asks for ~50k sequences.  Fewer will train the near heads fine and
leave the far ones under-fit, which shows up in Phase 1 validation as rho
collapsing with kappa — the first remedy the plan prescribes on a G1 failure
is more warm-up data, so record `--max-sequences` in the report.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from steer_f.mtp_heads import MTPHeads, mtp_ce_loss  # noqa: E402
from steer_f.mtp_sequential import SequentialMTPHeads, seq_mtp_ce_loss  # noqa: E402


class RolloutTextDataset(Dataset):
    """Tokenised rollout text, right-padded to a fixed length."""

    def __init__(self, path, tokenizer, max_length=1024, max_sequences=None, field="text"):
        self.samples = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get(field)
                if text is None:
                    text = (obj.get("prompt") or "") + (obj.get("response") or "")
                if not text:
                    continue
                self.samples.append(text)
                if max_sequences and len(self.samples) >= max_sequences:
                    break
        if not self.samples:
            raise ValueError(f"no usable rollouts in {path}")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.samples[i],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
        }


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--rollouts", required=True, help="JSONL of rollout text")
    p.add_argument("--out", required=True, help="where to write the head checkpoint")
    p.add_argument("--num-heads", type=int, default=8,
                   help="K. Train the largest K you may want; kappa is chosen post hoc")
    p.add_argument("--head-hidden", type=int, default=1024)
    p.add_argument("--sequential", action="store_true",
                   help="DeepSeek-V3 style heads: depth k consumes Emb(y_{t+k}) as well "
                        "as the previous depth's hidden. Plan §3 remedy 3 for a G1 "
                        "failure. Costs one embedding lookup and one block per depth, "
                        "and makes H_togo conditional on the realised continuation")
    p.add_argument("--no-tie-unembedding", action="store_true",
                   help="give the heads their own unembedding (much larger checkpoint)")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--max-sequences", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=2048,
                   help="positions per unembedding call; lower it if memory is tight")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    print(f"[warmup] loading {args.model} ({args.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    ).to(args.device)
    model.eval()
    model.requires_grad_(False)  # policy frozen — this is a warm-up, not RL

    hidden_size = model.config.hidden_size
    vocab_size = model.config.vocab_size
    lm_head = model.get_output_embeddings()

    head_cls = SequentialMTPHeads if args.sequential else MTPHeads
    heads = head_cls(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_heads=args.num_heads,
        head_hidden=args.head_hidden,
        tie_unembedding=not args.no_tie_unembedding,
        dtype=dtype,
    ).to(args.device)
    embedding = model.get_input_embeddings() if args.sequential else None
    n_params = sum(p.numel() for p in heads.parameters())
    print(f"[warmup] heads: K={args.num_heads}, {n_params/1e6:.1f}M params")

    dataset = RolloutTextDataset(
        args.rollouts, tokenizer, max_length=args.max_length, max_sequences=args.max_sequences
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"[warmup] {len(dataset)} sequences, {len(loader)} batches/epoch")

    opt = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, (len(loader) * args.epochs) // args.grad_accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / max(1, args.warmup_steps))
        * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))),
    )

    step, started = 0, time.time()
    for epoch in range(args.epochs):
        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)

            # The body is frozen, so its hidden states need no graph.
            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
            hidden = out.hidden_states[-1].detach()

            if args.sequential:
                loss, metrics = seq_mtp_ce_loss(
                    heads, hidden, input_ids, attention_mask,
                    lm_head=lm_head, embedding=embedding, chunk_size=args.chunk_size,
                )
            else:
                loss, metrics = mtp_ce_loss(
                    heads, hidden, input_ids, attention_mask,
                    lm_head=lm_head, chunk_size=args.chunk_size,
                )
            (loss / args.grad_accum).backward()

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                if step % args.log_every == 0:
                    per_head = " ".join(
                        f"k{k}={metrics.get(f'mtp/ce_k{k}', float('nan')):.3f}"
                        for k in range(args.num_heads)
                    )
                    print(
                        f"[warmup] epoch {epoch} step {step}/{total_steps} "
                        f"loss={float(loss):.4f} lr={sched.get_last_lr()[0]:.2e} "
                        f"({time.time()-started:.0f}s) | {per_head}",
                        flush=True,
                    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": heads.state_dict(),
            "config": {
                "hidden_size": hidden_size,
                "vocab_size": vocab_size,
                "num_heads": args.num_heads,
                "head_hidden": args.head_hidden,
                "tie_unembedding": not args.no_tie_unembedding,
            },
            "sequential": bool(args.sequential),
            "model": args.model,
            "train_args": vars(args),
            "final_metrics": metrics,
        },
        out_path,
    )
    print(f"[warmup] wrote {out_path}")
    print("[warmup] per-head final CE (expect monotone increase with k):")
    for k in range(args.num_heads):
        print(f"           +{k+1}: {metrics.get(f'mtp/ce_k{k}', float('nan')):.4f}")


if __name__ == "__main__":
    main()
