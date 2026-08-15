#!/usr/bin/env python3
"""Phase 1: MTP 헤드 워밍업 (정책 freeze, 헤드만 cross-entropy 학습).

위치 t의 헤드 k 타깃 = 실제 토큰 y_{t+k}. 손실은 **응답 토큰에만** 건다
(프롬프트 구간의 예보는 RL에서 쓰지 않으므로 학습할 이유가 없다).

K=8로 학습해 두면 κ 선택은 사후 분석으로 끝난다 (헤드가 서로 독립이라 재학습 불필요).

사용 예:
    # 1) 롤아웃 생성 (없으면)
    python scripts/phase1_warmup_heads.py generate \\
        --model Qwen/Qwen2.5-Math-7B \\
        --prompts datasets/DAPO-Math-17k.parquet \\
        --n-prompts 6250 --n-samples 8 \\
        --out artifacts/rollouts.jsonl

    # 2) 헤드 학습
    python scripts/phase1_warmup_heads.py train \\
        --model Qwen/Qwen2.5-Math-7B \\
        --rollouts artifacts/rollouts.jsonl \\
        --num-heads 8 --epochs 1 \\
        --out artifacts/mtp_heads_qwen7b.pt
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts._common import (  # noqa: E402
    GenerationBackend,
    Rollout,
    apply_chat_template,
    get_last_hidden_states,
    get_lm_head,
    load_policy,
    load_prompts_from_parquet,
    load_rollouts_jsonl,
    save_rollouts_jsonl,
)
from steer_f.mtp_heads import MTPHeads  # noqa: E402


# ----------------------------------------------------------------------
# 롤아웃 생성
# ----------------------------------------------------------------------
def cmd_generate(args) -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    problems = load_prompts_from_parquet(args.prompts, limit=args.n_prompts)
    print(f"[generate] {len(problems)} prompts × {args.n_samples} samples")

    backend = GenerationBackend(
        args.model,
        prefer_vllm=not args.no_vllm,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
    )
    texts = [apply_chat_template(tok, p["messages"]) for p in problems]

    rollouts: list[Rollout] = []
    batch = args.gen_batch
    for start in range(0, len(texts), batch):
        stop = min(start + batch, len(texts))
        outs = backend.generate(
            texts[start:stop],
            n=args.n_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_response_length,
            seed=args.seed,
        )
        for local, completions in enumerate(outs):
            p = problems[start + local]
            for c in completions:
                rollouts.append(
                    Rollout(
                        prompt=texts[start + local],
                        response=c,
                        problem_id=p["problem_id"],
                        ground_truth=p["ground_truth"],
                    )
                )
        print(f"[generate] {stop}/{len(texts)} prompts, {len(rollouts)} rollouts")

    n = save_rollouts_jsonl(rollouts, args.out)
    print(f"[generate] wrote {n} rollouts -> {args.out}")
    return 0


# ----------------------------------------------------------------------
# 헤드 학습
# ----------------------------------------------------------------------
def encode_batch(tok, rollouts, max_len: int, device):
    """(input_ids, attention_mask, response_mask). response_mask는 응답 토큰만 1."""
    import torch

    ids_list, resp_start = [], []
    for r in rollouts:
        p_ids = tok(r.prompt, add_special_tokens=False)["input_ids"]
        r_ids = tok(r.response, add_special_tokens=False)["input_ids"]
        full = (p_ids + r_ids)[:max_len]
        ids_list.append(full)
        resp_start.append(min(len(p_ids), len(full)))

    t = max(len(x) for x in ids_list)
    pad = tok.pad_token_id or 0
    input_ids = torch.full((len(ids_list), t), pad, dtype=torch.long)
    attn = torch.zeros((len(ids_list), t), dtype=torch.long)
    resp = torch.zeros((len(ids_list), t), dtype=torch.long)
    for i, x in enumerate(ids_list):
        input_ids[i, : len(x)] = torch.tensor(x, dtype=torch.long)
        attn[i, : len(x)] = 1
        resp[i, resp_start[i] : len(x)] = 1
    return input_ids.to(device), attn.to(device), resp.to(device)


def cmd_train(args) -> int:
    import torch

    device = args.device
    model, tok = load_policy(args.model, dtype=args.dtype, device=device)
    lm_head = get_lm_head(model)
    for p in lm_head.parameters():
        p.requires_grad_(False)

    hidden_size = model.config.hidden_size
    heads = MTPHeads(
        hidden_size=hidden_size,
        num_heads=args.num_heads,
        head_hidden=args.head_hidden,
        vocab_size=model.config.vocab_size,
    ).to(device=device, dtype=torch.float32)

    n_params = sum(p.numel() for p in heads.parameters())
    print(f"[train] MTP heads: K={args.num_heads}, head_hidden={args.head_hidden}, {n_params/1e6:.1f}M params")

    rollouts = load_rollouts_jsonl(args.rollouts, limit=args.limit)
    print(f"[train] {len(rollouts)} rollouts, {args.epochs} epoch(s), batch {args.batch_size}")

    opt = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_steps = args.epochs * math.ceil(len(rollouts) / args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(n_steps, 1), pct_start=0.05)

    log: list[dict] = []
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for start in range(0, len(rollouts), args.batch_size):
            chunk = rollouts[start : start + args.batch_size]
            input_ids, attn, resp = encode_batch(tok, chunk, args.max_len, device)
            if resp.sum() == 0:
                continue

            with torch.no_grad():
                hidden, _ = get_last_hidden_states(model, input_ids, attn)
            hidden = hidden.detach().float()

            loss, per_head = heads.ce_loss(hidden, lm_head, input_ids, mask=resp)
            if not torch.isfinite(loss):
                print(f"[train] step {step}: non-finite loss, skipped")
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(heads.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            step += 1

            if step % args.log_every == 0:
                row = {"step": step, "epoch": epoch, "loss": float(loss),
                       "lr": sched.get_last_lr()[0], "elapsed_s": time.time() - t0}
                row.update({k: float(v) for k, v in per_head.items()})
                log.append(row)
                heads_str = " ".join(f"k{ i+1 }={row.get(f'mtp/ce_head_{i+1}', float('nan')):.3f}"
                                     for i in range(args.num_heads))
                print(f"[train] step {step}/{n_steps} loss={float(loss):.4f} | {heads_str}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": heads.state_dict(),
            "config": {
                "hidden_size": hidden_size,
                "num_heads": args.num_heads,
                "head_hidden": args.head_hidden,
                "vocab_size": model.config.vocab_size,
                "residual": True,
                "base_model": args.model,
            },
            "train_log": log,
        },
        out,
    )
    print(f"[train] saved -> {out}")
    (out.with_suffix(".log.json")).write_text(json.dumps(log, indent=2))
    return 0


# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="워밍업용 롤아웃 생성")
    g.add_argument("--model", required=True)
    g.add_argument("--prompts", required=True, help="DAPO-Math-17k.parquet 등")
    g.add_argument("--out", required=True)
    g.add_argument("--n-prompts", type=int, default=6250)
    g.add_argument("--n-samples", type=int, default=8, help="프롬프트당 샘플 수 (기본 8 → 50k 시퀀스)")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--max-response-length", type=int, default=3072)
    g.add_argument("--max-model-len", type=int, default=4096)
    g.add_argument("--gen-batch", type=int, default=512)
    g.add_argument("--tp", type=int, default=1)
    g.add_argument("--no-vllm", action="store_true")
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("train", help="헤드 CE 학습")
    t.add_argument("--model", required=True)
    t.add_argument("--rollouts", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--num-heads", type=int, default=8)
    t.add_argument("--head-hidden", type=int, default=1024)
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--max-len", type=int, default=4096)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--weight-decay", type=float, default=0.0)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--limit", type=int, default=None)
    t.add_argument("--log-every", type=int, default=20)
    t.add_argument("--dtype", default="bfloat16")
    t.add_argument("--device", default="cuda")
    t.set_defaults(func=cmd_train)

    return ap


if __name__ == "__main__":
    _args = build_parser().parse_args()
    raise SystemExit(_args.func(_args))
