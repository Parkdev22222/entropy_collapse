#!/usr/bin/env python3
"""Phase 3: MTP 헤드를 다른 모델 패밀리(Llama, EXAONE 등)로 이식.

모듈 자체는 공용이고 hidden_size/vocab만 다르므로, 이식은 (1) 새 모델 크기로
헤드를 재생성하고 (2) 워밍업을 다시 돌리는 것이 전부다. 하이퍼파라미터
(λ, κ, γ_H)는 **Qwen 값을 그대로 쓰는 것이 1차 시도** — "그대로 작동"이 주장
포인트이므로 재튜닝은 최소화하고, 캘리브레이션만 모델별로 재산출한다.

부가 기능: 학습 데이터의 그룹 pass rate 분포를 확인한다. 소형 Llama는 수학
baseline이 약해 전부-오답 그룹이 과다할 수 있고, 그러면 GRPO advantage가 0이라
STEER/STEER-F 모두 신호를 못 받는다 (계획서 §5.4). 조정이 필요하면 보고서에
명시할 것.

사용 예:
    # 1) 헤드 스캐폴드 점검 (형상만 확인, 학습 없음)
    python scripts/phase3_port_model.py inspect --model meta-llama/Llama-3.2-3B-Instruct

    # 2) 그룹 pass rate 분포 점검
    python scripts/phase3_port_model.py passrate \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --prompts datasets/DAPO-Math-17k.parquet --n-prompts 200 --group-size 8

    # 3) 이후는 Qwen과 동일:
    #    phase1_warmup_heads.py generate/train → phase1_validate.py (축소판)
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts._common import GenerationBackend, apply_chat_template, load_prompts_from_parquet  # noqa: E402
from scripts.phase1_validate import answers_match, extract_answer  # noqa: E402
from steer_f.mtp_heads import MTPHeads  # noqa: E402


def cmd_inspect(args) -> int:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd")
    vocab_size = cfg.vocab_size

    heads = MTPHeads(
        hidden_size=hidden_size,
        num_heads=args.num_heads,
        head_hidden=args.head_hidden,
        vocab_size=vocab_size,
    )
    n_params = sum(p.numel() for p in heads.parameters())
    tied = getattr(cfg, "tie_word_embeddings", None)

    print(json.dumps(
        {
            "model": args.model,
            "model_type": getattr(cfg, "model_type", "?"),
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "tie_word_embeddings": tied,
            "num_heads": args.num_heads,
            "head_hidden": args.head_hidden,
            "head_params_M": round(n_params / 1e6, 2),
            "head_params_pct_of_hidden_matmul": None,
        },
        indent=2,
    ))
    print(
        "\nunembedding 공유(tie_unembedding=True)를 쓰므로 헤드는 lm_head 파라미터를 "
        "복제하지 않는다. 위 head_params_M 만 새로 학습된다."
    )
    if vocab_size > 128_000:
        print(
            f"주의: vocab={vocab_size} 이 크다. forward_entropy 의 chunk_size 를 "
            "줄여 (K, chunk, V) 로짓 피크를 관리할 것."
        )
    return 0


def cmd_passrate(args) -> int:
    """그룹 pass rate 분포 — 전부-오답 그룹 비율이 신호 빈약의 직접 지표."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    problems = load_prompts_from_parquet(args.prompts, limit=args.n_prompts)
    texts = [apply_chat_template(tok, p["messages"]) for p in problems]

    backend = GenerationBackend(args.model, prefer_vllm=not args.no_vllm, tensor_parallel_size=args.tp)
    outs = backend.generate(
        texts,
        n=args.group_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_response_length,
        seed=args.seed,
    )

    rates = []
    for p, completions in zip(problems, outs):
        n_ok = sum(answers_match(extract_answer(c), p["ground_truth"]) for c in completions)
        rates.append(n_ok / max(len(completions), 1))

    hist = collections.Counter(round(r, 3) for r in rates)
    all_wrong = sum(1 for r in rates if r == 0.0) / len(rates)
    all_right = sum(1 for r in rates if r == 1.0) / len(rates)
    useful = 1.0 - all_wrong - all_right

    summary = {
        "model": args.model,
        "n_prompts": len(rates),
        "group_size": args.group_size,
        "mean_pass_rate": sum(rates) / len(rates),
        "all_wrong_frac": all_wrong,
        "all_correct_frac": all_right,
        "informative_frac": useful,
        "histogram": dict(sorted(hist.items())),
    }
    print(json.dumps(summary, indent=2))

    if all_wrong > 0.5:
        print(
            "\n경고: 전부-오답 그룹이 50%를 넘습니다 (계획서 §9 'Llama 신호 빈약').\n"
            "      GRPO advantage가 0이 되어 STEER/STEER-F 모두 신호를 못 받습니다.\n"
            "      난이도 하위 서브셋으로 조정하고, 그 사실을 보고서에 명시하세요."
        )

    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("inspect", help="새 모델의 헤드 형상/비용 점검")
    i.add_argument("--model", required=True)
    i.add_argument("--num-heads", type=int, default=8)
    i.add_argument("--head-hidden", type=int, default=1024)
    i.set_defaults(func=cmd_inspect)

    p = sub.add_parser("passrate", help="그룹 pass rate 분포 점검")
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--n-prompts", type=int, default=200)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-response-length", type=int, default=3072)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--no-vllm", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_passrate)

    return ap


if __name__ == "__main__":
    _args = build_parser().parse_args()
    raise SystemExit(_args.func(_args))
