#!/usr/bin/env python3
"""Phase 3: MTP 헤드를 다른 모델 패밀리(Llama, EXAONE 등)로 이식.

모듈 자체는 공용이고 hidden_size/vocab만 다르므로, 이식은 (1) 새 모델 크기로
헤드를 재생성하고 (2) 워밍업을 다시 돌리는 것이 전부다. 하이퍼파라미터
(λ, κ, γ_H)는 **Qwen 값을 그대로 쓰는 것이 1차 시도** — "그대로 작동"이 주장
포인트이므로 재튜닝은 최소화하고, 캘리브레이션만 모델별로 재산출한다.

부가 기능 -- 이쪽이 실제 게이트다: 학습 데이터의 그룹 pass rate 분포를 확인한다.
소형/비수학 백본은 전부-오답 그룹이 과다할 수 있고, 그러면 GRPO advantage가 0이라
STEER/STEER-F 모두 신호를 못 받는다 (계획서 §5.4).

★ 검증셋 점수와 혼동하지 말 것. 검증셋은 "우리가 백본을 **잴 수** 있나"를 재고,
여기 informative_frac 은 "백본이 **배울 수** 있나"를 잰다. 2026-09-17 에
meta-llama/Llama-3.2-3B(base)가 MATH500 mean@1 .020 으로 떨어진 것이 그 사례다 --
정답률 2% 면 .98^8 = 85% 의 그룹이 축퇴하므로 gradient 의 6분의 1만 쓴다.

임계는 절대값으로 고르지 않는다. 이 도구의 채점기가 학습 보상과 다르므로 절대값은
보상률이 아니다. 학습이 되는 것으로 알려진 백본(Qwen2.5-Math-1.5B)을 같은 명령으로
먼저 재고, 그 informative_frac 의 비율로 --min-informative 를 준다.

사용 예:
    # 1) 헤드 스캐폴드 점검 (형상만 확인, 학습 없음)
    python scripts/phase3_port_model.py inspect --model meta-llama/Llama-3.2-3B-Instruct

    # 2) 기준선을 먼저 잰다
    python scripts/phase3_port_model.py passrate \\
        --model Qwen/Qwen2.5-Math-1.5B \\
        --prompts datasets/DAPO-Math-17k.parquet --n-prompts 200 --group-size 8 \\
        --out docs/probe_qwen15.json

    # 3) 후보를 같은 명령으로 재고, 기준선의 비율로 게이트한다 (미달이면 exit 1)
    python scripts/phase3_port_model.py passrate \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --prompts datasets/DAPO-Math-17k.parquet --n-prompts 200 --group-size 8 \\
        --min-informative <기준선의 0.5배> --out docs/probe_llama32i.json

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


def gate_verdict(all_wrong: float, informative: float, min_informative):
    """(refuse, explain) for a pass-rate probe.

    Pure on purpose: cmd_passrate needs a GPU and a 17k-problem parquet, so the
    only part a test can reach is this one -- and it is the part a queue acts
    on. Refusing is opt-in (--min-informative), because the command shipped as a
    diagnostic and existing callers must keep their exit code.
    """
    refuse = min_informative is not None and informative < min_informative
    return refuse, bool(refuse or all_wrong > 0.5)


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
        # 이 도구는 phase1_validate 의 extract_answer/answers_match 로 채점한다.
        # 학습 보상은 verl/utils/reward_score/multi_datasets_eval.py 의
        # compute_score_both (DAPO or Qwen) 라 **다른 채점기**다. 그래서 이 숫자의
        # 절대값을 보상률로 읽으면 안 되고, 두 백본을 같은 채점기로 비교하는 데만
        # 쓴다. 기록에 안 남기면 나중에 이 구분이 사라진다.
        "grader": "phase1_validate.answers_match (NOT the training reward)",
        "histogram": dict(sorted(hist.items())),
    }
    print(json.dumps(summary, indent=2))

    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(summary, indent=2))

    # 게이트. --min-informative 를 안 주면 exit 0 으로 진단 전용이던 기존 동작
    # 그대로다. 주면 큐가 이걸로 막을 수 있다.
    refused, explain = gate_verdict(all_wrong, useful, args.min_informative)
    if refused:
        print(
            f"\nREFUSE: informative_frac {useful:.3f} < --min-informative "
            f"{args.min_informative:.3f}"
        )
    elif all_wrong > 0.5:
        print(f"\n경고: 전부-오답 그룹이 {all_wrong:.0%}로 50%를 넘습니다.")

    if explain:
        # 전부-오답 그룹은 GRPO advantage 가 0 이라 gradient 에 기여하지 않는다.
        # STEER 도 STEER-F 도 그 위에 얹히므로 같이 죽는다.
        print(
            "      GRPO advantage가 0이 되어 STEER/STEER-F 모두 신호를 못 받습니다.\n"
            "\n"
            "      처방은 **백본을 바꾸는 것**입니다 (base 체크포인트면 같은 계열의\n"
            "      instruct 변종부터). 난이도 하위 서브셋으로 갈아타지 마십시오 --\n"
            "      백본마다 학습셋이 달라지면 백본 간 비교가 무너지고, 원고 §11.1은\n"
            "      전 백본이 DAPO-Math-17k를 쓴다고 적습니다."
        )

    return 1 if refused else 0


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
    p.add_argument(
        "--min-informative",
        type=float,
        default=None,
        help="informative_frac 가 이 값 미만이면 exit 1. 절대 임계를 직접 고르지 "
             "말고, 학습이 되는 것으로 알려진 백본(Qwen2.5-Math-1.5B)을 같은 명령으로 "
             "먼저 재서 그 값의 비율로 정할 것.",
    )
    p.set_defaults(func=cmd_passrate)

    return ap


if __name__ == "__main__":
    _args = build_parser().parse_args()
    raise SystemExit(_args.func(_args))
