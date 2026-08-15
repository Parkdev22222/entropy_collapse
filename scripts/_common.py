"""스크립트 공용 유틸 (데이터 로딩, 모델 로딩, 생성 백엔드).

무거운 의존성(transformers/vllm)은 **함수 안에서** import 한다 — 그래야 GPU/의존성
없는 환경에서도 이 모듈을 import 하고 순수 로직을 테스트할 수 있다.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

__all__ = [
    "Rollout",
    "load_prompts_from_parquet",
    "load_rollouts_jsonl",
    "save_rollouts_jsonl",
    "load_policy",
    "get_last_hidden_states",
    "GenerationBackend",
]


@dataclass
class Rollout:
    """워밍업/검증에서 쓰는 최소 단위."""

    prompt: str
    response: str
    problem_id: str = ""
    ground_truth: Optional[str] = None
    correct: Optional[bool] = None

    def to_json(self) -> dict:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "problem_id": self.problem_id,
            "ground_truth": self.ground_truth,
            "correct": self.correct,
        }


# ----------------------------------------------------------------------
# 데이터
# ----------------------------------------------------------------------
def load_prompts_from_parquet(path: str | pathlib.Path, limit: Optional[int] = None) -> list[dict]:
    """STEER 동봉 parquet(DAPO-Math-17k 등)에서 프롬프트를 읽는다.

    스키마: data_source / prompt(list[{role, content}]) / ability / reward_model / extra_info
    """
    import pandas as pd

    df = pd.read_parquet(path)
    if limit is not None:
        df = df.head(limit)

    out = []
    for i, row in df.iterrows():
        messages = list(row["prompt"])
        rm = row.get("reward_model")
        gt = None
        if isinstance(rm, dict):
            gt = rm.get("ground_truth")
        extra = row.get("extra_info")
        pid = extra.get("index") if isinstance(extra, dict) else None
        out.append(
            {
                "problem_id": str(pid if pid is not None else i),
                "messages": [dict(m) for m in messages],
                "ground_truth": gt,
            }
        )
    return out


def save_rollouts_jsonl(rollouts: Iterable[Rollout], path: str | pathlib.Path) -> int:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in rollouts:
            f.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")
            n += 1
    return n


def load_rollouts_jsonl(path: str | pathlib.Path, limit: Optional[int] = None) -> list[Rollout]:
    out = []
    with pathlib.Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(
                Rollout(
                    prompt=d["prompt"],
                    response=d["response"],
                    problem_id=d.get("problem_id", ""),
                    ground_truth=d.get("ground_truth"),
                    correct=d.get("correct"),
                )
            )
            if limit is not None and len(out) >= limit:
                break
    return out


# ----------------------------------------------------------------------
# 모델
# ----------------------------------------------------------------------
def load_policy(model_path: str, dtype: str = "bfloat16", device: str = "cuda"):
    """(model, tokenizer) 반환. 정책은 항상 eval 모드로 둔다."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def get_last_hidden_states(model, input_ids, attention_mask):
    """마지막 레이어 히든 (B, T, H)와 정책 로짓 (B, T, V)을 함께 반환.

    위치 t의 히든은 s_t = (prompt, y_<t)를 조건으로 하며 y_t를 예측한다
    — MTP 헤드의 입력 정렬 기준 (docs/steer_code_map.md §5).
    """
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    return out.hidden_states[-1], out.logits


def get_lm_head(model):
    """본체 unembedding. 헤드가 공유해서 쓴다 (tie_unembedding=True)."""
    return model.get_output_embeddings()


# ----------------------------------------------------------------------
# 생성 백엔드
# ----------------------------------------------------------------------
class GenerationBackend:
    """vLLM이 있으면 vLLM, 없으면 HF generate로 폴백."""

    def __init__(self, model_path: str, prefer_vllm: bool = True, dtype: str = "bfloat16",
                 gpu_memory_utilization: float = 0.6, tensor_parallel_size: int = 1,
                 max_model_len: int = 4096):
        self.model_path = model_path
        self.kind = "hf"
        self._llm = None
        if prefer_vllm:
            try:
                from vllm import LLM

                self._llm = LLM(
                    model=model_path,
                    dtype=dtype,
                    gpu_memory_utilization=gpu_memory_utilization,
                    tensor_parallel_size=tensor_parallel_size,
                    max_model_len=max_model_len,
                )
                self.kind = "vllm"
            except Exception as exc:  # pragma: no cover - 환경 의존
                print(f"[GenerationBackend] vLLM 사용 불가 ({exc}), HF generate로 폴백")

        if self.kind == "hf":
            self._model, self._tok = load_policy(model_path, dtype=dtype)

    def generate(
        self,
        prompts: Sequence[str],
        n: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 3072,
        seed: Optional[int] = None,
    ) -> list[list[str]]:
        """prompts[i]마다 n개의 continuation 문자열."""
        if self.kind == "vllm":
            from vllm import SamplingParams

            params = SamplingParams(
                n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed
            )
            outs = self._llm.generate(list(prompts), params)
            return [[o.text for o in out.outputs] for out in outs]

        import torch  # pragma: no cover - 환경 의존

        results = []
        for prompt in prompts:
            enc = self._tok(prompt, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                gen = self._model.generate(
                    **enc,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    num_return_sequences=n,
                    pad_token_id=self._tok.pad_token_id,
                )
            cut = enc["input_ids"].shape[1]
            results.append([self._tok.decode(g[cut:], skip_special_tokens=True) for g in gen])
        return results


def apply_chat_template(tok: Any, messages: list[dict]) -> str:
    """chat template이 있으면 적용, 없으면 content를 이어붙인다."""
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n".join(m["content"] for m in messages)
