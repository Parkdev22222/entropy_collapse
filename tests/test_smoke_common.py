"""`scripts/_common.py` 의 스모크 경로 및 의존성 없는 프롬프트 로딩.

여기서 검증하는 것은 두 가지다.

1. `smoke:` 센티널 모델 경로가 transformers 없이 정책/생성 백엔드를 만든다.
2. 프롬프트 파일이 parquet 이 아니라 json/jsonl 이어도 로딩된다 (pandas/pyarrow 불필요).

둘 다 **실제 경로를 건드리지 않는 추가 분기**여야 한다 — 실모델 경로의 동작은
그대로 남아야 하므로, 센티널이 아닌 입력에서는 기존과 동일하게 transformers 를 탄다.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts._common import (  # noqa: E402
    GenerationBackend,
    get_last_hidden_states,
    get_lm_head,
    is_smoke_model,
    load_policy,
    load_prompts_from_parquet,
)


# ----------------------------------------------------------------------
# 센티널 판별
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "path,expected",
    [
        ("smoke:", True),
        ("smoke:h16,l2,seed3", True),
        ("Qwen/Qwen2.5-Math-7B", False),
        ("/models/smoke/qwen", False),  # 경로 중간의 'smoke' 는 센티널이 아니다
    ],
)
def test_is_smoke_model_only_matches_prefix(path, expected):
    assert is_smoke_model(path) is expected


# ----------------------------------------------------------------------
# 정책 로딩
# ----------------------------------------------------------------------
def test_load_policy_smoke_needs_no_transformers():
    model, tok = load_policy("smoke:h16,l2", dtype="float32", device="cpu")
    assert model.config.vocab_size == tok.vocab_size
    assert model.config.hidden_size == 16


def test_load_policy_smoke_parses_options():
    model, _ = load_policy("smoke:h24,l3,seed5", dtype="float32", device="cpu")
    assert model.config.hidden_size == 24
    assert len(model.blocks) == 3


def test_load_policy_smoke_defaults_are_usable_without_options():
    model, tok = load_policy("smoke:", dtype="float32", device="cpu")
    assert model.config.hidden_size > 0 and tok.vocab_size > 0


def test_load_policy_smoke_returns_frozen_eval_policy():
    model, _ = load_policy("smoke:h16,l2", dtype="float32", device="cpu")
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())


def test_get_last_hidden_states_works_on_smoke_policy():
    """`get_last_hidden_states` 는 실모델과 스모크 모델에서 동일하게 동작해야 한다."""
    import torch

    model, tok = load_policy("smoke:h16,l2", dtype="float32", device="cpu")
    ids = torch.tensor([tok("2+2=", add_special_tokens=False)["input_ids"]])
    hidden, logits = get_last_hidden_states(model, ids, torch.ones_like(ids))
    assert hidden.shape == (1, ids.shape[1], 16)
    assert logits.shape == (1, ids.shape[1], tok.vocab_size)
    assert get_lm_head(model)(hidden).shape == logits.shape


# ----------------------------------------------------------------------
# 생성 백엔드
# ----------------------------------------------------------------------
def test_generation_backend_smoke_returns_n_completions_per_prompt():
    backend = GenerationBackend("smoke:h16,l2", prefer_vllm=False, dtype="float32")
    outs = backend.generate(["a+b=", "c*d="], n=3, temperature=1.0, max_tokens=6, seed=0)
    assert len(outs) == 2
    assert all(len(o) == 3 for o in outs)
    assert all(isinstance(s, str) for o in outs for s in o)


def test_generation_backend_smoke_never_tries_vllm():
    """vLLM 이 설치돼 있어도 스모크 모델을 vLLM 에 넘기면 안 된다 (경로가 없다)."""
    backend = GenerationBackend("smoke:h16,l2", prefer_vllm=True, dtype="float32")
    assert backend.kind == "smoke"


def test_generation_backend_smoke_respects_seed():
    backend = GenerationBackend("smoke:h16,l2", prefer_vllm=False, dtype="float32")
    a = backend.generate(["x="], n=2, temperature=1.0, max_tokens=5, seed=11)
    b = backend.generate(["x="], n=2, temperature=1.0, max_tokens=5, seed=11)
    assert a == b


# ----------------------------------------------------------------------
# 프롬프트 로딩 (parquet 없이)
# ----------------------------------------------------------------------
def _write_problems(path: pathlib.Path, n: int) -> None:
    rows = [
        {
            "problem_id": f"p{i}",
            "prompt": [{"role": "user", "content": f"question {i}"}],
            "reward_model": {"ground_truth": str(i)},
            "extra_info": {"index": f"p{i}"},
        }
        for i in range(n)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))


def test_load_prompts_reads_jsonl_without_pandas(tmp_path):
    p = tmp_path / "problems.jsonl"
    _write_problems(p, 3)
    got = load_prompts_from_parquet(p)
    assert [g["problem_id"] for g in got] == ["p0", "p1", "p2"]
    assert got[0]["messages"] == [{"role": "user", "content": "question 0"}]
    assert got[0]["ground_truth"] == "0"


def test_load_prompts_reads_json_array(tmp_path):
    p = tmp_path / "problems.json"
    rows = [{"problem_id": "a", "prompt": [{"role": "user", "content": "q"}],
             "reward_model": {"ground_truth": "7"}}]
    p.write_text(json.dumps(rows))
    got = load_prompts_from_parquet(p)
    assert got[0]["ground_truth"] == "7"


def test_load_prompts_honors_limit(tmp_path):
    p = tmp_path / "problems.jsonl"
    _write_problems(p, 5)
    assert len(load_prompts_from_parquet(p, limit=2)) == 2


def test_load_prompts_falls_back_to_index_when_id_missing(tmp_path):
    p = tmp_path / "problems.jsonl"
    p.write_text(json.dumps({"prompt": [{"role": "user", "content": "q"}]}))
    assert load_prompts_from_parquet(p)[0]["problem_id"] == "0"
