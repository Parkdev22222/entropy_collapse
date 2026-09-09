"""`scripts/smoke_model.py` 의 HF 호환 계약 테스트.

스모크 스택은 transformers/GPU 없이 Phase 1 스크립트를 완주시키기 위한 것이므로,
실제 코드가 **실제로 호출하는 인터페이스**만 정확히 흉내내면 된다.
여기서 검증하는 계약의 출처는 전부 `scripts/_common.py` 와 두 Phase 1 스크립트다.
"""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts.smoke_model import TinyCausalLM, TinyTokenizer, build_smoke_stack  # noqa: E402


# ----------------------------------------------------------------------
# 토크나이저 계약
# ----------------------------------------------------------------------
def test_tokenizer_call_returns_input_ids_list():
    """`tok(text, add_special_tokens=False)["input_ids"]` — _common/두 스크립트 공통 경로."""
    tok = TinyTokenizer()
    ids = tok("hello", add_special_tokens=False)["input_ids"]
    assert isinstance(ids, list)
    assert len(ids) == len("hello")
    assert all(isinstance(i, int) and 0 <= i < tok.vocab_size for i in ids)


def test_tokenizer_roundtrip_preserves_text():
    tok = TinyTokenizer()
    text = "step 1: 2+2=4\n"
    assert tok.decode(tok(text, add_special_tokens=False)["input_ids"]) == text


def test_tokenizer_return_tensors_pt_is_movable_batch():
    """GenerationBackend HF 폴백이 `tok(prompt, return_tensors='pt').to(device)` 를 쓴다."""
    tok = TinyTokenizer()
    enc = tok("abc", return_tensors="pt").to("cpu")
    assert enc["input_ids"].shape == (1, 3)
    assert enc["attention_mask"].shape == (1, 3)


def test_tokenizer_has_pad_and_eos_ids():
    tok = TinyTokenizer()
    assert isinstance(tok.pad_token_id, int)
    assert tok.pad_token is not None and tok.eos_token is not None


def test_apply_chat_template_joins_messages():
    """`_common.apply_chat_template` 는 chat_template 이 없으면 content 를 잇는다."""
    from scripts._common import apply_chat_template

    tok = TinyTokenizer()
    out = apply_chat_template(tok, [{"role": "user", "content": "q1"}])
    assert "q1" in out


# ----------------------------------------------------------------------
# 모델 계약
# ----------------------------------------------------------------------
def test_model_forward_exposes_logits_and_hidden_states():
    """`get_last_hidden_states` 가 요구하는 정확한 형상."""
    model = TinyCausalLM(vocab_size=17, hidden_size=8, num_layers=2)
    ids = torch.randint(0, 17, (2, 5))
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                output_hidden_states=True, use_cache=False)
    assert out.logits.shape == (2, 5, 17)
    assert out.hidden_states[-1].shape == (2, 5, 8)


def test_model_forward_without_hidden_states_still_gives_logits():
    """stage_ground_truth 는 output_hidden_states 를 주지 않고 로짓만 쓴다."""
    model = TinyCausalLM(vocab_size=17, hidden_size=8, num_layers=2)
    ids = torch.randint(0, 17, (1, 4))
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    assert out.logits.shape == (1, 4, 17)


def test_model_is_causal_so_hidden_at_t_ignores_future_tokens():
    """위치 t의 히든이 y_{>t} 를 보면 A_H 신호 전체가 미래 누설로 오염된다.

    `docs/steer_code_map.md` §5 의 정렬 전제(위치 t = s_t 조건)를 스모크 모델도 지켜야
    검증 파이프라인의 상관계수가 의미를 갖는다.
    """
    model = TinyCausalLM(vocab_size=17, hidden_size=8, num_layers=2)
    a = torch.tensor([[3, 1, 4, 1, 5]])
    b = a.clone()
    b[0, -1] = 9  # 마지막 토큰만 변경
    ha = model(input_ids=a, attention_mask=torch.ones_like(a), output_hidden_states=True).hidden_states[-1]
    hb = model(input_ids=b, attention_mask=torch.ones_like(b), output_hidden_states=True).hidden_states[-1]
    torch.testing.assert_close(ha[:, :-1, :], hb[:, :-1, :])


def test_model_config_and_output_embeddings_match_shapes():
    """헤드가 본체 lm_head 를 공유한다 (tie_unembedding=True)."""
    model = TinyCausalLM(vocab_size=17, hidden_size=8, num_layers=2)
    assert model.config.hidden_size == 8
    assert model.config.vocab_size == 17
    lm_head = model.get_output_embeddings()
    assert lm_head(torch.zeros(3, 8)).shape == (3, 17)


def test_model_generate_returns_prompt_plus_new_tokens():
    """GenerationBackend HF 폴백 경로가 쓰는 시그니처."""
    model = TinyCausalLM(vocab_size=17, hidden_size=8, num_layers=2)
    ids = torch.randint(0, 17, (1, 3))
    gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                         do_sample=True, temperature=1.0, top_p=0.9,
                         max_new_tokens=5, num_return_sequences=4, pad_token_id=0)
    assert gen.shape == (4, 8)
    assert torch.equal(gen[:, :3], ids.expand(4, 3))


def test_model_generate_is_deterministic_under_manual_seed():
    model = TinyCausalLM(vocab_size=17, hidden_size=8, num_layers=2)
    ids = torch.randint(0, 17, (1, 3))
    kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
              temperature=1.0, top_p=1.0, max_new_tokens=4, num_return_sequences=2,
              pad_token_id=0)
    torch.manual_seed(0)
    first = model.generate(**kw)
    torch.manual_seed(0)
    assert torch.equal(first, model.generate(**kw))


# ----------------------------------------------------------------------
# 스택 빌더
# ----------------------------------------------------------------------
def test_build_smoke_stack_returns_matched_model_and_tokenizer():
    model, tok = build_smoke_stack(hidden_size=8, num_layers=2, seed=0)
    assert model.config.vocab_size == tok.vocab_size
    ids = torch.tensor([tok("hi", add_special_tokens=False)["input_ids"]])
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
    assert out.logits.shape[-1] == tok.vocab_size


def test_build_smoke_stack_is_reproducible_across_seeds():
    m1, _ = build_smoke_stack(hidden_size=8, num_layers=2, seed=7)
    m2, _ = build_smoke_stack(hidden_size=8, num_layers=2, seed=7)
    m3, _ = build_smoke_stack(hidden_size=8, num_layers=2, seed=8)
    ids = torch.tensor([[1, 2, 3]])
    kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
    torch.testing.assert_close(m1(**kw).logits, m2(**kw).logits)
    assert not torch.allclose(m1(**kw).logits, m3(**kw).logits)


def test_smoke_model_parameters_are_frozen_for_policy_use():
    """정책은 freeze 상태여야 한다 (`load_policy` 계약). 헤드만 학습한다."""
    model, _ = build_smoke_stack(hidden_size=8, num_layers=2, seed=0)
    assert all(not p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("bad", [0, -1])
def test_tiny_model_rejects_nonpositive_layers(bad):
    with pytest.raises(ValueError):
        TinyCausalLM(vocab_size=17, hidden_size=8, num_layers=bad)


# ----------------------------------------------------------------------
# 추론 스텝 구조
# ----------------------------------------------------------------------
def test_generate_emits_line_structured_text():
    """`phase1_validate.split_steps` 는 줄 단위로 '추론 스텝'을 센다.

    난수 모델이 줄바꿈을 거의 내지 않으면 prefix 풀이 0개가 되어 검증 파이프라인이
    통째로 무의미해진다. 스모크 픽스처는 스텝 구조를 가진 텍스트를 내야 한다.
    """
    model, tok = build_smoke_stack(hidden_size=8, num_layers=2, seed=0, newline_period=4)
    ids = torch.tensor([tok("go", add_special_tokens=False)["input_ids"]])
    gen = model.generate(input_ids=ids, max_new_tokens=32, num_return_sequences=1,
                         do_sample=True, temperature=1.0, top_p=1.0)
    text = tok.decode(gen[0, ids.shape[1]:])
    lines = [ln for ln in text.split("\n") if ln.strip()]
    assert len(lines) >= 4, f"스텝이 {len(lines)}개뿐 — prefix 풀이 비게 된다: {text!r}"


def test_newline_period_zero_disables_forced_structure():
    """구조 주입은 옵션이어야 한다 (기본 동작을 바꾸지 않음을 확인)."""
    model, tok = build_smoke_stack(hidden_size=8, num_layers=2, seed=0, newline_period=0)
    ids = torch.tensor([[5, 6]])
    torch.manual_seed(3)
    gen = model.generate(input_ids=ids, max_new_tokens=12, num_return_sequences=1)
    nl = tok("\n", add_special_tokens=False)["input_ids"][0]
    forced_positions = gen[0, ids.shape[1]:][3::4]
    assert not (forced_positions == nl).all()
