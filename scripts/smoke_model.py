"""의존성 없는 소형 합성 정책 스택 (스모크 전용).

**연구 결과를 내기 위한 것이 아니다.** 목적은 단 하나 — `transformers`/vLLM/GPU 없이
Phase 1 파이프라인(`phase1_warmup_heads.py` → `phase1_validate.py`)을 **끝까지 완주**시켜
배관(형상, 정렬, 캐시, 리포트 생성, 게이트 종료코드)이 실제로 동작하는지 확인하는 것이다.
GPU 노드의 첫 실행에서 터질 수 있는 문제를 CPU에서 미리 잡는 용도다.

여기서 흉내내는 인터페이스는 실제 코드가 호출하는 것만으로 한정한다 (출처는
`scripts/_common.py` 와 두 Phase 1 스크립트). 계약은 `tests/test_smoke_model.py` 가 강제한다.

합성 모델의 상관계수/게이트 판정은 **아무 의미도 없다**. 난수 가중치의 예보는 실측
엔트로피와 상관이 없으므로 G1은 보통 실패한다 — 그것이 정상이며, 스모크가 확인하는
것은 "판정이 내려지고 리포트가 쓰였는가"이지 "통과했는가"가 아니다.
"""

from __future__ import annotations

import math
import string
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TinyTokenizer", "TinyCausalLM", "CausalLMOutput", "build_smoke_stack", "SMOKE_PREFIX"]

# `--model smoke:...` 형태로 스모크 스택을 요청하는 센티널.
SMOKE_PREFIX = "smoke:"


# ----------------------------------------------------------------------
# 토크나이저
# ----------------------------------------------------------------------
class _Encoding(dict):
    """`tok(...)` 반환값. HF BatchEncoding 처럼 `.to(device)` 를 지원한다."""

    def to(self, device):
        for k, v in self.items():
            if torch.is_tensor(v):
                self[k] = v.to(device)
        return self


class TinyTokenizer:
    """문자 단위 토크나이저. id 0=pad, 1=eos, 2=unk, 3.. = 문자.

    문자 단위라 어휘가 작고(≈100) 로짓 텐서가 작아 CPU에서 즉시 돈다. 실제 모델의
    subword 경계는 재현하지 못하지만, 배관 검증에는 토큰 경계의 의미가 필요 없다.
    """

    CHARS = string.printable[:100]

    def __init__(self):
        self.pad_token, self.eos_token, self.unk_token = "<pad>", "<eos>", "<unk>"
        self.pad_token_id, self.eos_token_id, self.unk_token_id = 0, 1, 2
        self._itos = {i + 3: c for i, c in enumerate(self.CHARS)}
        self._stoi = {c: i for i, c in self._itos.items()}
        self.vocab_size = len(self.CHARS) + 3
        # 없어야 `_common.apply_chat_template` 가 content 이어붙이기로 폴백한다.
        self.chat_template = None

    def __call__(self, text, add_special_tokens: bool = False, return_tensors: Optional[str] = None):
        texts = [text] if isinstance(text, str) else list(text)
        seqs = [[self._stoi.get(c, self.unk_token_id) for c in t] for t in texts]
        if add_special_tokens:
            seqs = [s + [self.eos_token_id] for s in seqs]

        if return_tensors is None:
            return _Encoding(
                input_ids=seqs[0] if isinstance(text, str) else seqs,
                attention_mask=[1] * len(seqs[0]) if isinstance(text, str) else [[1] * len(s) for s in seqs],
            )

        width = max((len(s) for s in seqs), default=0)
        ids = torch.full((len(seqs), width), self.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attn[i, : len(s)] = 1
        return _Encoding(input_ids=ids, attention_mask=attn)

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if torch.is_tensor(ids):
            ids = ids.tolist()
        out = []
        for i in ids:
            i = int(i)
            if i in (self.pad_token_id, self.eos_token_id):
                if skip_special_tokens:
                    continue
                out.append(self.pad_token if i == self.pad_token_id else self.eos_token)
            else:
                out.append(self._itos.get(i, self.unk_token))
        return "".join(out)

    def batch_decode(self, seqs, skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(s, skip_special_tokens) for s in seqs]


# ----------------------------------------------------------------------
# 모델
# ----------------------------------------------------------------------
@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    hidden_states: Optional[tuple[torch.Tensor, ...]] = None


class _CausalBlock(nn.Module):
    """단일 헤드 causal self-attention + MLP."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size), nn.SiLU(), nn.Linear(2 * hidden_size, hidden_size)
        )
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        # causal mask: 위치 t 는 y_{>t} 를 보지 못한다 (steer_code_map §5 정렬 전제).
        t = x.shape[1]
        mask = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        att = att.masked_fill(~mask, float("-inf"))
        x = x + self.proj(torch.softmax(att, dim=-1) @ v)
        return x + self.mlp(self.ln2(x))


class TinyCausalLM(nn.Module):
    """HF CausalLM 의 최소 흉내. 실제로 호출되는 표면만 구현한다."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 32,
        num_layers: int = 2,
        max_pos: int = 4096,
        newline_token_id: Optional[int] = None,
        newline_period: int = 0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {hidden_size}")
        self.config = SimpleConfig(hidden_size=hidden_size, vocab_size=vocab_size)
        # 난수 모델은 줄바꿈을 거의 내지 않는다. 그러면 `phase1_validate.split_steps`
        # 가 스텝을 못 세어 prefix 풀이 0개가 되고 파이프라인 전체가 무의미해진다.
        # period > 0 이면 period 토큰마다 줄바꿈을 강제해 '추론 스텝' 구조를 만든다.
        self.newline_token_id = newline_token_id
        self.newline_period = newline_period
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pos = nn.Embedding(max_pos, hidden_size)
        self.blocks = nn.ModuleList([_CausalBlock(hidden_size) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        **_ignored,
    ) -> CausalLMOutput:
        t = input_ids.shape[1]
        pos = torch.arange(t, device=input_ids.device)
        x = self.embed(input_ids) + self.pos(pos)[None, :, :]

        hiddens = [x]
        for block in self.blocks:
            x = block(x)
            hiddens.append(x)
        x = self.ln_f(x)
        hiddens[-1] = x

        return CausalLMOutput(
            logits=self.lm_head(x),
            hidden_states=tuple(hiddens) if output_hidden_states else None,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 16,
        num_return_sequences: int = 1,
        pad_token_id: Optional[int] = None,
        **_ignored,
    ) -> torch.Tensor:
        """프롬프트 + 새 토큰을 붙인 (num_return_sequences, T+max_new_tokens).

        조기 종료를 하지 않는다 — 스모크에서는 길이가 예측 가능한 편이 낫고,
        `_common.GenerationBackend` 가 프롬프트 길이로 잘라 쓰기 때문이다.
        """
        seq = input_ids.repeat_interleave(num_return_sequences, dim=0)
        force_nl = self.newline_period > 0 and self.newline_token_id is not None
        for step in range(max_new_tokens):
            if force_nl and (step + 1) % self.newline_period == 0:
                nxt = torch.full((seq.shape[0], 1), self.newline_token_id,
                                 dtype=seq.dtype, device=seq.device)
                seq = torch.cat([seq, nxt], dim=1)
                continue
            logits = self(input_ids=seq).logits[:, -1, :]
            if not do_sample or temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                nxt = _sample_top_p(logits / max(temperature, 1e-6), top_p)
            seq = torch.cat([seq, nxt], dim=1)
        return seq


@dataclass
class SimpleConfig:
    hidden_size: int
    vocab_size: int


def _sample_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    if top_p >= 1.0:
        return torch.multinomial(probs, num_samples=1)
    ordered, idx = probs.sort(dim=-1, descending=True)
    keep = (ordered.cumsum(dim=-1) - ordered) < top_p  # 항상 최소 1개는 남는다
    ordered = ordered * keep
    ordered = ordered / ordered.sum(dim=-1, keepdim=True)
    return idx.gather(-1, torch.multinomial(ordered, num_samples=1))


# ----------------------------------------------------------------------
def build_smoke_stack(
    hidden_size: int = 32,
    num_layers: int = 2,
    seed: int = 0,
    max_pos: int = 4096,
    newline_period: int = 8,
):
    """(model, tokenizer). 정책은 freeze + eval — `load_policy` 계약과 동일.

    `newline_period` 는 합성 응답에 추론 스텝(줄) 구조를 준다. 0이면 끈다.
    """
    tok = TinyTokenizer()
    gen_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        model = TinyCausalLM(
            vocab_size=tok.vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            max_pos=max_pos,
            newline_token_id=tok("\n", add_special_tokens=False)["input_ids"][0],
            newline_period=newline_period,
        )
    finally:
        torch.random.set_rng_state(gen_state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok
