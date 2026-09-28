"""Medusa 스타일 병렬 MTP(Multi-Token Prediction) 헤드.

정책 모델의 마지막 히든에서 +1..+K 위치의 토큰 분포를 병렬 예측한다.
용도는 토큰 예측 정확도가 아니라 **분포의 엔트로피 예보**임에 유의.

핵심 설계 제약 (docs/steer_code_map.md §5 참조):
    K개 헤드의 로짓 `[K, B, T, V]`를 동시에 상주시키면 Qwen2.5(V≈152k) 기준
    수십 GB가 된다. RL 학습 경로에서는 반드시 `forward_entropy()`를 쓸 것 —
    헤드/청크 단위로 로짓을 만들고 즉시 엔트로피로 축약한 뒤 버린다.
    `forward_logits()`는 소형 검증 배치 전용이다.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """H(softmax(logits)) — 마지막 차원에 대해. 반환 shape은 logits.shape[:-1].

    logsumexp 기반이라 fp16/bf16에서도 수치적으로 안전하다.
    """
    logits = logits.float()
    logsumexp = torch.logsumexp(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    return logsumexp - (probs * logits).sum(dim=-1)


class MTPHeads(nn.Module):
    """+1..+K 위치 예측 헤드 묶음.

    Args:
        hidden_size: 본체 모델의 hidden dim.
        num_heads: K. 헤드 k는 위치 t에서 y_{t+k}를 예측한다 (k = 1..K).
        head_hidden: 각 헤드 MLP의 중간 폭.
        residual: True면 proj 출력을 입력 히든에 더한다 (Medusa 기본, 학습 초기 안정).
        vocab_size: 선택. `forward_*`에 lm_head를 주입하므로 필수는 아니지만,
            형상 검증과 리포팅에 쓰인다.

    lm_head는 소유하지 않는다. 본체의 unembedding을 호출 시점에 주입받아 공유한다
    (계획서의 `tie_unembedding=True`에 해당). 별도 unembedding을 쓰고 싶으면
    호출부에서 다른 모듈을 넘기면 된다.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        head_hidden: int = 1024,
        vocab_size: Optional[int] = None,
        residual: bool = True,
    ):
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.residual = residual

        self.projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, head_hidden),
                    nn.SiLU(),
                    nn.Linear(head_hidden, hidden_size),
                )
                for _ in range(num_heads)
            ]
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """마지막 Linear를 0으로 초기화 → residual 모드에서 초기 출력이 본체와 동일.

        헤드가 학습되기 전 예보가 본체의 next-token 분포와 같은 지점에서 출발하므로,
        워밍업 초기에 터무니없는 H_togo가 나오는 것을 막는다.
        """
        for proj in self.projs:
            last = proj[-1]
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    # ------------------------------------------------------------------
    # 내부: 헤드 k의 히든 변환
    # ------------------------------------------------------------------
    def _head_hidden(self, hidden_states: torch.Tensor, k: int) -> torch.Tensor:
        out = self.projs[k](hidden_states)
        if self.residual:
            out = out + hidden_states
        return out

    # ------------------------------------------------------------------
    # 검증/워밍업용: 전체 로짓
    # ------------------------------------------------------------------
    def forward_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: Callable[[torch.Tensor], torch.Tensor],
        heads: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """[K, ..., V] 로짓을 한 번에 반환. **메모리 주의 — 소형 배치 전용.**

        Args:
            hidden_states: (..., H). 보통 (B, T, H) 또는 packed (N, H).
            lm_head: 히든 → 로짓 호출가능 객체 (본체 lm_head 공유).
            heads: 계산할 헤드 인덱스(0-based). None이면 전부.
        """
        idxs = list(range(self.num_heads)) if heads is None else heads
        return torch.stack([lm_head(self._head_hidden(hidden_states, k)) for k in idxs])

    # 계획서 시그니처 호환용 별칭
    def forward(self, hidden_states: torch.Tensor, lm_head) -> torch.Tensor:  # noqa: D102
        return self.forward_logits(hidden_states, lm_head)

    # ------------------------------------------------------------------
    # RL 경로용: 로짓을 상주시키지 않고 엔트로피만
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_entropy(
        self,
        hidden_states: torch.Tensor,
        lm_head: Callable[[torch.Tensor], torch.Tensor],
        kappa: Optional[int] = None,
        chunk_size: int = 4096,
    ) -> torch.Tensor:
        """헤드별 예측 엔트로피만 반환. shape (kappa, *hidden_states.shape[:-1]).

        로짓은 (chunk_size, V) 단위로만 존재했다가 즉시 해제된다.
        `no_grad`인 이유: 이 값은 STEER의 `compute_token_weights`(전체가 no_grad)로
        들어가는 재가중 계수 재료다. MTP 헤드의 학습은 별도 CE 보조 손실 경로가
        담당한다 (docs/steer_code_map.md §7.3).
        """
        k_max = self.num_heads if kappa is None else kappa
        if not (1 <= k_max <= self.num_heads):
            raise ValueError(f"kappa must be in [1, {self.num_heads}], got {kappa}")

        lead_shape = hidden_states.shape[:-1]
        flat = hidden_states.reshape(-1, self.hidden_size)
        n = flat.shape[0]

        out = torch.empty((k_max, n), dtype=torch.float32, device=hidden_states.device)
        for k in range(k_max):
            for start in range(0, n, chunk_size):
                stop = min(start + chunk_size, n)
                h = self._head_hidden(flat[start:stop], k)
                out[k, start:stop] = entropy_from_logits(lm_head(h))
                del h
        return out.reshape(k_max, *lead_shape)

    # ------------------------------------------------------------------
    # 워밍업 / 보조 손실
    # ------------------------------------------------------------------
    def ce_loss(
        self,
        hidden_states: torch.Tensor,
        lm_head: Callable[[torch.Tensor], torch.Tensor],
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kappa: Optional[int] = None,
    ) -> tuple[torch.Tensor, dict]:
        """헤드 k의 타깃 = 위치 t 기준 실제 토큰 y_{t+k}인 cross-entropy 평균.

        Args:
            hidden_states: (B, T, H).
            input_ids: (B, T) — hidden_states와 **같은 정렬**의 토큰 id.
                위치 t의 헤드 k 타깃은 `input_ids[:, t + k]`.
            mask: (B, T) 0/1. 유효 위치(응답 토큰). None이면 전부 유효.
            kappa: 사용할 헤드 수. None이면 전부.

        Returns:
            (loss, per_head_loss_dict). 유효 타깃이 하나도 없으면 loss=0.
        """
        b, t, _ = hidden_states.shape
        k_max = self.num_heads if kappa is None else kappa
        if mask is None:
            mask = torch.ones((b, t), dtype=torch.bool, device=hidden_states.device)
        mask = mask.bool()

        total = hidden_states.new_zeros(())
        counted = 0
        per_head = {}
        for k in range(1, k_max + 1):
            if t - k <= 0:
                break
            # 위치 t = 0 .. T-1-k 만 타깃 y_{t+k}가 존재한다.
            h = hidden_states[:, : t - k, :]
            target = input_ids[:, k:]
            valid = mask[:, : t - k] & mask[:, k:]
            if not valid.any():
                continue
            logits = lm_head(self._head_hidden(h[valid], k - 1))
            loss_k = F.cross_entropy(logits.float(), target[valid])
            per_head[f"mtp/ce_head_{k}"] = loss_k.detach()
            total = total + loss_k
            counted += 1

        if counted == 0:
            return hidden_states.new_zeros(()), per_head
        return total / counted, per_head


def build_mtp_heads_for(model, num_heads: int = 8, head_hidden: int = 1024, residual: bool = True) -> MTPHeads:
    """HF CausalLM에서 hidden_size/vocab_size를 읽어 헤드를 만든다 (Phase 3 이식 헬퍼)."""
    cfg = model.config
    hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd")
    vocab_size = getattr(cfg, "vocab_size")
    heads = MTPHeads(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_hidden=head_hidden,
        vocab_size=vocab_size,
        residual=residual,
    )
    param_dtype = next(model.parameters()).dtype
    return heads.to(device=next(model.parameters()).device, dtype=param_dtype)
