"""미래 엔트로피 예보 유틸 — H_togo, 엔트로피 advantage A_H, 헤드 캘리브레이션.

정의 (계획서 §0.2):

    H_togo^κ(s) = Σ_{k=1..κ} γ_H^k · H( p_MTP(y_{t+k} | s) )
    A_H(s_t, y_t) = H_togo(s_t ⊕ y_t) - H̄_togo(s_t)

구현 노트: 정책 forward는 위치 t에서 **이미 y_t를 조건으로 한** 히든을 만들지
않는다. verl의 rmpad 경로에서 위치 t의 히든은 s_t = (prompt, y_<t) 를 조건으로
하며 y_t를 예측한다. 따라서 `H_togo(s_t ⊕ y_t)`는 위치 t+1의 히든에서 얻은
예보 — 즉 head_entropies를 **한 칸 당겨서** 쓴다 (`shift=True`, 기본값).
이 한 칸을 놓치면 A_H가 "샘플된 토큰의 분기 가치"가 아니라 "그 이전 상태의
가치"가 되어 신호가 통째로 어긋난다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from .mtp_heads import entropy_from_logits

__all__ = [
    "HeadCalibration",
    "h_togo",
    "entropy_advantage",
    "sibling_mask",
    "masked_zscore",
]


# ----------------------------------------------------------------------
# 캘리브레이션
# ----------------------------------------------------------------------
@dataclass
class HeadCalibration:
    """헤드별 아핀 보정: H_hat_k = scale_k * H_raw_k + bias_k.

    먼 헤드(+k 큼)는 예측이 뭉개져 엔트로피를 체계적으로 **과대추정**한다.
    보정 없이 쓰면 "먼 미래는 항상 다양해 보이는" 편향이 생기므로,
    Phase 1에서 실측 경험분포 엔트로피에 대해 최소제곱으로 (scale, bias)를 맞춘다.
    """

    scale: list[float] = field(default_factory=list)
    bias: list[float] = field(default_factory=list)

    def __post_init__(self):
        if len(self.scale) != len(self.bias):
            raise ValueError("scale and bias must have the same length")

    @property
    def num_heads(self) -> int:
        return len(self.scale)

    @staticmethod
    def identity(num_heads: int) -> "HeadCalibration":
        return HeadCalibration(scale=[1.0] * num_heads, bias=[0.0] * num_heads)

    @staticmethod
    def fit(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> "HeadCalibration":
        """헤드별 최소제곱 적합.

        Args:
            pred: (K, N) 헤드 k의 원시 예보 엔트로피.
            target: (K, N) 대응 실측 엔트로피.
            mask: (K, N) 또는 (N,) 유효 마스크.
        """
        pred = pred.float()
        target = target.float()
        if mask is None:
            mask = torch.ones_like(pred, dtype=torch.bool)
        elif mask.dim() == 1:
            mask = mask.unsqueeze(0).expand_as(pred)
        mask = mask.bool()

        scales, biases = [], []
        for k in range(pred.shape[0]):
            x = pred[k][mask[k]]
            y = target[k][mask[k]]
            if x.numel() < 2 or torch.var(x) < 1e-12:
                scales.append(1.0)
                biases.append(0.0)
                continue
            xm, ym = x.mean(), y.mean()
            slope = ((x - xm) * (y - ym)).sum() / ((x - xm).pow(2).sum() + 1e-12)
            scales.append(float(slope))
            biases.append(float(ym - slope * xm))
        return HeadCalibration(scale=scales, bias=biases)

    def apply(self, head_entropies: torch.Tensor) -> torch.Tensor:
        """(K, ...) 예보에 헤드별 아핀 보정 적용."""
        k = head_entropies.shape[0]
        if self.num_heads < k:
            raise ValueError(f"calibration has {self.num_heads} heads but got {k}")
        shape = (k,) + (1,) * (head_entropies.dim() - 1)
        scale = torch.tensor(self.scale[:k], dtype=head_entropies.dtype, device=head_entropies.device).view(shape)
        bias = torch.tensor(self.bias[:k], dtype=head_entropies.dtype, device=head_entropies.device).view(shape)
        return head_entropies * scale + bias

    def to_dict(self) -> dict:
        return {"scale": self.scale, "bias": self.bias}

    @staticmethod
    def from_dict(d: dict) -> "HeadCalibration":
        return HeadCalibration(scale=list(d["scale"]), bias=list(d["bias"]))


# ----------------------------------------------------------------------
# H_togo
# ----------------------------------------------------------------------
def h_togo(
    head_entropies: torch.Tensor,
    kappa: int,
    gamma_h: float,
    calib: Optional[HeadCalibration] = None,
    is_logits: bool = False,
) -> torch.Tensor:
    """H_togo^κ = Σ_{k=1..κ} γ_H^k · H_k.

    Args:
        head_entropies: (K, B, T) 헤드별 예측 엔트로피.
            `is_logits=True`면 (K, B, T, V) 로짓으로 받아 내부에서 엔트로피화한다
            (소형 검증 배치 전용 — 메모리 주의).
        kappa: 사용할 호라이즌 (1 <= kappa <= K).
        gamma_h: 할인 계수.
        calib: 헤드별 보정. None이면 보정 없음.

    Returns:
        (B, T) float32.
    """
    if is_logits:
        head_entropies = entropy_from_logits(head_entropies)
    if head_entropies.dim() < 2:
        raise ValueError(f"expected (K, ...) tensor, got shape {tuple(head_entropies.shape)}")
    k_total = head_entropies.shape[0]
    if not (1 <= kappa <= k_total):
        raise ValueError(f"kappa must be in [1, {k_total}], got {kappa}")

    h = head_entropies[:kappa].float()
    if calib is not None:
        h = calib.apply(h)

    shape = (kappa,) + (1,) * (h.dim() - 1)
    weights = torch.tensor(
        [gamma_h ** (k + 1) for k in range(kappa)], dtype=h.dtype, device=h.device
    ).view(shape)
    return (h * weights).sum(dim=0)


# ----------------------------------------------------------------------
# sibling baseline
# ----------------------------------------------------------------------
def sibling_mask(response_ids: torch.Tensor, group_size: int, response_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """위치별 sibling 관계 마스크.

    rollout i와 j가 위치 t에서 sibling ⟺ 같은 프롬프트 그룹이고 prefix y_<t 가 동일.
    (t=0에서는 프롬프트만 공유하므로 그룹 전원이 sibling.)

    Args:
        response_ids: (B, T) 응답 토큰 id. B는 group_size의 배수여야 한다
            (verl GRPO는 프롬프트당 `rollout.n`개를 연속 배치한다).
        group_size: G = `actor_rollout_ref.rollout.n`.
        response_mask: (B, T) 0/1. 패딩 위치는 비교에서 제외한다.

    Returns:
        (n_groups, G, G, T) bool.
    """
    b, t = response_ids.shape
    if b % group_size != 0:
        raise ValueError(f"batch {b} is not divisible by group_size {group_size}")
    n_groups = b // group_size

    ids = response_ids.view(n_groups, group_size, t)
    eq = ids.unsqueeze(2) == ids.unsqueeze(1)  # (n_groups, G, G, T)
    if response_mask is not None:
        m = response_mask.view(n_groups, group_size, t).bool()
        both_valid = m.unsqueeze(2) & m.unsqueeze(1)
        # 한쪽이라도 패딩이면 그 위치에서 이미 갈라진 것으로 본다(길이 차이 = 분기).
        eq = eq & both_valid

    # prefix_eq[..., t] = (0..t 전부 일치)
    prefix_eq = torch.cumprod(eq.int(), dim=-1).bool()
    # 위치 t의 sibling 판정은 y_<t 기준 → 한 칸 밀어서 t=0은 전부 True
    sib = torch.ones_like(prefix_eq)
    sib[..., 1:] = prefix_eq[..., :-1]
    return sib


def _shift_forward(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """위치 t에 t+1의 값을 놓는다 (마지막 위치는 자기 자신 유지)."""
    out = torch.empty_like(x)
    out[:, :-1] = x[:, 1:]
    out[:, -1] = x[:, -1]
    if mask is not None:
        # t+1이 패딩이면 t의 값을 그대로 둔다 (시퀀스 끝).
        m = mask.bool()
        next_valid = torch.zeros_like(m)
        next_valid[:, :-1] = m[:, 1:]
        out = torch.where(next_valid, out, x)
    return out


def entropy_advantage(
    h_togo_vals: torch.Tensor,
    response_ids: Optional[torch.Tensor] = None,
    group_size: int = 1,
    response_mask: Optional[torch.Tensor] = None,
    baseline: str = "sibling",
    shift: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A_H(s_t, y_t) = H_togo(s_t ⊕ y_t) - H̄_togo(s_t).

    Args:
        h_togo_vals: (B, T) `h_togo()` 출력.
        response_ids: (B, T) — `baseline="sibling"`일 때 필수.
        group_size: G. 같은 프롬프트에서 나온 rollout이 연속 G개라고 가정.
        response_mask: (B, T) 0/1.
        baseline: "sibling" | "group" | "none".
            - sibling: 위치 t까지 prefix를 공유하는 rollout들의 평균 (계획서 1차 구현).
            - group: 같은 프롬프트 그룹 전체 평균 (근사, 저렴). Ablation A5.
            - none: baseline 0 (디버깅용).
        shift: True면 h_togo를 한 칸 당겨 "y_t를 조건으로 한" 예보로 정렬한다.
            docstring 상단의 정렬 노트 참조. **기본값을 바꾸지 말 것.**

    Returns:
        (a_h, baseline_vals) 둘 다 (B, T). 패딩 위치는 0.
        sibling이 자기 자신뿐이면 baseline = 자기 값 → A_H = 0 (계획서 §8.4).
    """
    b, t = h_togo_vals.shape
    branch = _shift_forward(h_togo_vals, response_mask) if shift else h_togo_vals

    if baseline == "none":
        base = torch.zeros_like(branch)
    elif baseline == "group":
        if group_size <= 1:
            base = branch.clone()
        else:
            g = branch.view(b // group_size, group_size, t)
            if response_mask is not None:
                m = response_mask.view(b // group_size, group_size, t).float()
                denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)
                mean = (g * m).sum(dim=1, keepdim=True) / denom
            else:
                mean = g.mean(dim=1, keepdim=True)
            base = mean.expand_as(g).reshape(b, t)
    elif baseline == "sibling":
        if response_ids is None:
            raise ValueError("baseline='sibling' requires response_ids")
        sib = sibling_mask(response_ids, group_size, response_mask).float()  # (n_g, G, G, T)
        g = branch.view(b // group_size, group_size, t)
        if response_mask is not None:
            m = response_mask.view(b // group_size, group_size, t).float()
        else:
            m = torch.ones_like(g)
        # base[i, t] = Σ_j sib[i,j,t] * m[j,t] * g[j,t] / Σ_j sib[i,j,t] * m[j,t]
        w = sib * m.unsqueeze(1)
        num = (w * g.unsqueeze(1)).sum(dim=2)
        den = w.sum(dim=2).clamp(min=1e-6)
        base = (num / den).reshape(b, t)
    else:
        raise ValueError(f"unknown baseline: {baseline}")

    a_h = branch - base
    if response_mask is not None:
        m = response_mask.to(a_h.dtype)
        a_h = a_h * m
        base = base * m
    return a_h, base


# ----------------------------------------------------------------------
# 정규화 유틸
# ----------------------------------------------------------------------
def masked_zscore(x: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-6) -> torch.Tensor:
    """마스크 내에서 z-정규화. 0-분산이면 0을 반환 (계획서 §8.3)."""
    if mask is None:
        valid = torch.ones_like(x, dtype=torch.bool)
    else:
        valid = mask.bool()
    if not valid.any():
        return torch.zeros_like(x)
    vals = x[valid].float()
    std = vals.std()
    if not torch.isfinite(std) or std < eps:
        return torch.zeros_like(x)
    out = (x - vals.mean()) / std
    return out * valid.to(out.dtype)
