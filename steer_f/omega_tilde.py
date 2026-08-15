"""Ω̃ — STEER의 로컬 엔트로피 변화 예측 Ω에 미래 엔트로피(방문 이동) 항을 더한다.

    Ω̃_{i,t} = z(Ω_{i,t}) + λ · z( Δlogπ̂_{i,t} · clip(A_H, -c, c) )
    Δlogπ̂_{i,t} ≈ η · w_{i,t} · (1 - π_{i,t}),   w = A / π_old

`w`는 STEER 원 코드의 `advantage_over_old_prob`과 **같은 양**이다
(docs/steer_code_map.md §2.1의 1차 전개 참조).

모드별 결합 규칙 (docs/steer_code_map.md §7.2):
    symmetric  : base가 |Ω| 이므로 미래 항도 크기로   → z(|Ω|) + λ·z(|visit|)
    asymmetric : base가 부호 있는 Ω 이므로 부호 그대로 → z(Ω)   + λ·z(visit)

λ=0이면 입력 metric을 **그대로 반환**한다 (z-norm도 건너뜀) — 순수 STEER와 비트 동일.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

__all__ = ["SteerFConfig", "compute_omega_tilde", "normalize"]


@dataclass
class SteerFConfig:
    """STEER-F 하이퍼파라미터. verl config(`policy_loss.*`)에서 채운다."""

    lam: float = 0.0            # λ — 미래 항 가중. 0이면 순수 STEER.
    eta: float = 1.0            # η — 1차 로그확률 이동 스케일. z-norm이 뒤따르므로
                                #     λ와 축퇴한다. 진단용으로만 조정할 것.
    clip_c: float = 1.0         # A_H 클리핑 경계.
    kappa: int = 4              # 예보 호라이즌.
    gamma_h: float = 0.85       # 예보 할인.
    baseline: str = "sibling"   # "sibling" | "group" | "none"
    norm: str = "z"             # "z" | "robust" (median/IQR)
    beta_mtp: float = 0.05      # MTP CE 보조 손실 계수.
    drift_kl_threshold: float = 0.5   # 예보-정책 표류 문턱. 초과 시 λ 감쇠.
    drift_lam_decay: float = 0.5      # 초과 시 λ에 곱할 값.
    local_scale: float = 1.0          # z(Ω) 계수. 0 = 미래 항만 (Ablation A4).

    def __post_init__(self):
        if self.lam < 0:
            raise ValueError(f"lam must be >= 0, got {self.lam}")
        if self.clip_c <= 0:
            raise ValueError(f"clip_c must be > 0, got {self.clip_c}")
        if self.baseline not in {"sibling", "group", "none"}:
            raise ValueError(f"unknown baseline: {self.baseline}")
        if self.norm not in {"z", "robust"}:
            raise ValueError(f"unknown norm: {self.norm}")


def normalize(x: torch.Tensor, mask: Optional[torch.Tensor], how: str = "z", eps: float = 1e-6) -> torch.Tensor:
    """마스크 내 정규화. 0-분산/전부-마스크 배치에서 0을 반환한다.

    - "z"      : (x - mean) / std
    - "robust" : (x - median) / IQR — STEER의 배치 min-max가 이상치 하나에
                 좌우되는 문제(docs/steer_code_map.md §6.1)를 완화하고 싶을 때.
    """
    valid = torch.ones_like(x, dtype=torch.bool) if mask is None else mask.bool()
    if not valid.any():
        return torch.zeros_like(x)
    vals = x[valid].float()

    if how == "z":
        center = vals.mean()
        scale = vals.std()
    elif how == "robust":
        center = vals.median()
        q = torch.quantile(vals, torch.tensor([0.25, 0.75], device=vals.device, dtype=vals.dtype))
        scale = q[1] - q[0]
    else:
        raise ValueError(f"unknown norm: {how}")

    if not torch.isfinite(scale) or scale < eps:
        return torch.zeros_like(x)
    out = (x.float() - center) / scale
    return (out * valid.to(out.dtype)).to(x.dtype)


def compute_omega_tilde(
    omega_local: torch.Tensor,
    w: torch.Tensor,
    pi_sampled: torch.Tensor,
    h_togo_vals: Optional[torch.Tensor],
    baseline_h_togo: Optional[torch.Tensor],
    response_mask: Optional[torch.Tensor] = None,
    *,
    mode: str = "symmetric",
    eta: float = 1.0,
    lam: float = 0.0,
    clip_c: float = 1.0,
    norm: str = "z",
    local_scale: float = 1.0,
    return_stats: bool = False,
):
    """Ω̃ 계산.

    Args:
        omega_local: (B, T) STEER의 metric. **symmetric 모드에서는 이미 |Ω|이다**
            (core_algos.py:650-652 이후 값을 넘길 것).
        w: (B, T) advantage / π_old — STEER 내부의 `advantage_over_old_prob`.
        pi_sampled: (B, T) 샘플된 토큰의 현재 확률 π_θ(y_t|s_t).
        h_togo_vals: (B, T) 예보값. λ=0이면 None 가능.
        baseline_h_togo: (B, T) sibling/group baseline. λ=0이면 None 가능.
        response_mask: (B, T) 0/1.
        mode: "symmetric" | "asymmetric".
        eta, lam, clip_c, norm: SteerFConfig 참조.
        local_scale: 로컬 항 z(Ω)의 계수. 0으로 두면 미래 항만 남는다
            (Ablation A4: 두 항의 개별 기여 분해). 기본 1.0.
        return_stats: True면 (omega_tilde, stats) 반환.

    Returns:
        (B, T) Ω̃ — 또는 return_stats=True면 (Ω̃, stats dict).

    Note:
        λ=0 우회는 `local_scale` 보다 우선한다 — λ=0이면 미래 항이 없어
        `local_scale`만 남고, 그것은 min-max 매핑에서 아핀 불변이라 무의미하다.
    """
    if mode not in {"symmetric", "asymmetric"}:
        raise ValueError(f"unknown mode: {mode}")

    # λ=0 → 완전 우회. 순수 STEER와 비트 동일성 보장 (계획서 §8.1).
    if lam == 0.0:
        if return_stats:
            return omega_local, {"steerf/lam": 0.0, "steerf/future_frac": 0.0}
        return omega_local

    if h_togo_vals is None or baseline_h_togo is None:
        raise ValueError("h_togo_vals and baseline_h_togo are required when lam != 0")

    delta_log_pi = eta * w * (1.0 - pi_sampled)
    a_h = torch.clamp(h_togo_vals - baseline_h_togo, -clip_c, clip_c)
    visit_term = delta_log_pi * a_h

    if mode == "symmetric":
        # omega_local은 이미 |Ω| — 미래 항도 크기로 맞춘다.
        visit_term = visit_term.abs()

    visit_term = torch.where(torch.isfinite(visit_term), visit_term, torch.zeros_like(visit_term))

    z_local = local_scale * normalize(omega_local, response_mask, norm)
    z_future = normalize(visit_term, response_mask, norm)
    omega_tilde = z_local + lam * z_future

    if not return_stats:
        return omega_tilde

    valid = torch.ones_like(omega_tilde, dtype=torch.bool) if response_mask is None else response_mask.bool()
    if valid.any():
        loc = z_local[valid].abs().mean()
        fut = (lam * z_future[valid]).abs().mean()
        stats = {
            "steerf/lam": float(lam),
            "steerf/z_local_absmean": float(loc),
            "steerf/z_future_absmean": float(fut),
            # 두 항의 상대 크기 (계획서 §4.2 로깅 요구사항)
            "steerf/future_frac": float(fut / (loc + fut + 1e-12)),
            "steerf/a_h_absmean": float(a_h[valid].abs().mean()),
            "steerf/a_h_clip_frac": float((a_h[valid].abs() >= clip_c - 1e-9).float().mean()),
            "steerf/delta_logpi_absmean": float(delta_log_pi[valid].abs().mean()),
        }
    else:
        stats = {"steerf/lam": float(lam), "steerf/future_frac": 0.0}
    return omega_tilde, stats
