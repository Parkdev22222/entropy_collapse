"""STEER-F 학습 중 감시 지표.

계획서 §4.2가 요구하는 로깅:
  - α(token_weights) 분포 — 원 STEER는 mean/min/max만 올린다 (code_map §3.4).
  - 분기 토큰(그룹 A_H 상위 10%)만의 엔트로피 곡선.
  - Ω̃ 두 항의 상대 크기 → `omega_tilde.compute_omega_tilde(return_stats=True)`.
  - MTP 헤드 +1 예측과 정책 실제 next-token 분포의 KL 표류.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "token_weight_stats",
    "branch_token_entropy",
    "forecast_policy_kl",
    "LambdaDriftController",
]


def token_weight_stats(
    token_weights: torch.Tensor,
    response_mask: torch.Tensor,
    weight_min: float,
    weight_max: float,
    n_bins: int = 10,
) -> dict:
    """α 분포 통계.

    원 STEER의 매핑은 이산 3단이 아니라 [weight_min, weight_max] 연속이므로
    "γ/1/1/γ 비율"은 정의되지 않는다. 대신 (a) 3구간 요약과 (b) 히스토그램을 낸다.
    계획서의 "α 비율" 게이트는 이 3구간 요약으로 읽는다.
    """
    valid = response_mask.bool()
    if not valid.any():
        return {}
    w = token_weights[valid].float()
    span = weight_max - weight_min
    lo_edge = weight_min + span / 3.0
    hi_edge = weight_max - span / 3.0

    stats = {
        "steerf/alpha_mean": float(w.mean()),
        "steerf/alpha_std": float(w.std()) if w.numel() > 1 else 0.0,
        "steerf/alpha_min": float(w.min()),
        "steerf/alpha_max": float(w.max()),
        # 3구간 요약: 강한 감쇠 / 중립 / 강한 증폭
        "steerf/alpha_frac_low": float((w <= lo_edge).float().mean()),
        "steerf/alpha_frac_mid": float(((w > lo_edge) & (w < hi_edge)).float().mean()),
        "steerf/alpha_frac_high": float((w >= hi_edge).float().mean()),
        # 극단 쏠림 경보 (계획서 §9 "밴드 오설정" 징후: 한쪽 90%+)
        "steerf/alpha_saturated": float(
            max((w <= lo_edge).float().mean(), (w >= hi_edge).float().mean())
        ),
    }
    hist = torch.histc(w, bins=n_bins, min=weight_min, max=weight_max)
    hist = hist / hist.sum().clamp(min=1.0)
    for i, v in enumerate(hist.tolist()):
        stats[f"steerf/alpha_hist_{i}"] = v
    return stats


def branch_token_entropy(
    entropys: torch.Tensor,
    a_h: torch.Tensor,
    response_mask: torch.Tensor,
    group_size: int = 1,
    top_frac: float = 0.1,
) -> dict:
    """분기 토큰(A_H 상위 top_frac)의 엔트로피 vs 전체 엔트로피.

    STEER-F의 메커니즘 주장(분기 지점의 다양성 보존)을 직접 확인하는 곡선이다.
    상위 선택은 계획서대로 **그룹 단위**로 한다 (프롬프트마다 난이도가 달라
    A_H 스케일이 다르므로 배치 전역 분위수는 쉬운 프롬프트로 쏠린다).
    """
    valid = response_mask.bool()
    if not valid.any():
        return {}

    b, t = a_h.shape
    if group_size > 1 and b % group_size == 0:
        a_g = a_h.view(b // group_size, group_size * t)
        v_g = valid.view(b // group_size, group_size * t)
        sel = torch.zeros_like(v_g)
        for i in range(a_g.shape[0]):
            vals = a_g[i][v_g[i]]
            if vals.numel() == 0:
                continue
            thr = torch.quantile(vals.float(), 1.0 - top_frac)
            sel[i] = v_g[i] & (a_g[i] >= thr)
        branch = sel.view(b, t)
    else:
        vals = a_h[valid].float()
        thr = torch.quantile(vals, 1.0 - top_frac)
        branch = valid & (a_h >= thr)

    out = {
        "steerf/entropy_all": float(entropys[valid].float().mean()),
        "steerf/branch_token_frac": float(branch.float().sum() / valid.float().sum()),
    }
    if branch.any():
        out["steerf/entropy_branch"] = float(entropys[branch].float().mean())
        out["steerf/entropy_branch_minus_all"] = out["steerf/entropy_branch"] - out["steerf/entropy_all"]
    return out


@torch.no_grad()
def forecast_policy_kl(
    head1_logits: torch.Tensor,
    policy_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """KL( 정책 next-token ‖ MTP 헤드 +1 예측 ).

    헤드 +1은 위치 t에서 y_{t+1}을 예측하고, 정책은 위치 t+1에서 y_{t+1}을 예측한다.
    따라서 **정책 로짓을 한 칸 당겨서** 넘겨야 같은 타깃을 비교한다 — 호출부가
    정렬을 맞춰서 넘긴다(`scripts`/패치 참조).

    Args:
        head1_logits: (N, V) 헤드 +1 로짓.
        policy_logits: (N, V) 정렬된 정책 로짓.
        mask: (N,) 유효 마스크.
    Returns:
        스칼라 평균 KL.
    """
    if mask is not None:
        m = mask.bool()
        head1_logits = head1_logits[m]
        policy_logits = policy_logits[m]
    if head1_logits.numel() == 0:
        return torch.zeros((), device=head1_logits.device)
    log_p = torch.log_softmax(policy_logits.float(), dim=-1)
    log_q = torch.log_softmax(head1_logits.float(), dim=-1)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=-1).mean()


class LambdaDriftController:
    """예보-정책 표류가 커지면 λ를 감쇠시키는 안전장치 (계획서 §4.1).

    KL이 문턱을 넘으면 λ에 `decay`를 곱하고, 문턱 아래로 안정적으로 돌아오면
    (`recover_patience` 스텝 연속) 원래 값으로 복구한다.
    """

    def __init__(
        self,
        lam_base: float,
        threshold: float = 0.5,
        decay: float = 0.5,
        recover_patience: int = 20,
        lam_floor: float = 0.0,
    ):
        self.lam_base = lam_base
        self.lam = lam_base
        self.threshold = threshold
        self.decay = decay
        self.recover_patience = recover_patience
        self.lam_floor = lam_floor
        self._ok_streak = 0
        self.n_decays = 0

    def update(self, kl: float) -> float:
        """현재 KL을 넣고 갱신된 λ를 받는다."""
        if kl > self.threshold:
            self._ok_streak = 0
            new_lam = max(self.lam * self.decay, self.lam_floor)
            if new_lam < self.lam:
                self.n_decays += 1
            self.lam = new_lam
        else:
            self._ok_streak += 1
            if self._ok_streak >= self.recover_patience and self.lam < self.lam_base:
                self.lam = self.lam_base
                self._ok_streak = 0
        return self.lam

    def stats(self, kl: Optional[float] = None) -> dict:
        out = {
            "steerf/lam_effective": self.lam,
            "steerf/lam_decay_count": self.n_decays,
        }
        if kl is not None:
            out["steerf/forecast_policy_kl"] = float(kl)
        return out
