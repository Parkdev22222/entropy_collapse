"""STEER-F 버전의 `compute_token_weights`.

원본(`verl/trainer/ppo/core_algos.py:580-713`)을 **줄 단위로 보존**하고, metric이
확정된 직후(원본 653행 위치)에 Ω̃ 훅을 딱 한 번 삽입했다. min-max 매핑 이하
(원본 659-710)는 한 글자도 바꾸지 않았다 — docs/steer_code_map.md §7.1의 통합 계약.

λ=0이면 훅이 metric을 그대로 반환하므로 원본과 비트 동일하다
(`tests/test_lambda_zero_equiv.py`가 강제한다).

verl에 적용할 때는 `patches/core_algos_steerf.patch`를 쓴다. 이 모듈은 verl 없이도
임포트 가능하도록 순수 torch로만 작성되어 있어, 단위 테스트가 GPU/verl 없이 돈다.
"""

from __future__ import annotations

from typing import Optional

import torch

from .omega_tilde import compute_omega_tilde

__all__ = ["compute_token_weights_steerf"]


def compute_token_weights_steerf(
    advantages: torch.Tensor,
    entropys: torch.Tensor,
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    token_weight_min: float = 0.8,
    token_weight_max: float = 1.2,
    linear: bool = True,
    mode: str = "symmetric",
    # ---- STEER-F 추가 인자 (전부 기본값 → 미지정 시 원본 동작) ----
    h_togo_vals: Optional[torch.Tensor] = None,
    baseline_h_togo: Optional[torch.Tensor] = None,
    lam: float = 0.0,
    eta: float = 1.0,
    clip_c: float = 1.0,
    steerf_norm: str = "z",
    steerf_local_scale: float = 1.0,
    return_stats: bool = False,
):
    """원본과 동일한 (B, T) 토큰 가중치. STEER-F 인자를 주면 Ω→Ω̃로 교체된다."""

    steerf_stats: dict = {}

    with torch.no_grad():

        # Calculate \delta
        x = torch.exp(log_prob)  # Convert log_prob to probability values
        x = torch.clamp(x, min=1e-8, max=1.0 - 1e-8)

        x_one_minus_x_squared = x * (1 - x)
        ln_x_plus_h = torch.log(x) + entropys

        f_x = x_one_minus_x_squared * ln_x_plus_h

        # Calculate A / old_log_prob
        old_prob = torch.exp(old_log_prob)
        old_prob = torch.clamp(old_prob, min=1e-8, max=1.0)

        advantage_over_old_prob = advantages / old_prob

        # Calculate \Omega
        if mode == "asymmetric":
            metric = -advantage_over_old_prob * f_x
        else:
            metric = advantage_over_old_prob * f_x

        if not torch.isfinite(metric).all():
            print("[Token Weighting] Warning: Found non-finite values in metric")
            metric = torch.where(torch.isfinite(metric), metric, torch.zeros_like(metric))

        if mode == "symmetric":
            # Calculate abs(\Omega)
            metric = torch.abs(metric)
        # else: mode == "asymmetric" -> keep the sign of Ω

        # ================= STEER-F 훅 (여기 한 곳만 추가) =================
        # Ω → Ω̃.  lam == 0.0 이면 metric이 그대로 반환되어 원본과 비트 동일.
        metric, steerf_stats = compute_omega_tilde(
            omega_local=metric,
            w=advantage_over_old_prob,
            pi_sampled=x,
            h_togo_vals=h_togo_vals,
            baseline_h_togo=baseline_h_togo,
            response_mask=response_mask,
            mode=mode,
            eta=eta,
            lam=lam,
            clip_c=clip_c,
            norm=steerf_norm,
            local_scale=steerf_local_scale,
            return_stats=True,
        )
        # ================================================================

        valid_metric = metric[response_mask.bool()]
        if valid_metric.numel() == 0:
            zero = torch.zeros_like(response_mask, dtype=torch.float)
            return (zero, steerf_stats) if return_stats else zero

        metric_min = valid_metric.min()
        metric_max = valid_metric.max()

        # Initialize token weights
        token_weights = torch.zeros_like(metric, dtype=torch.float)

        # Calculate weights only for valid tokens
        valid_mask = response_mask.bool()
        if valid_mask.any():
            valid_metric = metric[valid_mask]

            if linear:
                # Linear mapping strategy: map to [token_weight_min, token_weight_max]
                if metric_max > metric_min:
                    scale_factor = (token_weight_max - token_weight_min) / (metric_max - metric_min)
                    if mode == "asymmetric":
                        valid_weights = token_weight_min + (valid_metric - metric_min) * scale_factor
                    else:
                        valid_weights = token_weight_max - (valid_metric - metric_min) * scale_factor
                else:
                    valid_weights = torch.full_like(valid_metric, (token_weight_min + token_weight_max) / 2)
                valid_weights = torch.clamp(valid_weights, min=token_weight_min, max=token_weight_max)
            else:
                if mode == "asymmetric":
                    abs_max = torch.maximum(
                        valid_metric.abs().max(),
                        torch.tensor(0.02, dtype=metric.dtype, device=metric.device),
                    )
                    k = -torch.log(torch.tensor(token_weight_min, dtype=metric.dtype, device=metric.device)) / abs_max
                    valid_weights = torch.exp(k * valid_metric)
                    valid_weights = torch.clamp(valid_weights, min=token_weight_min, max=token_weight_max)
                else:
                    k = -torch.log(torch.tensor(token_weight_min, dtype=metric.dtype, device=metric.device)) / \
                        torch.maximum(
                            metric_max,
                            torch.tensor(0.02, dtype=metric.dtype, device=metric.device)
                        )
                    valid_weights = torch.exp(-k * valid_metric)
                    valid_weights = torch.clamp(valid_weights, min=token_weight_min, max=1.0)

            # Apply computed weights to valid token positions
            token_weights[valid_mask] = valid_weights

        token_weights = token_weights * response_mask.float()

        return (token_weights, steerf_stats) if return_stats else token_weights
