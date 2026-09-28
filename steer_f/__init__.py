"""STEER-F: STEER with Future-entropy Forecasting.

STEER의 근시안적 토큰 엔트로피 변화 추정 Ω를, MTP 헤드 기반 미래 엔트로피
예보로 확장한 Ω̃로 대체한다.

모듈:
    mtp_heads          Medusa 스타일 병렬 MTP 헤드 + CE 워밍업 손실
    entropy_forecast   H_togo, A_H(sibling/group baseline), 헤드 캘리브레이션
    omega_tilde        Ω̃ 결합 (λ=0이면 순수 STEER와 비트 동일)
    monitors           α 분포 / 분기 토큰 엔트로피 / 예보 표류 KL / λ 감쇠 제어
"""

from .entropy_forecast import HeadCalibration, entropy_advantage, h_togo, masked_zscore, sibling_mask
from .monitors import LambdaDriftController, branch_token_entropy, forecast_policy_kl, token_weight_stats
from .mtp_heads import MTPHeads, build_mtp_heads_for, entropy_from_logits
from .omega_tilde import SteerFConfig, compute_omega_tilde, normalize

__version__ = "0.1.0"

__all__ = [
    "MTPHeads",
    "build_mtp_heads_for",
    "entropy_from_logits",
    "HeadCalibration",
    "h_togo",
    "entropy_advantage",
    "sibling_mask",
    "masked_zscore",
    "SteerFConfig",
    "compute_omega_tilde",
    "normalize",
    "token_weight_stats",
    "branch_token_entropy",
    "forecast_policy_kl",
    "LambdaDriftController",
]
