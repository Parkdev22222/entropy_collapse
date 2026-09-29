# Copyright 2026 STEER-F authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""STEER-F — STEER with Future-entropy Forecasting.

Extends STEER's myopic token-entropy estimate ``Omega`` with a forecast of the
entropy that lies *ahead* of each branching decision, so that RLVR training
preserves trajectory-level entropy rather than only per-state entropy.

Layout:
    ``mtp_heads``        Medusa-style parallel MTP heads + their CE loss.
    ``entropy_forecast`` ``H_togo``, head calibration, ``A_H`` baselines.
    ``omega_tilde``      ``Omega_tilde`` and the patched token-weighting.
    ``monitors``         drift / branch-entropy / recall diagnostics.
"""

from .entropy_forecast import (
    HeadCalibration,
    entropy_advantage,
    fit_head_calibration,
    group_mean_baseline,
    h_togo,
    sibling_prefix_baseline,
)
from .monitors import (
    LambdaDriftController,
    branch_recall_at_k,
    branch_token_entropy,
    mtp_policy_kl,
    token_weight_distribution,
)
from .mtp_heads import MTPHeads, entropy_from_logits, mtp_ce_loss
from .omega_tilde import (
    SteerFConfig,
    compute_omega_tilde,
    compute_token_weights_steerf,
    local_omega_signed,
    normalize,
    visit_term,
)

__version__ = "0.1.0"

__all__ = [
    "HeadCalibration",
    "LambdaDriftController",
    "MTPHeads",
    "SteerFConfig",
    "branch_recall_at_k",
    "branch_token_entropy",
    "compute_omega_tilde",
    "compute_token_weights_steerf",
    "entropy_advantage",
    "entropy_from_logits",
    "fit_head_calibration",
    "group_mean_baseline",
    "h_togo",
    "local_omega_signed",
    "mtp_ce_loss",
    "mtp_policy_kl",
    "normalize",
    "sibling_prefix_baseline",
    "token_weight_distribution",
    "visit_term",
]
