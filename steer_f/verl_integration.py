"""verl 러너에 MTP 헤드를 붙이는 헬퍼.

`patches/core_algos_steerf.patch`가 손실 경로를 준비해 두면, 남는 일은
배치에 `h_togo` / `baseline_h_togo` 를 채워 넣고 MTP 보조 손실을 흘리는 것뿐이다.
이 모듈이 그 두 가지를 담당한다.

**적용 위치** (docs/steer_code_map.md §5 참조):

1. `verl/workers/fsdp_workers.py` — 액터 모델을 FSDP로 감싸는 지점 근처에서
   `attach_mtp_heads(actor_module, config)` 를 호출하고, 반환된 헤드 파라미터를
   액터 옵티마이저 param group에 추가한다. 헤드는 정책과 **함께** 업데이트되어야
   낡지 않는다 (계획서 §0.3).

2. `verl/workers/actor/dp_actor.py::_forward_micro_batch` — 본체 forward를
   `output_hidden_states=True` 로 바꾸고 `output.hidden_states[-1]` 를 반환값에
   추가한다. rmpad(packed) 레이아웃 그대로 `compute_h_togo_packed()` 에 넘긴다.

3. `verl/trainer/ppo/ray_trainer.py` — 롤아웃 직후 `compute_log_prob` 패스에서
   나온 h_togo를 `batch.batch["h_togo"]` 로 넣고, `attach_entropy_advantage()` 로
   sibling baseline을 계산해 `batch.batch["baseline_h_togo"]` 에 넣는다.

이 파일의 함수들은 verl 없이 임포트 가능하며(순수 torch), 형상 계약은
`tests/test_verl_integration.py` 가 모의 텐서로 검증한다. 다만 **실제 verl 런에서
end-to-end로 검증되지 않았다** — GPU가 있는 환경에서 첫 실행 시 확인이 필요하다.
"""

from __future__ import annotations

import pathlib
from typing import Callable, Optional

import torch

from .entropy_forecast import HeadCalibration, entropy_advantage, h_togo
from .monitors import forecast_policy_kl
from .mtp_heads import MTPHeads

__all__ = [
    "load_heads_checkpoint",
    "attach_mtp_heads",
    "compute_h_togo_packed",
    "attach_entropy_advantage",
    "mtp_auxiliary_loss",
    "measure_forecast_drift",
]


def load_heads_checkpoint(path: str | pathlib.Path, map_location="cpu") -> tuple[MTPHeads, dict, Optional[HeadCalibration]]:
    """`scripts/phase1_warmup_heads.py train` 산출물을 로드."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ckpt["config"]
    heads = MTPHeads(
        hidden_size=cfg["hidden_size"],
        num_heads=cfg["num_heads"],
        head_hidden=cfg["head_hidden"],
        vocab_size=cfg.get("vocab_size"),
        residual=cfg.get("residual", True),
    )
    heads.load_state_dict(ckpt["state_dict"])
    calib = HeadCalibration.from_dict(ckpt["calibration"]) if ckpt.get("calibration") else None
    return heads, cfg, calib


def attach_mtp_heads(actor_module, heads_path: str, optimizer: Optional[torch.optim.Optimizer] = None, lr: Optional[float] = None):
    """헤드를 로드해 액터와 같은 device/dtype에 올리고, 옵티마이저에 등록한다.

    Returns:
        (heads, calibration)

    Note:
        FSDP를 쓰는 경우 헤드는 본체와 **같은 wrap 정책**을 타야 한다.
        헤드가 작으므로(수십 M) 별도 wrap 없이 rank마다 복제해도 무방하지만,
        그 경우 그래디언트를 수동으로 all-reduce 해야 한다. 가장 단순한 경로는
        본체 FSDP wrap **이전에** `actor_module.mtp_heads = heads` 로 붙여
        같이 감싸지게 하는 것이다.
    """
    heads, _, calib = load_heads_checkpoint(heads_path)
    param = next(actor_module.parameters())
    heads = heads.to(device=param.device, dtype=param.dtype)

    if optimizer is not None:
        group = {"params": list(heads.parameters())}
        if lr is not None:
            group["lr"] = lr
        optimizer.add_param_group(group)
    return heads, calib


@torch.no_grad()
def compute_h_togo_packed(
    heads: MTPHeads,
    hidden_states: torch.Tensor,
    lm_head: Callable[[torch.Tensor], torch.Tensor],
    kappa: int,
    gamma_h: float,
    calib: Optional[HeadCalibration] = None,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """packed 또는 (B, T, H) 히든에서 H_togo를 계산.

    로짓을 K개 동시에 만들지 않는다 (`MTPHeads.forward_entropy` 참조).
    반환 shape은 `hidden_states.shape[:-1]`.
    """
    head_ent = heads.forward_entropy(hidden_states, lm_head, kappa=kappa, chunk_size=chunk_size)
    return h_togo(head_ent, kappa=kappa, gamma_h=gamma_h, calib=calib)


def attach_entropy_advantage(
    batch,
    group_size: int,
    baseline: str = "sibling",
    h_togo_key: str = "h_togo",
    responses_key: str = "responses",
) -> dict:
    """`batch.batch[h_togo_key]` 에서 baseline을 계산해 `baseline_h_togo` 를 채운다.

    verl의 `DataProto` 를 그대로 받도록 되어 있지만, `.batch` 가 dict-like이면
    무엇이든 동작한다 (테스트는 평범한 dict로 검증한다).

    Returns:
        로깅용 통계 dict.
    """
    td = batch.batch if hasattr(batch, "batch") else batch
    h = td[h_togo_key]
    resp = td[responses_key]
    mask = td.get("response_mask")
    if mask is None:
        attn = td.get("attention_mask")
        mask = attn[:, -resp.shape[1]:] if attn is not None else torch.ones_like(h)

    a_h, base = entropy_advantage(
        h,
        response_ids=resp,
        group_size=group_size,
        response_mask=mask.float(),
        baseline=baseline,
    )
    td["baseline_h_togo"] = base

    valid = mask.bool()
    if not valid.any():
        return {}
    return {
        "steerf/h_togo_mean": float(h[valid].mean()),
        "steerf/h_togo_std": float(h[valid].std()) if valid.sum() > 1 else 0.0,
        "steerf/a_h_std": float(a_h[valid].std()) if valid.sum() > 1 else 0.0,
        "steerf/a_h_zero_frac": float((a_h[valid].abs() < 1e-8).float().mean()),
    }


def mtp_auxiliary_loss(
    heads: MTPHeads,
    hidden_states: torch.Tensor,
    lm_head: Callable[[torch.Tensor], torch.Tensor],
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    beta_mtp: float = 0.05,
    kappa: Optional[int] = None,
) -> tuple[torch.Tensor, dict]:
    """L_total = L_STEER + β_mtp · L_mtp_ce 의 두 번째 항.

    헤드가 정책과 함께 진화해 예보가 낡는 것을 막는다 (계획서 §4.1).
    `beta_mtp=0` 이면 헤드가 freeze된 것과 같다 (Ablation A7).
    """
    if beta_mtp == 0.0:
        return hidden_states.new_zeros(()), {}
    ce, per_head = heads.ce_loss(hidden_states, lm_head, input_ids, mask=response_mask, kappa=kappa)
    stats = {f"steerf/{k}": float(v) for k, v in per_head.items()}
    stats["steerf/mtp_ce"] = float(ce.detach())
    return beta_mtp * ce, stats


@torch.no_grad()
def measure_forecast_drift(
    heads: MTPHeads,
    hidden_states: torch.Tensor,
    lm_head: Callable[[torch.Tensor], torch.Tensor],
    policy_logits: torch.Tensor,
    response_mask: Optional[torch.Tensor] = None,
    max_samples: int = 2048,
) -> float:
    """헤드 +1 예측과 정책 next-token 분포의 KL (표본 측정).

    헤드 +1은 위치 t에서 y_{t+1}을 예측하고 정책은 위치 t+1에서 y_{t+1}을 예측하므로,
    정책 로짓을 한 칸 당겨 정렬한다.

    Args:
        hidden_states: (B, T, H).
        policy_logits: (B, T, V) 본체 로짓.
        response_mask: (B, T).
    """
    b, t, _ = hidden_states.shape
    if t < 2:
        return 0.0
    h = hidden_states[:, :-1, :].reshape(-1, hidden_states.shape[-1])
    pol = policy_logits[:, 1:, :].reshape(-1, policy_logits.shape[-1])

    if response_mask is not None:
        m = (response_mask[:, :-1].bool() & response_mask[:, 1:].bool()).reshape(-1)
    else:
        m = torch.ones(h.shape[0], dtype=torch.bool, device=h.device)

    idx = torch.nonzero(m, as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return 0.0
    if idx.numel() > max_samples:
        # 결정적 서브샘플링 (RNG 상태를 건드리지 않는다)
        stride = idx.numel() // max_samples
        idx = idx[::stride][:max_samples]

    head1 = lm_head(heads._head_hidden(h[idx], 0))
    return float(forecast_policy_kl(head1, pol[idx]))
