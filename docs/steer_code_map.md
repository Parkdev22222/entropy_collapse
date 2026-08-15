# STEER 코드 맵 (Phase 0 산출물)

**대상**: https://github.com/zz-haooo/STEER (verl v0.4.1.x 동봉본)
**커밋**: `--depth 1` 클론 기준, 2026-08-15 시점 `main`
**작성 원칙**: 논문 서술이 아니라 **코드가 진실**. 계획서(§0.1)와 어긋나는 부분은 §6에 따로 모았다.

---

## 1. 파일·라인 인덱스

| 대상 | 위치 |
|---|---|
| Ω 계산 + 토큰 가중치 산출 | `verl/trainer/ppo/core_algos.py:580-713` (`compute_token_weights`) |
| STEER 정책 손실 | `verl/trainer/ppo/core_algos.py:717-851` (`compute_policy_loss_with_entropy`, `@register_policy_loss("entropy_control")`) |
| 바닐라 GRPO/PPO 손실 (비교 기준) | `verl/trainer/ppo/core_algos.py:854-937` (`compute_policy_loss`) |
| 손실 호출부 / 설정 읽기 | `verl/workers/actor/dp_actor.py:502-551` |
| 모델 forward (logits·entropy·log_prob) | `verl/workers/actor/dp_actor.py:82-260` (`_forward_micro_batch`) |
| token_weights 메트릭 로깅 | `verl/workers/actor/dp_actor.py:750-751` |
| 실행 스크립트 | `run/run_linear.sh`, `run/run_exp.sh`, `run/run_asymmetric.sh`, `run/run_*_extreme.sh` |

---

## 2. Ω 계산: 정확한 시그니처와 텐서 shape

```python
# core_algos.py:580
def compute_token_weights(
    advantages:     torch.Tensor,   # (B, T_resp)
    entropys:       torch.Tensor,   # (B, T_resp)  현재 정책의 토큰별 엔트로피 H(π(·|s_t))
    old_log_prob:   torch.Tensor,   # (B, T_resp)  log π_old(y_t | s_t)  (샘플된 토큰)
    log_prob:       torch.Tensor,   # (B, T_resp)  log π_θ(y_t | s_t)    (샘플된 토큰)
    response_mask:  torch.Tensor,   # (B, T_resp)  0/1
    token_weight_min: float = 0.8,
    token_weight_max: float = 1.2,
    linear: bool = True,
    mode: str = "symmetric",        # "symmetric" | "asymmetric"
) -> torch.Tensor                   # (B, T_resp) float, 마스크 밖은 0
```

`T_resp = responses.size(1)` (= `data.max_response_length`). 전 구간이 `torch.no_grad()` 안 (`core_algos.py:619`) — **가중치는 그래디언트를 흘리지 않는다**. 이 점이 STEER-F 설계에 중요하다 (§7.3).

### 2.1 Ω의 실제 수식 (core_algos.py:623-653)

```python
x   = clamp(exp(log_prob), 1e-8, 1-1e-8)        # π_θ(y_t|s_t), 샘플된 토큰의 현재 확률
f_x = x * (1 - x) * (log(x) + H_t)              # 변수명은 x_one_minus_x_squared 지만 실제로는 x(1-x)
old = clamp(exp(old_log_prob), 1e-8, 1.0)       # π_old(y_t|s_t)
w   = advantages / old                          # ★ "advantage_over_old_prob"

metric = w * f_x                  # mode == "symmetric"
metric = -w * f_x                 # mode == "asymmetric"  (부호 규약: Ω>0 = 엔트로피 증가)
```

즉

```
Ω_{i,t} = ( A_{i,t} / π_old(y_t|s_t) ) · π_θ(y_t|s_t)(1 - π_θ(y_t|s_t)) · ( log π_θ(y_t|s_t) + H_t )
```

- 비유한값은 0으로 치환하고 경고 출력 (`core_algos.py:646-648`).
- `mode == "symmetric"`이면 여기서 **`metric = |metric|`** (`core_algos.py:650-652`).
- 유효 토큰(`response_mask`)이 하나도 없으면 전부 0 반환 (`656-657`).

> **계획서와의 대조**: 계획서는 `w = clip(ratio) * advantage`라고 썼지만, 코드의 `w`는 **`advantage / π_old`**이며 ratio 클리핑은 여기 관여하지 않는다 (클리핑은 손실 쪽 `pg_losses1/2/3`에서만). STEER-F의 `Δlogπ̂ = η·w·(1-π)`는 코드의 `advantage_over_old_prob`을 그대로 재사용해야 1차 전개가 일관된다 — 실제로 그 형태가 맞다:
> Δπ_sampled ≈ η·(A/π_old)·π(1-π) ⇒ Δlogπ = Δπ/π ≈ η·(A/π_old)·(1-π).

---

## 3. Ω → 토큰 가중치 α: **밴드가 아니라 배치 min-max 연속 매핑**

`core_algos.py:659-710`. 배치 내 유효 토큰의 `metric_min`, `metric_max`를 매 마이크로배치마다 새로 구한다.

### 3.1 `linear=True` (논문 기본, `run_linear.sh`)

```python
scale = (token_weight_max - token_weight_min) / (metric_max - metric_min)

# symmetric: 내림차순 — |Ω| 최대 토큰이 최소 가중치
α = token_weight_max - (metric - metric_min) * scale

# asymmetric: 오름차순 — Ω 최소(가장 엔트로피 감소) 토큰이 최소 가중치
α = token_weight_min + (metric - metric_min) * scale

α = clamp(α, token_weight_min, token_weight_max)
```

`metric_max == metric_min`이면 전부 `(min+max)/2`.

### 3.2 `linear=False` (지수 매핑, `run_exp.sh`)

```python
# symmetric
k = -log(token_weight_min) / max(metric_max, 0.02)
α = clamp(exp(-k * metric), token_weight_min, 1.0)     # ★ 상한이 token_weight_max 가 아니라 1.0

# asymmetric
abs_max = max(|metric|.max(), 0.02)
k = -log(token_weight_min) / abs_max
α = clamp(exp(k * metric), token_weight_min, token_weight_max)
```

### 3.3 적용 지점

```python
# core_algos.py:825-830
pg_losses         = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
weighted_pg_losses = pg_losses * token_weights          # ← α가 곱해지는 유일한 지점
pg_loss = agg_loss(weighted_pg_losses, response_mask, loss_agg_mode)
```

`token_weights`는 마지막에 `* response_mask.float()` (`710`)이므로 마스크 밖은 0.

### 3.4 α 통계 로깅

`core_algos.py:832-848`이 `clip_stats`에 `token_weights_{mean,min,max}`를 담고, `dp_actor.py:750-751`이 `entropy_control` 모드일 때만 메트릭으로 올린다. **히스토그램/분위수는 없다** — Phase 0 게이트가 요구하는 "α 비율" 로깅은 우리가 추가해야 한다 (`steer_f/monitors.py`).

---

## 4. 하이퍼파라미터 노출 경로

Hydra 오버라이드 → `self.config.policy_loss.*` → `dp_actor.py:528-548` → `compute_policy_loss_with_entropy`.

| CLI 키 | 코드 기본값 | `run_linear.sh` 값 |
|---|---|---|
| `actor_rollout_ref.actor.policy_loss.loss_mode` | `vanilla` | `entropy_control` |
| `+...policy_loss.token_weight_min` | `0.95` (dp_actor) / `0.8` (core_algos 시그니처) | `0.8` |
| `+...policy_loss.token_weight_max` | `1.05` (dp_actor) / `1.2` (core_algos) | `1.2` |
| `+...policy_loss.linear` | `True` | `True` |
| `+...policy_loss.entropy_control_mode` | `symmetric` | (미지정 → symmetric) |

> `dp_actor.py`의 `.get()` 기본값(0.95/1.05)과 `core_algos.py` 시그니처 기본값(0.8/1.2)이 **다르다**. run 스크립트가 항상 명시하므로 실사용엔 문제없지만, 스크립트 없이 호출하면 조용히 약한 세팅이 된다. STEER-F 스크립트도 반드시 명시할 것.

`+` 접두사는 verl 기본 config에 없는 키를 추가하는 Hydra 문법이다. STEER-F 신규 키도 전부 `+`가 필요하다.

`run_linear.sh` 기타 핵심: `algorithm.adv_estimator=grpo`, `rollout.n=8` (= 그룹 크기 G), `train_batch_size=512`, `ppo_mini_batch_size=32`, `clip_ratio_low=0.2 / high=0.28 / c=10.0`, `entropy_coeff=0`, `use_kl_loss=False`.

---

## 5. 엔트로피가 어디서 오는가 (MTP 헤드 삽입 지점 파악용)

`dp_actor.py:510-526`:

- `entropy_control` 모드에서 배치에 `entropys`가 이미 있으면 그것을 쓰고 log_prob만 재계산 (`514`).
- 없으면 `_forward_micro_batch(..., calculate_entropy=...)`로 계산.

`_forward_micro_batch` (`dp_actor.py:82-260`)는 remove-padding(rmpad) 경로를 쓴다:

- `output = self.actor_module(input_ids=input_ids_rmpad, ...)` → `output.logits` `(1, total_nnz, V)` → squeeze → `(total_nnz, V)`.
- 온도 나눗셈은 `logits_rmpad.div_(temperature)` (in-place).
- 엔트로피는 `compute_entropy_from_logits(logits_rmpad)`, 이후 `pad_input(...)`으로 `(B, T)` 복원.
- Ulysses SP를 쓰면 시퀀스가 쪼개졌다가 `gather_outpus_and_unpad`로 모인다.

**STEER-F 삽입 지점**: MTP 헤드는 마지막 히든이 필요하다. `self.actor_module(...)` 호출에 `output_hidden_states=True`를 주고 `output.hidden_states[-1]` (`(1, total_nnz, H)`)를 packed 레이아웃 그대로 헤드에 통과시킨 뒤, log_prob과 동일하게 `pad_input`으로 `(B, T)` 복원하는 것이 가장 침습이 적다. SP가 켜져 있으면 히든도 log_prob과 같은 gather 경로를 타야 한다.

> **비용 경고**: `lm_head(proj(hidden))`를 K=8회 하면 `(total_nnz, V)` 로짓이 8개 생긴다. Qwen2.5 기준 V≈152k, total_nnz가 수만이면 bf16로도 수십 GB. **엔트로피만 필요하므로 로짓을 상주시키면 안 된다** — 헤드별로 로짓을 만들고 즉시 엔트로피로 축약한 뒤 버리는 청크 루프가 필수다. `steer_f/mtp_heads.py`의 `forward_entropy()`가 이 역할을 한다. 계획서에 적힌 `torch.stack([...])` 형태(`[K,B,T,V]`)는 **검증 스크립트용 소형 배치에서만** 쓸 것.

---

## 6. 계획서 서술과 코드가 어긋나는 지점 (Phase 2 설계에 직접 영향)

| 계획서 §0.1 서술 | 실제 코드 | 영향 |
|---|---|---|
| α ∈ {γ, 1, 1/γ} 이산 3단 | **연속 매핑**, α ∈ [0.8, 1.2] | "이산 α로 예보 오차를 유계 방어"라는 논거가 성립하지 않음. 유계성은 `token_weight_min/max` 클램프가 대신 제공한다 (여전히 유계이긴 하다). |
| 목표 밴드 `[ΔH_low, ΔH_high]`와 비교 | **절대 밴드 없음.** 배치 min-max 상대 순위만 사용 | 계획서 §4.1의 "밴드 재산정" 작업은 **불필요**. 대신 아래 두 가지가 새 이슈가 된다. |
| Ω̃ z-정규화 후 밴드 값 못 씀 | `linear=True`에서 min-max 매핑은 **아핀 불변** | z-정규화는 `linear=True`일 때 α를 **전혀 바꾸지 않는다**(수학적으로). λ=0 동치성이 공짜로 성립. 반대로 `linear=False`에서는 아핀 불변이 아니므로 z-norm이 결과를 바꾼다. |
| `w = clip(ratio) * advantage` | `w = advantage / π_old` | `Δlogπ̂` 정의를 코드 쪽에 맞춰야 함 (§2.1). |
| 대칭 모드에서 Ω의 부호 사용 | symmetric은 `|Ω|`를 씀 | 미래 항을 부호 있는 채로 더하면 대칭 모드의 의미론이 깨진다. §7.2의 모드별 결합 규칙 필요. |

### 6.1 새로 드러난 리스크: min-max의 이상치 민감성

`metric_max`는 마이크로배치 내 단 하나의 이상치로 결정된다. 이상치 하나가 크면 나머지 토큰 전부가 `token_weight_max` 근처로 압축되어 STEER가 사실상 무력화된다. 미래 항을 더하면 꼬리가 두꺼워질 수 있으므로 **분위수 기반 로버스트 정규화 옵션**을 두고, α 히스토그램을 반드시 로깅해야 한다 (`monitors.py`).

---

## 7. STEER-F 통합 계약 (이 코드 맵에서 도출)

### 7.1 교체 지점
`compute_token_weights`에서 `metric`이 확정된 직후(`core_algos.py:653`, `|·|` 적용 후) ~ `valid_metric` 추출 직전(`655`) 사이에 딱 한 번 훅을 넣는다. min-max 매핑 이하 로직(`659-710`)은 **한 줄도 건드리지 않는다**.

### 7.2 모드별 결합 규칙
- `symmetric`: base가 `|Ω|`(엔트로피 변화의 **크기**)이므로 미래 항도 크기로 결합 —
  `Ω̃ = z(|Ω|) + λ·z(|Δlogπ̂ · clip(A_H)|)`
- `asymmetric`: base가 부호 있는 Ω이므로 부호 그대로 —
  `Ω̃ = z(Ω) + λ·z(Δlogπ̂ · clip(A_H))`

`steer_f/omega_tilde.py`가 `mode`를 받아 이를 처리한다.

### 7.3 no_grad 제약
`compute_token_weights` 전체가 `no_grad`이므로, α 경로로 들어가는 `H_togo`는 **detach된 값**이어야 한다. MTP 헤드의 CE 보조 손실(`β_mtp · L_mtp_ce`)은 별도 경로로 흘려야 하며 α 계산과 그래디언트를 공유하지 않는다.

### 7.4 λ=0 동치성 보장 방식
`lam == 0.0`이면 훅이 `metric`을 **그대로 반환**(z-norm도 건너뜀)하여 비트 동일성을 보장한다. 추가로 `linear=True`에서 z-norm 적용본이 동일 α를 주는지 별도 테스트로 확인한다 (`tests/test_lambda_zero_equiv.py`).

---

## 8. Phase 0 게이트 G0 상태

- [x] `docs/steer_code_map.md` 작성 완료
- [ ] STEER 소형 재현 학습 (GRPO vs STEER, 100~150 스텝) — **미실행**. 이 개발 컨테이너에 GPU가 없다 (`nvidia-smi` 부재, CUDA 미가용). 실행 스크립트와 로깅 훅은 준비되어 있으므로 GPU 노드에서 `run/run_steerf_linear.sh`의 `LAMBDA=0` 세팅으로 그대로 돌리면 된다.

G0의 코드 이해 항목은 충족, 재현 실험 항목은 **미충족(환경 제약)**. 계획서 규칙대로 이 사실을 보고하며, Phase 1의 실행 단계(워밍업/MC 검증)도 같은 이유로 미실행이다.
