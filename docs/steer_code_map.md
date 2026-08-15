# STEER 코드 맵 (Phase 0 산출물)

> 대상: `https://github.com/zz-haooo/STEER` @ `08add1cc27f4d32a78a6c0d6cb857aa52b8f2a55`
> (verl `0.4.1.dev`, `verl/version/version`)
> 원칙: **코드가 진실.** 계획서 §0.1의 서술과 실제 구현이 어긋나는 곳은 전부 §3에 명시.

---

## 1. 한눈에 보기 — 파일과 행 번호

| 무엇 | 위치 | 비고 |
|---|---|---|
| Ω 계산 + 토큰 가중치 매핑 | `verl/trainer/ppo/core_algos.py:580-713` `compute_token_weights` | 계획서가 가리킨 "579~808행"의 앞 절반 |
| STEER 정책 손실 | `verl/trainer/ppo/core_algos.py:717-851` `compute_policy_loss_with_entropy` | `@register_policy_loss("entropy_control")` |
| 손실 모드 디스패치 | `verl/workers/actor/dp_actor.py:505-551` | `loss_mode == "entropy_control"` 분기 |
| 엔트로피 생성 (π_old 패스) | `verl/workers/actor/dp_actor.py:189-193, 244` | `_forward_micro_batch(calculate_entropy=True)` |
| 엔트로피 → 배치 주입 | `verl/trainer/ppo/ray_trainer.py:1198-1219` | `batch.batch["entropys"] = entropys` |
| worker 경유 | `verl/workers/fsdp_workers.py:725-728` | `tensors={"old_log_probs", "entropys", "max_prob_log_probs"}` |
| update 시 키 선택 | `verl/workers/actor/dp_actor.py:406` | `select_keys` — **STEER-F가 새 텐서를 넣어야 하는 곳** |
| 실행 스크립트 | `run/run_linear.sh`, `run_exp.sh`, `run_asymmetric.sh` (+`*_extreme`) | 차이는 §6 |

---

## 2. Ω 계산 함수 — 정확한 시그니처와 텐서 shape

```python
# core_algos.py:580
def compute_token_weights(
    advantages:      torch.Tensor,  # [B, T]  T = response_length
    entropys:        torch.Tensor,  # [B, T]  π_old 패스에서 계산된 토큰 엔트로피 (nats)
    old_log_prob:    torch.Tensor,  # [B, T]  log π_old(y_t | s_t)
    log_prob:        torch.Tensor,  # [B, T]  log π_θ(y_t | s_t)  (현재 파라미터)
    response_mask:   torch.Tensor,  # [B, T]  1 = 실토큰
    token_weight_min: float = 0.8,
    token_weight_max: float = 1.2,
    linear: bool = True,
    mode: str = "symmetric",
) -> torch.Tensor:                  # [B, T]  token_weights
```

- 전체가 `with torch.no_grad():` 안 (581-713). **토큰 가중치에는 그래디언트가 흐르지 않는다.**
- `B` = `ppo_micro_batch_size_per_gpu` (레포 기본 8), `T` = `data.max_response_length` (기본 3072).
- 반환 텐서 dtype은 `torch.float` 고정 (`torch.zeros_like(metric, dtype=torch.float)`, 663행).

### Ω의 실제 수식 (622-644행)

```python
x   = clamp(exp(log_prob), 1e-8, 1-1e-8)          # 현재 정책의 샘플 토큰 확률 π
f_x = x * (1 - x) * (log(x) + entropys)           # 1차 해석적 계수
adv_over_old = advantages / clamp(exp(old_log_prob), 1e-8, 1.0)

metric = +adv_over_old * f_x   if mode == "symmetric"
metric = -adv_over_old * f_x   if mode == "asymmetric"
```

주의할 점 세 가지:

1. **부호 규약이 mode마다 반대다.** `asymmetric`이 계획서 §0.1의 규약(Ω>0 = 엔트로피 증가)과 일치하고,
   `symmetric`은 그 부호를 뒤집은 뒤 곧바로 `abs()`를 취하므로(652행) 결과는 동일하다.
2. 변수명 `x_one_minus_x_squared`(626행)는 **오해를 부르는 이름**이다. 실제 값은 `x*(1-x)`이고 제곱은 없다.
3. 비유한값은 `metric` 단계에서 0으로 치환된다(646-648행). `abs()` **이전**이다.

---

## 3. ⚠️ 밴드/α 결정 로직 — 계획서와 코드가 다른 지점

**계획서 §0.1·§4.1은 다음을 전제한다:**
- 목표 밴드 `[ΔH_low, ΔH_high]`
- 이산 3단 계수 `α ∈ {γ, 1, 1/γ}`
- 밴드 값 재산정 필요

**실제 코드에는 밴드도, γ도, 이산 α도 존재하지 않는다.** 대신 배치 내 통계를 이용한
**연속 매핑** 두 가지가 있다 (670-705행). `linear` 플래그가 둘을 고른다.

### 3.1 `linear=True` — min-max 선형 재스케일

```python
scale = (token_weight_max - token_weight_min) / (metric_max - metric_min)
# symmetric  (내림차순): w = token_weight_max - (metric - metric_min) * scale
# asymmetric (오름차순): w = token_weight_min + (metric - metric_min) * scale
w = clamp(w, token_weight_min, token_weight_max)
```
`metric_min`/`metric_max`는 **해당 마이크로배치의 유효 토큰에서** 매번 새로 구한다(659-660행).
즉 임계값은 절대 스케일이 아니라 배치 상대적이다. `metric_max == metric_min`이면 전 토큰이
중점 `(min+max)/2`를 받는다(683행).

### 3.2 `linear=False` — 지수 매핑

```python
# symmetric:  k = -log(token_weight_min) / max(metric_max, 0.02);  w = exp(-k * metric), clamp[min, 1.0]
# asymmetric: k = -log(token_weight_min) / max(|metric|.max(), 0.02); w = exp(+k * metric), clamp[min, max]
```
`symmetric`의 상한 clamp가 `token_weight_max`가 아니라 **하드코딩 `1.0`** 이다(705행).
그래서 `run_exp.sh`가 `token_weight_max=1.0`을 넘기는 것이며, 이 조합에서 STEER는
"감쇠 전용"으로 동작한다 (어떤 토큰도 1.0을 넘는 가중치를 받지 못함).

### 3.3 STEER-F에 대한 함의 (중요)

Ω̃가 이 매핑에 들어갈 때 **정규화 방식이 결과를 바꿀 수 있다.** 각 매핑의 불변성:

| 매핑 | 불변인 변환 |
|---|---|
| `linear=True` | 임의의 아핀 `a·x + b` (a>0) — min-max 재스케일이므로 |
| `linear=False` | 양의 스케일 `a·x`만. `0.02` 하한이 걸리지 않는 동안. 지수는 평행이동 불변이 아님 |
| `mode="symmetric"` | 추가로 `abs()`가 붙으므로 **0이 의미를 갖는다 → 중심이동 불가** |

세 제약의 교집합은 **"중심이동 없는 양의 스케일링"** 뿐이다.
따라서 STEER-F 기본 정규화는 계획서가 적은 `z_norm`이 아니라 **RMS 스케일링**(`norm="scale"`)이다.
`z`도 선택 가능하지만 `mode="asymmetric"` + `linear=True`에서만 안전하며,
`SteerFConfig.validate()`가 그 외 조합에서 경고를 낸다.
`tests/test_lambda_zero_equiv.py::test_z_norm_would_break_symmetric_equivalence`가
"z를 쓰면 λ=0인데도 가중치가 달라진다"를 실제로 보여준다.

**계획서 §4.1의 "밴드 분위수 재산정" 절차는 그대로 적용할 수 없다.** 재산정할 밴드가 없기 때문이다.
대응물은 `steer_f.monitors.token_weight_distribution` — `[token_weight_min, token_weight_max]`를
3등분한 구간 점유율이다. 계획서가 감시하려던 실패(한쪽에 90% 이상 쏠림)를 동일하게 잡아낸다.

---

## 4. w 조립 위치와 α가 곱해지는 위치

`compute_policy_loss_with_entropy` (717-851) 안의 순서:

| 행 | 내용 |
|---|---|
| 784-794 | `token_weights = compute_token_weights(...)` — **손실 계산보다 먼저** |
| 798-800 | `ratio = exp(clamp(log_prob - old_log_prob, -20, 20))` |
| 803-808 | `pg_losses1 = -A·ratio`, `pg_losses2 = -A·clamp(ratio, 1-lo, 1+hi)` |
| 809 | `clip_pg_losses1 = max(pg_losses1, pg_losses2)` |
| 821-825 | dual-clip: `A<0`이면 `min(-A·clip_ratio_c, clip_pg_losses1)` |
| **828** | **`weighted_pg_losses = pg_losses * token_weights`** ← α가 곱해지는 유일한 지점 |
| 830 | `agg_loss(..., loss_agg_mode)` |

즉 계획서가 말한 `w = clip(ratio)·advantage`는 **`compute_token_weights` 호출 시점에는 아직 없다.**
STEER-F의 visit term은 이 `w`를 필요로 하므로, `compute_token_weights_steerf` 내부에서
`clip_ratio_low`/`clip_ratio_high`를 받아 동일한 클리핑으로 재계산한다
(`steer_f/omega_tilde.py`의 `compute_token_weights_steerf`).
첫 inner epoch에서는 `ratio ≈ 1`이라 `w ≈ advantage`다.

---

## 5. 엔트로피/예보 텐서의 데이터 경로 — STEER-F가 끼어들 자리

```
ray_trainer.py:1199  actor_rollout_wg.compute_log_prob(batch)
   └→ fsdp_workers.py:725  actor.compute_log_prob(data, calculate_entropy=True)
        └→ dp_actor.py:325-396  compute_log_prob
             └→ dp_actor.py:96-296  _forward_micro_batch  (torch.no_grad)
                  returns (entropy, log_probs, max_prob_log_probs)   # 반환 순서 주의
   ← fsdp_workers.py:728  DataProto{old_log_probs, entropys, max_prob_log_probs}
ray_trainer.py:1214  batch.batch["entropys"] = entropys
   ...
dp_actor.py:406  select_keys = [..., "entropys"]      ← 여기 "h_togo"를 추가
dp_actor.py:511  entropy = data["entropys"]           ← 여기 h_togo도 꺼냄
dp_actor.py:534  compute_policy_loss_with_entropy(entropys=entropy, ...)
```

**인덱스 정합성 (MTP 헤드 정렬의 근거).**
`_forward_micro_batch`는 `full_*.squeeze(-1)[:, -response_length-1 : -1]`로 잘라낸다(244-246행).
즉 인덱스 `i`의 log-prob/엔트로피는 `responses[:, i]`를 **예측한** 분포에서 나온 값이고,
그 분포를 만든 히든은 위치 `prompt_len + i - 1`의 것이다.
따라서 `responses[:, i]`를 **소비한 뒤**의 히든은 위치 `prompt_len + i`이며,
여기에 MTP 헤드 k를 붙이면 `y_{t+k+1}`을 예측한다 — 이것이 계획서의
`H_togo(s_t ⊕ y_t)`(샘플된 토큰까지 조건으로 한 예보)와 정확히 일치한다.
슬라이스를 한 칸 옮겨 `[:, -response_length : ]`를 쓰면 된다.

**sibling 그룹 키**: `ray_trainer.py:1168`이 `batch.non_tensor_batch["uid"]`에 프롬프트별
uuid를 넣는다. GRPO advantage도 이 키로 그룹핑한다(252, 266행). STEER-F의
`entropy_advantage(group_index=...)`에 그대로 넘기면 된다.

---

## 6. 하이퍼파라미터가 config로 노출되는 경로

`dp_actor.py:506, 529-532`에서 읽는다. 전부 `policy_loss` 하위:

| CLI 인자 | 코드상 기본값 | `run_linear.sh` | `run_exp.sh` | `run_asymmetric.sh` |
|---|---|---|---|---|
| `actor_rollout_ref.actor.policy_loss.loss_mode` | `"vanilla"` | `entropy_control` | `entropy_control` | `entropy_control` |
| `+...policy_loss.token_weight_min` | `0.95` | `0.8` | `0.8` | `0.9` |
| `+...policy_loss.token_weight_max` | `1.05` | `1.2` | `1.0` | `1.0` |
| `+...policy_loss.linear` | `True` | `True` | `False` | `True` |
| `+...policy_loss.entropy_control_mode` | `"symmetric"` | (기본) | (기본) | `asymmetric` |

> `dp_actor.py`의 기본값 `0.95/1.05`와 `core_algos.py`의 기본값 `0.8/1.2`가 **서로 다르다.**
> 실행 스크립트가 항상 명시적으로 넘기므로 실사용에는 영향이 없지만,
> 스크립트 없이 함수를 직접 호출하면 조용히 다른 세팅이 된다. STEER-F 스크립트는 전부 명시한다.

`+` 접두사는 hydra에서 "config에 없는 키를 새로 추가"를 뜻한다. STEER-F 인자도 같은 방식으로 붙인다.

기타 STEER 실험이 공유하는 세팅 (`run_linear.sh`):
`adv_estimator=grpo`, `train_batch_size=512`, `ppo_mini_batch_size=32`,
`ppo_micro_batch_size_per_gpu=8`, `rollout.n=8` (= 그룹 크기 G),
`max_prompt_length=1024`, `max_response_length=3072`, `lr=1e-6`,
`clip_ratio_low=0.2`, `clip_ratio_high=0.28`, `clip_ratio_c=10.0`,
`use_kl_loss=False`, `entropy_coeff=0`, `total_epochs=10`.

---

## 7. 반환값 계약 (패치 시 깨뜨리면 안 되는 것)

`compute_policy_loss_with_entropy`는 **9개**를 반환하고 `dp_actor.py:551`이 그대로 언팩한다:

```
pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, pg_clipfrac_high,
pg_clipfrac_low, clipped_by_high, clipped_by_low, clip_stats
```

`clip_stats`(838-848)는 dict이고 `dp_actor.py:751`이 `"token_weights_mean"` 유무로 로깅을 분기한다.
STEER-F 지표는 **`clip_stats`에 키를 추가**하는 방식으로 넣는다 — 반환 arity를 건드리면
`vanilla`/`gspo` 등 다른 경로의 언팩 분기(572-616행)까지 영향을 받는다.

---

## 8. 관찰된 이슈 (STEER-F 실험 설계에 영향)

1. **`dp_actor.py:554`에 `print(f"Entering vanilla branch")`가 남아 있다.** GRPO baseline 실험에서
   마이크로배치마다 stdout에 찍힌다. 로그가 GB 단위로 불어나므로 baseline 실행 전에 제거 권장
   (`patches/` 에 별도 패치로 분리해 두었다 — STEER-F 로직과 섞지 않기 위함).
2. **`compute_token_weights`의 임계값이 마이크로배치 상대적이다.** `ppo_micro_batch_size_per_gpu`를
   바꾸면 동일 하이퍼파라미터라도 가중치 분포가 달라진다. Phase 2의 소형/7B 비교에서
   micro batch size를 반드시 고정하거나, 최소한 보고서에 명시할 것.
3. `entropys`는 `temperature`로 나눈 로짓에서 계산된다(`dp_actor.py:171, 271`). MTP 헤드 예보도
   동일 temperature를 써야 `A_H`가 같은 척도가 된다 — `forecast_entropy(temperature=...)` 사용.
4. `run_linear.sh`의 `best_metric_key=val-core/math_dapo/acc/mean@32`는 검증셋이 `aime24`
   단일 파일임을 전제한다. Phase 2 평가에서 벤치마크를 늘리면 이 키도 같이 바꿔야 한다.

---

## 9. 게이트 G0 상태

- [x] `docs/steer_code_map.md` 작성 완료 — 이 문서.
- [ ] STEER 소형 재현 학습 (GRPO vs STEER, 100~150 스텝) — **미실행.**
      본 세션 환경에 GPU가 없다 (CPU 4코어 / RAM 15GB / `nvidia-smi` 없음).
      실행 방법은 `run/run_steerf_small.sh`와 `docs/experiment_log.md` 참조.

**G0는 아직 미통과이며, 이는 예상된 상태다.** 재현 학습 없이 Phase 1의 결론을 신뢰해서는 안 된다.
