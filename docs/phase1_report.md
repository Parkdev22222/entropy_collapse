# Phase 1 보고서 — 예보 검증과 (κ, γ_H) 결정

**상태: 미실행.** 이 문서는 결과가 들어갈 자리를 미리 확정해 둔 것이다.
숫자를 채우기 전에는 어떤 칸도 "잠정 결론"으로 인용하지 말 것.
GPU가 없는 환경에서 구현만 완료했다 — `docs/experiment_log.md` 참조.

---

## 1. 무엇을 재는가

계획서 §3.3의 프로토콜. 핵심은 **예보가 실측과 상관이 있는가**이며,
"토큰을 잘 맞히는가"가 아니다. MTP 헤드의 CE가 나빠도 엔트로피 예보는
유효할 수 있고 그 반대도 가능하다.

| 지표 | 정의 | 게이트 |
|---|---|---|
| 주 지표 | 문제-내 Spearman ρ(H_togo, GT_future_entropy), Fisher z 평균 | **≥ 0.2** |
| 보조 | ρ(H_togo, GT_branch_div) | 참고용 |
| 분기 회수율 | recall@10% (A_H 상위 10%가 실제 분기 위치를 얼마나 덮는가) | 랜덤 대비 **p < 0.05** |

**문제-내(within-problem)로 재는 이유.** 문제를 섞어 pooled로 재면 문제 간
난이도 차이가 상관을 부풀린다. `tests/test_validation.py::
test_within_problem_separates_a_simpson_reversal`이 그 반전을 실제로 보여준다
— 문제마다 관계가 −1인 데이터에서 pooled ρ는 +0.8이 나온다.

### GT 정의에 관한 계획서 수정

계획서 §3.3의 `GT_future_entropy`(할인 없는 엔트로피 합)를 그대로 쓰면
할인된 `H_togo^κ`를 무할인 실측에 맞춰 채점하게 되어, `γ_H < 1`이 예보 품질과
무관한 이유로 전부 불리해진다. 따라서:

- **주 지표**: (κ, γ_H) 셀마다 **동일한 할인**을 적용한 실측 오프셋별 엔트로피.
- **부가 기록**: 계획서 원안(`gt_sum_h`)도 함께 저장 (`docs/phase1_results.json`).

---

## 2. 실행 방법

```bash
# 1) 헤드 워밍업 (정책 freeze). 계획서 권장 규모: rollout 약 50k 시퀀스.
python scripts/phase1_warmup_heads.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --rollouts rollout_data/STEER-F-small/<run>/rollouts.jsonl \
    --out checkpoints/mtp_heads.pt \
    --num-heads 8 --head-hidden 1024 --epochs 1

# 2) MC 검증 + (κ, γ_H) 그리드 + 캘리브레이션
python scripts/phase1_validate.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --heads checkpoints/mtp_heads.pt \
    --problems third_party/STEER/datasets/math500.parquet \
    --n-problems 25 --n-trajectories 32 --n-continuations 16 \
    --max-prefixes 600 --calibrate \
    --out docs/phase1_results.json
```

`K = 8`로 학습해 두고 κ는 사후 선택한다 (헤드가 서로 독립이므로 재학습 불필요).

---

## 3. 결과 — 헤드 워밍업

| 항목 | 값 |
|---|---|
| 워밍업 시퀀스 수 | _(미기입)_ |
| 학습 스텝 / 시간 | _(미기입)_ |
| 헤드 파라미터 수 | _(미기입)_ |

헤드별 최종 CE (k가 커질수록 단조 증가해야 정상):

| head | +1 | +2 | +3 | +4 | +5 | +6 | +7 | +8 |
|---|---|---|---|---|---|---|---|---|
| CE | | | | | | | | |

> 단조 증가하지 않으면 워밍업이 수렴하지 않은 것이다. 검증으로 넘어가지 말 것.

---

## 4. 결과 — (κ, γ_H) 그리드

문제-내 ρ (Fisher z 평균):

| γ_H \ κ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| 0.70 | | | | | | | | |
| 0.85 | | | | | | | | |
| 1.00 | | | | | | | | |

**선택**: κ = _(미기입)_, γ_H = _(미기입)_

선택 규칙은 `steer_f.validation.select_kappa_gamma`에 고정되어 있다:
전역 최고 ρ와 `elbow_tolerance`(기본 0.01) 안에서 통계적으로 구분되지 않는
후보들 중 **가장 짧은 κ**를 택한다. 짧은 호라이즌은 예보 비용이 싸고,
먼 헤드의 과대추정 편향에 덜 노출된다.

_(곡선 플롯 자리 — `docs/phase1_results.json`의 `grid`에서 생성)_

---

## 5. 결과 — 캘리브레이션

먼 헤드는 예측이 뭉개져 엔트로피를 체계적으로 **과대추정**한다. 보정 없이 쓰면
"먼 미래는 항상 다양해 보이는" 편향이 생겨 A_H의 변별력이 사라진다
(계획서 §3.4, 리스크 표 4행).

| head | temperature | scale | bias |
|---|---|---|---|
| +1 | | | |
| +2 | | | |
| … | | | |

보정 전후 `H_togo`의 κ에 대한 단조성: _(미기입)_
→ 보정 후에도 κ에 단조 증가하고 A_H 분산이 작다면 **캘리브레이션 실패**이며,
계획서대로 κ를 줄이는 쪽으로 대응한다.

---

## 6. 게이트 G1 판정

판정은 `steer_f.validation.evaluate_gate_g1`이 내린다(테스트로 고정된 규칙).

| 기준 | 값 | 통과 |
|---|---|---|
| 문제-내 ρ ≥ 0.2 | | |
| 분기 recall@10% 유의 (p<0.05) | | |

**판정: _(미기입)_**

> recall 검정의 귀무 선택률은 명목 0.1이 아니라 **실제 상위 10분위 크기**를 쓴다.
> 동점·반올림 때문에 실제 선택률이 0.1에서 벗어나며, 명목값으로 검정하면
> 판정이 편향된다.
>
> `n_branch`가 30 미만이면 검정력이 부족하다. 이때의 FAIL은 "미래 항이
> 노이즈다"가 아니라 "표본이 부족하다"이므로, prefix를 더 모으고 재판정할 것.

### 실패 시 절차 (계획서 §3 게이트 G1)

순서대로 **1회씩만** 재시도하고, 그래도 실패하면 **중단하고 보고**한다.
미래 항이 노이즈라는 결론 자체가 유의미한 부정 결과다.

1. 헤드 워밍업 데이터 증량 → 결과: _(미기입)_
2. `head_hidden` 증가 → 결과: _(미기입)_
3. 순차형(DeepSeek-V3식) MTP → 결과: _(미기입)_

---

## 7. Phase 2로 넘길 값

```bash
STEERF_KAPPA=<미정> STEERF_GAMMA_H=<미정> STEERF_NORM=scale
STEERF_HEADS=checkpoints/mtp_heads.pt
STEERF_CALIB=checkpoints/mtp_calibration.json
```

`phase1_validate.py`가 쓴 `docs/phase1_results.json`의 `calibration` 필드를
`checkpoints/mtp_calibration.json`으로 저장하면 학습 스크립트가 그대로 읽는다.
