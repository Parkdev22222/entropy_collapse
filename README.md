# STEER-F: STEER with Future-entropy Forecasting

STEER의 근시안적 토큰 엔트로피 변화 추정 Ω를, MTP(Multi-Token Prediction) 헤드
기반 미래 엔트로피 예보로 확장한 Ω̃로 대체한다.

```
Ω̃_{i,t} = z(Ω_{i,t}) + λ · z( Δlogπ̂_{i,t} · clip(A_H, -c, c) )

Δlogπ̂_{i,t} ≈ η · w_{i,t} · (1 - π_{i,t}),     w = A / π_old
A_H(s_t, y_t) = H_togo(s_t ⊕ y_t) - H̄_togo(s_t)
H_togo^κ(s)   = Σ_{k=1..κ} γ_H^k · H( p_MTP(y_{t+k} | s) )
```

**λ = 0이면 순수 STEER와 비트 단위로 동일**하다. `tests/test_lambda_zero_equiv.py`가
업스트림 구현의 자동 추출본과 대조해 이를 강제한다.

---

## 현재 상태

| Phase | 코드/문서 | 실행 |
|---|---|---|
| 0 — 코드 맵 | ✅ `docs/steer_code_map.md` | 재현 학습 ❌ (GPU 없음) |
| 1 — MTP 헤드·검증 | ✅ 모듈 + 워밍업/검증 스크립트 | ❌ (GPU 없음) |
| 2 — Ω̃ 통합 | ✅ 패치 + 105개 단위 테스트 통과 | ❌ (GPU 없음) |
| 3 — 패밀리 전이 | ✅ 이식 헬퍼 | ❌ |
| 4 — Ablation | ✅ 전 축이 config 노출됨 | ❌ |

게이트 G0/G1/G2/G3는 **모두 미판정**이다 — 판정에 필요한 학습·샘플링이 이 개발
환경(CPU 전용)에서 실행 불가하기 때문. 자세한 내용은 `docs/experiment_log.md`.

---

## 저장소 구조

```
steer_f/
  mtp_heads.py          Medusa 스타일 병렬 헤드 + 메모리 안전 엔트로피 forward
  entropy_forecast.py   H_togo, A_H(sibling/group baseline), 헤드 캘리브레이션
  omega_tilde.py        Ω̃ 결합 (모드 인지, λ=0 우회)
  token_weights.py      STEER 함수 + 훅 1개 (매핑 로직 무수정)
  monitors.py           α 분포 / 분기 토큰 엔트로피 / 예보 표류 KL / λ 감쇠
  validation.py         Phase 1 지표 (Spearman, Fisher z, recall@k) + 게이트 G1
  verl_integration.py   헤드 부착, h_togo 산출, MTP 보조 손실, 표류 측정
scripts/
  extract_reference.py     STEER 원본 함수 verbatim 추출 (테스트 기준선)
  phase1_warmup_heads.py   롤아웃 생성 + 헤드 CE 워밍업
  phase1_validate.py       MC 검증 + (κ, γ_H) 그리드 + G1 판정 + 리포트
  phase3_port_model.py     패밀리 이식 점검 + 그룹 pass rate 분포
patches/
  core_algos_steerf.patch  verl 수정분 (git apply 검증 완료)
run/
  run_steerf_linear.sh     LAMBDA=0이면 STEER 재현, >0이면 STEER-F
docs/
  steer_code_map.md        Phase 0 산출물 — 실제 코드 기준 라인 단위 맵
  experiment_log.md        전 실험 러닝 로그 (실패 포함)
  phase1_report.md         phase1_validate.py가 생성 (현재는 자리표시자)
tests/                     105 tests, CPU에서 2초
```

---

## 빠른 시작

```bash
pip install torch pytest numpy pandas pyarrow
python -m pytest tests/ -q          # 105 passed
```

### 전체 파이프라인 (GPU 노드)

```bash
# 0. STEER 레포에 패치 적용
git clone https://github.com/zz-haooo/STEER && cd STEER
git apply /path/to/STEER-F/patches/core_algos_steerf.patch
export STEER_ROOT=$PWD

# 0b. baseline 재현 (STEER와 비트 동일)
LAMBDA=0 bash /path/to/STEER-F/run/run_steerf_linear.sh

# 1. MTP 헤드 워밍업
python scripts/phase1_warmup_heads.py generate --model Qwen/Qwen2.5-Math-7B \
    --prompts $STEER_ROOT/datasets/DAPO-Math-17k.parquet \
    --n-prompts 6250 --n-samples 8 --out artifacts/rollouts.jsonl
python scripts/phase1_warmup_heads.py train --model Qwen/Qwen2.5-Math-7B \
    --rollouts artifacts/rollouts.jsonl --num-heads 8 --out artifacts/mtp_heads.pt

# 2. 예보 검증 + (κ, γ_H) 결정 → 게이트 G1
python scripts/phase1_validate.py --model Qwen/Qwen2.5-Math-7B \
    --heads artifacts/mtp_heads.pt --problems $STEER_ROOT/datasets/math500.parquet \
    --calibrate
#   exit code 0 = G1 통과, 2 = G1 실패 (계획서 §3 실패 절차 수행)

# 3. RL 본실험 (λ 스윕)
for LAM in 0 0.25 0.5 1.0; do
  LAMBDA=$LAM KAPPA=4 GAMMA_H=0.85 MTP_HEADS=artifacts/mtp_heads.pt \
    bash run/run_steerf_linear.sh
done
```

---

## 계획서와 실제 코드가 달랐던 지점

`docs/steer_code_map.md` §6에 전부 기록했다. Phase 2 설계에 영향을 준 것들:

1. **α는 이산 3단({γ, 1, 1/γ})이 아니다.** 배치 min-max 연속 매핑으로
   [0.8, 1.2] 구간에 사상한다. 절대 밴드 `[ΔH_low, ΔH_high]`는 코드에 존재하지 않는다.
2. 따라서 계획서 §4.1의 "밴드 재산정" 작업은 **불필요**하다. 게다가 `linear=True`의
   min-max 매핑은 아핀 불변이므로 z-정규화가 α를 바꾸지 않는다
   (`test_zscore_is_affine_invariant_under_linear_mapping`이 확인).
3. `w = clip(ratio) · A`가 아니라 **`w = A / π_old`**다. Δlogπ̂ 정의를 코드에 맞췄다.
4. symmetric 모드는 `|Ω|`를 쓰므로 미래 항도 크기로 결합해야 의미론이 맞는다
   (모드별 결합 규칙: `steer_f/omega_tilde.py`).
5. 새 리스크: 배치 min-max는 이상치 하나에 좌우된다. `steerf_norm=robust`
   (median/IQR) 옵션과 α 히스토그램 로깅을 추가했다.

---

## 미검증 표면

`steer_f/verl_integration.py`는 형상 계약만 단위 테스트되어 있고, **실제 verl 런에서
end-to-end로 검증되지 않았다.** 특히 다음 세 곳은 GPU 노드에서 첫 실행 시 확인이 필요하다:

1. `_forward_micro_batch`에 `output_hidden_states=True`를 추가하고 rmpad 레이아웃의
   히든을 꺼내는 부분 (Ulysses SP가 켜져 있으면 gather 경로도 동일하게 타야 함).
2. FSDP wrap 시점의 헤드 파라미터 등록 및 그래디언트 동기화.
3. `ray_trainer.py`에서 `h_togo` / `baseline_h_togo`를 배치에 실어 나르는 부분
   (`attach_entropy_advantage`).

각 지점의 정확한 삽입 위치는 `steer_f/verl_integration.py` 상단 docstring과
`docs/steer_code_map.md` §5에 적어 두었다.
