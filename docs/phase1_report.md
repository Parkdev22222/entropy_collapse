# Phase 1 리포트 — MTP 예보 검증 및 (κ, γ_H) 결정

> **자리표시자.** 이 파일은 `scripts/phase1_validate.py` 가 실행 시 **덮어쓴다.**
> 현재 개발 컨테이너에 GPU가 없어 검증을 실행하지 못했다 (`docs/experiment_log.md` 참조).

## 실행 방법

```bash
python scripts/phase1_validate.py \
    --model Qwen/Qwen2.5-Math-7B \
    --heads artifacts/mtp_heads_qwen7b.pt \
    --problems $STEER_ROOT/datasets/math500.parquet \
    --workdir artifacts/phase1 \
    --n-problems 24 --n-trajectories 32 --n-mc 16 \
    --calibrate
```

종료 코드: `0` = 게이트 G1 통과, `2` = 실패.

## 생성될 내용

- 선정 문제 수 / prefix 수 / 실측 호라이즌
- 결정된 (κ, γ_H)와 within-problem Spearman ρ
- (κ, γ_H) 전체 그리드 표 + `artifacts/phase1/kappa_gamma_curve.png` 곡선
- 분기 토큰 recall@10% (recall / 랜덤 baseline / lift / 순열검정 p값)
- 헤드별 캘리브레이션 계수 (scale, bias)
- 게이트 G1 판정과 근거

## 게이트 G1 판정 기준 (계획서 §3)

| 조건 | 임계 |
|---|---|
| within-problem ρ(H_togo, GT_future_entropy) | ≥ 0.2 |
| 분기 토큰 recall@10% 가 랜덤 대비 유의 | p < 0.05 **그리고** lift > 1.0 |

두 조건을 **모두** 만족해야 통과다. 판정 로직은 `steer_f/validation.py::evaluate_gate_g1`,
테스트는 `tests/test_validation.py`.

## 실패 시 절차 (계획서 §3 게이트 G1)

1. 헤드 워밍업 데이터 증량 (`--n-prompts`, `--n-samples` 증가)
2. `head_hidden` 증가
3. 순차형 MTP (DeepSeek-V3식) — 현재 미구현, 필요 시 신규 작업

셋 다 실패하면 **중단하고 보고**한다. "미래 항이 노이즈"라는 결론 자체가 유의미한
결과이므로, 부정 결과도 이 파일과 `experiment_log.md` 에 그대로 남긴다.

## 해석 시 주의

- **측정 호라이즌**: `--gt-horizon` (기본 64토큰) 고정 창에서 실측 엔트로피를 합한다.
  전체 continuation 길이에 대해 합하면 길이와 교락되어 상관이 부풀려진다.
  리포트에는 길이정규화 평균 기준 ρ도 함께 나오므로 둘을 같이 볼 것.
- **문제-내 상관**: 문제 간 난이도 차이가 상관을 부풀리므로 반드시 문제 내에서 재고
  Fisher z로 평균한다 (`tests/test_validation.py::test_within_problem_beats_pooled_when_offsets_differ`
  가 pooled 상관이 부호까지 뒤집히는 사례를 보여준다).
- **먼 헤드 편향**: 보정 없이 쓰면 먼 헤드가 엔트로피를 과대추정해 "먼 미래는 항상
  다양해 보이는" 편향이 생긴다. `--calibrate` on/off 결과를 둘 다 기록할 것 (Ablation A6).
