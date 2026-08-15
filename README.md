# STEER-F — STEER with Future-entropy Forecasting

STEER reweights each token's RLVR learning signal by a first-order prediction
of how the update will change **that token's** entropy. That estimate is
myopic: it sees the local term of the trajectory entropy and misses the
visitation term, where a policy that concentrates at a branch point silently
deletes every future state down the abandoned branch.

STEER-F adds that missing term. MTP heads forecast the entropy that lies ahead
of each decision, the forecast becomes a branch score `A_H`, and the score
extends `Ω` into `Ω̃`:

```
Ω̃   = norm(Ω) + λ · norm( Δlogπ̂ · clip(A_H, -c, c) )
Δlogπ̂ = η · w · (1 - π)
A_H   = H_togo(s_t ⊕ y_t) - H̄_togo(s_t)
H_togo = Σ_{k=1..κ} γ_H^k · H( p_MTP(y_{t+k} | s) )
```

`λ = 0` is bit-identical to stock STEER — enforced by a test against a verbatim
copy of upstream's function, not by inspection.

## Status

| Phase | Deliverable | State |
|---|---|---|
| 0 | `docs/steer_code_map.md` | done |
| 0 | STEER small-model reproduction | **not run — needs a GPU** |
| 1 | `steer_f/mtp_heads.py`, warm-up + validation scripts | code done, unrun |
| 2 | `Ω̃` integration, verl patches, λ=0 equivalence test | code done and tested; RL unrun |
| 3 | `scripts/phase3_port_model.py` | done |
| 4 | Ablations | knobs exposed; no results |

Everything that can be checked without a GPU is checked: **213 tests pass**,
and the patches apply, revert, and reproduce upstream exactly at `λ = 0`.
No training has been run, so there are no experimental results yet, and no
gate (G0–G3) has been cleared. `docs/experiment_log.md` records the state and
the exact commands to continue.

## 실험 진행 순서

각 Phase 끝에는 **게이트**가 있다. 게이트가 실패하면 다음 Phase로 넘어가지 말고
해당 Phase의 "실패 시" 절차를 따른 뒤 보고할 것. 각 단계는 앞 단계의 **산출물**에
의존하므로 순서를 건너뛸 수 없다.

```
  Step 0  셋업·자체검증        (GPU 불필요)
     │
     ▼
  Phase 0  GRPO / STEER 재현 ─────────────► G0 ─┐  산출물: rollout JSONL, 엔트로피 곡선
     │                                          │
     ▼                                          │
  Phase 1  헤드 워밍업 → MC 검증 ─────────► G1 ─┤  산출물: mtp_heads.pt, κ, γ_H, 캘리브레이션
     │                                          │
     ▼                                          │
  Phase 2  λ 스윕 (소형) → 7B ────────────► G2 ─┤  산출물: 최적 λ, pass@1/@16
     │                                          │
     ▼                                          │
  Phase 3  Llama / EXAONE 전이 ───────────► G3 ─┤
     │                                          │
     ▼                                          │
  Phase 4  Ablation A1–A7 ────────────────────► 최종 표
```

---

### Step 0 — 셋업과 자체 검증 (GPU 불필요)

```bash
pip install -r requirements.txt
bash scripts/setup_steer.sh          # STEER를 핀 커밋으로 클론 + 패치 적용
pip install -e third_party/STEER     # verl 설치
python -m pytest tests/ -q           # 213 passed 여야 함
```

`213 passed`가 나오지 않으면 여기서 멈출 것. 특히
`test_lambda_zero_equiv.py`가 깨지면 λ=0 팔이 더 이상 stock STEER가 아니라는
뜻이고, 그 뒤의 모든 비교가 무의미해진다.

패치를 되돌리려면 `bash scripts/setup_steer.sh --revert`.

---

### Phase 0 — STEER baseline 재현 → **게이트 G0**

```bash
ARM=grpo  ./run/run_steerf_small.sh     # GRPO baseline
ARM=steer ./run/run_steerf_small.sh     # STEER (λ=0, stock과 비트 동일)
```

**확인할 것**
- 두 런 모두 에러 없이 완주 (계획서 기준 100~150 스텝)
- `actor/entropy` 곡선: GRPO는 **하락**, STEER는 **유지** — 논문 Figure와 정성적 일치
- `steerf/tw_frac_{low,mid,high}`가 한쪽에 90% 이상 쏠리지 않을 것
  (쏠리면 `TOKEN_WEIGHT_MIN/MAX` 재설정 후 재시작)

> `ARM=steer`가 원본과 비트 단위로 동일함은 테스트로 보장되므로,
> G0 재현을 위해 원본 STEER 레포를 따로 돌릴 필요는 없다.

**G0 통과 조건**: 위 엔트로피 곡선 대조가 성립 + `docs/steer_code_map.md` 작성 완료(✅ 이미 완료).

**산출물** → Phase 1 입력: `rollout_data/STEER-F-small/<run>/` 의 rollout 텍스트.
`phase1_warmup_heads.py`는 한 줄에 하나의 JSON, `text` 필드(또는 `prompt`+`response`)를
가진 JSONL을 기대한다. verl의 덤프 형식이 다르면 이 형태로 변환할 것.

---

### Phase 1 — MTP 헤드와 예보 검증 → **게이트 G1**

계획서가 "가장 싼 실패 지점"이라 부른 단계다. **절대 생략하지 말 것.**

```bash
# 1) 헤드 워밍업 (정책 freeze). K=8로 학습하고 κ는 사후 선택한다.
python scripts/phase1_warmup_heads.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --rollouts rollout_data/STEER-F-small/<run>/rollouts.jsonl \
    --out checkpoints/mtp_heads.pt \
    --num-heads 8 --head-hidden 1024 --epochs 1

# 2) Monte-Carlo 검증 + (κ, γ_H) 그리드 + 캘리브레이션
python scripts/phase1_validate.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --heads checkpoints/mtp_heads.pt \
    --problems third_party/STEER/datasets/math500.parquet \
    --n-problems 25 --n-trajectories 32 --n-continuations 16 \
    --max-prefixes 600 --calibrate \
    --out docs/phase1_results.json

# 3) 캘리브레이션을 학습 스크립트가 읽는 위치로 추출
python -c "import json;d=json.load(open('docs/phase1_results.json'));\
json.dump(d['calibration'],open('checkpoints/mtp_calibration.json','w'),indent=2)"
```

**중간 확인**: 워밍업의 헤드별 최종 CE가 k에 대해 **단조 증가**해야 정상이다.
그렇지 않으면 수렴하지 않은 것이니 검증으로 넘어가지 말 것.

**G1 통과 조건** (판정은 `steer_f.validation.evaluate_gate_g1`이 내린다)
1. 문제-내 Spearman ρ(H_togo, GT_future_entropy) **≥ 0.2**
2. 분기 recall@10%가 랜덤 대비 **p < 0.05**

> recall 검정의 귀무 선택률은 명목 0.1이 아니라 **실제 상위 10분위 크기**를 쓴다.
> `n_branch < 30`이면 검정력 부족이므로, 이때의 FAIL은 "미래 항이 노이즈"가 아니라
> "표본 부족"이다 — prefix를 더 모아 재판정할 것.

**실패 시** (계획서 §3, 순서대로 **1회씩만**):
1. 헤드 워밍업 데이터 증량 (`--max-sequences` 상향)
2. `--head-hidden` 증가
3. 순차형(DeepSeek-V3식) MTP

그래도 실패하면 **중단하고 보고.** 미래 항이 노이즈라는 결론 자체가 유의미한 부정 결과다.
결과는 실패 포함 전부 `docs/phase1_report.md`에 기록한다.

**산출물** → Phase 2 입력: `checkpoints/mtp_heads.pt`, `checkpoints/mtp_calibration.json`, κ, γ_H.

---

### Phase 2 — Ω̃ 통합과 RL 본실험 → **게이트 G2**

```bash
export STEERF_HEADS=checkpoints/mtp_heads.pt
export STEERF_CALIB=checkpoints/mtp_calibration.json
export STEERF_KAPPA=<Phase 1 결정값>  STEERF_GAMMA_H=<Phase 1 결정값>

# E2-1: 소형 모델 λ 스윕. λ=0이 기준선(= stock STEER).
for lam in 0 0.25 0.5 1.0; do
  ARM=steerf STEERF_LAM=$lam ./run/run_steerf_small.sh
done

# E2-2: 최적 λ로 7B 스케일 확인
ARM=steerf STEERF_LAM=<최적> MODEL_PATH=Qwen/Qwen2.5-Math-7B ./run/run_steerf.sh
```

**반드시 고정할 것**: `ppo_micro_batch_size_per_gpu`. STEER의 가중치 임계값은
**마이크로배치별로** 계산되므로(`docs/steer_code_map.md` §8.2), 이 값이 다르면
동일 하이퍼파라미터라도 다른 실험이 된다. 소형 vs 7B 비교에서 특히 주의.

**모니터링할 로그**
| 키 | 의미 |
|---|---|
| `actor/entropy` | 전체 정책 엔트로피 |
| `steerf/branch_entropy` / `branch_entropy_gap` | **분기 토큰만의 엔트로피** — 메커니즘 확인의 핵심 |
| `steerf/tw_frac_{low,mid,high}` | 가중치 분포 (한쪽 쏠림 감시) |
| `steerf/visit_rel_mag` | 미래 항 / 로컬 항 상대 크기 |
| `steerf/a_h_clip_frac` | A_H 클립 포화율 (높으면 `STEERF_CLIP_C` 완화) |
| `steerf/branch_recall` / `branch_lift` | 학습 중 A_H 유효성 |

**G2 통과 조건**
1. λ=0 대비 최적 λ에서 평균 **pass@16 +1.0pp 이상**, pass@1 비열화(−0.5pp 이내)
2. 분기 토큰 엔트로피 곡선이 λ=0 대비 높게 유지 (메커니즘 확인)

**실패 시** 진단 순서: 가중치 분포 재설정 → baseline 교체(`STEERF_BASELINE=group`) → κ 재조정.

> **현재 상태 주의**: MTP 보조 손실(`β_mtp`)이 아직 학습 루프에 연결되지 않아
> 헤드는 freeze 상태로 돈다. 즉 지금의 Phase 2는 사실상 **Ablation A7 조건**이다.
> 먼저 이 조건으로 G2를 돌리고, 표류(KL) 로그를 보고 헤드 공동학습의 필요성을 판단할 것.
> 상세는 `docs/experiment_log.md`의 "Phase 2 미해결 리스크".

---

### Phase 3 — 모델 패밀리 전이 → **게이트 G3**

하이퍼파라미터(λ, κ, γ_H, norm)는 **Qwen 값 그대로** 쓴다 — "그대로 작동"이 주장 포인트다.
모델별로 다시 구하는 것은 **캘리브레이션뿐**.

```bash
# 1) 이식 가능성 점검 + 빈 헤드 생성 + 훈련셋 pass rate 분포 확인
python scripts/phase3_port_model.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --out checkpoints/mtp_heads_llama3b_init.pt \
    --reference-heads checkpoints/mtp_heads.pt \
    --rollout-parquet rollout_data/.../rollouts.parquet

# 2) 헤드 워밍업 (Phase 1 스텝 1과 동일)
# 3) 축소 검증: 문제 10개 / prefix 200개 — 캘리브레이션만 재산출
python scripts/phase1_validate.py --model meta-llama/Llama-3.2-3B-Instruct \
    --heads checkpoints/mtp_heads_llama3b.pt --problems <...> \
    --n-problems 10 --max-prefixes 200 --calibrate --out docs/phase3_llama.json

# 4) 본학습 3종
for arm in grpo steer steerf; do ARM=$arm MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct \
    ./run/run_steerf.sh; done
```

**Llama 주의**: 소형 Llama는 수학 baseline이 약해 전부-오답 그룹이 과다할 수 있다.
`phase3_port_model.py`가 `frac_all_wrong > 0.5`면 `TOO HARD` 판정을 낸다.
그 경우 난이도 하위 서브셋으로 조정하고 **조정 사실을 보고서에 명시**할 것.

**EXAONE**: 한국어 수학/추론 벤치 1종(HRM8K 등)을 평가에 추가.

**G3 통과 조건**: 두 패밀리 모두에서 STEER-F ≥ STEER 경향 재현 (효과 크기는 작아도 방향 일치).

---

### Phase 4 — Ablation

| # | 변인 | 실행 방법 |
|---|---|---|
| A1 | κ ∈ {2,4,8} | `STEERF_KAPPA=` |
| A2 | γ_H ∈ {0.7,0.85,1.0} | `STEERF_GAMMA_H=` |
| A3 | λ ∈ {0,0.1,0.25,0.5,1.0,2.0} | `STEERF_LAM=` (큰 값에서 붕괴 예상) |
| A4 | 로컬 항 제거 | `STEERF_NORM=` + Ω 항 무효화 (미구현 — 코드 수정 필요) |
| A5 | baseline 방식 | `STEERF_BASELINE=sibling` vs `group` |
| A6 | 캘리브레이션 on/off | `STEERF_CALIB=` 제거 |
| A7 | 헤드 freeze | **현재 기본 동작** (β_mtp 미연결) |

---

### 재현성 체크리스트 (계획서 §10)

- [ ] 시드 3개 이상으로 각 팔 반복
- [ ] `ppo_micro_batch_size_per_gpu` 전 팔 동일
- [ ] 모든 하이퍼파라미터를 환경변수가 아닌 파일로 고정해 커밋
- [ ] 실패·부정 결과 포함 전부 `docs/experiment_log.md`에 기록

## Layout

```
steer_f/
  mtp_heads.py         parallel MTP heads; forecast_entropy never builds [K,B,T,V]
  entropy_forecast.py  H_togo, per-head calibration, sibling / group baselines
  omega_tilde.py       Ω̃ and the drop-in replacement for compute_token_weights
  monitors.py          forecast-drift KL, branch entropy, λ decay controller
  validation.py        gate-G1 statistics (within-problem ρ, exact binomial)
  verl_integration.py  index alignment and batch-level A_H assembly
patches/               the only channel by which STEER-F edits verl
scripts/               setup, head warm-up, MC validation, family porting
run/                   training arms: grpo / steer / steerf
docs/                  code map, phase reports, running log
tests/                 213 tests, incl. a verbatim upstream copy for equivalence
```

## Two things worth knowing before reading the code

**The paper's band does not exist in the code.** STEER's implementation has no
`[ΔH_low, ΔH_high]` band and no discrete `α ∈ {γ, 1, 1/γ}`; it min-max rescales
`Ω` onto `[0.8, 1.2]` per micro-batch. Every design choice downstream follows
from the real mapping. `docs/steer_code_map.md` §3.

**Normalisation is RMS, not z-score.** Recentring changes STEER's mapping —
`symmetric` mode takes `abs()`, so zero is a meaningful pivot, and the
exponential mapping is not shift-invariant. Scaling without recentring is the
only normalisation that leaves all four `(mode, mapping)` combinations
untouched as `λ → 0`. `docs/steer_code_map.md` §3.3.

## Licence

Apache-2.0, matching STEER and verl. `tests/reference_steer.py` is an
unmodified excerpt of STEER, retained for the equivalence test.
