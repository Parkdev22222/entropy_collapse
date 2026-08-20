# STEER-F on the official STEER release

이 브랜치는 [zz-haooo/STEER](https://github.com/zz-haooo/STEER) (커밋 `08add1cc`,
논문 *Rethinking Entropy Interventions in RLVR*, arXiv:2510.10150 / ACL 2026)를
**그대로 베이스**로 하고, 그 위에 STEER-F(미래 엔트로피 예보 항)를 직접 통합한
것이다. 첫 커밋이 upstream verbatim import이므로,

```bash
git diff $(git rev-list --max-parents=0 HEAD) -- verl/
```

가 우리 방법이 verl에 가한 변경의 전부다. 방법론 자체의 수식과 주석은
[`docs/STEERF_method.md`](docs/STEERF_method.md).

```
Ω̃      = norm(Ω_STEER) + λ · norm( Δlogπ̂ · clip(A_H, ±c) )
A_H     = H_togo(s_t ⊕ y_t) − H̄_togo(s_t)          # 형제 베이스라인
H_togo  = Σ_{k=1..κ} γ_H^k · H( p_MTP(y_{t+k} | s) )  # MTP 헤드 예보
```

`steerf_lam = 0`은 stock STEER와 **비트 단위로 동일**하다
(`tests/test_lambda_zero_equiv.py`, upstream verbatim 사본과 대조).

## Setup

```bash
pip install -e .                     # verl 포크 + steer_f (base 의존성만!)
pip install "vllm==0.8.5.post1"      # 롤아웃 백엔드 — base에 포함되지 않는 [vllm] extra.
                                     # 빠뜨리면 모든 스테이지가 ~50초 만에
                                     # "ModuleNotFoundError: msgspec" 으로 죽는다.
pip install math-verify word2number tensorboard==2.18.0
# flash-attn: verl의 dp_actor가 CUDA에서 무조건 import하므로 **필수**다.
# 반드시 torch 버전 + CUDA + python + cxx11-abi가 전부 맞는 휠을 써야 한다
# (안 맞으면 "undefined symbol: _ZN3c105Error..."로 워커가 죽는다).
# 내 torch 빌드에 맞는 정확한 휠 이름은 scripts/preflight.py가 찍어 준다:
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.compiled_with_cxx11_abi())"
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
python -m pytest tests/ -q
python scripts/preflight.py          # 큐를 걸기 전 환경 검증 (run_all이 자동 수행)
```

**검증된 버전 조합** (이 조합에서 실제 학습이 돌았다 — 다른 조합은 자기 책임):

| 패키지 | 버전 | 어긋나면 생기는 일 |
|---|---|---|
| torch | 2.6.0+cu124 | vllm 0.8.5가 요구 |
| vllm | 0.8.5.post1 | 상위 버전은 verl 0.4.1 rollout 어댑터와 불일치 |
| **ray[default]** | **2.46.0** | ≥2.47은 대시보드가 opentelemetry-prometheus를 import — vllm이 핀한 구버전 semconv와 충돌해 **raylet 등록 타임아웃**으로 죽는다 (`pip install "ray[default]==2.46.0" && ray stop --force`) |
| flash-attn | 2.7.4.post1 | **휠의 cxx11abi가 torch와 일치해야 한다.** PyPI의 `pip install flash-attn`은 ABI가 어긋난 휠을 집어올 수 있다 — Dao-AILab 릴리스에서 `cu12torch2.6cxx11abiFALSE-cp312` 처럼 4요소가 맞는 휠을 직접 지정할 것 |
| transformers | 4.51.3 | |
| tensorboard | 2.18.0 | 상위 버전은 protobuf 충돌 이력 |

`pip install -e .`가 필수인 이유: Ray 워커는 `steer_f`를 절대 경로로 import하며,
재사용된 raylet에는 부모 셸의 PYTHONPATH가 닿지 않는다 (실측된 사고).
`run/run_all_experiments.sh`는 시작 시 `scripts/preflight.py`로 위 전부를
검사하고, 하나라도 빠지면 **아무것도 큐에 넣지 않고** 수정 방법과 함께 멈춘다.

## 실험 프로토콜 — 논문과의 패리티

원칙: **STEER-F 팔만 새로 돌리고, 비교군 숫자는 논문 표의 값을 그대로 쓴다.**
세팅의 1차 근거는 논문 v4 (arXiv:2510.10150v4) §6.1 + Appendix E이며, 릴리스
스크립트와 다른 지점은 논문을 따른다 (아래 λ_min 주 참조).

| 항목 | 값 (논문 v4) |
|---|---|
| 학습 데이터 (수학) | DAPO-Math-17k, 17,398 프롬프트 (레포 동봉) |
| 학습 데이터 (코드 생성) | ArcherCodeR 6,753 태스크 — `scripts/prepare_code_data.py --archer` |
| 레시피 | GRPO, `train_batch=512`, `mini=32`, `micro/GPU=8`, rollout `n=8`, lr 1e-6, warmup 없음 |
| 롤아웃 샘플링 | temperature 1.0, top-p 1.0 |
| 길이 | prompt 1024 / response 3072 (수학·코드 생성 공통) |
| 정규화 | KL 없음(loss·reward), entropy loss 없음, token-level loss norm, dynamic sampling 없음 |
| STEER | symmetric + 지수 매핑, **λ_min = 0.7** |
| 스케줄 | **최대 200 rollout steps**, 체크포인트 10스텝마다, **AIME24 최고 체크포인트 선택** |
| 반복 | 메인 표는 **독립 2런 평균** (`SEED=1`, `SEED=2`) |
| 평가 샘플링 | temperature 1.0, top_p 0.7, max 3072, zero-shot, Math-Verify + Qwen-Verify |
| 평가 지표 (수학) | **avg@32**: AIME24/AIME25/AMC23 · **avg@1**: MATH500/Minerva/OlympiadBench |
| 평가 지표 (코드) | **avg@4**: LiveCodeBench v5 (279문제) |
| 하드웨어 | 8×H20 (TP=4) |

> **λ_min 주의**: 릴리스 `run_exp.sh`는 `token_weight_min=0.8`이지만 논문 v4
> §6.1은 "λ_min is set to **0.7**"이고 민감도 분석(Fig 18–19)에서도 0.7이
> 최적이다. 우리 스크립트 기본값은 **0.7**이며, 릴리스 재현이 필요하면
> `TOKEN_WEIGHT_MIN=0.8`로 오버라이드한다.

### 원샷 실행 — 전체 스위트

```bash
pip install -e . && python -m pytest tests/ -q     # 1회
bash run/run_all_experiments.sh                     # 전부 (재개 가능)
```

논문 표 전부(Table 3/4/5-LCB/6/12, Fig 6/20a, RLOO/OPO 팔)를 순차 큐로 돌린다.
스테이지별 완료 마커(`experiments_state/`)가 있어 **중단 후 재실행하면 끝난
스테이지는 건너뛴다**. 실패한 스테이지는 기록하고 큐는 계속 진행하며, 종료 시
요약과 함께 `results/summary.tsv`(표에 넣을 숫자, 시드별 + 평균)를 쓴다.

부분 실행 토글: `RUN_MATH/RUN_CODE/RUN_EXTREME/RUN_RL_ALGOS`(기본 1),
`RUN_PASSK`(기본 0 — 고비용), `MATH_MODELS/CODE_MODELS/SEEDS`.
예: 7B 메인만 → `MATH_MODELS=Qwen/Qwen2.5-Math-7B RUN_CODE=0 RUN_EXTREME=0 RUN_RL_ALGOS=0 bash run/run_all_experiments.sh`

체크포인트 선택(논문 규칙: AIME24 최고)은 `scripts/select_best_checkpoint.py`가
tensorboard 이벤트에서 자동으로 수행하고, 결과 수집은
`scripts/collect_results.py`가 eval 로그에서 자동으로 뽑는다.

### 실행 순서 (수동, 모델당)

```bash
# 0) 헤드 워밍업용 base-policy 롤아웃 (~32k 시퀀스, 짧은 GRPO 런의 덤프)
MODEL_PATH=Qwen/Qwen2.5-Math-7B bash run/collect_warmup_rollouts.sh

# 1) MTP 헤드 워밍업 + MC 검증 + 캘리브레이션 + (κ, γ_H) 선택  →  게이트 G1
MODEL_PATH=Qwen/Qwen2.5-Math-7B bash run/warmup_and_validate.sh
#    출력의 STEERF_KAPPA / STEERF_GAMMA_H 를 다음 단계에 전달할 것.
#    recall 판정은 반드시 _control(미학습 헤드) 파일과 비교해서 읽는다.

# 2) STEER-F 본학습 — 논문 세팅 그대로, 독립 2런
for SEED in 1 2; do
  MODEL_PATH=Qwen/Qwen2.5-Math-7B SEED=$SEED \
  STEERF_LAM=0.25 STEERF_KAPPA=<κ> STEERF_GAMMA_H=<γ> bash run/run_steerf.sh
done

# 3) 체크포인트 선택: 학습 로그의 val-core/aime_2024_dapo_boxed/acc/mean@32 가
#    최고인 스텝(10의 배수)의 hf_model — 논문의 선택 규칙 그대로.

# 4) 논문 지표 평가, 두 런 각각 → 표에는 평균 기입
MODEL_PATH=<선택된 체크포인트>/hf_model bash run/eval_steerf.sh
```

### 논문 표/그림 ↔ 이 레포의 매핑

**우리 행(STEER-F)을 추가할 수 있는 표** — 전부 위 파이프라인으로 채워진다:

| 논문 표/그림 | 세팅 | 우리 런 | 표에 넣을 숫자 |
|---|---|---|---|
| **Table 3** — Qwen2.5-Math-7B, 12행 (GRPO/SimpleRL-Zoo/Eurus-PRIME/OPO/clip-high/Entro.Loss/Fork Tokens/W-REINFORCE/Entro.Adv./Clip-Cov/KL-Cov/STEER) | 본문 메인 | `MODEL_PATH=Qwen/Qwen2.5-Math-7B run/run_steerf.sh` ×2런 → `eval_steerf.sh` | `val-core/<bench>/acc/mean@{32,1}` 2런 평균 |
| **Table 12** (App. F.3) — Qwen2.5-Math-1.5B, 6행 (Base/GRPO/OPO/Entro.Adv./Clip-Cov/STEER) | 동일 | `MODEL_PATH=Qwen/Qwen2.5-Math-1.5B ...` | 동일 |
| **Table 4** — Qwen2.5-14B, 6행 (Base/GRPO/OPO/Entro.Adv./Clip-Cov/STEER) | 동일 | `MODEL_PATH=Qwen/Qwen2.5-14B ...` | 동일 |
| **Table 5** — 코드, LCB-v5 행 (GRPO vs STEER × Coder-3B/7B/14B) | ArcherCodeR 학습 | `scripts/prepare_code_data.py --archer --lcb` → `MODEL_PATH=Qwen/Qwen2.5-Coder-{3B,7B,14B} run/run_steerf_code.sh` → `CODE=1 eval_steerf.sh` | `val-core/codecontests/acc/mean@4` |
| **Table 6** — 극한 시나리오 (ε_low=0.99, ε_high=5), 5행 (GRPO/Entro.Adv./Entro.Loss/Clip-Cov/STEER) | 7B | `run/run_steerf_extreme.sh` → `eval_steerf.sh` | 동일 6개 벤치 |
| **Figure 6** — Pass@256/512/1024, AIME24/25 | 7B 선택 체크포인트 | `PASSK=1 MODEL_PATH=... eval_steerf.sh` | `val-core/<aime>/acc/best@{256,512,1024}/mean` |
| **Figure 7** — 학습 중 test acc 곡선 | 7B | 학습 로그의 `val-core/aime_2024_dapo_boxed/acc/mean@32` per step | 곡선 |
| **Figure 8** — 극한 시나리오 엔트로피 곡선 | 7B | `run_steerf_extreme.sh` 로그의 `actor/entropy` | 곡선 |
| **App. F.3 Fig 20a** — Llama-3.2-3B 수학 (GRPO vs STEER) | 동일 레시피 | `MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct run/run_steerf.sh` | 6개 벤치 + Avg |
| **App. F.3** — RL 알고리즘 일반화 (RLOO 45.8→46.8, OPO 46.4→47.5) | 7B | `run/run_steerf.sh algorithm.adv_estimator=rloo` (또는 `opo`) — 후행 hydra 인자가 앞의 값을 이긴다 | 6개 벤치 평균 |

**추가 불가/비대상 표**:

| 표/그림 | 이유 |
|---|---|
| Table 1, 2, 7, Fig 1–5, 9–17 | Ω 추정 정확도·4분면 분석 등 **분석용** — 방법 성능 표가 아님. STEER-F의 대응물은 `docs/STEERF_method.md`의 검증 절 + `steerf/*` 로그 지표 |
| Table 5의 Internal/Zeta 행 (코드 편집) | 학습 코퍼스가 **내부 데이터 51,474건** — 공개 재현 불가. Mistral-7B 일반화(Fig 20b)도 동일 사유 |
| Table 10 (ε 클립 스윕), Table 11 (매핑 exp/linear/binary), Fig 18–19 (λ_min) | **STEER 자체의 ablation** — 우리 행을 추가하는 표가 아니다. STEER-F의 대응 ablation은 `STEERF_*` 노브로 동일하게 수행 (λ 스윕, `STEERF_MAPPING`, `STEERF_APPLY`, oracle 팔) |

비교군 숫자는 재실험하지 않고 논문 값을 그대로 옮긴다. 훈련·평가 스크립트가
저자 프로토콜과 동일함이 그 정당화이며, 유일하게 저자 릴리스와 다른 두 지점
(λ_min 0.7, 학습 중 val을 AIME24 avg@32로 측정해 선택 규칙을 논문 문면대로
구현)은 모두 논문 쪽을 따른 것이다.

## STEER-F 노브 (전부 env로 노출, 기본값 = 권장값)

| env | 기본 | 의미 |
|---|---|---|
| `STEERF_LAM` | 0.25 | 방문 항 가중. **0 = stock STEER (비트 동일)** |
| `STEERF_KAPPA` / `STEERF_GAMMA_H` | 2 / 0.7 | 예보 호라이즌·할인 — **Phase 1 출력값을 쓸 것** |
| `STEERF_APPLY` | weight | `metric`(Ω̃로 합산) / `weight`(매핑 후 tanh 보정) / `branch`(예보 없는 귀무 팔) |
| `STEERF_MAPPING` | minmax | `winsor`/`rank`: Ω 꼬리가 min-max를 독점해 가중치가 상수로 붕괴하는 병리의 처방. **minmax가 아니면 λ=0도 stock이 아님** (제2 기준선) |
| `STEERF_FORECAST` | mtp | `oracle`: 실현된 하류 엔트로피로 A_H를 만드는 **공짜 대조군** — MTP가 이걸 못 이기면 헤드는 무가치 |
| `STEERF_BASELINE` | sibling | `group`: 프리픽스 무관 그룹 평균 (ablation) |
| `STEERF_CLIP_C` / `STEERF_NORM` | 1.0 / scale | A_H 클립, 정규화(RMS) |

`STEERF_APPLY=weight`가 기본인 근거와 `mapping` 옵션의 동기(가중치 압축 실측)는
`docs/STEERF_method.md` §12–§15.

## 반드시 지킬 것

1. **`pip install -e .` 후 실행** — PYTHONPATH만으로 돌리면 재사용된 Ray
   워커에서 `ModuleNotFoundError: steer_f`가 난다 (실측된 사고).
2. `ppo_micro_batch_size_per_gpu=8` 고정 — STEER의 min-max가 마이크로배치
   단위라 이 값이 방법의 일부다.
3. λ>0 팔을 보고하기 전에 **oracle 팔과 λ=0 팔을 같은 시드로** 함께 돌린다.
4. **독립 2런 평균** — 논문 메인 표의 규약(§6.1). avg@1 500문제의 이항 SE가
   ≈2.2pp라 1런 차이는 대부분 잡음이며, 2런 평균으로도 빠듯하니 시드별 원값을
   함께 보고할 것.

## 감시 지표

| 키 | 정상 범위 밖이면 |
|---|---|
| `steerf/tw_std`, `steerf/tw_frac_*` | 가중치가 상수로 붕괴(std ≪ 밴드×1%) → `STEERF_MAPPING=rank` |
| `steerf/visit_rel_mag` | 미래 항이 로컬 항 대비 ≪1% → λ 스윕이 무의미 |
| `steerf/a_h_clip_frac` | 포화(>0.5) → `STEERF_CLIP_C` 완화 |
| `steerf/branch_recall` vs 미학습 대조 | 차이 없음 → 예보가 기여 없음 |
| `actor/entropy` | 붕괴 판정은 GRPO·λ=0 곡선과의 대조로만 |

## Licence

Apache-2.0 (upstream STEER/verl과 동일). `tests/reference_steer.py`는 동치성
테스트용 upstream 발췌 사본이다.
