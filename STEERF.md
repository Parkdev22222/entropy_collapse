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
pip install -e .          # verl 포크 + steer_f 를 함께 설치
                          # (Ray 워커가 steer_f 를 import 하므로 PYTHONPATH 의존 금지)
python -m pytest tests/ -q
```

## 실험 프로토콜 — 논문과의 패리티

원칙: **STEER-F 팔만 새로 돌리고, 비교군 숫자는 논문 표의 값을 그대로 쓴다.**
그러려면 학습·평가 세팅이 논문과 동일해야 하며, 아래 스크립트의 모든
하이퍼파라미터는 저자 릴리스 `run/run_exp.sh`·`run/eval.sh`에서 **STEER-F 노브를
제외하고 한 글자도 다르지 않다**:

| 항목 | 값 (저자 릴리스 = 논문) |
|---|---|
| 학습 데이터 | DAPO-Math-17k (레포 동봉) |
| 레시피 | GRPO, `train_batch=512`, `mini=32`, `micro/GPU=8`, rollout `n=8` |
| 길이 | prompt 1024 / response 3072 |
| 최적화 | lr 1e-6, KL 없음(loss·reward 모두), entropy_coeff 0 |
| 클리핑 | 0.2 / 0.28, dual-clip c=10 |
| STEER 매핑 | **symmetric + 지수 매핑 + token weights [0.8, 1.0]** |
| 스케줄 | 10 epochs (~340 steps), 8×H20, TP=4 |
| 평가 샘플링 | temperature 1.0, top_p 0.7, max 3072 tokens |
| 평가 지표 | **avg@32**: AIME24 / AIME25 / AMC23 · **avg@1**: MATH500 / Minerva / OlympiadBench / GSM8K |

### 실행 순서 (모델당)

```bash
# 0) 헤드 워밍업용 base-policy 롤아웃 (~32k 시퀀스, 짧은 GRPO 런의 덤프)
MODEL_PATH=Qwen/Qwen2.5-Math-7B bash run/collect_warmup_rollouts.sh

# 1) MTP 헤드 워밍업 + MC 검증 + 캘리브레이션 + (κ, γ_H) 선택  →  게이트 G1
MODEL_PATH=Qwen/Qwen2.5-Math-7B bash run/warmup_and_validate.sh
#    출력의 STEERF_KAPPA / STEERF_GAMMA_H 를 다음 단계에 전달할 것.
#    recall 판정은 반드시 _control(미학습 헤드) 파일과 비교해서 읽는다.

# 2) STEER-F 본학습 (논문 세팅 그대로, STEER-F 노브만 추가)
MODEL_PATH=Qwen/Qwen2.5-Math-7B STEERF_LAM=0.25 STEERF_KAPPA=<κ> STEERF_GAMMA_H=<γ> \
    bash run/run_steerf.sh

# 3) 논문 지표 평가 (avg@32 3종 + avg@1 4종, 두 패스)
MODEL_PATH=<체크포인트>/hf_model bash run/eval_steerf.sh
```

### 논문 표 ↔ 이 레포의 매핑

| 논문 표 | 세팅 | 우리 런 | 표에 넣을 숫자 |
|---|---|---|---|
| Table 1 (Qwen2.5-Math-1.5B, 비교군 10종) | 본문 메인 | `MODEL_PATH=Qwen/Qwen2.5-Math-1.5B run/run_steerf.sh` → `eval_steerf.sh` | `val-core/<bench>/acc/mean@{32,1}` 을 STEER-F 행으로 추가 |
| Table 3 (Qwen2.5-Math-7B) | 본문 메인 | `MODEL_PATH=Qwen/Qwen2.5-Math-7B ...` 동일 | 동일 |
| 14B 표 (Qwen2.5-14B) | 본문 메인 | `MODEL_PATH=Qwen/Qwen2.5-14B ...` 동일 | 동일 |
| 극한 시나리오 (clip 0.99/5) | 엔트로피 제어 스트레스 | `run/run_steerf_extreme.sh` | 엔트로피 곡선 (`actor/entropy`) |
| λ_min 스윕 (STEER의 유일 하이퍼) | ablation | `token_weight_min` 오버라이드는 스크립트 끝에 hydra 인자로 전달: `bash run/run_steerf.sh +actor_rollout_ref.actor.policy_loss.token_weight_min=0.9` — 오버라이드가 나중에 오므로 앞의 값을 이긴다 | 동일 |
| 코딩 벤치 (LiveCodeBench v5, avg@4) | ACL판 추가 실험 | **공식 릴리스에 코드 트랙의 데이터·스크립트·리워드가 없음.** 아래 "코드 트랙" 참조 | — |

비교군(GRPO, SimpleRL-Zoo, Eurus-PRIME, OPO, clip-high, entropy-loss,
Fork Tokens, W-REINFORCE, Entropy Adversarial, Clip-Cov, KL-Cov, STEER)은
**재실험하지 않는다** — 논문 표의 값을 그대로 옮긴다. 같은 세팅에서 돌았음이
스크립트 동일성으로 보장되는 것이 이 브랜치의 요점이다.

### 코드 트랙에 관하여

논문 ACL판은 LiveCodeBench v5 (avg@4)를 보고하지만, 공식 릴리스에는 코드 학습
데이터·리워드·스크립트가 포함되어 있지 않다. 저자 세팅을 추정으로 재구성하면
"동일 세팅" 주장이 깨지므로 이 브랜치는 코드 트랙을 **의도적으로 비워 둔다**.
논문 부록에서 코드 트랙의 데이터셋·리워드 구성을 확인해 오면
`run/run_steerf.sh`는 train_files/val_files 교체만으로 재사용 가능하다.

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
4. 시드 3개 이상 — avg@1 500문제의 이항 SE ≈ 2.2pp로, 1시드 차이는 대부분 잡음.

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
