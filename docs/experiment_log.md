# 실험 러닝 로그

> 계획서 §10: **실패·부정 결과 포함 전부 기록.** 시간 역순이 아니라 정순으로 적는다.

---

## 2026-08-15 — 구현 세션 (GPU 없음)

### 환경
| 항목 | 값 |
|---|---|
| GPU | **없음** (`nvidia-smi` 미설치) |
| CPU / RAM | 4 코어 / 15 GB |
| torch | 2.13.0 (CPU 실행) |
| STEER 핀 | `08add1cc27f4d32a78a6c0d6cb857aa52b8f2a55`, verl `0.4.1.dev` |

**결론: 이 세션에서 학습 실행은 불가능.** 따라서 GPU 없이 검증 가능한 전부
(코드 분석·라이브러리·패치·테스트)를 완료하고, 학습은 미실행으로 남긴다.

### 완료

- `docs/steer_code_map.md` — Phase 0 게이트 산출물. 원본 코드 정독 결과.
- `steer_f/` 6개 모듈, `scripts/` 4개, `patches/` 3개, `run/` 2개.
- 테스트 **213개 전부 통과** (`python -m pytest tests/ -q`).
- 패치 적용 → 되돌리기 → 재적용 왕복 검증 완료.

### 발견 1 — 계획서 §0.1의 STEER 서술이 코드와 다르다 (설계 영향 큼)

계획서는 밴드 `[ΔH_low, ΔH_high]`와 이산 α `{γ, 1, 1/γ}`를 전제하지만,
**실제 구현에는 밴드도 γ도 이산 α도 없다.** `compute_token_weights`는 Ω를
마이크로배치 min-max로 `[token_weight_min, token_weight_max]`에 연속 선형 매핑
(또는 지수 매핑)한다. 상세는 `docs/steer_code_map.md` §3.

영향:
- 계획서 §4.1의 "밴드 분위수 재산정" 절차는 **적용 대상이 없다.** 대응물로
  `token_weight_distribution`(가중치 범위 3등분 점유율)을 로깅한다.
- 계획서 §0.2의 "α의 이산 3단 구조 유지"라는 방어 논리는 성립하지 않는다.
  유계성은 대신 `clip(A_H, ±c)` + `[w_min, w_max]` clamp가 담당한다.

### 발견 2 — 정규화는 z-score가 아니라 RMS여야 한다

계획서 §4.1은 `z_norm`을 쓰라고 하지만, STEER 매핑의 불변성을 조사한 결과
**중심이동(centering)은 λ=0 동치성을 깨뜨린다:**

| 매핑 | 불변인 변환 |
|---|---|
| `linear=True` | 아핀 `a·x+b` (a>0) |
| `linear=False` | 양의 스케일 `a·x`만 (지수는 평행이동 불변 아님) |
| `mode="symmetric"` | `abs()`가 붙어 **0이 의미를 가짐 → 중심이동 불가** |

교집합은 "중심이동 없는 양의 스케일링"뿐. 기본값을 `norm="scale"`(RMS)로 하고,
`z`는 선택 가능하되 `SteerFConfig.validate()`가 위험 조합에서 경고한다.
실패 사례는 `test_z_norm_would_break_symmetric_equivalence`가 실증한다.

### 발견 3 — 원본 verl 포크의 잠재 크래시

`verl/workers/fsdp_workers.py:772`가 3-tuple을 반환하는
`compute_log_prob`에서 **4개를 언패킹**한다. STEER 스크립트는
`use_kl_loss=False`, `use_kl_in_reward=False`라 독립 reference policy가 뜨지
않아 발화하지 않지만, KL 항을 켜는 순간 `ValueError`로 죽는다.
`patches/steerf_forecast_pass.patch`가 반환값을 4개로 늘리면서 부수적으로 해소된다.
KL을 쓸 계획이 없어도, 이 사실을 모르고 켜면 원인 찾기 어려우니 기록해 둔다.

### 발견 4 — 마이크로배치 크기가 하이퍼파라미터다

Ω의 min/max가 **마이크로배치별로** 계산되므로 `ppo_micro_batch_size_per_gpu`를
바꾸면 동일 설정에서도 가중치 분포가 달라진다. Phase 2의 소형 vs 7B 비교에서
반드시 고정하거나 보고서에 명시할 것. `run/run_steerf_small.sh`에 주석으로 고정해 뒀다.

### 발견 5 — 계획서 §3.3의 GT 정의를 그대로 쓰면 γ_H가 부당하게 불리하다

`H_togo^κ`는 **할인된** 합인데 계획서의 `GT_future_entropy`는 할인 없는 합이다.
할인 예보를 무할인 실측에 맞춰 채점하면 `γ_H < 1`이 예보 품질과 무관한 이유로
전부 손해를 본다. `phase1_validate.py`는 (κ, γ_H) 셀마다 **같은 할인**을 적용한
실측치를 주 지표로 쓰고, 계획서 정의는 `gt_sum_h`로 함께 기록한다.

### 미실행 (전부 GPU 필요)

- [ ] **G0**: STEER 소형 재현 (GRPO vs STEER, 100~150 스텝). 엔트로피 곡선 대조.
- [ ] **G1**: 헤드 워밍업 → MC 검증 → (κ, γ_H, 캘리브레이션) 확정.
- [ ] **G2**: λ 스윕 및 7B 확인.
- [ ] **G3**: Llama / EXAONE 전이.
- [ ] Ablation A1–A7.

> 게이트를 하나도 통과하지 않았으므로, 현재 저장소에는 **STEER-F가 효과가 있다는
> 어떤 증거도 없다.** 있는 것은 "λ=0에서 STEER와 정확히 같고, λ>0에서 정의대로
> 동작한다"는 구현 수준의 보증뿐이다.

### Phase 2 미해결 리스크

1. **`_steerf_unembedding()`의 FSDP 경로가 미검증.** FSDP는 forward 밖에서
   파라미터를 샤딩하므로, lm_head 직접 호출이 실패하거나 샤드만 볼 수 있다.
   첫 다중 GPU 실행에서 여기서 죽으면 두 가지 처방:
   (a) 예보를 `FSDP.summon_full_params(..., writeback=False)`로 감싼다,
   (b) unembedding의 비샤딩 사본을 따로 유지한다.
   `patches/steerf_forecast_pass.patch` 헤더와 해당 docstring에 같은 내용을 적어 뒀다.
2. **예보용 추가 forward 1회**의 비용. verl의 기존 패스는 히든을 버리므로
   rmpad 경로를 수술하는 대신 no-grad forward를 한 번 더 도는 쪽을 택했다.
   backward는 없으므로 스텝당 증가는 대략 +1 forward. 실측 후 기록할 것.
3. **MTP 보조 손실(`β_mtp`)이 아직 학습 루프에 연결되지 않았다.** 헤드는 현재
   Phase 1 워밍업 상태로 고정(freeze)되어 돈다. 즉 지금 상태는 사실상
   **Ablation A7(헤드 freeze)** 이며, 계획서 §4.1의 "헤드가 정책과 함께 진화"는
   미구현이다. `mtp_ce_loss`는 준비돼 있고 `update_policy`에서 호출하면 되지만,
   그러려면 학습 forward에서 히든을 꺼내야 해서 리스크 1과 같은 문제를 만난다.
   **G2를 A7 조건에서 먼저 돌리고, 표류(KL) 로그를 보고 필요성을 판단할 것.**
4. `LambdaDriftController`도 같은 이유로 아직 루프에 연결되지 않았다
   (KL 표류 측정에 헤드-정책 동시 분포가 필요). 단위 테스트는 통과 상태.

### 다음 세션 시작 지점

```bash
bash scripts/setup_steer.sh
pip install -e third_party/STEER
python -m pytest tests/ -q                 # 213 passed 확인
ARM=grpo  ./run/run_steerf_small.sh        # G0 절반
ARM=steer ./run/run_steerf_small.sh        # G0 나머지
```

`ARM=steer`가 stock STEER와 비트 단위로 같다는 것은 테스트로 보장되므로,
G0 재현은 원본 레포를 따로 돌릴 필요 없이 이 저장소만으로 끝난다.

---

## 2026-08-15 — 실행 세션 (A100 80GB × 1)

### 환경

| 항목 | 값 |
|---|---|
| GPU | **A100-SXM4-80GB × 1** |
| CPU / RAM | 255 코어 / 1007 GB |
| torch | 2.6.0+cu124 |
| vLLM | 0.8.5.post1 |
| flash-attn | 2.7.4.post1 (torch2.6 cu12 cp312 사전빌드 휠) |
| transformers / datasets / ray | 4.51.3 / 3.5.0 / 2.46.0 |
| tensorboard | **2.18.0 고정** |
| STEER 핀 | `08add1cc27f4d32a78a6c0d6cb857aa52b8f2a55`, verl `0.4.1.dev` |
| 정책 모델 | Qwen2.5-1.5B-Instruct, **풀 파인튜닝** (`lora_rank=0`), FSDP |

Step 0 자체검증: **213 passed** (새 패치 적용 후 재실행).

### 발견 6 — `rollout_data_dir`이 조용히 아무것도 쓰지 않는다 (Phase 1 차단)

STEER 포크의 `ray_trainer.py`는 프롬프트/응답을 디코드해 놓고 `_dump_generations`
호출을 **주석 처리**해 뒀다. 따라서 `trainer.rollout_data_dir`은 명령줄에서
받아들여지지만 파일을 만들지 않는다. README가 "Phase 0 산출물 → Phase 1 입력"으로
지정한 rollout JSONL이 **존재하지 않게 되므로 Phase 1이 시작될 수 없다.**

`patches/rollout_dump_steerf.patch`로 주석만 해제. `reward_extra_infos_dict`는
같은 스코프에 이미 바인딩돼 있고 `_dump_generations`는 `_validate`가 쓰는
포크 자신의 헬퍼라 다른 수정은 필요 없다.

덤프 레코드는 `{input, output, score, step, acc, pred}` 형식이고
`phase1_warmup_heads.py`는 `text`(또는 `prompt`+`response`)를 기대하므로
`scripts/phase0_collect_rollouts.py`가 변환·병합을 담당한다 (README가 예고한
"형식이 다르면 변환할 것"이 실제로 필요했다).

### 발견 7 — `run_steerf_small.sh`가 호출자의 인자를 삼킨다

`run_steerf_small.sh`는 자기 오버라이드만 `run_steerf.sh`에 넘기고 `"$@"`를
전달하지 않는다. 그 결과 이 래퍼를 통해 준 hydra 오버라이드가 **에러 없이 전부
무시**된다. 스텝 수를 줄인 스모크 테스트가 조용히 풀 배치로 도는 식으로 발화한다.
`"$@"`를 추가했다.

### 발견 8 — `word2number` 미설치로 첫 validation에서 죽는다

`math500`·`aime24`의 `data_source`는 `multi_datasets_eval` 경로를 타는데, 이
모듈이 `word2number`를 임포트한다. STEER의 `requirements.txt`에 없다.
`run/run_steerf.sh`의 기본 검증셋이 aime24이므로 **원본 설정 그대로도**
`val_before_train`에서 `ModuleNotFoundError`로 죽는다.

### 발견 9 — 코어 수가 많으면 torch가 스래싱한다

255 코어에서 스레드 제한 없이 `pytest tests/`가 10분 이상 (2700% CPU) 걸린다.
`OMP_NUM_THREADS=8`에서 **6.2초**. `run/env_1gpu.sh`에 고정.

### 1-GPU 예산 결정 (전 팔 동일 적용, `run/phase0_1gpu.sh`에 파일로 고정)

| 항목 | 원래 | 변경 | 이유 |
|---|---|---|---|
| `n_gpus_per_node` | 4 | 1 | 호스트에 GPU 1장 |
| `max_response_length` | 2048 | 1024 | 시간 예산. 실측 `response_length/clip_ratio` = 0.104로 절단 영향 작음 |
| 학습 스텝 | ~265 | 100 | 계획서 §2.3의 하한 |
| 검증셋 | aime24 | **math500** | 1.5B-Instruct는 1024토큰 예산에서 AIME24가 ~0%라 곡선에 신호가 없다. math500 학습 전 baseline `acc/mean@1` = **0.516** |
| `save_freq` | 25 | -1 | G0는 곡선만 필요 |
| `ppo_micro_batch_size_per_gpu` | 8 | **8 (고정)** | 방법의 일부. §8.2 |

실측 스텝 시간 (λ=0, 100스텝 기준): **82초/step**
= gen 34s + old_log_prob 11s + update_actor 37s. MFU 0.39.
이 값이 미해결 리스크 #2("예보용 추가 forward 1회 비용")의 비교 기준선이다.

### 발견 10 — 현재 설정으로는 G2 판정이 성립하지 않는다

두 가지가 동시에 걸린다.

1. **pass@16이 산출되지 않는다.** G2 통과 조건 1번은 pass@16을 요구하지만
   verl 기본값 `rollout.val_kwargs.n = 1`에서는 `acc/mean@1`만 나온다.
   `n=16`으로 올려야 verl이 `best@16/mean`(= pass@16)을 계산한다.
   검증 비용은 16배(500 → 8000 생성, 약 9분/회).

2. **1시드로는 +1.0pp를 노이즈와 구분할 수 없다.** MATH500 500문제 `mean@1`의
   이항 표준오차는 약 **2.2pp**로 게이트 임계값(+1.0pp)보다 크다. 계획서 §10의
   "시드 3개 이상"은 부가 항목이 아니라 **G2 성립의 전제**다. 3시드 평균에서도
   SE ≈ 1.3pp라 여전히 빠듯하므로, 효과 크기와 함께 시드별 원값을 전부 싣는다.

**결정**: Phase 2를 λ ∈ {0, 0.25, 0.5, 1.0} × seed ∈ {1, 2, 3} = **12런** 전수로
돌린다 (`run/phase2_queue.sh`). λ=0은 stock STEER와 비트 동일하므로 별도 실험이
아니라 그리드의 한 점이다. 시드는 dataloader(`data.seed`)와 vLLM 샘플러
(`rollout.seed`) **양쪽**에 준다 — 한쪽만 주면 복제가 데이터 순서로만 갈린다.

이 결정의 부수 효과: Phase 0의 `ARM=steer` 런을 Phase 2의 λ=0 기준선으로
**재사용할 수 없다** (검증 설정이 n=1이라 pass@16이 없다). Phase 0 steer 런은
G0 산출물로만 쓴다.

큐는 **시드-메이저** 순서다. 중단되면 "완성된 팔 3개 + 빠진 팔 1개"가 아니라
"1시드에서 완성된 그리드"가 남아, 검정력은 부족해도 해석 가능한 결과가 된다.

### 미해결 리스크에 대한 조치

리스크 #1(`_steerf_unembedding()`의 FSDP 경로 미검증)은 λ>0과 헤드 체크포인트가
동시에 있어야 도달하는 경로라, 그대로 두면 Phase 2 첫 스텝에서 처음 발화한다.
`scripts/make_dummy_heads.py`(랜덤 초기화, `untrained: True` 플래그)로 Phase 0
직후 2스텝 배관 테스트를 끼워 넣었다. **결과가 아니라 배관만 보는 것**이며,
`run/phase2_1gpu.sh`는 `untrained` 플래그가 붙은 체크포인트를 거부한다.

### 미실행 코드 감사 1회차 (학습 대기 중 수행)

학습 경로는 스모크로 확인됐지만 Phase 1/3 스크립트와 λ>0 전용 모니터는 한 번도
실행된 적이 없다. 아래 4건은 전부 **게이트 판정을 조용히 무효화하는** 종류다.

#### 발견 11 — `phase1_validate.py`가 프롬프트를 Python repr로 넣는다

`prompts = [str(x) for x in df[prompt_column]]`. 이 parquet들의 `prompt`는 채팅
메시지 배열이라, 모델에 실제로 들어가는 문자열은
`[{'content': 'If $2^8=4^x$...', 'role': 'user'}]`가 된다. verl은 학습에서
`tokenizer.apply_chat_template(..., add_generation_prompt=True)`를 쓴다
(`rl_dataset.py`). 즉 검증이 **학습 분포 밖**에서 돌고, 여기서 나온 G1 FAIL은
"미래 항이 노이즈"로 읽히지만 실제 원인은 "프롬프트가 깨짐"이다.
→ `render_prompt()` 추가, verl과 동일 렌더링. 평문 컬럼도 계속 동작.

#### 발견 12 — EOS 이후 패딩이 ground truth를 오염시킨다

`sample_continuations`는 `generate`의 반환을 그대로 자른다. `generate`는 배치
전원이 끝날 때까지 돌고 먼저 끝난 시퀀스를 패딩하므로, 반환된 continuation은
"진짜 토큰 + 패딩"이다. `measured_future_entropy`는 텐서가 지평선보다 **짧을**
때만 NaN을 넣는데 패딩 때문에 절대 짧아지지 않는다. 결과적으로 offset 2에서
끝난 continuation이 `...<eos><pad><pad>` 상태의 엔트로피를 정상 실측치로
기여하고, 하류의 `isfinite` 필터가 그걸 걸러내지 못한다.
→ 각 continuation을 자기 EOS에서 절단(EOS 자체는 유지 — 그 분포는 실제 정책
상태다). 이제 NaN 경로가 설계대로 작동한다.

#### 발견 13 — G1의 recall 절반을 계산하는 스크립트가 없었다

`phase1_validate.py`는 `evaluate_gate_g1(recall=nan, n_branch=0, ...)`을 넘기고
"실제 rollout 그룹에서 재야 한다"는 안내만 출력한다. 그 계산을 하는 코드가
저장소에 없었으므로 **G1 조건 2는 구조적으로 항상 FAIL**이었고, 게이트를
통과할 방법이 없었다. 진짜 분기점은 형제(sibling) 정의라 독립 샘플링된 prefix에는
존재하지 않는 양이다 — Phase 0 덤프(`rollout.n=8`)에는 존재한다.
→ `scripts/phase1_branch_recall.py` 신규. `verl_integration.forecast_h_togo` /
`compute_a_h`를 그대로 써서 학습이 쓰는 인덱스 정렬을 오프라인에서 같이 검증한다.

#### 발견 14 — "상위 10분위"가 실제로는 99.7%를 고른다 (G1·G2 동시 무효화)

`branch_recall_at_k`와 `branch_token_entropy`는 `thresh = topk(a_v, k).values.min()`
뒤 `a_v >= thresh`로 선택한다. 그런데 `A_H`에는 **정확히 0인 점질량**이 있다:
`baseline="sibling"`에서 고유해진 rollout은 자기 자신이 유일한 형제라
`sibling_prefix_baseline`이 자기 값을 그대로 돌려주고 `A_H`가 정확히 0이 된다
(docstring에 명시된 동작). 첫 분기 이후 거의 모든 위치가 여기 해당한다.

형제 8개, 공유 prefix 5토큰으로 실측:

| T | `A_H == 0` 비율 | `>= thresh`가 고르는 비율 |
|---|---|---|
| 96 | 93.8% | 96.7% |
| 512 | 98.8% | 99.4% |
| **1024** | **99.4%** | **99.7%** |

결과: G1 이항검정의 귀무 선택률이 0.997이 되어 **유의해질 수 없고**,
G2의 `branch_entropy` / `nonbranch_entropy` 분할이 99.7% 대 0.3%가 되어
`branch_entropy_gap`이 분기점에 대해 아무것도 말하지 않는다. README가 G2의
"메커니즘 확인의 핵심"이라고 지정한 지표가 통째로 무의미해진다.

→ 값 임계 대신 **순위 기반 선택**(`_top_k_selection`)으로 교체. 정확히 k개를
고른다. 회귀 테스트 3개 추가, **216 passed**.
수정 후 실측: 선택률 98.0% → **10.03%**, 조건 2가 p=5.5e-20으로 정상 판정됨.

> **해석 주의**: 위 검증을 *미학습* 헤드로 돌려도 recall이 0.60이 나온다.
> `A_H`의 **support**(형제가 있는 위치)만으로도 분기 영역이 상당히 특정되기
> 때문이다. 즉 G1 조건 2의 PASS는 "예보가 분기점을 잘 **순위매긴다**"의 증거로는
> 약하다. Phase 1 보고서에 미학습 헤드 대조값을 함께 실어야 한다.

### 진행 상태

- [x] Step 0 셋업·자체검증 (216 passed)
- [ ] **G0**: `ARM=grpo` 진행 중 → `ARM=steer` 대기
- [ ] **G1**: Phase 1
- [ ] **G2**: Phase 2 12런
- [ ] **G3**, Ablation

## (다음 항목은 여기에)
