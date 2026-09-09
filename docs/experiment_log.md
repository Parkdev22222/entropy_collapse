# 실험 러닝 로그

계획서 §10: **실패·부정 결과 포함 전부 기록.**

---

## 2026-08-15 — Phase 0~2 코드 구축 (실행 없음)

### 환경 제약 (중요)

작업 컨테이너는 **CPU 전용**이다.

```
$ nvidia-smi
bash: nvidia-smi: command not found
$ python -c "import torch; print(torch.cuda.is_available())"
False
```

계획서 §1이 요구하는 H100/A100이 없으므로 다음이 **전부 미실행**이다:

- Phase 0 소형 재현 학습 (GRPO vs STEER, 100~150 스텝) → **게이트 G0 미판정**
- Phase 1 헤드 워밍업, MC 검증, (κ, γ_H) 결정 → **게이트 G1 미판정**
- Phase 2 λ 스윕, 7B 확인 → **게이트 G2 미판정**
- Phase 3, 4 전부 → **게이트 G3 미판정**

계획서 §11의 규칙("게이트 실패 시 다음 Phase로 넘어가지 말 것")은 게이트 *실패* 시
중단하라는 것이므로, 게이트가 **판정 불가**한 상황에서 코드를 미리 준비해 두는 것은
규칙 위반이 아니라고 판단해 진행했다. 다만 어떤 실험 결론도 주장하지 않는다.

### 수행한 것

1. STEER 레포 클론 및 정독. `docs/steer_code_map.md` 작성 (Phase 0 산출물).
2. `steer_f/` 6개 모듈 구현.
3. `patches/core_algos_steerf.patch` 작성 — `git apply --check` 통과 확인.
4. 105개 단위 테스트 작성, 전부 통과 (CPU, 약 2초).

### 발견 — 계획서 §0.1 서술과 코드 불일치

계획서는 STEER의 α를 "밴드 비교 후 이산 3단 {γ, 1, 1/γ}"로 서술했으나,
실제 구현(`core_algos.py:659-710`)은 **배치 min-max 연속 매핑**이다.
절대 밴드는 코드에 존재하지 않는다. 전체 목록은 `steer_code_map.md` §6.

파급:

- 계획서 §4.1의 "밴드 재산정" 작업 항목은 **삭제**했다 (재산정할 밴드가 없음).
- 계획서 §0.2의 "이산 α로 예보 오차를 유계 방어"라는 논거는 성립하지 않는다.
  유계성은 `token_weight_min/max` 클램프가 대신 제공하며, 이는 여전히 유효한 방어다.
  다만 논문 작성 시 이 근거를 다시 써야 한다.
- 예상 밖의 이득: `linear=True`에서 min-max 매핑이 **아핀 불변**이므로, z-정규화가
  α를 전혀 바꾸지 않는다. λ=0 동치성이 수학적으로 공짜로 성립한다
  (`test_zscore_is_affine_invariant_under_linear_mapping`이 수치로 확인, 오차 2e-4).
  단, `linear=False`(지수 매핑)에서는 불변이 아니므로 z-norm이 결과를 바꾼다 —
  지수 모드로 실험할 때 주의.

### 발견 — 새 리스크 (계획서 §9에 없던 것)

`metric_max`가 마이크로배치 내 단 하나의 이상치로 결정되므로, 이상치 하나가 크면
나머지 토큰 전부가 `token_weight_max` 근처로 압축되어 STEER가 사실상 무력화된다.
미래 항을 더하면 꼬리가 두꺼워질 수 있어 이 리스크가 커진다.

대응: `steerf_norm=robust`(median/IQR) 옵션과 α 히스토그램/포화 경보 로깅 추가.
Phase 2 첫 실행에서 `steerf/alpha_saturated`를 반드시 확인할 것.

### 설계 결정 기록

| 결정 | 근거 |
|---|---|
| symmetric에서 미래 항을 **크기**로 결합 | base가 `\|Ω\|`이므로 부호 있는 항을 더하면 의미론이 깨짐 |
| `A_H` 계산 시 h_togo를 **한 칸 shift** | 위치 t의 히든은 y_t를 아직 조건으로 하지 않음. 안 밀면 신호가 통째로 어긋남 |
| MTP 헤드 마지막 Linear를 **0 초기화** | residual 모드에서 학습 전 예보 = 본체 분포. 워밍업 초기 폭주 방지 |
| RL 경로에서 로짓을 K개 동시 생성 **금지** | Qwen V≈152k, `[K,B,T,V]`는 수십 GB. `forward_entropy`가 청크로 축약 |
| λ=0에서 z-norm까지 **우회** | 아핀 불변이 성립하더라도 부동소수점 오차가 남는다. 비트 동일성을 코드로 보장 |
| 실측 엔트로피를 **고정 창**(기본 64토큰)에서 합산 | 전체 길이 합은 continuation 길이와 교락됨. 길이정규화 평균도 함께 기록 |

### 미검증으로 남긴 것

`steer_f/verl_integration.py`의 세 지점(히든 추출, FSDP 헤드 등록, 배치 전달)은
형상 계약만 테스트했고 실제 verl 런에서 검증되지 않았다. README "미검증 표면" 참조.

---

## 2026-09-09 — Phase 1 배관 검증 (CPU 스모크). 여전히 실험 없음

### 환경

새 컨테이너. 여전히 **CPU 전용**이고, 이번엔 의존성도 비어 있었다.

```
$ nvidia-smi
bash: nvidia-smi: command not found
$ pip install torch --index-url https://download.pytorch.org/whl/cpu
ERROR: 프록시가 download.pytorch.org 를 차단 (403) — PyPI 기본 인덱스로는 설치됨
$ python -c "import transformers, pandas, vllm"
ModuleNotFoundError (셋 다 없음, 설치 불가/불필요)
```

torch 2.14.0+cu130(CPU 실행) + numpy + pytest 만 확보. 게이트 G0~G3 는 **여전히 전부
미판정**이며 이 세션에서도 어떤 실험 결론도 주장하지 않는다.

### 문제 인식

기존 105개 테스트는 `steer_f/` 모듈만 덮고 있었다. 반면 `scripts/phase1_warmup_heads.py`
와 `scripts/phase1_validate.py` 는 **한 번도 실행된 적이 없었다**. 즉 GPU 노드에서 가장
비싼 첫 실행이 곧 이 스크립트들의 첫 실행이 되는 상황이었다 — 배관 버그 하나로 GPU
시간을 통째로 버릴 수 있다.

### 수행한 것

의존성 없는 합성 스택(`scripts/smoke_model.py`)을 만들어 Phase 1 전 구간을 CPU에서
완주시켰다. 문자 단위 토크나이저 + 소형 causal transformer로, 실제 코드가 **실제로
호출하는 표면만** 흉내낸다(형상 계약은 `tests/test_smoke_model.py`가 강제).

```
$ bash run/run_smoke_cpu.sh
[stage1] kept 3 mixed-outcome problems
[stage2] 48 prefixes
[grid] selected kappa=1 gamma_h=0.85 rho=0.1251
[G1 FAIL] ... rho 0.1251 < 0.2 / recall lift 1.00
게이트 종료코드: 2   파이프라인 완주 — 배관 정상.
```

**이 숫자들은 아무 의미도 없다** (난수 가중치). 확인한 것은 "판정이 내려지고 리포트가
쓰이는가"이지 "통과했는가"가 아니다.

### 발견 — 실제 버그: 빈 prefix 풀이 조용히 통과한다

스모크 첫 실행에서 `[stage2] 0 prefixes` 가 나왔는데도 stage3/4/5 가 **빈 입력으로
그대로 진행**했고, 네 단계 뒤 `select_kappa_gamma` 에서 원인과 무관한 메시지로 죽었다:

```
[stage2] 0 prefixes
[stage3] sampling 2 continuations for 0 prefixes
UserWarning: The use of `x.T` on tensors of dimension other than 2 ...
ValueError: no finite rho in grid results        ← 진짜 원인과 무관
```

GPU 노드에서 이 경로를 밟으면 원인 추적에 사람 시간과 GPU 시간을 모두 버린다.
실모델에서도 충분히 발생 가능하다 — 모델이 줄바꿈 없는 짧은 응답만 내거나
(`split_steps` 가 줄 단위로 스텝을 센다), `--problem-pool` 이 작아 혼합 난이도 문제가
안 걸리는 경우.

대응: `check_pools()` 가드를 **발생 지점**(stage2 직후)에 넣고, 어떤 인자를 고칠지
메시지에 담았다. 종료코드는 게이트 판정(0=통과, 2=실패)과 구분되는 **3**을 쓴다 —
설정/데이터 오류를 게이트 실패로 오독하면 계획서 §3 실패 절차를 헛되이 밟게 된다.

### 발견 — 재현성 구멍

`GenerationBackend.generate` 의 `seed` 인자는 **vLLM 경로에서만** 쓰이고 HF 로컬 생성
경로에서는 무시되고 있었다. `--force` 로 캐시를 재계산하면 다른 결과가 나온다는 뜻이다.
로컬 경로에도 `torch.manual_seed(seed)` 를 적용했다.

### 관찰 — 진단 신호로 기록해 둘 것

스모크에서 **κ 와 γ_H 를 바꿔도 ρ가 0.1251 로 완전히 동일**했다. 원인은 헤드 막 Linear
0-초기화(설계 결정) + 학습 3스텝 → 모든 헤드가 사실상 같은 분포를 내고, 그러면 H_togo 는
단일 값의 단조 변환이라 Spearman 순위가 κ/γ에 불변이 된다. 스모크 픽스처의 산물이다.

다만 **실제 워밍업 후에도 그리드 전체에서 ρ가 동일하다면 그것은 헤드가 분화하지 않았다는
신호**다 (워밍업 부족 또는 헤드가 본체 분포를 그대로 복사). Phase 1 리포트에서 반드시
확인할 것 — 이 경우 ρ 값 자체보다 먼저 워밍업을 의심해야 한다.

### 상태

- 테스트 106 → **154 passed** (CPU, 8.4초). 실모델 경로는 전부 `smoke:` 접두사 가드
  뒤에 있어 동작 무변경.
- 게이트 G0~G3: **여전히 전부 미판정**. 이 세션이 줄인 것은 배관 리스크뿐이다.

---

## (다음 기록은 GPU 노드에서 Phase 0 재현 실험부터)

기록 시 반드시 포함할 것:

- 커밋 해시, 시드, 정확한 CLI
- 스텝별 정책 엔트로피, `steerf/alpha_*`, `steerf/future_frac`,
  `steerf/forecast_policy_kl`, reward 곡선
- 게이트 판정과 그 근거
- 실패한 시도와 그 이유 (재시도 순서는 계획서의 "실패 시" 절차를 따를 것)
