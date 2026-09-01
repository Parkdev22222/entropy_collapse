# 논문 실험 인덱스

`paper` 브랜치에 담긴 실험 로그와 코드가 각각 무엇인지, 그리고 **아직 없는 것이 무엇인지**
정리한다. 표의 모든 값은 로그 본문에서 기계적으로 추출했다 (hydra 인자 + `steerf/*` 지표).

## 1. 커밋된 학습 로그

| 로그 파일 | arm | λ | mapping | rollout | 마지막 step | val 지점 | 상태 |
|---|---|---|---|---|---|---|---|
| `train-steer-Qwen2.5-Math-1.5B-s1.log` | **STEER** | 0 | minmax | plain | 110 | 12 (0~110) | **완결** |
| `train-math-Qwen2.5-Math-1.5B-s1.log` | **STEER-F** | 0.25 | **rank** | plain | 141 | 12 (30~140) | **완결** (step 30에서 resume, 앞 30 step은 같은 설정) |
| `train-steer-f-Qwen2.5-Math-1.5B-s1-tree-rollout.log` | **STEER-F signed** | 0.25 | minmax | **tree** | 79 | 9 (0~70) | ⚠️ **오래됨** — 실제 실행은 ~110까지 갔다 |
| `train-Qwen2.5-Math-1.5B-s1-tree-rollout.log` | — | 0 | minmax | plain | 0 | 1 | 스텁 (step 110 val 한 줄만) |
| `train-math-Qwen2.5-Math-1.5B-s1-tree-rollout.log` | — | 0 | minmax | plain | 0 | 0 | 스텁 (기동 실패) |
| `train-math-Qwen2.5-Math-1.5B-s2.log` | — | 0.25 | minmax | plain | 0 | 0 | 스텁 (기동 실패) |
| `train-steerf-Qwen2.5-Math-1.5B-s1-tree-rollout.log` | — | 0.25 | — | — | 0 | 0 | 스텁 (기동 실패) |

`rollout` 열의 tree/plain은 hydra 인자가 아니라 `run_name`으로 구분한다 — tree rollout은
`patches/steerf_tree_rollout.patch`로 적용되는 코드 변경이라 config에 남지 않는다.

## 2. 없는 로그 — pod에서 가져와야 함

논문 표의 통제군 두 arm이 레포에 **전혀 없다.** 두 arm 모두 `run/run_uniform_ablation.sh`로 띄웠고
로그는 각 pod의 `logs/experiments/` 아래에 있다.

| 필요한 것 | 파일명 | 왜 필요한가 |
|---|---|---|
| **uniform** arm | `train-steer-f-...-tree-rollout-uniform.log` | `δ = −λ·band` 상수 개입. 크기 불일치(15배)를 보여주는 근거 |
| **permuted** arm | `train-steer-f-...-tree-rollout-permuted.log` | **유일한 단일변수 통제군.** 논문 핵심 주장의 직접 증거 |
| signed 연장분 | 위 signed 로그의 step 80~110 | 커밋본이 79에서 끊겨 있음 |

tensorboard 이벤트(`tensorboard_log/STEER-F/<run>/`)도 함께 가져와야
`scripts/select_best_checkpoint.py`가 동작한다.

## 3. `.ipynb_checkpoints/` 는 인용하지 말 것

`logs/experiments/.ipynb_checkpoints/` 에 12개의 Jupyter 자동저장본이 있다. 대부분
같은 이름 로그의 **오래된 잘린 사본**이다 (예: signed 로그의 사본은 61줄, 실제는 5,250줄).
논문 수치를 여기서 읽으면 안 된다. 예외는 하나다:

- `warmup-rollouts-Llama-3.2-3B-Instruct-checkpoint.log` (397줄) 은 같은 이름의 라이브 로그(105줄)와
  **다른 실행**이다. 라이브본은 KeyboardInterrupt로, 사본은 gated-model 접근 거부로 끝난다.
  둘 다 실패한 warmup이고 논문 결과가 아니다.

## 4. 평가 도구 (검증 완료)

| 도구 | 용도 |
|---|---|
| `run/eval_steerf.sh` | 논문 프로토콜 다중 벤치마크 평가. `MODEL_PATH`만 받는다 |
| `scripts/select_best_checkpoint.py` | 논문 선택 규칙(AIME24 argmax). 디스크에 남은 step으로 제한된다 |
| `scripts/collect_results.py` | eval 로그 → 논문 표 TSV. **파일명이 `eval-<arm>-s<seed>.log` 여야 한다** |
| `docs/paper_reference.tsv` | STEER 논문 원 수치, 같은 컬럼 모양 |

데이터셋 재고는 확인했다. `aime24`/`aime25`는 30문제 × 32 replica = 960행,
`amc23`는 40 × 32 = 1,280행이라 avg@32 집계가 유효하고, 7개 벤치마크의 `data_source`가
`collect_results.py`의 `COLUMNS`와 전부 일치한다.

## 5. 체크포인트 보존 주의

모든 학습 실행이 `max_actor_ckpt_to_keep=3` + `save_best_only=False`다.
`verl/trainer/ppo/ray_trainer.py:914`의 best-checkpoint 분기는 `save_best_only=True`일 때만
동작하므로 **별도로 보존되는 best 체크포인트가 없다.** 110 step 실행이 끝나면
`global_step_90/100/110` 세 개만 남는다. signed의 AIME24 최고 지점인 step 70은 이미
회전되어 사라졌을 가능성이 높다.
