# 인수인계 (HANDOFF)

> 다음 세션이 이 문서 하나만 읽고 이어받을 수 있도록 쓴다.
> 마지막 갱신: 2026-09-09 · 브랜치 `claude/blissful-sagan-5fm6t7`

---

## 1. 한 줄 요약

**코드는 Phase 0~4 전부 준비됐고, 실험은 하나도 못 했다.** 개발 환경에 GPU가 없어서다.
직전 세션에서 한 일은 "GPU 노드 첫 실행이 배관 버그로 깨지는 것"을 막은 것뿐이다.

---

## 2. 지금 상태

### 판정된 것 / 안 된 것

| 게이트 | 내용 | 상태 |
|---|---|---|
| G0 | STEER 소형 재현 (GRPO vs STEER, 100~150 스텝) | **미판정** — GPU 없음 |
| G1 | 예보 검증 ρ ≥ 0.2 & 분기 recall 유의 | **미판정** — GPU 없음 |
| G2 | λ 스윕, 7B 확인 | **미판정** |
| G3 | 패밀리 전이 | **미판정** |

**어떤 실험 결론도 주장된 바 없다.** 코드 준비와 실험 결과를 혼동하지 말 것.

### 검증 수준별 코드 현황

| 표면 | 검증 수준 |
|---|---|
| `steer_f/` 6개 모듈 | 단위 테스트 (154개 중 대부분) |
| λ=0 ≡ 순수 STEER 동치성 | 업스트림 verbatim 추출본과 대조 (`test_lambda_zero_equiv.py`) |
| `scripts/phase1_*.py` 배관 | **CPU 스모크로 end-to-end 완주 확인** (합성 모델) |
| `patches/core_algos_steerf.patch` | `git apply --check` 통과. 실제 학습 미실행 |
| `steer_f/verl_integration.py` | **형상 계약만.** 실제 verl 런 미검증 — §5 참조 |

```bash
python -m pytest tests/ -q     # 154 passed, 10초 내외
bash run/run_smoke_cpu.sh      # Phase 1 전 구간 CPU 완주, 종료코드 2(=G1 실패) 기대
```

---

## 3. 환경 제약 (반복해서 부딪힌 것)

개발 컨테이너는 CPU 전용이고 의존성이 비어 있다. 매 세션 다시 설치해야 한다.

```bash
pip install torch pytest numpy          # PyPI 기본 인덱스로 설치할 것
```

- `download.pytorch.org` 는 **프록시가 403으로 차단**한다. `--index-url` 을 주지 말 것.
- `transformers` / `pandas` / `pyarrow` / `vllm` / `matplotlib` 는 없다.
  스모크 경로와 테스트는 이 다섯 없이도 전부 돈다 (그렇게 설계했다).

---

## 4. 다음에 할 일

### GPU 노드가 생겼을 때 (우선순위 1)

계획서의 순서 그대로. 각 단계 전에 CPU 스모크를 한 번 돌려 배관이 성한지 먼저 확인한다.

```bash
# 0. 패치 적용
git clone https://github.com/zz-haooo/STEER && cd STEER
git apply /path/to/STEER-F/patches/core_algos_steerf.patch
export STEER_ROOT=$PWD

# 0b. G0 — baseline 재현 (λ=0 이면 STEER와 비트 동일해야 한다)
LAMBDA=0 bash /path/to/STEER-F/run/run_steerf_linear.sh

# 1. 헤드 워밍업 → 2. G1 판정 → 3. λ 스윕
#    정확한 CLI 는 README "전체 파이프라인" 절 참조
```

**첫 GPU 실행에서 반드시 확인할 것:**

1. `steerf/alpha_saturated` — min-max 이상치 민감성 (`steer_code_map.md` §6.1).
   포화가 보이면 `steerf_norm=robust` 로 전환.
2. §5의 미검증 3지점이 실제로 도는지.
3. **(κ, γ_H) 그리드에서 ρ가 전부 동일하게 나오는지** — 그렇다면 헤드가 분화하지
   않은 것이다. ρ 값을 해석하기 전에 워밍업부터 의심할 것 (근거는 experiment_log
   2026-09-09 "관찰" 절).

### GPU 없이 계속할 수 있는 일 (우선순위 2)

- **순차형 MTP (DeepSeek-V3식)** — 현재 미구현. G1 실패 시 세 번째 대응책으로
  `phase1_report.md` 에 적혀 있는 유일한 기능 공백이다.
- **`verl_integration.py` 의 mock-verl 스모크** — Phase 1 에 한 것과 같은 방식으로
  rmpad 레이아웃·FSDP 등록·배치 전달을 합성 텐서로 완주시킬 수 있다. §5의 리스크를
  실제 GPU 없이 더 줄이는 방법.
- Phase 3 `phase3_port_model.py` 도 아직 실행된 적이 없다 — 스모크 대상 후보.

---

## 5. 남아 있는 최대 리스크: verl 통합 3지점

`steer_f/verl_integration.py` 는 **형상 계약만** 테스트돼 있다. 실제 verl 런에서
검증되지 않았고, GPU 노드 첫 실행에서 깨진다면 십중팔구 여기다.

1. `dp_actor.py::_forward_micro_batch` — `output_hidden_states=True` 추가 후 rmpad
   (packed) 레이아웃에서 히든 추출. Ulysses SP 가 켜져 있으면 히든도 log_prob 과
   **같은 gather 경로**를 타야 한다.
2. FSDP wrap 시점의 헤드 파라미터 등록과 그래디언트 동기화. 가장 단순한 경로는 본체
   FSDP wrap **이전에** `actor_module.mtp_heads = heads` 로 붙여 함께 감싸지게 하는 것.
3. `ray_trainer.py` 에서 `h_togo` / `baseline_h_togo` 를 배치에 실어 나르는 부분
   (`attach_entropy_advantage`).

정확한 삽입 위치는 `verl_integration.py` 상단 docstring 과 `steer_code_map.md` §5.

**비용 경고**: 헤드 K개의 로짓을 동시에 만들면 Qwen(V≈152k) 기준 수십 GB다.
RL 경로에서는 반드시 `forward_entropy()` (청크 축약)를 쓸 것. `forward_logits()` 는
소형 검증 배치 전용이다.

---

## 6. 직전 세션(2026-09-09)에서 바뀐 것

CPU 스모크 하니스를 만들고, 그 과정에서 실제 버그 2개를 잡았다.

**추가**
- `scripts/smoke_model.py` — 의존성 없는 합성 정책 스택 (문자 토크나이저 + causal LM).
- `scripts/make_smoke_data.py`, `run/run_smoke_cpu.sh`.
- 테스트 4개 파일 48개 (106 → 154).

**고친 버그**
- `phase1_validate.py`: prefix 풀이 **0개여도 stage3~5 가 조용히 진행**하다가 네 단계 뒤
  무관한 메시지(`no finite rho in grid results`)로 죽었다. `check_pools()` 가드를
  발생 지점에 넣고 **종료코드 3**(설정/데이터 오류)을 게이트 판정 0/2 와 분리했다.
- `_common.GenerationBackend.generate`: `seed` 가 vLLM 경로에서만 적용되고 로컬 생성
  경로에서는 무시되고 있었다 (`--force` 재계산이 재현되지 않음). 이제 둘 다 적용.

**주의** — 스모크의 ρ·recall·게이트 판정에는 **아무 의미도 없다**. 난수 가중치이므로
G1 은 거의 항상 실패한다(종료코드 2). 확인하는 것은 "판정이 내려지고 리포트가 쓰이는가"다.

---

## 7. 프로젝트 규칙 (계속 지킬 것)

- **코드가 진실이다.** 계획서 서술과 어긋나면 코드를 따르고 `steer_code_map.md` §6 에 기록.
- **실패·부정 결과도 전부 `docs/experiment_log.md` 에 남긴다.** "미래 항이 노이즈"라는
  결론 자체가 유의미한 결과다.
- **게이트가 실패하면 다음 Phase 로 넘어가지 않는다.** 단, 게이트가 *판정 불가*(환경
  제약)인 상황에서 코드를 준비해 두는 것은 위반이 아니다 — 다만 어떤 실험 결론도
  주장하지 않는다.
- 실험 기록에는 커밋 해시·시드·정확한 CLI 를 반드시 포함한다.
