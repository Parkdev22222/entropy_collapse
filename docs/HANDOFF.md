# HANDOFF — STEER-F 세션 인계 문서

**작성 2026-09-09.** 다른 세션(빈 컨텍스트)에서 이 작업을 이어가기 위한 문서다.
새 세션은 레포만 클론하고 대화는 하나도 모른다. **이 문서를 먼저 읽어라.**
설계 근거·검증 절차·폐기된 대안의 전체 기록은 `docs/session_plan_2026-09.md`에 있다.

---

## 0. 새 세션 시작하는 법

첫 프롬프트를 이렇게 준다:

```
docs/HANDOFF.md 와 docs/session_plan_2026-09.md 를 읽고 이어서 하자.
```

학습을 다시 띄우는 게 목적이면 **`docs/RUNBOOK.md`**가 절차다 — 상태 점검부터
기동·감시까지. 이 문서는 배경과 수치를, 런북은 명령을 담는다.

---

## 1. 지금 상태

| | |
|---|---|
| 작업 브랜치 | `claude/3b-text-generation-models-thz2vl` |
| 마지막 커밋 | `a97972c` — paper: fill the result tables from the logs instead of by hand |
| working tree | clean, 전부 푸시됨 |
| 원격 | `https://github.com/Parkdev22222/entropy_collapse` |

### ⚠️ 레포가 두 브랜치로 쪼개져 있다 — 제일 먼저 알아야 할 것

**어느 한 브랜치만 클론하면 파이프라인이 안 돈다.**

| | `claude/3b-...-thz2vl` (이 브랜치) | `origin/paper` |
|---|---|---|
| 오케스트레이션 | ✅ `run_paper.sh`, `_arms.sh`, `run_campaign.sh`, `run_followups.sh`, `run_phase2.sh`, `run_eval_all.sh`, `run_0905_chain.sh` | ❌ |
| **실제 런처** | ❌ **없음** | ✅ `run_steerf.sh`, `run_grpo.sh`, `run_uniform_ablation.sh`, `eval_steerf.sh`, `run_steerf_extreme.sh`, `warmup_and_validate.sh`, `_gpu_defaults.sh` |
| **학습 로그** | ❌ `logs/`가 비어 있다 | ✅ `logs/experiments/train-*.log` |
| 분석·논문 | ✅ `scripts/analyze_seeds.py`, `paper/steerf.tex`, `results/*.tsv` | ❌ |
| `steer_f/`, `verl/`, `tests/` | ✅ | ✅ |

pod의 작업 디렉토리는 두 쪽을 `git checkout <ref> -- run/...`로 섞어놓은 상태다.
새 환경을 만들 때는 **`origin/paper`를 베이스로 클론한 뒤 이 브랜치에서 `run/`·`scripts/`·`paper/`를
체크아웃**하는 순서가 맞다. 로그 분석만 할 거면 `origin/paper`만으로 충분하다.

### 돌고 있는 것 (2026-09-09 기준)

- **STEER `_0905` 복구 런** — `steer-Qwen2.5-Math-1.5B-s1_0905`, step 98/110.
  best 체크포인트 `global_step_80` (.1323). ~950 s/train-step.
  ⚠️ 이 런은 `validation_data_dir: None`이라 **문제별 점수가 안 남는다** (P0-1 계측 미적용).
- 그 뒤에 **20런 캠페인**(4 seed × 5 arm)이 tmux로 대기 중.

---

## 2. 이 프로젝트가 뭔가 — 3분 요약

**STEER-F**: 수학 추론 RL(GRPO)에서 엔트로피 붕괴의 *총량*이 아니라 *방향*을 고르는 방법.

### 2채널 분해 (항등식, 근사 없음)

```
ℋ(π_θ) = Σ_y π_θ(y|x)·f_θ(y),     f_θ(y) := Σ_t H(π_θ(·|s_t))

∇_θ ℋ = E[∇_θ f_θ(y)]          ← (L) local     : 각 분포가 뾰족해지는 것
       + E[f_θ(y)·∇_θ log π_θ]  ← (V) visitation: 어느 경로를 더 자주 밟는가
```

**STEER의 `Ω`는 (L)만 본다.** 그들 논문 Eq. (20) `∂H/∂z_{s,a} = −π(log π + H)`는
균등분포에서 모든 `a`에 대해 0이 된다 → **엔트로피가 가장 높은 바로 그 지점에서 눈이 먼다.**
우리는 (V)를 `A_H`로 근사해 더한다.

### `A_H` — 형제 프리픽스 베이스라인

```
A_H(y_t) = Ĥ_togo(s_t ⊕ y_t) − mean over siblings sharing prefix s_t
```

- 베이스라인이 `s_u`만의 함수 → **무편향**
- **plain rollout에서는 `A_H ≡ 0`** (형제 집합이 자기 자신뿐) → **tree rollout이 enabling condition이지 원인이 아니다**
- 분기 비율 상한 `2(n−1)/T = 2·7/984 = .0142`, 실측 tree arm이 .0120–.0136 (84–96% 포화)
- 형제 집합 위에서 `Σ A_H = 0` → **설계상 평균 감쇠 불변, 분산만 확대.**
  "분기점 평균 엔트로피 상승"은 이 경로의 작동 증거가 **될 수 없다**

### 최대 약점 (리뷰어가 제일 먼저 짚을 곳)

**유효 지평이 사실상 1스텝이다**: `H_togo = 0.7206·H₁ + 0.0636·H₂ + 0.106` (head 2 기여 8%).
"그럼 그냥 로컬 항 아니냐"는 반론에 답이 필요하다.
→ `scripts/measure_forecast_locality.py`가 **같은 연산자**(`compute_a_h`)를 `h_togo`와
모델 자신의 next-token 엔트로피에 각각 적용해 `corr(A_H, A_local)`과 R²를 낸다. 이게 방어다.

---

## 3. 확정된 수치 (전부 `origin/paper` 로그에서 재계산·검증됨)

### plateau step 40–110 (arm당 8 val 지점) — 논문 Table 1

| arm | acc@32 | maj@32 | uplift | entropy | resp len | s/step |
|---|---|---|---|---|---|---|
| GRPO | .1350 | .1963 | .0612 | **.1491** | 963.9 | (다른 박스) |
| STEER | .1330 | .1984 | .0654 | .1214 | 988.1 | 833 |
| uniform | .1313 | .1953 | .0640 | .1374 | 946.0 | 1480 |
| permuted | .1370 | .2094 | .0724 | .1190 | 960.2 | 1492 |
| **STEER-F** | **.1495** | **.2349** | **.0854** | .1210 | 983.9 | 1516 |

⚠️ **GRPO 행은 커밋된 로그로 검증되지 않았다** — 학습 로그가 어느 브랜치에도 없다.
세션 기록값이다. pod에서 push해야 `\num{}` 매크로가 `[pending]`을 벗는다.

### 대조 (8개 plateau step 짝지음) — 논문 Table 2

| 대조 | Δacc | paired t | Δmaj | paired t |
|---|---|---|---|---|
| STEER-F − GRPO | **+.0145** | — | **+.0385** | — |
| STEER-F − STEER | **+.0165** | **6.68** | +.0365 | 5.09 |
| STEER-F − uniform | **+.0182** | **7.18** | +.0396 | 5.50 |
| STEER-F − permuted | **+.0125** | 2.56 | +.0255 | 5.26 |
| permuted − STEER | +.0040 | 0.83 | +.0110 | 1.75 |
| uniform − STEER | −.0017 | −0.62 | −.0031 | −0.52 |

**헤드라인**: 위 셋은 유의, 아래 둘은 0. tree rollout과 감쇠 장치를 다 켜도(uniform) 이득이
0이고, **`A_H`의 실제 값을 넣어야만(signed) 이득이 생긴다.**

### 새 STEER `_0905` 복구 런 비교 (2026-09-09에 한 분석)

공통 창 **step 40–90 (6점)**:

| | acc@32 | maj@32 |
|---|---|---|
| 기존 STEER (s1) | .1310 | .1998 |
| 새 `_0905` | .1193 | .1992 |
| Δ | **−.0117** | **−.0007** |

산술: acc `.716/6 = .11933`, maj `1.195/6 = .19917`.

**판정: 재현 실패가 아니라 두 번째 시드 추출이다.**
−.0117은 대응차 SD ≈ .0088의 1.3배이고, maj는 사실상 동일하며 엔트로피 궤적이 겹친다
(step 90: .122 vs .123). 롤아웃 샘플링이 실행 간 시드 고정이 아니므로
**`data.seed`는 재현 핸들이 아니라 독립 반복의 라벨이다.**

같은 창에서 STEER-F signed s1 = `.1480 / .2378`:

| 대조 | Δacc | Δmaj |
|---|---|---|
| STEER-F − 기존 STEER | +.0170 | +.0380 |
| **STEER-F − 새 STEER** | **+.0287** | **+.0386** |

격차가 줄지 않고 acc에서 1.7배로 벌어졌다. **maj 격차는 +.038로 두 비교에서 동일** —
헤드라인 지표가 STEER 쪽 실행 변동에 둔감하다는 좋은 신호다. 단 여전히 n=1 대 n=1.

전망: 기존 런의 100/110은 `.146/.132`였다. 새 런이 8점 plateau에서 기존 `.1330`을
맞추려면 남은 두 점 평균이 `.174`여야 하는데 이 런이 보여준 적 없는 수준이다.
**`≈ .123`에 안착할 것으로 본다.**

### 노이즈 척도 (헷갈리기 쉬운 지점)

| 척도 | 값 | 언제 쓰나 |
|---|---|---|
| 같은 30문제 위 샘플링 노이즈 | ±.004 acc | 두 런이 같은 궤적인가 |
| 시드 간 SD | ≈ .0062 | 시드 재추출 변동 |
| 대응차 SD | ≈ .0088 | 두 런 plateau 차이 판정 |
| `SE_prob = std@32/√30` | .0288 | **절대 수준의 정밀도만.** 격차를 이걸로 나누지 말 것 |

### 검정력 — 시드를 줄이면 안 되는 이유

헤드라인 `STEER-F − GRPO = +.0145`에 대해:

| n | t | t_crit(.05) | |
|---|---|---|---|
| 3 | 2.85 | 4.30 | **부족** |
| 4 | 3.30 | 3.18 | 간당 |
| 5 | 3.68 | 2.78 | 유의 |

**n=3으로 줄이면 헤드라인이 유의하지 않다. 시드는 자르지 마라.**

---

## 4. 되돌리지 않는 결정들

1. **pass@32 전면 폐기.** 전 arm이 .378–.428에 뭉치고 GRPO가 최고다. 대체 통계는
   `uplift = maj@32 − acc@32`.
2. **"엔트로피 억제가 성능의 원인"이라는 주장 폐기.** 개입한 arm이 GRPO보다 엔트로피가
   **낮다**. 새 주장: **엔트로피 붕괴는 학습 그 자체다. 문제는 총량이 아니라 방향이다.**
   결정적 증거: 엔트로피가 0.1 SE로 동일한 STEER와 STEER-F가 acc 3.5 SE / maj 5.0 SE로 갈린다.
3. **통계 단위 = 시드.** 대조는 **시드로 짝지어** 검정, df = n−1, 대조의 n = 양쪽 arm 시드 수의 min.
   기존의 "8개 step 짝지은 paired t"는 부차 지표(실행 내 안정성)로 강등.
4. **시드 5개 유지** (위 검정력 표).
5. **예측·미측정 수치를 논문에 쓰지 않는다.** 대신 `paper/steerf.tex`가 `results/numbers.tex`를
   `\input`하고 `\num{...}` 매크로가 없는 값을 `[pending]`으로 렌더한다. 표는 자동으로 채워진다.
6. **STEER arm은 110 step 결과 그대로 사용** (200까지 재실행 안 함).
7. **STEER 원논문의 실험 수치는 논문에 넣지 않는다.** 방법 출처로만 인용.
8. **오버헤드 수치(+86%)는 같은 실행 안의 STEER↔STEER-F 비교에서만 인용한다.**
   train-only s/step 785 → 1459. 다른 박스의 s/step을 분모로 쓰면 과소평가된다.

---

## 5. SFT 확장 검토 — 전문

> 이 절이 이 문서를 만든 직접적인 이유다. 원 대화가 압축되면서 요약에서 빠졌고,
> 컨테이너 로컬 기록에서 회수했다. **사용자 질의: "내가 지금 제시하는 방법을 sft에서도
> 쓸 수 있는 방법이 있을까?? 방법만 생각해봐."** 아래는 그 답변 원문 그대로다.

---

SFT로 가져갈 수 있다. 다만 **그대로는 안 되고**, 어디가 끊기는지부터 정확히 봐야 한다.

## 1. 먼저 끊기는 곳 — 방문항이 gradient에서 사라진다

RL에서 두 채널이 나오는 이유는 기댓값이 **π_θ 자신에 대한 것**이기 때문이다. 곱미분이 성립하는 건 `π_θ(y|x)·f_θ(y)`에서 두 인자가 **둘 다** θ에 의존해서다.

SFT는 다르다:

```
L_SFT = −E_{(x,y)~D}[ Σ_t log π_θ(y_t|s_t) ]
∇L_SFT = −E_D[ Σ_t ∇log π_θ(y_t|s_t) ]
```

`D`는 고정이라 score-function 항이 **없다.** 즉 SFT 목적함수에는 방문 채널에 대응하는 항이 아예 존재하지 않는다.

## 2. 그런데 이게 오히려 더 강한 프레이밍이다

핵심은 이거다 — **채널은 목적함수의 성질이 아니라 ℋ 자체의 성질이다.**

```
ℋ(π_θ) = Σ_y π_θ(y|x)·f_θ(y)
∇ℋ     = E[f_θ·∇log π_θ]  +  E[∇f_θ]        ← 항등식, 학습 방식과 무관
```

SFT 한 스텝의 엔트로피 변화는 `⟨∇ℋ, Δθ⟩`이고 `Δθ = −η∇L_SFT`다. **∇L에 방문항이 없어도 ∇ℋ에는 있으므로, SFT는 매 스텝 방문 채널을 통해 엔트로피를 무너뜨리면서 그걸 볼 수단이 목적함수 안에 하나도 없다.**

STEER는 보려고 시도했으나 절반만 본다. **SFT는 아예 안 본다.** 지금 논문의 논지가 GRPO에 국한된 게 아니라는 걸 보여주는 자리다.

## 3. 방법 A — 형제를 만들어낸다 (직접 이식)

SFT 손실에 토큰 가중치를 붙이는 건 그대로 된다:

```
L = −Σ_t α_t · log π_θ(y_t|s_t),    α_t ∈ [w_min, 1]
```

문제는 **데이터에 형제가 없다**는 것. 한 prefix에 정답 continuation 하나뿐이다. 해결책: 모델 자신이 형제를 제안하게 한다. 위치 t에서 `π_θ(·|s_t)`의 top-k를 후보로 삼으면

```
A_H^SFT(y_t) = Ĥ_togo(s_t ⊕ y_t) − E_{a~π_θ(·|s_t)}[ Ĥ_togo(s_t ⊕ a) ]
```

의미: **"정답 토큰이 대안들보다 좁은 미래로 들어가는가?"**

- `A_H < 0` → 정답인데 미래를 좁힌다. 따라가긴 해야 하지만 **덜 날카롭게** → 감쇠
- `A_H > 0` → 미래를 넓힌다 → 그대로 학습

베이스라인이 `s_t`만의 함수라는 성질(무편향 논증)은 **그대로 유지된다.** `E_{a~π_θ(·|s_t)}[·]`는 실제로 어느 토큰이 뽑혔는지에 의존하지 않는다.

### 여기서 SFT가 RL보다 **유리한** 지점이 하나 있다

정답 경로의 `H_togo`는 **예보할 필요가 없다. 정확히 계산된다.** 미래 토큰이 데이터에 이미 있으니 teacher forcing 한 번으로 나머지 구간의 per-position 엔트로피를 할인 합산하면 끝이다.

MTP 헤드가 필요한 건 **반사실 가지(top-k 대안)뿐**이다. 결과적으로:

- `A_H`의 절반이 근사가 아니라 **정확값**
- 정답 가지는 지평 절단(A1)도 풀 수 있다 — 남은 시퀀스 전체를 쓰면 된다
- "유효 지평이 1스텝"이라는 지금 논문의 최대 약점이 **정답 가지에서는 사라진다**

이게 SFT 이식의 가장 강한 기술적 논거다.

## 4. 방법 B — 토큰이 아니라 데이터 수준 (싸다)

시퀀스 단위로 점수를 매겨 코퍼스를 재가중/필터링한다. "분기 같은 위치에서 미래 엔트로피를 체계적으로 좁히는 예제"가 모델을 가장 많이 무너뜨린다는 가설. 훨씬 싸고, 채택되기 쉽고, 평가도 간단하다. 방법 A의 ablation으로도 쓸 수 있다.

## 5. 방법 C — 측정만 하는 논문 (제일 싸다)

SFT 도중 두 채널을 **분리해서 추적**한다. 주기적으로 모델에서 샘플링해 Δℋ를 (L)/(V)로 분해하고, **"SFT의 엔트로피 붕괴는 대부분 (V)이고 목적함수는 그걸 못 본다"**를 보인다. 방법이 하나도 없어도 이건 발견이다.

## 6. 진짜 장애물

| | |
|---|---|
| **비용** | RL에선 롤아웃 위에서만 돌아 +86%였다. SFT는 forward가 비용의 전부라 top-k 후보마다 헤드를 돌리면 K배가 된다. → 엔트로피 높은 위치(forking token)로 제한해야 한다. 80/20 논문이 이 제한을 정당화해준다 |
| **정책 이동** | SFT는 RL보다 모델을 훨씬 빨리 움직인다. base에서 warm-up한 MTP 헤드가 금방 낡는다. drift-KL 게이트가 더 자주 걸릴 것이고, 주기적 헤드 재학습이 필요할 수 있다 |
| **평가 설계** | 주장은 "greedy 정확도는 같거나 낫고 pass@k 여지는 훨씬 많이 남아서 후속 RL이 더 멀리 간다"가 자연스럽다. 그런데 그건 **SFT→RL 2단계 실험**이라 비싸다. 싼 대체: SFT 후 pass@k와 엔트로피만 보고, RL 단계는 future work |
| **선행연구** | 토큰 가중 SFT 자체는 새롭지 않다 — 참조 모델로 토큰을 고르는 RHO-1(*Not All Tokens Are What You Need*) 계열이 있다. 확인해보고, **신규성을 "가중 메커니즘"이 아니라 "신호가 미래 엔트로피라는 것"에 둬야 한다.** 지금 논문의 주장 구조와 같다 |

## 7. 권고

- **지금 논문에는** 두세 문장으로 future work에 넣어라. "이 분해는 GRPO의 성질이 아니라 ℋ의 성질이고, SFT는 방문 채널에 대응하는 항을 목적함수에 아예 갖지 않는다" — 실험 없이 프레이밍만 넓힌다. 공짜다.
- **후속 논문으로는** C(측정) + A(방법), B를 싼 ablation으로. 측정이 프레이밍을 정당화하고 방법이 기여가 된다.
- **가장 먼저 확인할 것**: 정답 가지의 정확한 `H_togo`와 top-k 대안의 예보값이 실제로 갈리는가. GPU 몇 시간이면 재고, 안 갈리면 방법 A는 거기서 끝난다. 지금 만들어둔 `measure_forecast_locality.py`가 거의 그대로 쓰일 수 있다 — 형제를 롤아웃 대신 top-k로 바꾸면 된다.
---

## 6. 인프라 지도

### 단일 진입점

```bash
tmux new -d -s paper "cd /workspace/entropy_collapse && \
  REPO=DSDSh/steer-f_2 bash run/run_paper.sh > logs/experiments/paper.log 2>&1"
```

`DRY=1 bash run/run_paper.sh`로 계획만 출력. `STAGES=analysis ...`로 한 스테이지만 재실행.
**모든 스테이지가 재개 가능하다** — 끝난 것은 건너뛰므로 죽여도 도는 런 하나 말고는 안 잃는다.

스테이지 순서: `preflight → recover → campaign → measure → eval → followups → eval2 → analysis`

| 스테이지 | 하는 일 |
|---|---|
| `preflight` | 환경·계측·디스크 검사. verl import 안 되는 박스에서 큐를 태우기 전에 **거부**한다 |
| `recover` | seed 1의 steer·uniform (체크포인트 소실분). signed/grpo/permuted는 살아 있어 skip |
| `campaign` | seed 2–5 × 5 arm = **20런**. 본 표 |
| `measure` | 리뷰어 방어용 GPU-저비용 측정 (locality, omega 등). 학습 뒤에 돌아 GPU 경합 없음 |
| `eval` | 완료된 25런 × 6 벤치마크 |
| `followups` | seed 1에서 10개 ablation (compute-matched `grpo-long` 포함) |
| `eval2` | 그 10개 × 6 벤치마크 |
| `analysis` | 논문의 모든 표를 `results/`에 생성 |

### 핵심 파일

| 파일 | 역할 |
|---|---|
| `run/_arms.sh` | **single source of truth.** `run_name_for()` 14개 arm, `ckpt_alias_for()`, `steps_for_arm()`, `train_log_done()`, `env_preflight()`, 락/busy 검사 |
| `run/run_campaign.sh` | 시드-우선 순회로 20런. 완료 감지·건너뛰기·락·디스크 가드 |
| `run/run_followups.sh` | 10개 ablation 큐 |
| `run/run_eval_all.sh` | 6-벤치마크 평가. `MODEL_PATH` 충돌 검사 포함 |
| `run/run_phase2.sh` | 캠페인 락이 풀릴 때까지 대기 후 eval → followups |
| `run/instrument_phase2.sh` | `eval_steerf.sh`에 `validation_data_dir` 패치, `run_uniform_ablation.sh`에 `"$@"` 포워딩 |
| `run/setup_env.sh` | 환경 복구. **핀 검사 포함 — 파드 재시작 후 필수** |
| `run/hf_backup.sh` | 체크포인트 HF Hub 업로드·바이트 검증·삭제. `_0905` 이름은 거부(학습 보호) |
| `scripts/analyze_seeds.py` | 학습 로그 → `results/{per_seed,arm_means,contrasts,compute_match}.tsv` + `tables.tex` + `numbers.tex`. t-분포를 자체 구현(불완전 베타), 6개 임계값으로 검증됨 |
| `scripts/measure_forecast_locality.py` | **최대 약점 방어.** 같은 연산자를 `h_togo`와 로컬 엔트로피에 적용해 `corr(A_H, A_local)` |
| `scripts/check_env_pins.py` | 설치된 METADATA에서 핀을 읽어 검사(하드코딩 없음). 붙여넣을 `pip install` 한 줄 출력 |
| `tests/test_run_names.py` | `train_log_done` 글롭 회귀 테스트. **arm 접미사가 형제 arm을 DONE으로 오판정하던 버그** 고정 |
| `paper/steerf.tex` | `results/numbers.tex`를 `\input`, `\num{}`로 자동 채움. 17쪽, 에러 0 |

### 알려진 함정

- **`huggingface_hub` 1.30.0이 스택을 깨뜨린다.** transformers가 import 시점에
  `>=0.34.0,<1.0`을 강제 → `import verl` 실패 → 모든 `main_ppo`가 몇 초 만에 사망.
  복구: `pip install "huggingface_hub>=0.34,<1.0"` 또는 `bash run/setup_env.sh`.
  **파드 재시작 시 `/usr/local/lib/python3.12/dist-packages`(컨테이너 오버레이)가 날아가
  이미지 버전이 돌아온다. `/workspace`는 네트워크 볼륨이라 살아남는다.**
  → **재시작했으면 학습 전에 반드시 `bash run/setup_env.sh`.** 두 번 발생했다.
- `run_steerf.sh`가 SCALE case에서 `STEPS=200`을 하드코딩한다 → `steer_plain_args()`를
  trailing override로 붙여야 한다.
- `train_log_done`의 글롭은 `train-<run>.log`와 `train-<run>_*.log`만 허용해야 한다.
  `train-<run>*.log`로 넓히면 `-permuted`/`-uniform`/`-lam0.1`을 삼켜 **주력 arm이 조용히 건너뛰어진다.**

### 비용 (A100×2 순차)

남은 것 ≈ **1,188 box-hours ≈ 49일 ≈ $2,900–4,500**.
안전한 절감: ablation 9개 중 8개(307 h) + 통제군 3시드(185 h) → 697 h ≈ 29일.
**시드는 자르지 마라.** H100×4는 ~3.2배 빠르지만 마이크로배치 구성이 바뀌어
STEER의 min–max 그룹이 달라진다 → **박스 하나에 시드 하나를 통째로** 배정해야 대조가 깨끗하다.

---

## 7. 미해결 항목

1. **GRPO 학습 로그가 어느 브랜치에도 없다.** pod에서 `paper` 브랜치로 push해야
   논문 표의 GRPO 행이 `[pending]`을 벗는다. **제일 값싸고 제일 중요하다.**
2. **`checkpoints/mtp_calibration_Qwen2.5-Math-1.5B-paper.json` 회수** — 원고의 `a_k`
   수치 정합(발견된 4번째 오류). pod에서만 가능.
3. **seed 1에 중복 draw가 생겼다** — STEER와 signed가 각각 두 런(기존, `_0905`).
   arm별로 좋은 쪽을 고르면 체리피킹이다. 롤아웃이 시드 고정이 아니므로 사실상 독립
   추출이니, **`_0905`를 "재실행"이 아니라 별도 draw로 라벨링**해 arm 평균의 n을 올리되
   시드-대응 대조에는 짝이 있는 시드만 쓰는 쪽이 방어하기 쉽다.
   `scripts/analyze_seeds.py`가 지금은 run 이름 글롭으로 파싱하므로 **규칙이 없다.**
   캠페인이 끝난 뒤에 정하면 된다.
4. **작업 8(예방)은 미승인.** `constraints.txt` + `PIP_CONSTRAINT`로 hub 업그레이드를
   원천 차단하는 계획. 사용자가 ExitPlanMode를 거부했다 — **다시 제안하지 말 것.**
   현재는 탐지(`check_env_pins.py`, `env_preflight`)만 있고 예방은 없다.
5. **`_0905` STEER 런은 `validation_data_dir: None`** — 문제별 점수가 안 남는다.
   캠페인의 나머지 arm은 반드시 계측이 걸린 상태로 돌아야 한다:
   `grep validation_data_dir logs/experiments/train-*.log`로 확인.
6. **SFT 확장은 §5의 검토 단계까지만.** 사용자가 "방법만 생각해봐"라고 했다. 실험 없음.

---

## 8. 이 문서의 출처

- 원 세션: `session_014wUL1tA6wZVXuNL5VhD9mB` (Claude Code on the web)
- 전체 설계 기록: `docs/session_plan_2026-09.md` (작업 1–9)
- §5의 SFT 답변은 원 대화 기록에서 **글자 단위로 그대로** 옮겼다(3,072자). 요약·재작성이 아니다.
