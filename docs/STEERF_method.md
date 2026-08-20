# STEER-F 방법론 — 수식과 주석 전부

> 이 문서는 `steer_f/`에 **실제로 구현된** 수식을 계산 순서대로 펼쳐 놓고,
> 각 식의 모든 기호·텐서 모양·단위·설계 근거(코드 docstring의 논증)를 주석으로 단다.
> 논문/계획서의 수식이 아니라 코드의 수식이 기준이다 — 둘이 다른 지점은 그때마다 명시한다.
>
> 파일 기준: `steer_f/{mtp_heads, mtp_sequential, entropy_forecast, omega_tilde, monitors, validation, verl_integration}.py`

---

## 0. 표기법과 텐서 규약

| 기호 | 의미 | 코드에서 |
|---|---|---|
| `B` | 마이크로배치의 시퀀스 수 | `ppo_micro_batch_size_per_gpu` (기본 8) |
| `T` | 응답 길이 | `data.max_response_length` |
| `K` | MTP 헤드 수 (학습 시) | `num_heads`, 기본 8 |
| `κ` | 예보 호라이즌 (사용 시, `κ ≤ K`) | `SteerFConfig.kappa` |
| `V` | 어휘 크기 | Qwen2.5 기준 ≈152k |
| `H` | 정책 hidden 폭 | `hidden_size` |
| `π = π_θ(y_t\|s_t)` | 현재 정책이 샘플 토큰에 주는 확률 | `exp(log_prob)` |
| `π_old` | 롤아웃 시점 정책의 확률 | `exp(old_log_prob)` |
| `r` | 중요도 비율 `π/π_old` | `exp(clamp(log_prob − old_log_prob, ±20))` |
| `A` | GRPO 어드밴티지 | `advantages [B,T]` |
| `mask` | 응답 마스크 (1=실토큰) | `response_mask [B,T]` |
| `uid` | 프롬프트 그룹 키 | `non_tensor_batch["uid"]` — GRPO가 쓰는 것과 동일 |
| `s_t ⊕ y_t` | 상태 `s_t`에서 토큰 `y_t`를 소비한 뒤의 상태 | hidden index 규약, §4.3 |

규약 세 가지:

- **엔트로피 단위는 전부 nats.** 상한은 `log V ≈ 11.93` (Qwen). `clip_c = 1.0`,
  `kl_drift_threshold = 0.5` 같은 상수는 이 스케일 위에서 읽는다.
- **토큰 가중치 계산 전체가 `torch.no_grad()` 안이다.** 가중치는 손실에 곱하는
  상수 계수이지 손실의 일부가 아니다 — 그래디언트가 흐르지 않는다.
- **모든 `[B,T]` 통계는 `mask > 0`에서만** 계산되고, 결과는 마스크 밖에서 0이다.

---

## 1. 한 줄 요약과 전체 사슬

STEER는 토큰 하나의 엔트로피가 **지금** 얼마나 변할지를 1차 근사한다(로컬 항 `Ω`).
STEER-F는 거기에 **앞으로** 남은 엔트로피의 예보를 더한다(방문 항):

```
Ω̃      = norm(Ω_local) + λ · norm( Δlogπ̂ · clip(A_H, −c, +c) )
Δlogπ̂  = η · w · (1 − π)
A_H     = H_togo(s_t ⊕ y_t) − H̄_togo(s_t)
H_togo  = Σ_{k=1..κ} γ_H^k · H( p_MTP(y_{t+k} | s) )
```

계산 사슬 (숫자는 이 문서의 절 번호):

```
[4] MTP 헤드 → [5] H_togo → [7] 형제 베이스라인 H̄ → [8] A_H
                                                        │
[9] Ω (STEER 로컬 항) ──[11] 정규화──┐                  │
                                     ▼                  ▼
                          [12] Ω̃ = local + λ·visit ←─ [10] visit
                                     │
                          [13] metric reshaping (minmax│winsor│rank)
                                     │
                          [14] STEER 밴드 매핑 → token_weights [B,T]
                                     │
                          손실:  L = Σ w_t · pg_loss_t / Σ mask   (token-mean)
```

`λ = 0`이면 [10]–[12]가 통째로 생략되고 `mapping="minmax"`(기본)이면 [13]도
항등이므로, **기본 설정의 λ=0은 stock STEER와 비트 단위로 동일**하다.
이것은 희망이 아니라 구조다: `compute_omega_tilde`는 `lam == 0.0`일 때
정규화가 실행되기 *전에* `omega_local` 객체를 그대로 반환한다.

---

## 2. 원자 단위: 엔트로피 `H(l)`

`mtp_heads.entropy_from_logits`

```
lse  = logsumexp(l)              # 스칼라(마지막 축 축약)
p    = softmax(l)
H(l) = lse − ⟨p, l⟩              # [.., V] → [..]
```

**주석.**

- 정의식 `H = −Σ p log p`는 `log p`를 명시적으로 만들어야 하고, `p`가
  언더플로로 0이면 `log 0 = −inf`가 샌다. `log p = l − lse`를 대입하면
  `H = −Σ p(l − lse) = lse − Σ p·l` — 로그를 한 번도 취하지 않고 같은 값을
  얻는다. docstring의 "numerically stable without ever forming log(p)".
- 결과는 부동소수점 오차 한도 안에서 항상 `≥ 0`.

---

## 3. STEER의 로컬 항 `Ω` — 1차 엔트로피 변화 추정

`omega_tilde.local_omega_signed` (verl `core_algos.compute_token_weights`의
산술을 부호 규약만 통일해 재현)

### 3.1 유도

소프트맥스에서 로짓 `z_a`에 대한 엔트로피의 도함수는

```
∂H/∂z_a = −p_a (log p_a + H)
```

한 번의 그래디언트 스텝은 **샘플된 토큰의 로짓만** `η·w`만큼 움직이므로
(Appendix G Step 3: "for non-sampled actions the logits remain unchanged"),
1차 근사로 그 위치의 엔트로피 변화는 계수 × `π(1−π)(log π + H)` 꼴이 된다.

### 3.2 구현된 식

```
x   = clamp(exp(log_prob), 1e−8, 1−1e−8)        # π_θ(y_t|s_t)
f   = x · (1 − x) · (log x + H_t)               # 해석적 1차 계수
                                                # H_t = entropys[b,t] (π_old 패스에서 측정)
coef = A / clamp(exp(old_log_prob), 1e−8, 1)    # released 형태 (기본)
coef = A · r                                    # use_ratio=True (Theorem 1 형태)

Ω_signed = − coef · f
Ω_signed[¬finite] = 0                           # 방문 항을 더하기 *전에* 소독
Ω_signed = Ω_signed · I_clip                    # use_iclip=True일 때만
```

**기호 주석.**

| 항 | 뜻 | 왜 이 형태인가 |
|---|---|---|
| `x(1−x)` | 소프트맥스 야코비안의 대각 성분 | `π≈0`이나 `π≈1`이면 로짓을 밀어도 확률이 안 움직인다 → Ω≈0 |
| `log x + H` | **부호 결정자** | `log π > −H`(평균보다 확률 높은 토큰)를 더 밀면 엔트로피 **하락**, 꼬리 토큰을 밀면 **상승** |
| 선행 `−` | 부호 규약 | verl `asymmetric` 분기와 일치. **Ω > 0 = 이 위치 엔트로피가 오를 것으로 예측** |
| `A / π_old` | released 코드의 계수 | 논문 공개 구현 그대로. λ=0 동치성 테스트의 기준 |
| `A · r` | Theorem 1의 계수 | 기대값을 π_old 표본으로 추정하는 중요도 보정. released와 정확히 `π_θ` 한 겹 차이 |

**stock 두 모드의 복원.** stock `symmetric`은 이 값의 부호를 뒤집은 뒤 곧바로
`abs()`를 취하는데 IEEE-754에서 `|−x| == |x|`가 정확하므로, 부호를 signed
규약으로 통일해도 두 모드 모두 비트 동일하게 복원된다.

**`use_ratio`가 리스케일이 아닌 이유 (docstring의 실측).** `π_θ`는 응답 안에서
수십 자릿수를 오가므로 `1/π_old`는 희귀 토큰을 무한정 부풀린다. 실측
(`docs/omega_forms.json`, 20그룹 pooled): 배치 최댓값을 받는 토큰의 확률이
released에서 `2.0e−6`, theorem에서 `0.30`; max/p99 `6.4 → 3.7`; `w_max` 1%
이내 고정 토큰 `83.4% → 66.7%`; 두 형태의 순위상관 `0.945` — **어느 토큰을
감쇠할지에 대해서도 다른 답**을 낸다. 기본값이 `False`인 유일한 이유는 λ=0
동치성과 논문 수치가 released 기준이기 때문.

### 3.3 클립 지시자 `I_clip`

`omega_tilde.clip_indicator` — Theorem 1이 갖고 있고 공개 구현이 떨어뜨린 인자.

```
r = exp(clamp(log_prob − old_log_prob, −20, +20))

I_clip = 0   if A > 0 and r > 1 + ε_high        # 위로 클립된 토큰
I_clip = 0   if A < 0 and r < 1 − ε_low         # 아래로 클립된 토큰
I_clip = 1   otherwise
```

**주석.** 비율이 신뢰구간을 벗어난 토큰은 PPO 서로게이트가 클리핑해 그래디언트가
0이다 — 그 토큰의 실제 엔트로피 변화는 **정확히 0**. 이 인자가 없으면 움직이지도
않을 토큰이 배치 최대 `|Ω|`를 차지할 수 있고, 그 최댓값은 min-max의 분모이므로
**실제로 움직일 모든 토큰의 감쇠를 약화**시킨다. 첫 inner epoch에서는 `r ≡ 1`이라
전부 1(무효)이고, 미니배치 간 비율 표류가 생겨야 작동한다. `clamp(±20)`은 verl이
다른 곳에서 쓰는 것과 같은 오버플로 방어.

---

## 4. MTP 헤드 — 미래 분포의 예측기

### 4.1 병렬 헤드 (Medusa식, 기본) — `mtp_heads.MTPHeads`

헤드 `k`(0-index)는 위치 `t`의 최종 hidden `h_t`를 읽어 `y_{t+k+1}`의 분포를 예측한다.

```
proj_k(h) = Linear(H→d) → SiLU → Linear(d→H)      # d = head_hidden (기본 1024)
h^k_t     = h_t + proj_k(h_t)                     # residual=True
logits_k  = lm_head(h^k_t)                        # 정책 unembedding 공유 (tie_unembedding)
H_k(t)    = H( logits_k / T )                     # T: 온도, §6의 T_k가 여기 곱해짐
```

**주석.**

- **zero-init + residual이 만드는 초기 조건.** 마지막 `Linear`를 0으로 초기화하면
  `proj_k(h) = 0`이므로 학습 전 모든 헤드가 **정책 자신의 next-token 예측기와
  완전히 동일**하다. 랜덤 초기화라면 `H_k ≈ log V ≈ 11.9`라는 무의미한 상수로
  시작한다. docstring: "sane (if myopic) forecast from step 0 instead of the
  ~log(V) noise of a random head."
- **`[K,B,T,V]`는 절대 만들지 않는다.** `forward()`는 테스트 전용.
  실사용 `forecast_entropy()`는 헤드별·`chunk_size`(기본 4096) 위치별로 로짓을
  만들어 즉시 엔트로피로 접고 해제한다. 추가 피크 메모리는 `chunk_size × V`로
  **T와 무관**. 출력은 fp32 — 엔트로피를 κ개 합산하므로 bf16 누적은 정밀도를 잃는다.
- **온도는 엔트로피 계산 안에 들어간다** (`logits / temperature` 후 H). 이미
  스칼라가 된 엔트로피에서 온도를 되돌릴 수 없기 때문 (§6).

### 4.2 헤드 학습 손실 — `mtp_heads.mtp_ce_loss`

헤드 `k`, 위치 `t`의 타깃은 `labels[:, t+k+1]`:

```
valid_k[b,t] = mask[b,t] · mask[b,t+k+1] > 0      # 소스와 타깃 둘 다 실토큰
CE_k = Σ_{valid} CE(logits_k, y_{t+k+1}) / n_valid_k
L    = Σ_k w_k · CE_k / Σ_k w_k                   # w_k: head_weights (기본 전부 1)
```

**주석.**

- `valid`가 **곱**인 이유: 한쪽만 검사하면 패딩을 정답으로 학습하거나 존재하지
  않는 hidden에서 예측하게 된다.
- 타깃이 시퀀스 밖인 헤드(`offset ≥ T`)는 `nan`을 기록하고 손실에서 빠진다 —
  0을 넣으면 "완벽한 헤드"로 오독된다.
- `ce_loss_inputs()`가 projection과 unembedding을 분리해 두는 것은 fused
  `linear + cross_entropy` 커널(Liger 등)을 쓸 수 있게 하기 위해서다.
  V=152k에서 보조 손실을 감당 가능하게 만드는 지점.
- 정상성 판정: 워밍업의 헤드별 최종 CE는 **k에 대해 단조 증가**해야 한다
  (먼 미래일수록 어렵다). 실측(41.5k 시퀀스, Qwen2.5-1.5B):
  `0.554 / 1.396 / 2.371 / 3.038 / 3.546 / 3.885 / 4.075 / 4.199` — 정상.

### 4.3 인덱스 정렬 — `verl_integration.slice_response_hidden`

**이 코드베이스에서 가장 틀리기 쉬운 한 줄.**

```
verl의 로그확률 슬라이스:  hidden[:, −T−1 : −1]   # index i = responses[i]를 "예측한" 분포
헤드가 필요한 슬라이스:    hidden[:, −T :    ]    # index i = responses[i]를 "소비한 뒤"의 상태
```

hidden `h_t`는 `y_t`를 소비한 뒤 만들어지므로, 헤드의 예보는 자동으로
`H_togo(s_t ⊕ y_t)` — 즉 **샘플된 토큰에 조건부**다. 이것이 `A_H`가 "이 토큰을
골랐을 때"의 점수가 되는 근거이며, 이 +1 오프셋은 코드 전체에서 이 함수 **한
곳**에만 적혀 있다.

### 4.4 순차 헤드 (DeepSeek-V3식 대안) — `mtp_sequential.SequentialMTPHeads`

병렬 헤드는 형제 롤아웃이 같은 `h_t`를 보므로, 분기점에서 형제를 구별하는 정보가
"발산 토큰 하나가 hidden에 남긴 흔적"뿐이다. 순차 헤드는 깊이 `k`에게 실제로
나온 중간 토큰을 먹인다:

```
h⁰_t = h_t + head0(h_t)                                        # 깊이 0은 병렬형과 동일
h^k_t = merged + Block_k(merged),
        merged = W_k [ RMSNorm(h^{k−1}_t) ; RMSNorm(Emb(y_{t+k})) ]
p^k   = softmax( lm_head(h^k_t) )                              # y_{t+k+1} 예측
```

**주석.**

- `Emb`은 정책의 입력 임베딩 공유(`tie_embedding`) — 체크포인트를 줄이고 토큰
  표현을 블록이 이미 아는 공간에 둔다.
- 시퀀스 끝에서 `y_{t+k}`가 없는 위치는 마지막 유효 토큰으로 채워지고 호출자가
  마스크로 걸러낸다 (`h_togo`는 응답이 계속되는 곳에서만 읽힌다).
- **teacher forcing의 개념적 비용** (docstring이 스스로 경고): 실현된
  `y_{t+1..t+k}`를 소비하므로 `H_togo`의 의미가 "정책 하의 **기대** 미래
  엔트로피"에서 "**이 연속열을 따라간** 엔트로피"로 이동한다. 그 방향의 극한이
  §8의 오라클이고 오라클은 헤드가 필요 없다. 따라서 순차 헤드는 "병렬을 이기고
  **동시에** 오라클에 근접"할 때만 파라미터 값을 한다.
  실측(`docs/phase1_rank_test.json`): pooled ρ — oracle 1.000, **parallel
  0.819, sequential 0.796**, untrained 0.711 — 순차형이 병렬을 이기지 못했다.

---

## 5. `H_togo` — 앞으로 남은 엔트로피의 할인 합

`entropy_forecast.h_togo`

```
H_togo^κ(s) = Σ_{k=1..κ}  γ_H^k · H( p_MTP(y_{t+k} | s) )

# 0-index 구현:
weights[k] = γ_H^(k+1),  k = 0..κ−1              # 헤드 k는 오프셋 +k+1 담당
out = clamp_min( Σ_k weights[k] · ent[k], 0 )
```

**기호 주석.**

| 기호 | 뜻 | 주석 |
|---|---|---|
| `κ` | 호라이즌 | `1 ≤ κ ≤ K` 검사. K=8로 학습해 두고 κ는 사후 선택 — 헤드가 독립이라 재학습 불필요 |
| `γ_H` | 할인 | RL 리턴 할인과 같은 역할: 먼 헤드일수록 예보가 부정확(§6)하므로 기하급수적으로 눌러 준다. `γ_H > 0` 필수 |
| `ent` | `[K,...]` 헤드별 엔트로피 | **로짓 `[K,B,T,V]`는 입력으로 받지 않는다** — docstring: "the logit tensor is too large to exist" |
| `clamp_min(0)` | 하한 | 엔트로피는 음수 불가. 음수는 §6 캘리브레이션의 `bias_k`가 과하게 음수일 때만 나오므로 0에서 잘라 해석 가능성을 지킨다. `-inf`를 넘기면 비활성화 |

**스케일 감각.** `γ_H=0.85, κ=4`면 가중치는 `0.85/0.72/0.61/0.52`, 합 ≈2.70.
`H_togo`의 절대값은 평균 엔트로피의 ~2.7배 스케일이며 그 자체로는 의미가 없다 —
**같은 위치의 형제끼리 비교될 때만** 의미가 생긴다 (§7–8).

**직관.** "지금 이 상태에서 앞으로 κ 스텝 동안 정책이 얼마나 갈팡질팡할 것인가"를
스칼라 하나로 압축한 값. 크면 앞길이 열려 있고(다양한 전개 가능), 작으면 앞길이
정해져 있다(수식 하나 마저 쓰는 중).

---

## 6. 헤드 캘리브레이션 — 먼 헤드의 체계적 과대추정 보정

`entropy_forecast.HeadCalibration`, `fit_head_calibration`

헤드 `k`가 `y_{t+k+1}`을 예측할 때, 그 사이에 일어날 수 있는 모든 일이 예측
분포를 뭉갠다 → `k`가 클수록 `H_k`가 `log V` 쪽으로 밀린다. 보정하지 않으면
"먼 미래는 전부 다양해 보이는" 상태가 되어 `A_H`의 변별력이 사라진다
(계획서 §3.4 리스크 표 4행).

```
H_cal_k = scale_k · H( softmax(logits_k / T_k) ) + bias_k

# 적합 (헤드별 최소제곱):
measured_k ≈ scale_k · forecast_k + bias_k
scale_k = Σ(x−x̄)(y−ȳ) / Σ(x−x̄)²      bias_k = ȳ − scale_k·x̄
```

**주석.**

- **세 파라미터의 역할이 다르다.** `T_k`(온도)는 엔트로피 계산 **전에** 로짓에
  들어가야 한다 — 스칼라가 된 엔트로피에서 온도를 되돌릴 수 없다. 그래서
  `HeadCalibration.apply()`는 **일부러 온도를 적용하지 않고** scale/bias만
  적용하며 docstring이 그 사실을 명시한다. `scale/bias`는 온도로 못 잡은
  잔차의 아핀 보정.
- **measured의 정의.** Phase 1에서 `measured_k`는 몬테카를로 값: 프리픽스에서
  연속열 n개를 샘플링하고 각 연속열의 오프셋 `+k+1`에서 정책이 실제 가졌던
  엔트로피를 재서 `nanmean` — **헤드가 맞히려는 바로 그 양**과 직접 비교.
- **온도 탐색** (`fit_temperature=True`): 온도는 사후 적용이 불가능하므로
  `forecast_fn(k, T)`로 재계산하며 그리드(기본 `0.7..2.0`) 탐색, SSE 최소를 채택.
- **퇴화 방어.** `var(x) < 1e−12`(분산 없는 예보)면 `scale=1, bias=ȳ−x̄` —
  0으로 나누지 않는다. 이런 헤드는 상수를 내뱉는 헤드이므로 `A_H`에 기여가 없다.
- 실측 적합값 (Qwen2.5-1.5B, `phase1_queue_v2.log`):
  `scale ≈ [0.99, 0.44, 0.34, 0.34, 0.36, 0.36, 0.42, 0.42]`,
  `bias ≈ [0.00, −0.11, −0.32, −0.56, −0.83, −0.94, −1.29, −1.31]` —
  +1 헤드는 보정이 거의 필요 없고(항등에 가깝고), 멀수록 크게 눌러 내린다.
  과대추정 가설의 실측 확인.

---

## 7. 베이스라인 `H̄_togo(s_t)` — "높다"를 판정할 기준

### 7.1 쌍별 최초 발산 — `entropy_forecast.first_divergence`

```
div(i,j) = min{ t : y_i[t] ≠ y_j[t] }     (없으면 T; 대각선은 항상 T)
```

**마스크 의미론.** 한쪽만 살아 있는 위치(한 롤아웃이 먼저 끝남)는 발산으로
**친다**; 둘 다 죽은 위치는 치지 않는다. bool `argmax`는 all-False 행에서 0을
반환하므로 "never differs → T" 폴백이 명시적으로 있다.

### 7.2 형제 프리픽스 베이스라인 — `sibling_prefix_baseline` (기본, `baseline="sibling"`)

```
sibling(i,j,t)  ⇔  div(i,j) > t−1  ⇔  div(i,j) ≥ t     # [0,t)가 동일하면 형제
H̄_togo_i(t) = Σ_j sibling(i,j,t)·mask_j(t)·H_togo_j(t) / Σ_j sibling·mask
```

**주석.**

- **부등호 하나가 정의를 만든다.** `[0, t)`까지만 같으면 되고 `t`에서는 서로
  **달라도 된다** — 이 엄격한 `< t` 조건이 `H_togo(s_t⊕y_t) − H̄(s_t)`를 분기
  점수로 만든다 (docstring: "exactly what makes the difference a branch score").
- **자기 자신은 항상 형제로 포함**된다 (대각선 강제 True). 형제가 자신뿐이면
  베이스라인 = 자기 값 → `A_H`가 **정확히 0** (계획서 §8.4의 퇴화 케이스).
  긴 응답에서는 첫 발산 이후 거의 전 위치가 여기 해당한다 — 실측 `T=1024`에서
  99.4%. 이 점질량이 §16의 순위 선택과 `sibling_support`의 존재 이유다.
- **자기 포함의 스케일 효과.** 형제 n명일 때
  `A_H_i = (1 − 1/n)(h_i − 나머지 평균)` — 편차가 `(1−1/n)`배 축소된다. 순위는
  보존되므로 랭킹 지표에는 무해하나, `clip_c`가 무는 절대 스케일은 형제 수에
  의존한다.
- 살아 있는 형제만 기여한다 (`sibling ∧ mask_j`).

### 7.3 그룹 평균 베이스라인 — `group_mean_baseline` (`baseline="group"`, Ablation A5의 B팔)

```
H̄(t) = Σ_{j∈그룹, alive at t} H_togo_j(t) / #alive
```

프리픽스 일치를 요구하지 않는 값싼 근사. 두 가지 방어:

- **살아 있는 롤아웃만** 평균에 넣는다 — 짧은 롤아웃의 마스크 0을 그냥 합치면
  베이스라인이 뒤로 갈수록 0으로 끌려 `A_H`에 "뒤쪽일수록 양수"라는 가짜 추세가 생긴다.
- `t`의 생존자가 자신뿐이면 자기 값을 베이스라인으로 (빈 집합 대비 점수 방지).

### 7.4 형제 support — `sibling_support`

```
support[i,t] = ( #(alive, prefix-matched siblings of i at t, self 포함) ≥ 2 ) ∧ mask
```

`A_H`가 정의되는 영역(자신 외 형제가 있는 곳)의 마스크. **선택과 귀무 양쪽에**
이 마스크를 걸지 않으면, top-decile 검정은 "A_H가 어디서 정의되는가"를 잴 뿐
"A_H가 어떻게 순위 매기는가"를 재지 않는다 — 실측: 미학습 헤드 recall 0.4711 vs
학습 헤드 0.4693, 즉 예보에 대한 정보 0 (docstring에 수치로 박제).

---

## 8. 분기 점수 `A_H` 와 오라클 대조군

### 8.1 `A_H` — `entropy_forecast.entropy_advantage`

```
A_H(s_t, y_t) = H_togo(s_t ⊕ y_t) − H̄_togo(s_t)
a_h = (h_togo_vals − base) · mask        # clip은 여기서 하지 않는 것이 관례
                                          # (compute_token_weights_steerf가 담당)
```

**부호 읽기.** `A_H > 0`: 내가 고른 갈래의 앞길이 형제 평균보다 다양하다.
`A_H < 0`: 막다른 갈래이고 포기된 형제 쪽에 다양성이 있었다. GRPO 어드밴티지와
같은 문법이며 실제로 같은 `uid` 그룹 키를 쓴다.

**구성상 zero-centred.** 형제 평균에서의 편차이므로 한 분기점의 형제들 합은 0 —
§15의 signed 보정이 "분기 토큰의 평균 가중치를 옮길 수 없고 퍼뜨리기만 한다"는
주장이 여기서 나온다.

### 8.2 오라클 — `entropy_forecast.oracle_h_togo`

```
H_togo^oracle(t) = Σ_{k=1..κ} γ_H^k · H_policy(t+k) · mask(t+k)
```

RLVR은 롤아웃이 끝난 뒤 업데이트하므로 `t+1..t+κ`의 실제 정책 엔트로피는 이미
알려져 있다 — verl이 같은 패스에서 `entropys[B,T]`를 통째로 계산해 둔다.
**헤드도, 워밍업도, 추가 forward도 없이 공짜로** 얻는 대조군이며, docstring:
"It is the *control* the forecast has to beat."

- 오라클 = **실현된 한 표본**의 엔트로피, MTP = **기대값**의 추정 — 헤드가 살 수
  있는 것은 이 차이뿐이다.
- 끝 부분 편향: 응답 끝에서 κ 안쪽은 누락항이 0으로 들어가 아래로 치우친다.
  형제 베이스라인이 같은 `t`끼리 비교하므로 형제 간에는 공통이라 상쇄된다.

---

## 9. 방문 항 — `Δlogπ̂` 와 `visit`

`omega_tilde.delta_logpi_hat`, `visit_term`

```
w       = clamp(r, 1−ε_low, 1+ε_high) · A      # 정책 손실이 실제로 쓸 계수와 동일한 클리핑
Δlogπ̂  = η · w · (1 − π)                      # 한 스텝이 샘플 토큰의 log-prob을 미는 양
visit   = Δlogπ̂ · clip(A_H, −c, +c)
```

**주석.**

| 항 | 근거 |
|---|---|
| `η·w` | 한 그래디언트 스텝이 샘플 토큰의 **로짓**을 미는 크기 |
| `(1−π)` | 소프트맥스 야코비안 `∂log p_a/∂z_a = 1 − p_a` — 로짓 변화를 log-prob 변화로 환산. `π≈1`이면 더 몰아줄 여지가 없어 자동으로 0 |
| `η` | 정규화(§11)가 스케일을 지우므로 **λ와의 곱으로만** 의미가 있다 → 1.0에 두고 λ만 스윕 (config docstring) |
| `clip(A_H)` **먼저** | 예보가 아무리 틀려도 한 토큰의 영향은 `\|η·w\|·c`로 유계. 곱한 뒤 자르면 `w` 큰 토큰에서 예보 오차가 그대로 증폭된다 |

**부호 4분면 (붕괴 채널의 정의).**

| `Δlogπ̂` | `A_H` | 해석 |
|---|---|---|
| >0 | >0 | 앞이 다양한 갈래로 질량 집중 → 궤적 엔트로피 상승 |
| >0 | **<0** | **막다른 갈래로 집중, 포기된 형제의 다양성 삭제 — 로컬 Ω가 원리적으로 볼 수 없는 붕괴 경로. STEER-F의 존재 이유** |
| <0 | — | 부호가 뒤집혀 같은 논리가 반대로 |

---

## 10. `Ω̃` — 두 항의 결합

`omega_tilde.compute_omega_tilde`

```
if λ == 0:  return Ω_local (그대로, 산술 없음)      # 구조적 비트 동치

visit    = Δlogπ̂ · clip(A_H, ±c)
Ω̃       = normalize(Ω_local) + λ · normalize(visit)
```

로깅되는 진단:

```
steerf/visit_rel_mag  = mean|λ·visit_n| / (mean|local_n| + 1e−8)   # 미래 항의 상대 크기
steerf/a_h_clip_frac  = mean( |A_H| ≥ c )                          # 클립 포화율
```

포화율이 높으면 `A_H`가 상수 `±c`가 되어 분기를 구별하지 못한다 → `c` 완화 신호.

---

## 11. 정규화가 z-score가 아니라 RMS인 이유

`omega_tilde.normalize` — 이 저장소에서 docstring이 가장 길게 논증하는 지점이며
결론이 계획서(`z_norm`)와 다르다.

```
"scale" (기본):  x / sqrt(mean_{mask}(x²))         # RMS. 0 보존
"z"           :  (x − mean) / std                   # 중심이동 발생
"none"        :  x
# 퇴화(분모 < eps): 전부 0 반환 — 상수 지표는 어차피 전 토큰 동일 가중치이므로 잃는 정보 없음
```

**세 제약과 그 교집합.**

| STEER 매핑 | 불변인 변환 | 이유 |
|---|---|---|
| `linear=True` | 아핀 `a·x+b` (a>0) | min-max 재스케일이라 원점 무의미 |
| `linear=False` | **양의 스케일 `a·x`만** (0.02 하한이 안 물릴 때) | `exp`는 평행이동 불변이 아님 |
| `mode="symmetric"` | 추가로 **중심이동 불가** | `abs()`가 붙어 0이 의미 있는 축 |

교집합 = **중심이동 없는 양의 스케일링** → RMS. z를 쓰면 λ=0인데도 가중치가
달라져 "λ=0은 stock STEER" 기준선이 무너진다 —
`tests/test_lambda_zero_equiv.py::test_z_norm_would_break_symmetric_equivalence`가
실증. `z`는 선택 가능하되 `SteerFConfig.validate()`가 `symmetric`·`linear=False`
조합에서 경고를 낸다. `norm="none"` + `λ>0`도 경고 — 스케일이 무관한 두 항을
그대로 더하면 한쪽이 지배하거나 소멸한다.

---

## 12. STEER의 밴드 매핑 — `Ω̃`가 가중치가 되는 곳

`compute_token_weights_steerf`의 후반부. **stock verl과 산술이 동일한 구간**이며,
논문의 밴드 `[ΔH_low, ΔH_high]`와 이산 `α ∈ {γ, 1, 1/γ}`는 코드에 존재하지 않는다
(`docs/steer_code_map.md` §3). 실제로 있는 것은 마이크로배치 통계 기반 연속 매핑 둘:

```
metric = |Ω̃|          if mode == "symmetric"     # 0이 축이 되는 지점
metric = Ω̃            if mode == "asymmetric"

# ---- linear=True: 마이크로배치별 min-max ----
scale = (w_max − w_min) / (metric_max − metric_min)
symmetric :  w = w_max − (metric − metric_min)·scale    # |변화| 클수록 감쇠 (내림차순)
asymmetric:  w = w_min + (metric − metric_min)·scale    # Ω̃ 클수록 가중 (오름차순)
w = clamp(w, w_min, w_max)
# metric_max == metric_min 이면 전 토큰 = (w_min+w_max)/2

# ---- linear=False: 지수 매핑 ----
symmetric :  k = −log(w_min)/max(metric_max, 0.02);   w = exp(−k·metric), clamp[w_min, 1.0]
asymmetric:  k = −log(w_min)/max(|metric|max, 0.02);  w = exp(+k·metric), clamp[w_min, w_max]
```

**주석.**

- `metric_min/max`가 **마이크로배치마다** 새로 계산된다 → 임계값이 절대
  스케일이 아니라 배치 상대적이고, `ppo_micro_batch_size_per_gpu`가
  **하이퍼파라미터**가 된다 (전 팔 고정 필수).
- `symmetric` 지수 매핑의 상한은 `w_max`가 아니라 **하드코딩 `1.0`** — 이
  조합에서 STEER는 감쇠 전용이다 (원본 `run_exp.sh`가 `token_weight_max=1.0`을
  넘기는 이유).
- `0.02` 하한은 상수 지표에서 `k → ∞` 발산을 막는 방어.

**알려진 병리 (이 문서의 §13이 존재하는 이유).** `Ω`는 `1/π_old`(하한 1e−8)
때문에 꼬리가 극단적으로 두껍고, min-max는 그 배치의 이상치 **두 개**에 밴드
양 끝을 내준다. 실측: `max|Ω|/median|Ω| ≈ 3.4e4`, 학습 로그의 토큰 가중치
`tw_std ≈ 0.001` (0.1폭 밴드의 1%) — **STEER가 계산한 순서 정보가 손실까지
도달하지 못한다.** 같은 진단이 분기 항에 대해서는 `branch_weight_correction`
docstring에 이미 있었고("Ω is heavy-tailed … the bulk of tokens land within
~5% of the weight range"), 로컬 항 본체도 같은 병을 갖는다는 것이 학습 로그로
확인됐다.

---

## 13. 지표 재성형 — `mapping ∈ {minmax, winsor, rank}` (신규)

`omega_tilde.reshape_metric` — §12의 병리에 대한 처방. 매핑이 보는 지표를
매핑 **직전에** 바꾼다. `compute_token_weights_steerf` 안에서 단 한 줄:

```
metric = reshape_metric(metric, mask, mapping=mapping, winsor_q=winsor_q, mode=mode)
```

### 13.1 `"minmax"` — 항등 

stock 동작. **기본값인 이유는 λ=0 동치성 테스트가 이것을 기준으로 비교하기
때문**이다. `mapping != "minmax"`인 팔은 λ=0이어도 더 이상 stock STEER가
아니며(그게 이 옵션의 목적), `validate()`가 그 사실을 경고 문자열로 알린다 —
그 팔의 λ=0은 **제2의 기준선**이지 stock 기준선이 아니다.

### 13.2 `"winsor"` — 꼬리만 자르는 최소 수정

```
lo = quantile(metric_valid, q)          # 기본 q = 0.01
hi = quantile(metric_valid, 1−q)
metric ← clamp(metric, lo, hi)
```

벌크의 모양은 그대로 두고, min-max의 분모를 쥐고 있는 꼬리의 손아귀만 끊는다.
분위수는 정렬 인덱스로 뽑는다(`_quantile_sorted`) — `torch.quantile`은 ~16M
원소 위에서 거부하므로, 마이크로배치는 한참 아래지만 오프라인 pooled 배치에서도
그대로 쓸 수 있게.

### 13.3 `"rank"` — 순위 매핑 (기본)

```
r_i = avg_rank(metric_i) / (n−1)        ∈ [0, 1]        # 동점은 평균 순위
metric ← r          if mode == "symmetric"
metric ← 2r − 1     if mode == "asymmetric"
```

**주석.**

- 밴드 점유가 **구조적으로 균등**해지고 꼬리에 완전 면역이다. 비용: 크기 정보를
  전부 버린다 — 1등과 2등 `|Ω|`의 간격이 벌크 인접 토큰의 간격과 동일 취급.
- **동점 평균 순위**(`_average_ranks`)가 필수인 이유: `Ω`에는 마스크·클립으로
  생기는 `0` 점질량이 있다. `argsort` 순위를 그대로 쓰면 값이 같은 토큰들이
  메모리 배치 순서라는 임의 기준으로 서로 다른 가중치를 받는다. 평균 순위는
  같은 값 → 같은 가중치를 보장한다.
- **범위 선택의 근거.** `symmetric`은 지표가 비음수이고 0이 의미 있는 바닥이므로
  순위를 `[0,1]`에 둔다. `asymmetric`은 부호 있는 지표라 `[−1,1]`로 중심을
  유지한다. `linear=True` 아래에서는 이 선택이 보이지 않지만(민맥스는 아핀
  불변), `linear=False`(지수)는 평행이동에 민감하므로 중요하다 — 그래서
  `validate()`가 `rank + linear=False` 조합에 "감쇠 중심이 배치 중앙값으로
  이동한다"는 경고를 낸다.
- `n = 1`이면 순위가 정의되지 않으므로 0을 반환한다(그 배치는 어차피 중점 처리).

### 13.4 config 연결

`SteerFConfig`에 두 필드 추가:

```
mapping:  "minmax" (기본) | "winsor" | "rank"
winsor_q: 0.01                     # winsor일 때 각 꼬리에서 자르는 분위
```

`validate()`의 추가 규칙: `mapping` 값 검사(예외), `0 ≤ winsor_q < 0.5`(예외),
`mapping != "minmax"` 경고(제2 기준선), `rank + linear=False` 경고.

---

## 14. 전체 조립 — `compute_token_weights_steerf`

verl `compute_token_weights`의 드롭인 대체. 시그니처 앞부분(stock 인자)은 동일,
STEER-F 인자는 전부 기본값이 "stock과 동일"이 되도록 잡혀 있다.

```
1. iclip  = clip_indicator(...)   if use_iclip else None          # §3.3
2. Ω      = local_omega_signed(A, H, logp_old, logp; iclip, use_ratio)   # §3.2
3. λ>0 이고 a_h 가 주어졌으면:
     w  = clamp(r, 1−ε_l, 1+ε_h) · A
     π  = clamp(exp(logp), 1e−8, 1−1e−8)
     apply="metric":  metric = Ω̃ = norm(Ω) + λ·norm(Δlogπ̂·clip(A_H))    # §10
     apply="weight"|"branch": metric = Ω 그대로, visit은 매핑 뒤로 보류    # §15
   아니면: metric = Ω
4. mode="symmetric" 이면 metric = |metric|
5. metric = reshape_metric(metric, mask, mapping, winsor_q, mode)         # §13
6. STEER 밴드 매핑 (stock 산술 그대로)                                    # §12
7. 보류된 visit이 있으면 branch_weight_correction                          # §15
8. return token_weights [B,T] float32 (+ stats)
```

`apply` 세 값의 의미:

| `apply` | 분기 신호가 들어가는 곳 | 성격 |
|---|---|---|
| `"metric"` | 지표에 더해져 min-max를 **같이** 통과 | 계획서 원안. 실측으로는 min-max가 신호를 뭉갬 (`docs/weight_forms.json` — 4개 지표-수준 정식화 전부 동일하게 실패) |
| `"weight"` | 매핑이 끝난 가중치에 tanh 보정 (signed) | 분기 신호가 Ω 꼬리와 경쟁하지 않음 |
| `"branch"` | 동일하되 uniform 모드 | **예보 없는 귀무가설 팔** — 아래 §15 |

---

## 15. 가중치 공간 보정 — `branch_weight_correction`

`apply="weight"` / `"branch"` 전용. STEER 가중치는 stock 그대로 계산하고,
그 **위에** 유계 보정을 더한다.

```
support = mask ∧ (visit ≠ 0)
rms     = sqrt( mean( visit[support]² ) )                  # support에서만!

signed :  w ← clamp( w + λ·(w_max−w_min)·tanh(visit/rms), w_min, w_max )
uniform:  w ← clamp( w − λ·(w_max−w_min)·𝟙[support],      w_min, w_max )
```

**주석.**

- `tanh`가 `(−1,1)`로 묶으므로 한 토큰의 조정폭 상한은 `λ·(w_max−w_min)` —
  조정 크기가 **Ω 꼬리의 모양이 아니라 λ만으로** 결정된다. `λ=0.5`면 밴드의 절반.
- **rms를 support에서만** 잡는 이유: `A_H`는 형제 없는 위치에서 정확히 0이고
  그게 대다수다. 전체 rms는 신호 크기가 아니라 희소성으로 나누는 꼴이 된다.
- **signed는 변별, uniform은 보호.** signed는 `A_H`가 zero-centred이므로 분기
  토큰들의 **평균** 가중치를 못 옮기고 서로 **벌리기만** 한다 (실측: 평균차
  +0.002, 가중치 std 0.0034→0.0043). uniform은 "형제가 남아 있는 위치는 전부
  같은 양만큼 감쇠" — **예보도 MTP 헤드도 Phase 1도 필요 없는** 더 약한 가설이다.
  uniform이 signed와 같은 성능을 내면 예보는 효과를 만들고 있지 않은 것 —
  방법 자체에 내장된 반증 장치.
- 퇴화 방어: `λ=0`, 빈 support, rms 비유한/미소 → 가중치 무변경 + 통계 0.

---

## 16. 학습 중 모니터와 λ 안전밸브

`monitors.py`

### 16.1 예보 드리프트 — `mtp_policy_kl`

```
KL(policy ‖ head₁) = Σ_v p_v (log p_v − log q_v)      # 위치 무작위 subsample 평균
```

+1 헤드와 정책은 **같은 확률변수**(다음 토큰)를 기술하므로 정렬 보정이 필요 없는
직접 비교다. 값이 커진다 = 정책은 움직였는데 예보기가 못 따라왔다 = `A_H`가
현재 정책을 설명하지 않는다. subsample(기본 4096 위치)인 이유: 전 위치에서
V-차원 분포 두 개를 만드는 것이 정확히 STEER-F가 피하려는 비용이라서.

### 16.2 λ 감쇠 컨트롤러 — `LambdaDriftController`

```
kl_ema ← ema·kl_ema + (1−ema)·kl                       # ema=0.9
kl_ema > threshold 가 patience(2)번 연속  →  λ ← λ·decay(0.5)
λ ≤ lam_min(0.01)  →  λ ← 0                            # 깔끔한 "완전 꺼짐" 상태
```

계획서의 "임계 넘으면 절반"에 두 장치를 추가: **EMA + patience**(노이즈 배치
하나가 λ를 깎지 못하게), **자동 복구 없음**(운 좋은 배치에 λ가 되살아나면 런이
재현 불가 — 복구는 명시적 `reset()`뿐).

### 16.3 순위 기반 top-k — `_top_k_selection`

```
sel[ topk(values, k).indices ] = True          # 정확히 k개
# (임계값 방식 values ≥ kth 는 금지)
```

`A_H`의 `0` 점질량 때문에 임계값 방식은 상위 10분위가 실제로는 96.7–99.7%를
삼킨다(실측, T=96/512/1024). 그러면 G1 이항검정은 귀무 선택률 0.997로 유의
불가능, G2의 branch/nonbranch 분할은 99.7% 대 0.3%로 무의미해진다. 순위 선택이
둘 다 고친다. 동점 절단은 `torch.topk`의 결정적이지만 임의인 인덱스 순서 —
`A_H`만으로는 더 나은 답이 없다.

### 16.4 나머지 지표

| 지표 | 식 | 잡는 실패 |
|---|---|---|
| `branch_entropy_gap` | mean H(top-10% A_H) − mean H(나머지) | G2 메커니즘 확인: 전체 엔트로피가 평평해도 이 값이 λ=0보다 높아야 "분기점에서만 보존" |
| `tw_frac_{low,mid,high}` | `[w_min,w_max]` 3등분 점유율 | 이산 α가 없으므로 계획서 "γ/1/1-over-γ 비율"의 대응물. 한쪽 90%+ 쏠림 = 재가중이 사실상 상수 |
| `branch_recall_at_k` | 진짜 분기점(∃j: div=t) 중 top-decile에 든 비율, `lift = recall/top_frac` | 학습 중 A_H 유효성. **support를 걸어야** 랭킹 검정이 된다 (§7.4) |

---

## 17. 게이트 G1 통계

`validation.py` — 판정 규칙을 GPU 스크립트가 아니라 **테스트되는 모듈**에 두는
것이 설계 원칙이다 (docstring: "the pass/fail rule is the experiment's
contract with itself").

### 17.1 문제-내 Spearman + Fisher z

```
문제 p마다 (n_p ≥ 5):   ρ_p = spearman(forecast, truth)     # 동점 평균 순위
ρ̄ = tanh( Σ_p n_p·atanh(clamp(ρ_p, ±0.999999)) / Σ_p n_p )
```

- **문제별로 쪼개는 이유**: pooling은 문제 간 난이도 차이로 상관을 부풀린다 —
  쉬운 문제는 둘 다 낮고 어려운 문제는 둘 다 높아, 문제 안에서 아무것도 구별
  못 해도 큰 ρ가 나온다. 실제 결정(어느 토큰을 감쇠할까)은 **문제 안에서**
  내려진다. `tests/test_validation.py`에 Simpson 반전 실증(문제마다 −1인데
  pooled +0.8)이 있다.
- **Fisher z**: 상관은 가법적이지 않다. `atanh` 공간에서 n 가중 평균 후 복귀.
  `nan`(상수 입력)은 버리고, `ρ=±1`은 도메인 안으로 밀어 한 문제가 평균을
  무한대로 보내지 못하게 한다.
- 지상 진실은 **(κ, γ_H) 셀마다 같은 할인을 적용해** 재계산한다 — 할인 예보를
  무할인 실측으로 채점하면 `γ_H < 1`이 예보 품질과 무관하게 전부 벌점을 받는다
  (계획서 §3.3 수정 사항, `gt_sum_h`로 원안도 병기).

### 17.2 정확 이항검정 — `binomial_test_greater`

```
p = Σ_{j=n_hit..n_branch} C(n,j)·p₀^j·(1−p₀)^(n−j)      # lgamma로 로그공간 합산
p₀ = 실현 선택률 (명목 0.1이 아님!)
```

- 정확 검정인 이유: Phase 1의 `n_branch`는 작고, 정규근사는 그 영역에서
  낙관적이다.
- 귀무가 실현 선택률인 이유: 반올림·동점으로 실제 top-decile 크기가 0.1에서
  벗어나며, 명목값 검정은 판정을 편향시킨다.

### 17.3 판정 — `evaluate_gate_g1`

```
G1 = ( ρ̄ ≥ 0.2 )  AND  ( p < 0.05  AND  recall > 선택률 )
```

`n_branch < 30`이면 note 부착 — 그때의 FAIL은 "미래 항이 잡음"이 아니라
"표본 부족"이다. **알려진 한계**: support 안에서 `A_H ≠ 0 ⇔ 분기점`이므로 조건
2는 현재 형태로는 예보기와 무관하게 같은 recall을 낸다(학습/미학습 완전 동일
실측). 귀무를 "`A_H ≠ 0`인 위치"로 좁혀 분기점 **사이의** 순위를 재는 형태로
바꿔야 하며, 그 형태의 검정은 `scripts/phase1_rank_test.py`(§4.4)와
`scripts/analyze_ah_vs_correctness.py`에 이미 있다.

### 17.4 (κ, γ_H) 선택 — `select_kappa_gamma`

```
① 전역 최고 ρ의 elbow_tolerance(0.01) 안에 드는 γ 중 최고를 고른다
② 그 γ 안에서, 그 γ 최고 ρ의 tolerance 안에 드는 가장 작은 κ
③ 단 κ ≥ min_kappa = 2
```

**`min_kappa=2`가 하드 제약인 이유**: 헤드 0은 정책의 next-token 예측기
복사본으로 초기화되고(§4.1), 오프셋 +1의 측정값이 바로 그 엔트로피다 —
`κ=1`의 상관은 예보가 아니라 **항등식**이다. 실측 `ρ=0.997`(κ=1, 모든 γ에서 —
κ=1이면 γ가 순위상관에서 소거된다) vs `0.71–0.84`(κ≥2). 무제약 엘보는 항상
κ=1에 착지하고, 그걸 고르면 `A_H`가 "로컬 엔트로피의 형제 평균 대비 편차"로
축소된다 — STEER-F가 고치려던 근시안 그 자체. 실제 Phase 1 선택: **κ=2, γ_H=0.7**.

---

## 18. 설정 요약 — `SteerFConfig`

| 필드 | 기본 | 뜻 | 비고 |
|---|---|---|---|
| `lam` | 0.0 | 방문 항 가중. **0 = stock STEER와 비트 동일** | env `STEERF_LAM` |
| `eta` | 1.0 | `Δlogπ̂`의 스케일 | 정규화 후 λ와의 곱으로만 유효 — 고정하고 λ 스윕 |
| `clip_c` | 1.0 | `A_H` 대칭 클립 (nats) | 포화율은 `a_h_clip_frac`으로 감시 |
| `norm` | `"scale"` | RMS / z / none | z는 `asymmetric`+`linear=True`에서만 안전 (§11) |
| `mapping` | `"minmax"` | 매핑 직전 지표 재성형 (§13) | minmax 외에는 λ=0도 stock이 아님 (경고) |
| `winsor_q` | 0.01 | winsor의 꼬리 분위 | `[0, 0.5)` |
| `baseline` | `"sibling"` | H̄의 정의 (§7) | `"group"`은 A5의 B팔 |
| `kappa` | 4 | 호라이즌 | Phase 1 실선택은 2 — 실행 스크립트가 export해야 전달됨 |
| `gamma_h` | 0.85 | 할인 | Phase 1 실선택은 0.7 |
| `beta_mtp` | 0.05 | MTP 보조 손실 가중 | **아직 학습 루프 미연결** — 현재 헤드는 freeze (A7 조건) |
| `kl_drift_threshold` | 0.5 | λ 감쇠 트리거 (nats) | §16.2 |
| `lam_decay` / `lam_min` | 0.5 / 0.01 | 감쇠율 / 완전 꺼짐 문턱 | 자동 복구 없음 |

---

## 19. 알려진 실증 상태 (2026-08 기준, 요약)

방법 문서이므로 결과는 한 단락만: 소형(1.5B, MATH500, 100스텝, 1시드) λ 스윕에서
pass@16은 λ에 단조 감소했고 G2 미달, 원인 후보는 λ가 아니라 §12의 매핑 병리
(`tw_std ≈ 밴드의 1%`)로 특정됐다 — §13의 `mapping` 옵션이 그 처방이다.
`A_H`와 정답의 정렬은 **음(−)** 이었다(oracle 승률 0.355, 우연 0.5): 미래
엔트로피가 높은 갈래가 오답 쪽에 있었다. 상세와 수치 전부는 로그 해석 문서와
`docs/{omega_forms, weight_forms, ah_vs_correctness, phase1_rank_test}.json`.
