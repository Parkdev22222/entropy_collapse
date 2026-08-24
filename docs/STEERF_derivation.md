# STEER-F 유도 — 새로 근사한 부분

이 문서는 **STEER-F가 stock STEER 위에 새로 도입한 근사**만을 1차 원리에서 유도한다.
구현 세부(텐서 규약, config 연결, 코드 대응)는 [`STEERF_method.md`](STEERF_method.md)에 있고,
여기서는 *왜 그 식이어야 하는가*와 *각 근사가 무엇을 버리는가*만 다룬다.

표기는 `STEERF_method.md` §0을 따른다.

---

## 0. 결론 미리보기

궤적 엔트로피의 그래디언트는 **정확히** 두 항으로 갈라진다.

$$
\nabla_\theta \mathcal{H}(\pi_\theta \mid x)
= \underbrace{\mathbb{E}\Big[\textstyle\sum_t \nabla_\theta H(\pi_\theta(\cdot \mid s_t))\Big]}_{(\mathrm{L})\ \text{local}}
+ \underbrace{\mathbb{E}\Big[\textstyle\sum_u \nabla_\theta \log \pi_\theta(y_u \mid s_u)\, A_H(s_u, y_u)\Big]}_{(\mathrm{V})\ \text{visitation}}
$$

**(L) 로컬 채널**이 stock STEER가 잡는 것, **(V) 방문 채널**이 STEER-F가 추가하는 것이다.

(V)는 근사가 아니라 **항등식**이다. STEER-F의 근사는 전부 "(V)를 롤아웃 없이 계산 가능한 양으로 바꾸는" 과정에서만 발생한다.

---

## 1. 출발점: 궤적 엔트로피의 연쇄 분해

프롬프트 $x$, 응답 $y = y_{1:T}$, 상태 $s_t = (x, y_{<t})$. 정책이 응답 전체에 부여하는 분포의 엔트로피를

$$
\mathcal{H}(\pi_\theta \mid x) \;=\; -\sum_{y} \pi_\theta(y \mid x) \log \pi_\theta(y \mid x)
$$

로 두자. 자기회귀 분해 $\pi_\theta(y \mid x) = \prod_t \pi_\theta(y_t \mid s_t)$ 에 엔트로피 연쇄법칙을 적용하면

$$
\boxed{\;\mathcal{H}(\pi_\theta \mid x) \;=\; \mathbb{E}_{y \sim \pi_\theta}\Big[\sum_{t=1}^{T} H\big(\pi_\theta(\cdot \mid s_t)\big)\Big]\;}
\tag{1}
$$

이것은 **정확한 항등식**이다. 여기서 $H(\pi_\theta(\cdot \mid s_t)) = -\sum_a \pi_\theta(a \mid s_t)\log \pi_\theta(a \mid s_t)$ 는 코드가 로짓에서 직접 재는 per-token 엔트로피(`entropys`)다.

식 (1)이 이 문서 전체의 축이다. 좌변은 우리가 지키고 싶은 것(응답 다양성), 우변은 우리가 잴 수 있는 것(토큰별 엔트로피)이며, **기댓값 $\mathbb{E}_{y\sim\pi_\theta}$ 가 $\theta$ 에 의존한다**는 사실이 STEER-F의 존재 이유 전부다.

---

## 2. 두 채널의 분리 (항등식)

$f_\theta(y) := \sum_t H(\pi_\theta(\cdot \mid s_t(y)))$ 로 두면 $\mathcal{H} = \mathbb{E}_{y\sim\pi_\theta}[f_\theta(y)]$ 이고, 피적분함수와 측도가 **둘 다** $\theta$ 에 의존하므로 곱미분(score-function 항등식)이 적용된다.

$$
\nabla_\theta \mathcal{H}
= \underbrace{\mathbb{E}\big[\nabla_\theta f_\theta(y)\big]}_{(\mathrm{L})}
+ \underbrace{\mathbb{E}\big[f_\theta(y)\,\nabla_\theta \log \pi_\theta(y \mid x)\big]}_{(\mathrm{V})}
\tag{2}
$$

**(L) 로컬 채널.** 방문하는 상태를 고정한 채, 각 상태의 조건부 분포가 평평해지거나 뾰족해지는 효과. stock STEER의 $\Omega$ 가 추정하는 것이 정확히 이것이다 (`STEERF_method.md` §3).

**(V) 방문 채널.** 각 상태의 엔트로피를 고정한 채, **어떤 상태를 방문할 확률이 바뀌는** 효과. 업데이트가 질량을 어떤 갈래로 몰아주면 그 갈래 이후의 상태만 자주 방문하게 되고, 포기된 형제 갈래가 담고 있던 다양성은 궤적 분포에서 사라진다. **$\Omega$ 는 이 항을 원리적으로 볼 수 없다** — $\Omega$ 는 각 위치의 조건부 엔트로피만 보고, 그 위치에 *도달할 확률*은 보지 않기 때문이다.

> stock STEER가 로컬 엔트로피를 지켜내면서도 응답 다양성이 붕괴할 수 있는 이유가 이것이다. 식 (2)의 두 항은 부호가 반대일 수 있다.

---

## 3. (V) 를 entropy-to-go 형태로 정리

$\nabla_\theta \log \pi_\theta(y \mid x) = \sum_u \nabla_\theta \log \pi_\theta(y_u \mid s_u)$ 를 대입하면

$$
(\mathrm{V}) = \sum_t \sum_u \mathbb{E}\big[\,H(\pi_\theta(\cdot \mid s_t))\, \nabla_\theta \log \pi_\theta(y_u \mid s_u)\,\big]
$$

**인과성에 의한 소거.** $H(\pi_\theta(\cdot \mid s_t))$ 는 $y_{<t}$ 만의 함수다. $u \ge t$ 이면 $y_{<t}$ 는 $y_{<u}$ 에 대해 가측이고

$$
\mathbb{E}\big[\nabla_\theta \log \pi_\theta(y_u \mid s_u) \,\big|\, y_{<u}\big] = \sum_a \pi_\theta(a\mid s_u)\,\frac{\nabla_\theta \pi_\theta(a \mid s_u)}{\pi_\theta(a\mid s_u)} = \nabla_\theta \sum_a \pi_\theta(a \mid s_u) = 0
$$

이므로 반복기댓값에 의해 해당 항은 정확히 0이다. $u < t$ 만 남는다:

$$
(\mathrm{V}) = \mathbb{E}\Big[\sum_u \nabla_\theta \log \pi_\theta(y_u \mid s_u) \underbrace{\sum_{t > u} H\big(\pi_\theta(\cdot \mid s_t)\big)}_{\textstyle =:\; H_{\text{togo}}(s_u \oplus y_u)}\Big]
\tag{3}
$$

**$H_{\text{togo}}$ 는 정의를 고른 것이 아니라 유도의 결과다.** (V)는 "보상 = 앞으로 남은 조건부 엔트로피의 합"인 REINFORCE 추정량과 정확히 같은 모양이며, 그 reward-to-go가 $H_{\text{togo}}$ 다.

**베이스라인.** 상태만의 함수 $b(s_u)$ 는 위와 같은 이유로 기댓값에 기여하지 않는다:
$\mathbb{E}[\nabla_\theta \log \pi_\theta(y_u\mid s_u)\, b(s_u)] = 0$. 따라서 **편향 없이**

$$
\boxed{\;(\mathrm{V}) = \mathbb{E}\Big[\sum_u \nabla_\theta \log \pi_\theta(y_u \mid s_u)\; A_H(s_u, y_u)\Big],\qquad
A_H(s_u, y_u) := H_{\text{togo}}(s_u \oplus y_u) - \bar{H}_{\text{togo}}(s_u)\;}
\tag{4}
$$

여기까지 근사는 **하나도 없다.** $A_H$ 의 정의와 "형제 평균을 뺀다"는 선택은 분산 감소를 위한 베이스라인의 표준 논증에서 그대로 나온다.

**형제 프리픽스 베이스라인이 적법한 이유.** $b$ 는 $s_u$ 만의 함수여야 한다. 같은 프롬프트 그룹에서 $y_{<u}$ 가 **완전히 일치하는** 롤아웃들은 정의상 같은 $s_u$ 를 공유하므로, 그들의 $H_{\text{togo}}$ 평균은 $s_u$ 만의 함수다 (`sibling_prefix_baseline`. 조건이 $y_{<u}$ 일치이지 $y_{\le u}$ 일치가 아닌 것이 핵심 — $u$ 에서는 갈라져야 그 차이가 분기 점수가 된다).

> **미세 편향 하나.** 구현은 자기 자신을 형제 집합에 포함시킨다(`sibling[i,i,:] = True`). 그러면 $b$ 가 $y_u$ 에 약하게 의존하므로 엄밀하게는 leave-one-out이 아니고 $O(1/m)$ 편향이 생긴다 ($m$ = 생존 형제 수). GRPO의 그룹 평균 어드밴티지가 갖는 것과 같은 성질이며, $m$ 이 1이면 $A_H \equiv 0$ 이 되어 그 위치는 자동으로 침묵한다.

---

## 4. 근사 A1 — 지평 절단과 할인

식 (3)의 $H_{\text{togo}} = \sum_{t>u} H(\pi(\cdot \mid s_t))$ 는 응답 끝까지의 합이다. 이를

$$
H_{\text{togo}}(s_u \oplus y_u) \;\approx\; \sum_{k=1}^{\kappa} \gamma_H^{\,k}\, H\big(\pi(\cdot \mid s_{u+k})\big)
\tag{A1}
$$

로 자른다. $\gamma_H < 1$ 은 (i) 먼 항일수록 뒤에 오는 예보(A2)의 오차가 커지므로 그만큼 신뢰를 낮추고, (ii) 합을 $\kappa$ 에 무관하게 유계로 만든다.

**버리는 것:** 꼬리 $\sum_{k>\kappa}$, 그리고 남긴 항에도 곱해진 $\gamma_H^k$ 만큼의 축소.

**왜 치명적이지 않은가:** 식 (4)에서 쓰이는 것은 $H_{\text{togo}}$ 의 절댓값이 아니라 **형제 간 차이** $A_H$ 다. 같은 $s_u$ 를 공유하는 형제들에게 공통으로 걸리는 성분은 차분에서 소거된다. 절단·할인이 남기는 편향 중 형제 간 공통 부분은 $A_H$ 에 나타나지 않는다.

이 저장소의 Phase 1은 $(\kappa, \gamma_H)$ 를 이 근사가 오라클 $H_{\text{togo}}$ 와 갖는 순위상관 $\rho$ 로 고른다 (`docs/phase1_results_*.json`).

---

## 5. 근사 A2 — 실현된 미래를 MTP 예보로 대체 (핵심 근사)

(A1)조차 여전히 **미래 상태 $s_{u+k}$ 를 알아야** 한다. 업데이트 시점에는 그것이 없다. 롤아웃은 $\pi_{\text{old}}$ 에서 나왔고, 각 상태마다 연속열은 단 하나만 표본되어 있으며, 새로 굴리는 것은 비용상 불가능하다.

STEER-F는 **현재 hidden state $h_u$ 하나에서 $k$-스텝 앞 분포를 직접 예측하는 MTP 헤드**로 대체한다:

$$
H\big(\pi(\cdot \mid s_{u+k})\big) \;\longrightarrow\; H\big(p^{(k)}_{\text{MTP}}(\cdot \mid h_u)\big)
\tag{A2}
$$

이것이 STEER-F가 추가 롤아웃 없이 (V)를 잡는 방법이며, forward 한 번의 비용만 든다.

### 5.1 이 근사의 방향은 부호가 정해져 있다

$S_u$ 를 고정하고 기댓값을 취하면, (A1)의 $k$-번째 항의 기댓값은 조건부 엔트로피

$$
\mathbb{E}\big[H(\pi(\cdot\mid S_{u+k-1}))\big] \;=\; H\big(Y_{u+k} \,\big|\, Y_{u+1:u+k-1},\, S_u\big)
$$

인 반면, MTP 헤드가 모델링하는 것은 **주변 분포** $p(Y_{u+k} \mid S_u)$ 이므로 그 엔트로피는 $H(Y_{u+k} \mid S_u)$ 다. 조건화는 엔트로피를 줄이므로

$$
\boxed{\;H\big(Y_{u+k} \mid S_u\big) \;-\; \mathbb{E}\big[H(\pi(\cdot \mid S_{u+k-1}))\big] \;=\; I\big(Y_{u+k};\, Y_{u+1:u+k-1} \,\big|\, S_u\big) \;\ge\; 0\;}
\tag{5}
$$

즉 **(A2)는 항상 과대추정이며, 그 초과분은 정확히 중간 토큰들과의 상호정보량**이다. $k$ 가 커질수록 중간에 낀 토큰이 많아지므로 이 상호정보량은 단조 증가한다.

### 5.2 그래서 헤드별 아핀 캘리브레이션이 필요하다 (근사 A3)

식 (5)는 "헤드 $k$ 의 엔트로피는 체계적으로 부풀려져 있고, 그 정도는 $k$ 에 따라 커진다"를 **예측**한다. 이를 헤드별 아핀 보정으로 흡수한다:

$$
\hat{H}_k \;=\; a_k\, H\big(p^{(k)}_{\text{MTP}}(\cdot \mid h_u)\big) + b_k,
\qquad (a_k, b_k) = \arg\min \sum \big\| a_k H_k + b_k - H^{\text{oracle}}_k \big\|^2
\tag{A3}
$$

Phase 0 워밍업 롤아웃에서 최소제곱으로 적합한다 (`fit_head_calibration`).

**실측이 식 (5)를 확인한다.** `docs/phase1_results_Qwen2.5-Math-1.5B-paper.json` 의 적합 결과:

| $k$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| $a_k$ | 1.029 | 0.130 | 0.187 | 0.129 | 0.151 | 0.119 | 0.147 | 0.104 |

$a_1 \approx 1$ (1-스텝은 중간 토큰이 없으므로 식 (5)의 상호정보량이 0 — 근사가 정확), $k \ge 2$ 에서 $a_k \approx 0.1\text{–}0.2$ 로 급락. 유도가 예측한 그대로다.

### 5.3 최종 예보식

(A1)+(A2)+(A3)을 합치면 구현된 식이 된다:

$$
\boxed{\;\hat{H}_{\text{togo}}(s_u \oplus y_u) \;=\; \max\Big(0,\; \sum_{k=1}^{\kappa} \gamma_H^{\,k}\big(a_k H_k(h_u) + b_k\big)\Big)\;}
\tag{6}
$$

`clamp_min=0` 은 엔트로피가 비음수라는 사실을 되먹이는 안전장치다 — 음수는 캘리브레이션 적합 오차에서만 나올 수 있다.

> **현재 런의 실제 계수.** $\kappa=2,\ \gamma_H=0.7$ 에 위 표를 대입하면
> $\hat{H}_{\text{togo}} = 0.7206\,H_1 + 0.0636\,H_2 + 0.106$.
> head 2의 기여가 전체의 8%에 불과하다 — 유효 지평은 사실상 1스텝이다.
> 이는 식 (5)의 직접적 귀결이다: $k$ 가 커지면 캘리브레이션이 그 헤드를 스스로 꺼버린다.

---

## 6. 그래디언트에서 토큰 가중치로

식 (4)는 그래디언트에 더할 항의 모양이다. 그러나 STEER 계열은 그래디언트를 더하지 않고 **정책 손실을 토큰별로 재가중**한다. 따라서 필요한 것은 (V)의 그래디언트가 아니라, **한 번의 업데이트가 토큰 $u$ 를 통해 궤적 엔트로피를 얼마나 바꿀지에 대한 1차 예측**이다.

### 6.1 한 스텝이 로짓을 미는 양

클립된 대리손실의 계수는 정책 손실이 실제로 쓰는 것과 동일하게

$$
w_u \;=\; \mathrm{clip}\big(r_u,\; 1-\epsilon_{\text{lo}},\; 1+\epsilon_{\text{hi}}\big)\, A_u
$$

이고, 한 스텝은 **샘플된 토큰의 로짓만** 움직인다 (Appendix G Step 3):

$$
\Delta z_{u,a} \;=\; \eta\, w_u\, \delta_{a, y_u} + O(\eta^2)
$$

### 6.2 로짓 변화 → log-prob 변화

소프트맥스 야코비안 $\dfrac{\partial \log \pi_a}{\partial z_a} = 1 - \pi_a$ 를 써서

$$
\boxed{\;\widehat{\Delta \log \pi}_u \;=\; \eta\, w_u\,(1 - \pi_u)\;}
\tag{7}
$$

$\pi_u \to 1$ 이면 자동으로 0이 된다 — 이미 질량을 독점한 토큰에는 더 몰아줄 여지가 없다.

### 6.3 방문 채널을 통한 1차 엔트로피 변화

식 (4)에 $\Delta\theta$ 를 대입하면, 토큰 $u$ 를 통한 (V)의 1차 변화는 $\nabla_\theta \log\pi_u \cdot \Delta\theta = \Delta\log\pi_u$ 이므로

$$
\boxed{\;\widehat{\Delta \mathcal{H}}^{(\mathrm{V})}_u \;=\; \widehat{\Delta\log\pi}_u \cdot \mathrm{clip}\big(A_H(s_u,y_u),\, -c,\, +c\big) \;=:\; \mathrm{visit}_u\;}
\tag{8}
$$

**클립을 곱하기 *전에* 거는 이유 (근사 A4).** 그래야 한 토큰의 영향이 $|\eta w_u| \cdot c$ 로 유계다. 곱한 뒤 자르면 $w_u$ 가 큰 토큰에서 예보 오차가 그대로 증폭된다. 이는 편향을 감수하고 영향력을 유계로 만드는 robust-statistics식 절충이다.

### 6.4 부호 4분면 — 붕괴 채널의 정의

| $\widehat{\Delta\log\pi}_u$ | $A_H$ | 해석 |
|---|---|---|
| $>0$ | $>0$ | 형제 평균보다 풍부한 갈래로 질량 집중 → 궤적 엔트로피 상승 |
| $>0$ | $<0$ | **막다른 갈래로 집중. 포기된 형제가 담고 있던 다양성이 소멸** — 식 (2)의 (L)이 원리적으로 볼 수 없는 붕괴 경로 |
| $<0$ | — | 부호가 뒤집혀 같은 논리가 반대로 |

---

## 7. 두 채널의 결합

식 (2)를 토큰 단위 1차 예측으로 옮기면

$$
\tilde{\Omega}_u \;=\; \mathrm{norm}\big(\Omega_u\big) \;+\; \lambda \cdot \mathrm{norm}\big(\mathrm{visit}_u\big)
\tag{9}
$$

- $\Omega_u = \widehat{\Delta\mathcal{H}}^{(\mathrm{L})}_u$ 는 stock STEER의 로컬 항 (`STEERF_method.md` §3).
- $\lambda$ 는 신뢰 가중치다. 두 항이 같은 물리 단위(nats)임에도 정규화가 필요한 이유는 $\Omega$ 가 $1/\pi_{\text{old}}$ 를 품어 꼬리가 두껍기 때문이며, 정규화가 스케일을 지우므로 $\eta$ 는 $\lambda$ 와의 곱으로만 의미가 있다 ($\eta \equiv 1$ 고정, $\lambda$ 만 스윕).
- $\mathrm{norm}$ 이 z-score가 아니라 RMS인 것은 $\lambda \to 0$ 에서 stock STEER와 비트 동치를 유지하기 위한 제약이다 (`STEERF_method.md` §11).

### 7.1 가중치 공간 보정 (`apply="weight"`)

식 (9)는 $\mathrm{visit}$ 을 $\Omega$ 에 더한 뒤 STEER의 밴드 매핑에 넘긴다. 실측상 그 매핑이 신호를 소멸시키므로 (`branch_weight_correction` docstring), 구현의 기본 경로는 **매핑 이후** 유계 보정을 건다:

$$
w^{\text{final}}_u \;=\; \mathrm{clamp}\Big(w^{\text{STEER}}_u \;+\; \lambda\,(w_{\max}-w_{\min})\,\tanh\big(\mathrm{visit}_u / \mathrm{rms}(\mathrm{visit})\big),\; w_{\min},\, w_{\max}\Big)
\tag{10}
$$

$\tanh$ 는 홀함수·단조·유계이므로 (i) 부호를 보존하고, (ii) 한 토큰의 이동을 $\lambda(w_{\max}-w_{\min})$ 로 묶으며, (iii) $\mathrm{visit}$ 의 절대 스케일에 의존하지 않는다. rms는 $A_H \ne 0$ 인 **support 위에서만** 취한다 — 전체 위치로 나누면 보정 크기가 그 희소성에 따라 임의로 축소되기 때문이다.

### 7.2 이 보정은 평균을 움직이지 않는다 — 설계상

$A_H$ 는 형제 평균으로부터의 편차이므로 같은 $s_u$ 를 공유하는 형제 집합 위에서 **구성상 평균 0**이다:

$$
\sum_{j \in \text{sib}(s_u)} A_H(s_u, y^{(j)}_u) = 0
$$

따라서 식 (10)의 보정은 분기 토큰들의 **평균 가중치를 바꿀 수 없고, 오직 형제들을 서로 벌려놓을 뿐이다.** 이것은 결함이 아니라 (V)의 정의에서 오는 필연이다 — 베이스라인이 있어야 편향이 없고, 베이스라인이 있으면 평균은 0이다.

> **평가 기준에 대한 함의.** 그러므로 "분기점 평균 엔트로피가 올라가는가"는 이 경로(`apply="weight"`, `mode="signed"`)에서 STEER-F의 작동 증거로 쓸 수 없다. 올바른 판정은 *형제 간 분산*이 벌어지는가다. 평균을 올리고 싶다면 `apply="branch"`(`mode="uniform"`)가 그 가설의 구현이며, 그것은 예보를 전혀 쓰지 않는 더 약한 가설이다.

---

## 8. 근사 원장 (assumption ledger)

| # | 근사 | 무엇을 버리는가 | 부호/유계 | 완화 장치 |
|---|---|---|---|---|
| — | 식 (1)(2)(3)(4) | 없음 — 항등식 | — | — |
| A1 | 지평 $\kappa$ 절단 + $\gamma_H$ 할인 | 꼬리 $\sum_{k>\kappa}$, 남긴 항의 축소 | 과소추정 | 형제 차분에서 공통 성분 소거; Phase 1이 $\rho$ 로 $(\kappa,\gamma_H)$ 선택 |
| A2 | 실현된 미래 → MTP 주변 예보 | 중간 토큰과의 상호정보량, 식 (5) | **과대추정, $k$ 에 단조 증가** | A3 |
| A3 | 헤드별 아핀 캘리브레이션 | A2의 잔차 중 비아핀 성분 | 최소제곱 잔차 | 오라클 대조군 (`steerf_forecast=oracle`) |
| A4 | $A_H$ 를 $\pm c$ 로 클립 | 극단 분기 점수의 크기 정보 | 유계, $\lvert \cdot \rvert \le c$ | `a_h_clip_frac` 로 포화 감시 |
| B1 | 1차 전개 ($\eta$ 에 대해) | $O(\eta^2)$ | — | 작은 lr |
| B2 | 샘플된 토큰의 로짓만 이동 | 파라미터 공유로 인한 다른 위치·다른 토큰으로의 파급 | — | **stock STEER와 공유하는 가정** |
| B3 | 롤아웃이 $\pi_{\text{old}}$ 에서 나옴 | $\pi_\theta$ 기댓값과의 차이 | — | `use_ratio=True` 가 $r$ 을 복원 (`STEERF_method.md` §3.2) |
| B4 | 형제 집합에 자기 포함 | leave-one-out 대비 $O(1/m)$ 편향 | — | GRPO 어드밴티지와 동일한 성질 |

**가장 강한 가정은 A2와 B2다.** A2는 (5)로 부호와 단조성이 확정되어 A3가 흡수하지만, B2 — 한 토큰의 업데이트가 다른 위치의 조건부 분포를 바꾸지 않는다는 가정 — 는 파라미터가 공유되는 신경망에서는 성립하지 않으며 보정 장치가 없다. 이는 stock STEER가 이미 하고 있는 가정이므로 STEER-F가 새로 도입한 위험은 아니지만, 두 채널을 더할 때 오차도 함께 더해진다.

---

## 9. 검증 게이트와의 대응

| 게이트 | 무엇을 검사하는가 | 유도상 대응 | 구현 |
|---|---|---|---|
| G1 | $\hat{H}_{\text{togo}}$ 가 실제 분기점을 랭킹하는가 | A2+A3의 유용성 | `branch_recall_at_k` (반드시 `support=` 와 함께) |
| G2 | 분기점 엔트로피가 **전체 엔트로피가 평평한 채로** 오르는가 | (V) 채널이 (L)과 구분되어 작동하는가 | `branch_token_entropy` |
| — | $\lambda = 0$ 이 stock STEER와 비트 동치인가 | 식 (9)가 stock의 진부분확장인가 | `tests/test_lambda_zero_equiv.py` |

**G2의 조건절이 핵심이다.** "분기점 엔트로피 상승"만으로는 부족하고 "전체 엔트로피가 평평한 동안"이어야 한다 — 그렇지 않으면 균일 엔트로피 보너스와 구별되지 않는다. §7.2에서 보였듯 `apply="weight", mode="signed"` 경로는 평균을 올리도록 설계되지 않았으므로, 이 경로에서 G2는 상한이 아니라 **하한 확인**(분기점이 상대적으로 손해보지 않았는가)으로 읽어야 한다.

---

## 10. 참고

- 구현·텐서 규약·config: [`STEERF_method.md`](STEERF_method.md)
- $\Omega$ 형태 실측 비교: `docs/omega_forms.json`
- 가중치 공간 정식화 비교: `docs/weight_forms.json`
- Phase 1 $(\kappa, \gamma_H)$ 선택과 캘리브레이션: `docs/phase1_results_*.json`
- 분기 recall의 support 의존성: `docs/phase1_recall*.json`
