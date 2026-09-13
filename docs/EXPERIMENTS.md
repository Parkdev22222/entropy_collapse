# 실험 현황판

**무엇이 어디서 도는가, 언제 끝나는가, 지금 어디까지 왔는가.**
절차는 `docs/RUNBOOK.md`, 분석 계획과 예측은 `docs/preregistration_5seed.md`,
프로젝트 전체 맥락은 `docs/HANDOFF.md`.

최종 갱신 2026-09-13.

---

## 1. 한 눈에

| | A100 × 2 | H100 × 4 |
|---|---|---|
| **지금** | 캠페인 20런 (5 arm × seed 2–5) | `measure` + followups 9개 + `eval2` |
| **그 뒤** | `grpo-long` 1런 | — |
| **마지막** | 두 박스 로그를 합쳐 `eval` + `analysis` (박스 무관) | |

`grpo-long`만 A100에 남는 이유: `scripts/analyze_seeds.py`의 `compute_matched_step()`이
그 런의 `perf/time_per_step`을 **적분해서 STEER-F가 쓴 초와 맞춘다.** 박스가 다르면 초가
같은 단위가 아니다. STEER-F seed-1 기준선이 A100에 있다.

GPU 장수는 처치를 바꾸지 않는다 — `ppo_micro_batch_size_per_gpu=8`이 하드코딩이고
그게 STEER의 min–max 그룹이라 풀 크기가 2장에서도 4장에서도 8이다
(`run_steerf.sh:125-126`).

---

## 2. A100 — 캠페인 (헤드라인 표)

5 arm × 5 시드. 시드 1은 이미 있으므로 새로 도는 건 **seed 2–5 × 5 arm = 20런**.

| arm | 무엇을 켜는가 | 런처 | 런 이름 |
|---|---|---|---|
| `grpo` | 없음 (`loss_mode=vanilla`) | `run_grpo.sh` | `grpo-<tag>-s<N>` |
| `steer` | 로컬 Ω만 (λ=0, plain) | `run_steerf.sh` | `steer-<tag>-s<N>` |
| `permuted` | tree + 감쇠, `A_H`를 형제끼리 셔플 | `run_uniform_ablation.sh` | `steer-f-<tag>-s<N>-tree-rollout-permuted` |
| **`signed`** = STEER-F | tree + `A_H` 그대로 (λ=.25) | `run_uniform_ablation.sh` | `steer-f-<tag>-s<N>-tree-rollout` |
| `uniform` | tree + 감쇠, `A_H` 값 무시 | `run_uniform_ablation.sh` | `steer-f-<tag>-s<N>-tree-rollout-uniform` |

**시드 우선 순회**다. seed 2의 5 arm → seed 3의 5 arm → … 이라서 **아무 때나 끊어도 모든
arm의 시드 수가 같다.** 페어드 대조가 `min(n)`에 묶이므로 이 순서가 본질이다.

```bash
tmux new -d -s campaign \
  "cd /workspace/entropy_collapse && WAIT=1 REPO=DSDSh/steer-f_2 \
   bash run/run_campaign.sh > logs/experiments/campaign.log 2>&1"
```

| | |
|---|---|
| 시드 1개(5 arm) | 186.9 h ≈ **7.8일** |
| 20런 | 748 h ≈ **31일** |

### ⚠️ 시드 1이 정말 5 arm 다 있는지 확인할 것

| arm | 시드 1 |
|---|---|
| signed | ✅ step 110 완주 |
| steer | ✅ `_0905` exit 0 |
| permuted | ✅ git에 로그 있음 |
| grpo | ⚠️ pod엔 있으나 **git에 로그가 없다** → 논문 GRPO 행이 `[pending]` |
| **uniform** | ❓ `_0905`가 09-09 08:23 시작 배너까지만 |

```bash
grep -c 'step:110 - global_seqlen' \
  logs/experiments/train-steer-f-Qwen2.5-Math-1.5B-s1-tree-rollout-uniform_0905.log
```

`0`이면 캠페인 앞에 `run_0905_chain.sh`를 세운다(끝난 arm은 건너뛴다). 미완이면
uniform만 4시드가 되고 `STEER-F − uniform` 대조가 n=4로 묶인다.

---

## 3. H100 — 나머지

```bash
tmux new -d -s paper \
  "cd /workspace/entropy_collapse && \
   REPO=DSDSh/steer-f_2 \
   STAGES='preflight measure followups eval2' \
   FOLLOWUP_ARMS='lam0-tree lam0.1 lam0.5 xclip-signed xclip-steer rloo-signed rloo-steer opo-signed opo-steer' \
   bash run/run_paper.sh > logs/experiments/paper_h100.log 2>&1"
```

### `measure` — 표가 답할 수 없는 반론에 답하는 측정

| 스크립트 | 막는 반론 | 비용 |
|---|---|---|
| `measure_forecast_locality.py` | **"유효 지평이 1스텝이면 그냥 로컬 항 아니냐"** — 논문 최대 약점 | ~1 GPU-h |
| `measure_omega_at_branches.py` | "분기점에서 정말 \|Ω\|가 작냐" — Table 1을 예측에서 관측으로 | ~1 GPU-h |
| `phase1_branch_recall.py` | G1 recall을 실제로 쓴 κ=2 / γ_H=0.7에서 재측정 | ~0.5 GPU-h |
| `measure_ah_support.py` | 조합론적 상한 `2(n−1)/T` 대 실측 분기 비율 | CPU |

### `followups` — ablation 9개 (seed 1)

| arm | λ | 무엇을 묻는가 | 런처 |
|---|---|---|---|
| `lam0-tree` | 0 | **빠진 칸.** tree rollout만의 효과를 격리 | tree |
| `lam0.1` / `lam0.5` | .1 / .5 | .25가 튜닝된 값인가 임의값인가 | tree |
| `xclip-signed` / `xclip-steer` | .25 / 0 | 클리핑을 거의 푼 극단 세팅 (`ε_hi=5, ε_lo=0.99`) | tree / plain |
| `rloo-signed` / `rloo-steer` | .25 / 0 | RL 알고리즘 일반화 | tree / plain |
| `opo-signed` / `opo-steer` | .25 / 0 | 〃 | tree / plain |

**전부 seed 1 한 개씩이다.** 논문에 그렇게 명시할 것 — 여기서 나온 대조는 전부 n=1이고
효과 크기가 아니라 메커니즘을 말한다.

`lam0-tree`가 제일 중요하다. 지금 어느 arm도 tree rollout 자체의 효과를 격리하지 못한다
(uniform·permuted 둘 다 λ=.25가 켜져 있다). 리뷰어가 "tree가 그냥 더 좋은 샘플러 아니냐"고
물으면 이 칸이 없으면 답이 논증뿐이다.

---

## 4. 마지막 — 로그를 합친 뒤

`analyze_seeds.py`와 `collect_results.py`는 `logs/experiments/`만 읽는다. 두 박스 로그를
한쪽에 모으고(`paper` 브랜치가 제일 안전):

```bash
STAGES='eval analysis' REPO=DSDSh/steer-f_2 bash run/run_paper.sh
```

`eval`은 체크포인트를 HF에서 받아오므로 박스를 안 가린다.

### 숫자가 논문까지 가는 경로

```
logs/experiments/train-*.log
      │  scripts/analyze_seeds.py
      ▼
results/{per_seed,arm_means,contrasts,compute_match}.tsv
results/numbers.tex          ← \providecommand 매크로
      │  paper/steerf.tex 가 \input
      ▼
표의 \num{...} 가 자동으로 채워진다 (없는 값은 [pending])
```

**표를 손으로 고치지 마라.** `analyze_seeds.py`를 다시 돌리면 덮인다.

---

## 5. 진행 확인

```bash
tail -f logs/experiments/campaign.log            # A100
tail -f logs/experiments/paper_h100.log          # H100
grep -E '^\[campaign\].*exit' logs/experiments/campaign.log     # arm별 종료 코드
DRY=1 bash run/run_campaign.sh | tail -3         # 남은 큐 (20에서 줄어든다)

# 계측이 실제로 걸렸는지 — 안 걸리면 문제별 점수가 통째로 없다
grep -l validation_data_dir logs/experiments/train-*.log | wc -l
```

시작 5분 안에 `step:1 - global_seqlen`이 안 보이면 학습 크래시가 아니라 **시작 실패**다.
큐가 `diagnose_startup_failure`로 로그에 남긴다 — 보통 파드 재시작 후 `setup_env.sh`를
건너뛴 경우다.

---

## 6. 미해결

1. **GRPO 시드-1 학습 로그를 `paper` 브랜치에 push** — 제일 값싸고 제일 중요하다.
   논문 GRPO 행이 `[pending]`을 벗는다
2. **`checkpoints/mtp_calibration_*-paper.json` 회수·커밋** — 원고 `a_k` 수치 정합
3. **시드 1의 중복 draw 처리 규칙** — STEER와 signed가 시드 1에 두 런씩 있다
   (기존, `_0905`). arm별로 좋은 쪽을 고르면 체리피킹이다. 롤아웃이 시드 고정이 아니므로
   사실상 독립 추출이니 `_0905`를 **별도 draw로 라벨링**해 arm 평균의 n을 올리되
   시드-대응 대조에는 짝이 있는 시드만 쓰는 쪽이 방어하기 쉽다.
   `analyze_seeds.py`는 지금 run 이름 글롭으로 파싱하므로 **규칙이 없다**
4. **체크포인트가 런당 1개뿐** (`save_best_only=True` + argmax). "선택 규칙을 바꿔도
   arm 순서가 안 바뀐다"는 사전등록 §4의 rule 2를 지금 설정으로는 **확인할 수 없다**

---

## 7. 논문에 숫자가 꽂힐 자리 (2026-09-13 기준)

`paper/steerf.tex`의 문장·표는 **전부 써져 있다.** 실행되면 `\num{...}` 매크로가
`results/numbers.tex`에서 값을 읽어 자동으로 채워진다. 지금은 **47개 슬롯**이
`[pending]`(빨간 글씨)로 렌더된다.

```bash
cd paper && pdflatex steerf && pdftotext steerf.pdf - | grep -c pending
```

### 기존 생성기가 이미 채우는 것
`scripts/analyze_seeds.py`가 `R*`(평균) `E*`(SE) `N*`(시드 수) `C*`(대조) `T*`(t)를
캠페인 로그에서 뽑는다 → 표 1·2·3은 캠페인이 끝나면 손 안 대고 채워진다.

### 아직 생성기가 안 뱉는 것 — 채우려면 확장이 필요하다

| 슬롯 | 무엇 | 어디서 나오나 |
|---|---|---|
| `Seedsd` `Pairedsd` | 시드 간 SD, 대응차 SD | `analyze_seeds.py` (per_seed.tsv에 이미 있는 값) |
| `Rlam*` `Rxclip*` `Rrloo*` `Ropo*` | followups 9개 | followups 로그 — `analyze_seeds.py` 확장 |
| `Wsigned` `Wgrpo` `Wgrpolong` `Matchstep` `Longsteps` `Rgrpolongacc` | compute-matched | `compute_match.tsv`에 이미 계산돼 있다 |
| `Dirconsistency` `Dirconsistencyperm` | 6벤치 방향 일관성 | `collect_results.py` 확장 |
| `W*` `WT*` | 부록의 실행 내 안정성 (8-step paired t) | `analyze_seeds.py` 확장 |
| `C*grpo*` `T*grpo*` `N*grpo` | GRPO 대조 | **GRPO 시드-1 로그를 git에 올려야 한다** |
| `benchmarks_table.tex` | 6벤치 표 전체 | `collect_results.py`가 .tex를 뱉게 |

확장 전까지는 이 슬롯들을 손으로 채워야 한다. **표를 직접 고치지 말고**
`results/numbers.tex`에 `\providecommand{\이름}{값}` 줄을 추가하는 쪽이 안전하다 —
생성기를 다시 돌려도 안 덮인다.
