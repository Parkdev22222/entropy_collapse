# RUNBOOK — 4-시드 캠페인 기동

`run/run_campaign.sh`로 seed 2·3·4·5 × 5 arm = **20런**을 하나의 재개 가능한 큐로 돌린다.
절차는 2026-09-13에 검증했다. 배경과 수치는 `docs/HANDOFF.md`.

> **31일짜리 큐다.** 3단계(`DRY=1`)에서 `20`이 안 나오면 기동하지 마라.
> 큐가 조용히 짧아지는 것이 이 파이프라인의 대표적인 실패 모드이고, 에러가 안 난다.

---

## 0. 상태 점검 (30초)

```bash
cd /workspace/entropy_collapse
tmux ls
tail -5 logs/experiments/chain_0905.log
grep -c 'step:110 - global_seqlen' \
  logs/experiments/train-steer-f-Qwen2.5-Math-1.5B-s1-tree-rollout-uniform_0905.log
pgrep -af main_ppo | head -3
uptime -s                      # 컨테이너 시작 시각
git log --oneline -1
```

읽는 법:

| 관측 | 뜻 |
|---|---|
| `grep -c` = 1 | seed 1 uniform 완료 → 4단계 **A** |
| `grep -c` = 0 | 미완 → 4단계 **B**로 복구를 먼저 세운다 |
| `uptime -s`가 마지막 학습 시작보다 나중 | **파드가 재시작됐다.** 1단계를 건너뛰지 마라 |
| `chain_0905.log`에 `chain finished`가 없다 | 체인이 중간에 죽었다(보통 tmux/파드 재시작) |

`run_0905_chain.sh`는 arm이 끝나면 무조건 `[chain] <arm> exit N`과 `chain finished`
요약을 찍는다. **둘 다 없으면 그 arm은 끝난 게 아니다.**

## 1. 환경 — 재시작했으면 필수

`/usr/local/lib/python3.12/dist-packages`는 컨테이너 오버레이라 **재시작마다 날아가고
이미지의 `huggingface_hub`가 돌아온다.** 그 버전이 1.x면 transformers가 import 시점에
거부하고 **모든 `main_ppo`가 몇 초 만에 죽는다.** 2026-09-08과 09-09에 두 번 발생했다.
(`/workspace`는 네트워크 볼륨이라 살아남는다.)

```bash
bash run/setup_env.sh
python3 scripts/check_env_pins.py && echo "PINS OK"
```

## 2. 코드 동기화

```bash
git fetch origin claude/3b-text-generation-models-thz2vl
git checkout origin/claude/3b-text-generation-models-thz2vl -- run scripts docs
bash run/instrument_campaign.sh --apply
```

`--apply`가 없으면 캠페인이 `REFUSE`한다. 이 패치는 `validation_data_dir`(문제별 점수),
`save_best_only` 파라미터화, `seq_entropy` 로깅을 켠다 — 셋 다 gradient를 안 건드린다.

## 3. 큐 확인 — **20이 아니면 멈춘다**

```bash
DRY=1 bash run/run_campaign.sh | tail -3
```

```
20 run(s) queued, 5 arm(s) x 4 seed(s)
```

이 한 줄이 코드 버전·로그 상태·arm 이름을 한꺼번에 검증한다. 20보다 작으면 어떤 arm이
이미 DONE으로 판정된 것이고, 그대로 두면 **주력 arm이 표에서 통째로 빠진 채 31일이 지나간다.**
어느 것이 빠졌는지는 출력의 `skip` 줄에 그대로 나온다.

## 4. 기동

### A. seed 1 uniform이 완료된 경우

```bash
tmux new -d -s campaign \
  "cd /workspace/entropy_collapse && WAIT=1 REPO=DSDSh/steer-f_2 \
   bash run/run_campaign.sh > logs/experiments/campaign.log 2>&1"
```

### B. 미완인 경우 — 복구를 먼저 세우고 캠페인을 뒤에 줄 세운다

```bash
tmux new -d -s recover \
  "cd /workspace/entropy_collapse && bash run/run_0905_chain.sh \
   > logs/experiments/chain_0905_retry.log 2>&1"

tmux new -d -s campaign \
  "cd /workspace/entropy_collapse && WAIT=1 REPO=DSDSh/steer-f_2 \
   bash run/run_campaign.sh > logs/experiments/campaign.log 2>&1"
```

체인은 끝난 arm을 건너뛰므로 uniform만 돈다.

**`WAIT=1`이 핵심이다.** 없으면 캠페인은 도는 학습을 보고 `REFUSE` 하고 그냥 종료한다.
있으면 `main_ppo`가 사라질 때까지 120초마다 폴링하다가, GPU가 실제로 풀리도록 30초 더
기다린 뒤 시작한다.

## 5. 감시

```bash
tail -f logs/experiments/campaign.log
grep -E '^\[campaign\].*exit' logs/experiments/campaign.log     # arm별 종료 코드
```

시작 5분 안에 `step:1 - global_seqlen`이 안 뜨면 학습 크래시가 아니라 **시작 실패**다.
캠페인이 `diagnose_startup_failure`로 그 판정을 로그에 남긴다 — 보통 1단계를 건너뛴 것이다.

---

## 일정·비용

**시드-우선 순회다.** seed 2의 5 arm → seed 3의 5 arm → … 이므로 **아무 때나 끊어도
모든 arm의 시드 수가 같다.** 페어드 대조가 `min(n)`에 묶이므로 이 순서가 본질이다.
arm-우선으로 돌리면 STEER-F만 5시드고 GRPO는 1시드로 남는다.

| arm | h/run | | |
|---|---|---|---|
| GRPO | 24.4 | 시드 1개(5 arm) | **186.9 h ≈ 7.8일** |
| STEER | 25.4 | 4시드 20런 | **748 h ≈ 31일** |
| STEER-F | 46.3 | | |
| uniform | 45.2 | | |
| permuted | 45.6 | | |

**시드를 3개로 줄이면 헤드라인 대조가 유의하지 않다**(n=3에서 t=2.85 < 임계 4.30).
자를 거면 ablation과 통제군 시드를 먼저 자른다. `docs/HANDOFF.md` §3 참조.

`REPO`를 주면 런이 끝날 때마다 HF Hub에 올리고 바이트 단위로 검증한 뒤 로컬을 지운다
→ 디스크 정상 상태가 체크포인트 1개(3.1 GB). 빼면 런당 3.1 GB가 쌓인다.

## 알려진 함정

| | |
|---|---|
| **hub 핀** | 재시작마다 되돌아간다. 1단계가 유일한 방어. 두 번 당했다 |
| **`pip install wandb` 금지** | vllm 0.8.4의 opentelemetry 핀(`<1.27.0`)을 깨고 protobuf를 4.25→7.36으로 올린다. 2026-09-13 H100 박스에서 실제로 그랬다. 그리고 **어차피 안 쓴다** — 큐가 도는 세 경로가 전부 tensorboard다(`run_grpo.sh:207`, `run_uniform_ablation.sh:170`, `_arms.sh:102`). `setup_env.sh` 5b절이 이제 설치 후 핀을 다시 검사해서 이런 걸 잡는다 |
| **`run_steerf.sh`의 `STEPS=200`** | SCALE case에 하드코딩이라 `export STEPS`가 안 닿는다. 캠페인은 `steer_plain_args()`를 trailing override로 붙여 막는다 |
| **로그 글롭** | `train-<run>.log` + `train-<run>_*.log`만 허용해야 한다. 밑줄이 없으면 `-permuted`를 삼켜 signed를 DONE으로 오판정한다. `tests/test_run_names.py`가 고정한다 |
| **`run_paper.sh` preflight** | `-paper` 접미사 MTP 파일은 `measure` 스테이지 전용이다. 학습 arm은 안 쓴다 |

---

# 두 박스로 나눠 돌리기 (A100×2 캠페인 / H100×4 나머지)

캠페인은 A100 박스에 두고 나머지를 H100 박스로 넘길 때의 절차.

## GPU 장수는 처치를 바꾸지 않는다 — 확인함

`run_steerf.sh:192`가 `ppo_micro_batch_size_per_gpu=8`을 하드코딩하고, 같은 파일 125행이
이유를 적어뒀다: **그게 STEER의 min–max가 도는 그룹이다.** 풀 크기가 8로 고정이므로
GPU 장수가 바뀌어도 α의 분포가 체계적으로 이동하지 않는다.

| | GPU당 mini-batch | 8짜리 마이크로배치 | **풀 크기** |
|---|---|---|---|
| A100 2장 | 32/2 = 16 | 2개 | **8** |
| H100 4장 | 32/4 = 8 | 1개 | **8** |

남는 차이는 "어떤 시퀀스끼리 한 풀에 묶이나"뿐이고, 롤아웃 샘플링이 실행 간 시드 고정이
아니라서 **이미 존재하는 런-투-런 변동과 같은 층**이다. `TP_SIZE=4`는 vLLM 생성 수치만
건드리고 min–max에는 닿지 않는다.

→ **H100 박스는 `N_GPUS=4 TP_SIZE=4`를 그대로 쓴다.** `_gpu_defaults.sh`가 자동 감지하므로
따로 지정할 것도 없다.

## 예외: `grpo-long`은 A100에 남는다

`scripts/analyze_seeds.py:186`의 `compute_matched_step()`이 grpo-long의
`perf/time_per_step`을 적분해서 **STEER-F가 쓴 초**와 맞춘다. 박스가 다르면 초가 같은
단위가 아니다. STEER-F seed-1 기준선이 A100에 있으므로 grpo-long도 거기서 돈다.

나머지 9개 ablation은 전부 H100에서 도는 것끼리, 또는 자기 짝 baseline과 비교되므로
(`xclip-signed`↔`xclip-steer`, `rloo-*`, `opo-*`) 박스가 갈려도 문제없다.

## 분업표

| | A100×2 | H100×4 |
|---|---|---|
| 지금 | campaign 20런 (~31일) | `measure` + followups 9개 + `eval2` |
| 캠페인 후 | `grpo-long` | `eval` (체크포인트를 HF에서 받아서) |
| 마지막 | — | 로그 합류 후 `analysis` |

## 1. 이전 — A100(보내는 쪽)

`--export`는 **읽기·업로드만** 한다. 지우지 않으므로 도는 캠페인에 안전하다.

```bash
cd /workspace/entropy_collapse
git add -A && git commit -m "wip" && git push     # --export는 더러운 트리를 거부한다
bash run/migrate_pod.sh --check                   # 먼저 점검만
REPO=DSDSh/steer-f_2 bash run/migrate_pod.sh --export
```

옮기는 것은 **git에 없고 GPU 시간이 드는 것**뿐이다:

| | |
|---|---|
| **필수** | `checkpoints/mtp_heads_<tag>-paper.pt`, `mtp_calibration_<tag>-paper.json` — `run_uniform_ablation.sh:95`이 이 경로를 하드코딩하고 :149가 없으면 REFUSE한다. **λ=0이어도** 거부하므로 signed/uniform/permuted와 tree followups 6개가 전부 막힌다 |
| 선택 | 접미사 없는 2종, `mtp_heads_control_<tag>.pt`, `rollouts.jsonl` — 지금 큐 중 아무도 안 연다 |
| 안 옮김 | pip 환경, HF 모델 캐시 — 새 박스에서 다시 만든다. pip 트리를 복사하면 flash-attn이 엉뚱한 torch에 링크된다 |

sha256과 바이트 크기를 매니페스트에 적고 `--import`가 대조한다.

## 2. 이전 — H100(받는 쪽)

```bash
git clone https://github.com/Parkdev22222/entropy_collapse /workspace/entropy_collapse
cd /workspace/entropy_collapse
git checkout claude/3b-text-generation-models-thz2vl

bash run/bootstrap_pod.sh                 # 두 브랜치를 합쳐 실행 가능한 트리를 만든다
bash run/setup_env.sh                     # vllm / ray / flash-attn / 핀
REPO=DSDSh/steer-f_2 bash run/migrate_pod.sh --import      # MTP 헤드
```

**`bootstrap_pod.sh`가 왜 필요한가.** 이 브랜치에는 `verl/`·`datasets/`·`logs/`·
`requirements.txt`가 **없다**(§HANDOFF 1절의 분할). 그냥 클론하면 `import verl`이
`ModuleNotFoundError`로 죽고 학습이 한 줄도 안 돈다. 2026-09-13에 새 H100 박스가 정확히
이 증상을 냈다.

`DRY=1 bash run/bootstrap_pod.sh`로 무엇을 가져올지 먼저 볼 수 있다.

> ⚠️ **`git checkout origin/paper -- run` 을 통째로 치지 마라.** 두 브랜치가 모두
> `run/setup_env.sh`를 갖고 있어서, 도너의 옛 버전이 이 브랜치 것을 덮고 **hub 핀 검사를
> 잃는다** — 지금까지 런을 두 번 죽인 그 실패에 대한 방어다. `bootstrap_pod.sh`는
> **HEAD가 추적하는 파일은 절대 안 덮는다**(차집합만 가져온다). 수동으로 하려면 파일을
> 하나씩 지정해야 한다.

> ⚠️ **`git apply patches/steerf_tree_rollout.patch` 를 치지 마라.** `origin/paper`의
> `verl/`에 이미 적용돼 있다(`steerf_tree_depths` 4군데). `setup_env.sh`는 "적용됨"과
> "패치 파일 있음"을 구별하지 못해서 이걸 `NEED`에 넣는데, 실행하면 충돌한다.
> `bootstrap_pod.sh`가 끝에서 어느 쪽인지 알려준다.

`--import`가 sha256까지 맞는지 확인하고 `verl + steer_f import`까지 본다. **`migration
verified`가 뜨기 전에는 옛 박스를 지우지 마라.**

## 3. H100에서 남은 실험 한 번에

```bash
tmux new -d -s paper \
  "cd /workspace/entropy_collapse && \
   REPO=DSDSh/steer-f_2 \
   STAGES='preflight measure followups eval2' \
   FOLLOWUP_ARMS='lam0-tree lam0.1 lam0.5 xclip-signed xclip-steer rloo-signed rloo-steer opo-signed opo-steer' \
   bash run/run_paper.sh > logs/experiments/paper_h100.log 2>&1"
```

`FOLLOWUP_ARMS`가 학습 큐와 `eval2` 양쪽에 같이 전달되므로, 이 박스는 자기가 학습한 9개만
평가한다. `grpo-long`은 빠져 있다.

`eval`을 여기 넣지 않은 이유: 캠페인 체크포인트가 아직 HF에 다 올라오지 않았다.

## 4. A100에서 캠페인이 끝난 뒤

```bash
FOLLOWUP_ARMS=grpo-long STAGES=followups REPO=DSDSh/steer-f_2 bash run/run_paper.sh
```

## 5. 마지막 합류 — 로그를 한 곳에 모은 뒤

`analyze_seeds.py`와 `collect_results.py`는 `logs/experiments/`만 읽는다. 두 박스의 로그를
한쪽에 모으고(git `paper` 브랜치가 제일 안전하다) 거기서:

```bash
STAGES='eval analysis' REPO=DSDSh/steer-f_2 bash run/run_paper.sh
```

⚠️ **`s/step` 비용 표에 두 박스를 섞지 마라.** 논문의 오버헤드 수치(+86%)는 같은 실행
안의 STEER↔STEER-F 비교다. H100에서 잰 초를 그 표에 넣으면 오버헤드가 과소평가된다.
