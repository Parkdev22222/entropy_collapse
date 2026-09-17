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
bash run/run_status.sh          # 돌고 있나 / 몇 장 쓰나 / 어디까지 갔나 / 죽었으면 왜
WATCH=1 bash run/run_status.sh  # 60초마다 갱신
```

읽기 전용이다 — 아무것도 안 죽이고 안 지운다. `train_log_done`·`is_busy`·
`diagnose_startup_failure`를 큐와 **같은 함수로** 쓰므로 "끝났다"의 정의가 어긋날 수 없다.
2절의 토폴로지 줄이 arm마다 다른 GPU 장수를 잡아낸다.

원본 로그가 필요하면:

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

### 디스크가 찼을 때 (학습 중에도 안전)

큐는 여유가 `MIN_FREE_GB`(기본 20) 미만이면 **다음 arm을 시작하지 않고 멈춘다**
(`run_campaign.sh:176`). 도는 런이 죽는 게 아니라 그 다음이 안 뜬다.

```bash
bash run/hf_backup.sh                    # 인자 없이 = 런 목록
PRUNE=1 bash run/hf_backup.sh <끝난-런>   # optimizer/extra state 삭제, 네트워크 없음
REPO=DSDSh/steer-f_2          bash run/hf_backup.sh <끝난-런>   # 업로드 + 검증
REPO=DSDSh/steer-f_2 DELETE=1 bash run/hf_backup.sh <끝난-런>   # 검증 후 삭제
```

**`PRUNE=1`을 먼저 친다.** 체크포인트 용량의 대부분이 optimizer/extra state이고
평가에도 Hub 업로드에도 쓰이지 않는다(`actor/huggingface/`만 있으면 된다). 네트워크가
필요 없어서 즉시 회수된다. 도는 런은 세 검사가 알아서 막으므로 런 이름을 고를 때
무엇이 도는지 몰라도 된다.

## 로그를 git에 올리기

로그가 **실험 기록 그 자체**다. 원고의 모든 수치가 여기서 재계산된다
(`scripts/analyze_seeds.py`, `scripts/seed1_table.py --git-ref origin/paper`).
박스가 회수되면 커밋 안 된 로그는 같이 사라진다 — GRPO seed-1이 그렇게 사라져서
지금 `docs/seed1_grpo_transcript.json`에 붙여넣기 기록으로만 남아 있다.

```bash
cd /workspace/entropy_collapse
bash run/publish_logs.sh                  # 무엇이 올라갈지만 보여준다
bash run/publish_logs.sh --push           # 커밋 + 푸시 (기본 브랜치: paper)
bash run/publish_logs.sh --push --done-only   # 완주한 런만
bash run/publish_logs.sh --push --queue-logs  # 큐 드라이버 로그도
```

`git add logs && git commit`을 직접 치면 안 되는 이유가 셋이다:

| | |
|---|---|
| **학습 트리를 다른 브랜치로 체크아웃하면 안 된다** | 도는 트레이너가 `run/`과 `steer_f/`를 워킹 트리에서 읽는다. 브랜치를 바꾸면 **런 도중에 코드가 바뀐다.** 스크립트는 `git worktree`를 쓰므로 학습 트리를 건드리지 않는다 |
| **도는 런의 로그는 아직 쓰이는 중이다** | 커밋하면 반쪽짜리가 기록으로 굳는다. 트레이너가 붙어 있는 런은 건너뛴다(`--force`로 우회) |
| **`validation_data/`는 gitignore에 없다** | 런당 ~150 MB인데 무시 목록에 없어서 `git add -A`면 영구히 히스토리에 들어간다. 스크립트는 **명시한 파일만** 스테이징한다 |

완주 못 한 로그도 **기본으로 올린다** — 크래시한 런의 로그가 왜 크래시했는지의 증거이고
`diagnose_run_failure`가 그걸 읽는다. 표에는 `~`로 표시된다. A100과 H100이 같은 브랜치에
올려도 된다: 커밋 메시지가 박스를 적고, 파일 집합이 겹치지 않으며, 푸시가 rebase한다.

## 알려진 함정

| | |
|---|---|
| **★ `steer_f/`는 도너 것이어야 한다** | 두 브랜치는 **공통 조상이 없고** 양쪽 다 `steer_f/`를 갖는다. 같은 패키지의 두 버전이 아니라 **두 계보**이고, `origin/paper`의 `verl`은 그쪽 `steer_f`를 이름으로 부른다(`dp_actor.py:407` → `forecast_h_togo` 외 7개). 이 브랜치 것을 쓰면 **모든 tree·steer arm이 워커 초기화에서 죽는다.** 두 계보가 바이트까지 같은 파일은 `tree_rollout.py` 하나뿐인데 하필 그게 모든 게이트가 import하던 파일이라, 게이트는 통과하고 런만 죽었다. `bash run/bootstrap_pod.sh`가 이제 덮어쓰고, `python3 run/_check_steer_f.py`가 `verl` 소스의 import 문과 대조한다. 손으로는 `git checkout origin/paper -- steer_f` |
| **백업은 이름이 아니라 상태로 막힌다** | `hf_backup.sh`의 `LIVE_TAG`는 기본값이 `_0905`였고 그 접미사는 `run_0905_chain.sh`만 붙인다 — 캠페인 런은 전부 통과했다. 그리고 검증이 못 잡는다: 로컬 크기와 Hub 크기를 비교하는데 잘린 파일은 **자기 자신의 잘린 사본과 크기가 같다.** 지금은 (1) 이 런 이름을 커맨드라인에 가진 트레이너, (2) `FRESH_MIN`(기본 30분) 안에 쓰인 디렉토리, (3) `LIVE_TAG` 셋 중 하나라도 걸리면 REFUSE하고, 대신 올릴 수 있는 런을 찍는다. 완주하지 않은 런은 **경고만** 한다(죽은 런의 체크포인트도 보관 가치가 있다). `FORCE=1`이 전부를 덮는다 |
| **★ OOM에서 micro-batch를 줄이면 처치가 바뀐다** | `ppo_micro_batch_size_per_gpu=8`은 메모리 상수가 아니라 **STEER의 min–max 풀**이다(`run_steerf.sh:125`). 줄이면 그 시드는 옆 시드들과 다른 방법이 된다. `update_policy` OOM에서 쓸 수 있는 건 `OFFLOAD=1`(optimizer/param을 CPU로, ~12 GiB)과 `GPU_MEM_UTIL`(vLLM KV 풀)뿐이다. `LOGP_MBS`는 log-prob 패스 전용이라 정책 업데이트 OOM에는 효과가 없다. 큐가 이제 OOM을 판정해 `OFFLOAD=1`로 한 번 재시도한다 |
| **★ CUDA 에러가 권하는 `expandable_segments:True`를 따라가면 안 된다** | vLLM sleep mode가 cuMem API로 풀을 잡는데 expandable segments와 충돌해 **엔진 생성에서 assert로 죽는다**(pytorch#147851, `run_steerf.sh:38-41`). 지금 값은 `garbage_collection_threshold:0.8`이고, 에러 메시지는 이 스택을 모른다 |
| **크래시가 런 전체를 날렸다 (고침)** | `SAVE_CONTENTS="['hf_model']"`는 optimizer/rng state를 안 쓰므로 `resume_mode`가 재개할 대상이 없다(`run_steerf.sh:154`). 큐는 그러면서 "재개할 수 있게 남긴다"고 찍었다 — step 47 OOM이 25시간을 통째로 날린 이유다. 세 큐가 이제 `['hf_model','model','optimizer','extra']` + `RESUME_MODE=auto` + `MAX_CKPT_KEEP=1`로 돌고, 체크포인트가 3.1 → ~25 GiB가 되므로 `MIN_FREE_GB`가 20→30이다. 끝난 런은 `PRUNE=1 bash run/hf_backup.sh <run>`으로 차액을 돌려받는다 |
| **보상 채점기 의존성** | `word2number`가 없으면 **step-0 검증에서** 죽는다 — 모델 다 올리고 생성까지 끝난 뒤다. `verl/utils/reward_score/__init__.py:58`이 지연 import를 하고 그 끝에 `qwen_math_eval_toolkit/parser.py:7 → from word2number import w2n`가 있다. 로그만 보면 환경 문제로 안 보인다(런이 한참 돌다 죽는다). `pip install word2number sympy`. `_check_deps.py`가 이제 둘 다 요구하고, `env_preflight`이 그 모듈을 **큐 시작 전에** import해본다 |
| **심볼이 있다고 부를 수 있는 건 아니다** | 두 계보는 이름뿐 아니라 **시그니처**도 다르다. `measure_ah_support.py`가 `entropy_advantage`를 import하는 데는 성공하고 `TypeError: unexpected keyword argument 'response_ids'`로 죽었다 — 도너 쪽은 `(h_togo_vals, group_index, mask, responses=…)`를 받고 텐서를 돌려주는데 이 브랜치 쪽은 `response_ids=`/`group_size=`를 받고 2-튜플을 준다. `bootstrap_pod.sh`가 이 파일도 도너 것으로 덮고, `_check_steer_f.py`가 큐가 부르는 스크립트의 **호출부까지** 대조한다 |
| **`run_uniform_ablation.sh`는 `_gpu_defaults.sh`를 안 본다** | :83-84가 `N_GPUS=${N_GPUS:-2}`를 하드코딩하고 **export**한다. 자식 `run_steerf.sh`의 감지도 같이 막힌다. 4장 박스에서 tree arm 6개가 2장, steer arm 3개가 4장으로 갈렸다 — 같은 표 안에서. 수치는 안 바뀐다(`ppo_micro_batch_size_per_gpu=8`이 min–max 풀이고 장수와 무관)지만 **arm 간 불일치**가 문제다. `_arms.sh`의 `gpu_topology`가 한 곳에서 정하고 `topology_guard`가 거부한다. `bash run/run_status.sh` 2절이 로그에서 잡아낸다 |
| **★ base 체크포인트는 에러 없이 조용히 죽는다** | `meta-llama/Llama-3.2-3B`(base)로 프로브를 돌렸더니 MATH500 `.020`, GSM8K `.024`가 나왔다. 원인은 수학 실력이 아니라 **포맷**이다: Meta가 base 토크나이저에도 chat template을 실어 배포했고, `scripts/_common.py:221`과 verl의 `RLHFDataset`은 템플릿이 **있으면** 적용하므로 명령어 튜닝을 한 적 없는 모델이 명령어 포맷을 받는다. 프롬프트를 이어 쓰고 지시문을 반복한다. 보상은 멀쩡하다 — `multi_datasets_eval.py:327`이 `dapo_correct OR qwen_correct`라 `dapo:[INVALID]`만으로는 안 죽고, `reward/mean@1 = -0.96`은 정확히 `0.02·1 + 0.98·(−1)`이다. **`Mistral-7B-v0.3`도 base다.** 새 백본은 base인지 instruct인지 먼저 볼 것 |
| **★ 검증셋 점수는 "배울 수 있나"를 답하지 않는다** | 검증셋은 **잴 수** 있나를 재고, 학습 가능성은 **학습셋의 그룹 pass rate**가 정한다. 정답률 2%면 `.98^8 = .85`의 GRPO 그룹이 전부-오답 → advantage 0 → gradient 기여 0이고, STEER도 STEER-F도 그 위에 얹히므로 같이 죽는다. 백본을 넣기 전에 `scripts/phase3_port_model.py passrate`를 돌린다. 임계는 절대값으로 고르지 말고 **학습이 되는 백본(Qwen2.5-Math-1.5B)을 같은 명령으로 먼저 재서 그 `informative_frac`의 비율**로 준다 — 그 도구의 채점기(`phase1_validate.answers_match`)는 학습 보상과 다르므로 절대값이 보상률이 아니다. 40분이 140 GPU-시간을 산다 |
| **`--min-informative` 미달의 처방은 백본 교체다** | 난이도 하위 서브셋으로 갈아타면 백본마다 학습셋이 달라져 **백본 간 비교 자체가 무너진다**. 원고 §11.1이 전 백본 DAPO-Math-17k를 쓴다고 적는다. base 체크포인트면 같은 계열의 instruct 변종부터 본다 |
| **hub 핀** | 재시작마다 되돌아간다. 1단계가 유일한 방어. 두 번 당했다 |
| **`Cuda failure 401` / NCCL unhandled cuda error** | NCCL 버그가 아니다. `401 = CUDA_ERROR_ILLEGAL_STATE`. **로그에서 먼저 `nvls`를 찾아라** — `grep -n "nvls\|NCCL WARN" logs/experiments/train-*.log \| tail -20`. `transport/nvls.cc:158`이 있으면 NVLink SHARP(멀티캐스트)이고, 컨테이너가 멀티캐스트를 못 쓰면 정확히 거기서 401이 난다 → **`NCCL_NVLS_ENABLE=0`으로 기동**. H100+NVSwitch에서만 나는 경로라 A100×2는 애초에 안 탄다 — 같은 NCCL 2.21.5가 한쪽에서만 죽는 이유가 이것이다. `nvls`가 없을 때만 ①죽은 런이 남긴 `ray::`·vLLM 프로세스가 VRAM을 잡고 있나 (`nvidia-smi`) ②`/dev/shm`이 작나 (`df -h /dev/shm`)를 본다. 정리: `ray stop --force; pkill -f 'main_ppo\|ray::\|raylet'; sleep 10`. 진짜 이유는 `NCCL_DEBUG=INFO`로 |
| **flash-attn 은 선택이 아니다** | `verl/workers/actor/dp_actor.py:43`이 `if is_cuda_available:` 아래에서 `flash_attn.bert_padding`을 **무조건** import한다(옆의 `elif`는 Ascend NPU용, sdpa 폴백 아님). torch가 바뀌면 `.so`가 ABI 불일치로 죽고 **워커 초기화에서** 런이 끝난다 — 아래 §flash-attn 참조 |
| **`run_steerf.sh`의 모델 기본값이 7B** | `run_grpo.sh`·`run_uniform_ablation.sh`는 1.5B인데 `run_steerf.sh:58`만 `Qwen2.5-Math-7B`다. 캠페인의 steer arm이 그걸 직접 부른다. 캠페인이 시작된 파드에선 그 파일이 손으로 1.5B로 고쳐져 있었고 **커밋되지 않았다** — 새 파드가 커밋된 트리를 받자 steer arm이 `steer-...-1.5B-s5` 이름으로 **7B를 학습**하려 했다. 지금은 `_arms.sh`가 `MODEL_PATH`를 export하고 `model_guard`가 이름과 모델이 다르면 REFUSE한다 |
| **ray 핀 / opentelemetry 줄다리기** | `ray 2.58.0` + `vllm 0.8.4`가 정답이다 — 학습 중인 박스가 그 조합이다. otel을 vllm 선언(`<1.27`)에 맞추면 **ray 대시보드가 import에서 죽고** `ray.init()`이 타임아웃한다(에러에 ray도 opentelemetry도 안 나온다). ray에 맞추면 vllm 선언이 깨지는데, **vllm은 그걸 재검사하지 않는다** — `check_env_pins.py`가 `DECLARED ONLY`로 분류하는 쪽이고 실제로 학습된다. **ray 쪽이 이긴다.** `RAY_PIN=2.58.0`으로 고정해 두 박스를 같은 스택으로 유지할 것 |
| **Ray가 안 뜨면 먼저 잔해부터** | 죽은 `ray.init()`은 `gcs_server`·`raylet`·`/tmp/ray/session_*`을 남기고 다음 런이 그걸 물려받아 같은 자리에서 죽는다. `ray stop --force && rm -rf /tmp/ray`. 큐는 이제 런마다 자동으로 한다 |
| **opentelemetry가 Ray를 죽인다** | `pip install wandb`가 올린 1.44가 vllm 핀만이 아니라 **Ray 대시보드**를 깨서 `ray.init()`이 타임아웃한다. 에러 메시지에 opentelemetry도 pip도 안 나온다. `check_env_pins.py`가 이제 `ray`도 감시하고, `env_preflight`이 `ray.init`까지 확인한다 |
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

→ **H100 박스는 `N_GPUS=4 TP_SIZE=4`를 쓴다.**

> ⚠️ **"자동 감지하므로 따로 지정할 것 없다"고 여기 적혀 있었는데 틀렸다.**
> `run_steerf.sh:48`과 `run_grpo.sh:69`는 `_gpu_defaults.sh`를 source하지만
> **`run_uniform_ablation.sh:83-84`는 안 한다** — `N_GPUS=${N_GPUS:-2}`를 하드코딩하고
> 그걸 **export**해서, 자식으로 부르는 `run_steerf.sh`의 감지까지 막는다.
> 그래서 2026-09-14 H100에서 followups **9개 중 6개(tree arm)가 2장**, 3개가 4장으로
> 돌았다. 하필 `xclip-signed`(2장)와 `xclip-steer`(4장)처럼 **직접 짝지어 비교하는
> 쌍**이 갈렸다. 지금은 `_arms.sh`의 `gpu_topology`가 한 곳에서 정하고 큐가 export하며
> `topology_guard`가 확인한다. 기동 명령에도 남겨둔다 — 나중에 "이 박스가 왜 이
> 토폴로지인가"에 명령줄이 답한다.

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
hf auth login                                        # 이 파드에서 따로 해야 한다
REPO=DSDSh/steer-f_2 bash run/migrate_pod.sh --share
```

**파드마다 따로 로그인해야 한다.** 한쪽에서 `hf auth login` 해도 다른 쪽은 모른다.
그리고 **올리는 쪽 토큰은 write 권한이 있어야 한다** — fine-grained 토큰이면 그 레포가
Write로 명시돼 있어야 하고, read 전용이면 업로드가 401로 죽는다.

토큰은 **로그인 시점의 `HF_HOME` 아래**에 저장된다. `setup_env.sh`가 `HF_HOME`을 바꾸라고
권하므로, 로그인한 뒤에 다른 `HF_HOME`을 export하면 **토큰이 사라진 게 아니라 안 보이게**
된다. 스크립트 0절이 지금 어떤 `HF_HOME`이 걸려 있고 토큰 파일이 거기 있는지 찍어준다.

⚠️ 토큰을 `--token` 같은 인자로 주지 마라 — `~/.bash_history`와 `ps aux`에 남는다.
대화형 프롬프트에 붙여넣는다.

**`--export`가 아니라 `--share`다.** 두 파드가 **동시에** 도는 건 이전이 아니라 분업이고,
`--export`는 그 전제로 쓰이지 않았다:

| | `--share` | `--export` |
|---|---|---|
| 올리는 것 | 매니페스트 + gitignore된 산출물 | 거기에 **학습 체크포인트 전부** |
| 더러운 git | 경고만 (파드를 안 지우니 유실 위험이 없다) | 거부 |
| 학습 중 안전한가 | ✅ 체크포인트를 아예 안 건드린다 | 끝난 런만 올린다(상태로 판정) |

`--export`를 학습 중인 박스에서 돌리면 위험했다: `hf_backup.sh`의 라이브 가드는 런 이름에
`_0905`가 들어갈 때만 발동하는데 캠페인 런 이름엔 접미사가 없어서, **verl이 쓰는 중인
디렉토리를 올리고 바이트 검증까지 통과시킨다.** 지금은 `--export`도 `hf_backup.sh`도
이름이 아니라 **상태**로 판정한다 — 아래 함정 표 참조.

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
  "cd /workspace/entropy_collapse && NCCL_NVLS_ENABLE=0 N_GPUS=4 TP_SIZE=4 \
   ROLE=followups REPO=DSDSh/steer-f_2 \
   bash run/run_paper.sh > logs/experiments/paper_h100.log 2>&1"
```

> `NCCL_NVLS_ENABLE=0`은 **이 박스에만** 필요하다. H100+NVSwitch의 NVLink SHARP
> (멀티캐스트) 전송을 끄고 일반 NVLink로 폴백한다 — 컨테이너에 멀티캐스트가 없으면
> FSDP의 첫 브로드캐스트가 `transport/nvls.cc:158`에서 `Cuda failure 401`로 죽는다.
> 대용량 allreduce 처리량만 줄고 **수치는 안 바뀐다**. A100×2에는 NVSwitch가 없어
> 이 경로를 애초에 안 타므로, 이 변수는 두 박스를 갈라놓는 게 아니라 **오히려
> 같은 전송으로 맞춘다**. 코드에 안 박고 기동 명령에 두는 이유는 박스 특성이기
> 때문이다 — 나중에 "왜 이게 켜져 있지"에 명령줄이 답한다.

`ROLE`이 그 박스의 몫에 이름을 붙인다. **기본 `STAGES`는 "전부"라서, 두 번째 박스를
그냥 띄우면 캠페인까지 돈다** — 2026-09-13에 실제로 그랬다.

| ROLE | STAGES | 도는 것 |
|---|---|---|
| `campaign` | `preflight recover campaign` | 5 arm × 5 시드 본 표. **한 박스가 소유** |
| `followups` | `preflight measure followups eval2` | 측정 4종 + ablation 9개 + 그 평가 |
| `final` | `eval analysis` | 6벤치 평가 + 분석. 학습 없음 |

`ROLE`과 `STAGES`를 같이 주면 거부한다.

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


---

# flash-attn 다시 깔기

`undefined symbol: _ZN3c105ErrorC2E...` 는 "없음"이 아니라 **지금 torch와 다른 버전으로
빌드된 `.so`**라는 뜻이다. verl이 무조건 import하므로 건너뛸 수 없다.

## 1. 박스의 세 값을 읽는다

```bash
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI)"
# 예: 2.6.0+cu124 12.4 False
```

## 2. 미리 빌드된 wheel (1분) — 가능하면 이쪽

파일명 규칙:

```
flash_attn-<ver>+cu12torch<M.m>cxx11abi<TRUE|FALSE>-cp<XY>-cp<XY>-linux_x86_64.whl
                 └cu12      └torch 2.6   └위의 세 번째 값  └python 3.12 면 312
```

`torch 2.6.0+cu124 / ABI False / py3.12`이면:

```bash
pip uninstall -y flash-attn flash_attn
pip install --no-cache-dir \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
```

(2026-09-13 확인: 200, zip 매직 `PK`, 179 MB. `2.7.3`·`2.8.0.post2`도 같은 패턴으로 존재.)

미리 빌드된 wheel은 빌드를 안 하므로 `--no-build-isolation`이 필요 없다.

**버전은 학습 중인 박스에 맞춰라** — flash-attn은 어텐션 커널이라 수치에 직접 닿는다:

```bash
python3 -c "import flash_attn; print(flash_attn.__version__)"
```

## 3. 소스 빌드 (30~60분) — wheel이 없을 때만

```bash
pip uninstall -y flash-attn flash_attn
pip install flash-attn --no-cache-dir --no-build-isolation
```

**두 플래그가 다 필요하다.** `--no-cache-dir`는 예전 torch로 빌드된 캐시 wheel을 막고,
`--no-build-isolation`은 **설치된** torch에 대고 빌드하게 한다. 후자가 없으면 pip이 격리
환경에 자기 torch를 받아서 같은 ABI 불일치를 그대로 다시 만든다.

## 4. 확인

```bash
bash -c '. run/_arms.sh; env_preflight "$PWD"' && echo "PREFLIGHT OK"
```
