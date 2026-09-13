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
| **`run_steerf.sh`의 `STEPS=200`** | SCALE case에 하드코딩이라 `export STEPS`가 안 닿는다. 캠페인은 `steer_plain_args()`를 trailing override로 붙여 막는다 |
| **로그 글롭** | `train-<run>.log` + `train-<run>_*.log`만 허용해야 한다. 밑줄이 없으면 `-permuted`를 삼켜 signed를 DONE으로 오판정한다. `tests/test_run_names.py`가 고정한다 |
| **`run_paper.sh` preflight** | `-paper` 접미사 MTP 파일은 `measure` 스테이지 전용이다. 학습 arm은 안 쓴다 |
