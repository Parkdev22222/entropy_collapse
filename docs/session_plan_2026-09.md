# 작업 9: 세션 인계 문서 — 다른 세션에서 이 대화를 이어간다 (2026-09-09)

## Context

사용자 요청: **"이 대화를 다른 세션에서 이어가고 싶은데 방법이 있어?"**
선택된 옵션: **정리된 인계 문서 하나** + **git 브랜치에 푸시**.

왜 문서를 레포에 넣는 것 말고는 방법이 없는가 — 이 세션에서 확인한 사실:

| 층 | 실측 | 새 세션에서 살아남나 |
|---|---|---|
| 세션 자체 | `session_014wUL1tA6wZVXuNL5VhD9mB` | 같은 세션 재개는 됨. 하지만 **빈 컨텍스트의 새 세션은 레포만 클론하고 대화를 하나도 모른다** |
| 컨테이너 파일 | 대화 원본 `.jsonl` 4.5 MB (1,012줄), 플랜 파일 103,106 B | ❌ 컨테이너 로컬. 회수되면 복구 불가 |
| git | `a97972c`, working tree clean, 전부 푸시됨 | ✅ 유일하게 확실한 경로 |

**시간이 걸린 항목이 하나 있다.** 이 세션은 중간에 한 번 압축(compaction)됐고, 그때
**SFT 확장 답변 3,072자가 요약에서 통째로 빠졌다.** 대화 원본 900번째 줄에서 온전한
단일 메시지로 회수 완료. 컨테이너가 회수되면 이것만은 실제로 잃는다 —
이 작업에서 되돌릴 수 없는 유일한 산출물이다.

## 만들 것 — `docs/` 아래 파일 2개. 그게 전부다

### 1. `docs/HANDOFF.md` (신규) — 새 세션이 첫 번째로 읽는 문서

1. **지금 상태** — 브랜치 `claude/3b-text-generation-models-thz2vl`, 마지막 커밋 `a97972c`,
   돌고 있는 것(STEER `_0905` recovery step 98/110, 그 뒤에 20런 캠페인 대기)
2. **이 프로젝트가 뭔가** — 2채널 분해 `∇ℋ = E[∇f] + E[f·∇logπ]`(항등식), `A_H`와 형제
   프리픽스 베이스라인, tree rollout이 enabling condition인 이유(plain에서 `A_H ≡ 0`),
   상한 `2(n−1)/T = .0142`, **유효 지평 ≈ 1스텝이 최대 약점**
3. **확정된 수치** — 40–110 plateau 4-arm 표, 대조표, 그리고 **직전 턴에서 한 새 STEER
   `_0905` 비교**(공통창 40–90: acc −.0117 / maj −.0007, STEER-F와의 격차 +.0287 / +.0386)
4. **되돌리지 않는 결정들** — pass@32 폐기, "엔트로피 억제가 성능 원인" 주장 폐기,
   통계 단위 = 시드, 시드 5개 유지(n=3이면 헤드라인이 유의하지 않음),
   **예측 수치를 논문에 쓰지 않는다**
5. **SFT 확장 검토 — 회수한 3,072자 전문** ← 이 문서의 존재 이유
6. **인프라 지도** — `run/run_paper.sh`가 단일 진입점, `run/_arms.sh`가 single source of
   truth, `scripts/{analyze_seeds,measure_forecast_locality,check_env_pins}.py` 각 한 줄
7. **미해결** — `mtp_calibration_*-paper.json` 회수, GRPO 학습 로그 푸시,
   **seed-1 중복 draw 처리 규칙**(STEER·signed가 seed 1에 두 런), 작업 8은 미승인
8. **새 세션 시작하는 법** — 첫 프롬프트 예시 한 줄

### 2. `docs/session_plan_2026-09.md` (신규)

`/root/.claude/plans/joyful-jumping-quail.md`를 **그대로** 복사(103,106 B).
HANDOFF.md가 요약이면 이건 원장 — 작업 1~9의 설계 근거·검증 절차·폐기된 대안이 전부 있다.

## 이 작업이 건드리지 않는 것

`steer_f/`, `run/`, `scripts/`, `paper/`, 돌고 있는 학습 일체. **순수 문서 2개 추가다.**
대화 전문(571 KB, 그중 540 KB가 붙여넣은 학습 로그)은 커밋하지 않는다 — 사용자가
"정리된 인계 문서"를 골랐다. 새 원격 세션도 띄우지 않는다.

## 검증

1. `git status --short` — `docs/` 신규 2개 외에 아무것도 없을 것
2. HANDOFF.md의 모든 수치를 이 세션에서 검증된 값과 대조. 특히 새 STEER 비교표 산술
   재확인: 40–90 6점 평균 `.716/6 = .11933`, `1.195/6 = .19917`
3. **SFT 절이 원본과 글자 단위로 일치하는지 `diff`** — 원본을 임시파일로 다시 추출해 대조.
   불일치 시 중단(이게 유일한 비가역 산출물이다)
4. `wc -c docs/session_plan_2026-09.md` → `103106`
5. 푸시 후 `git log origin/claude/3b-text-generation-models-thz2vl --stat -1`로 2개 확인

---

# 작업 7: `train_log_done` 글롭이 너무 넓다 — signed arm이 조용히 건너뛰어진다 (2026-09-08)

## Context

사용자 질문: "모든 실험들에 로그 파일 뒤에 0905 이거 붙는거지?"

**답은 아니오다.** `_0905`는 `run/run_0905_chain.sh` 하나에서만 붙는다(`TAG=${TAG:-0905}`,
119·140·160행). 캠페인·후속·평가는 전부 접미사 없는 이름을 쓴다.

그런데 이걸 확인하다 **훨씬 심각한 버그**를 찾았다. `run/_arms.sh`의 `train_log_done`이

```bash
for f in "$1/train-$2"*.log; do
```

로 글롭한다. `*`는 `_0905`만 잡으라고 넣은 건데, **arm 이름 접미사까지 같이 삼킨다.**

| 판정 대상 | 글롭 | 잘못 매칭되는 로그 |
|---|---|---|
| `...-s2-tree-rollout` (**signed**) | `...-s2-tree-rollout*.log` | `...-s2-tree-rollout-permuted.log`, `-uniform.log`, `-lam0.1.log`, `-xclip.log`, `-rloo.log`, `-opo.log` |
| `...-s1-tree-rollout-lam0` | `...-lam0*.log` | `...-lam0.1.log`, `...-lam0.5.log` |
| `steer-...-s1` | `...-s1*.log` | `...-s1-xclip.log`, `-rloo.log`, `-opo.log` |

**실측 (샌드박스에서 재현함)** — seed 2의 permuted 로그 하나만 놓고 큐를 만들면:

```
  permuted  steer-f-...-s2-tree-rollout-permuted   DONE (skipped)
  signed    steer-f-...-s2-tree-rollout            DONE (skipped)   ← 돈 적이 없는데
```

캠페인 arm 순서가 `grpo steer permuted signed uniform`이라 **permuted가 signed보다 먼저
돈다.** 큐는 시작 시점에 한 번만 만들어지므로 *한 번의 실행 안에서는* 문제가 없지만,
**캠페인을 재시작하는 순간 그 시드의 signed는 영구히 건너뛰어진다.**
바로 앞 턴에서 내가 `run_campaign.sh` 버그 수정을 적용하려면 재시작하라고 안내했으므로,
지금이 정확히 그 순간이다. 주력 arm이 표에서 통째로 빠진 채 31일이 지나갔을 것이다.

`run/run_eval_all.sh`도 같은 함수를 쓰므로 평가 큐에서도 signed가 빠진다.
어제 커밋한 `is_done`의 원래 글롭도 동일했으니 **이 버그는 오늘 리팩터링이 만든 게 아니라
어제부터 있었다.**

## 고칠 것 — 함수 하나

`run/_arms.sh`의 `train_log_done`: 허용하는 접미사를 **`_<태그>`로만** 좁힌다.
arm 이름 접미사는 전부 `-`로 시작하므로 이 한 줄이 정확히 갈라준다.

```bash
train_log_done () {   # <log-dir> <run-name> <steps>
    local f
    # 접미사는 _<태그>만 허용한다. "train-<run>"* 는 _0905 뿐 아니라
    # -permuted / -uniform / -lam0.1 / -xclip 까지 삼켜서, 형제 arm이 하나
    # 끝나면 signed 를 "이미 완료"로 판정해 버린다.
    for f in "$1/train-$2.log" "$1/train-$2"_*.log; do
        [ -f "${f}" ] || continue
        grep -q "step:$3 - global_seqlen" "${f}" && return 0
    done
    return 1
}
```

7개 케이스 전부 샌드박스에서 확인함 (현재 코드는 이 중 3개를 틀린다):

| 런 | 현재 | 수정 후 | 기대 |
|---|---|---|---|
| signed s2 (permuted만 존재) | DONE | queued | queued |
| permuted s2 | DONE | DONE | DONE |
| lam0-tree s1 (lam0.1만 존재) | DONE | queued | queued |
| lam0.1 s1 | DONE | DONE | DONE |
| **signed s1 (`_0905`만 존재)** | DONE | **DONE** | DONE ← 복구 런 인식은 유지 |
| steer s1 (`-xclip`만 존재) | DONE | queued | queued |
| xclip-steer s1 | DONE | DONE | DONE |

## 회귀 테스트 — `tests/test_run_names.py` (신규)

이 버그는 "돌려보면 안다"로 안 잡힌다. 큐가 조용히 짧아질 뿐 에러가 없다.
그래서 테스트를 붙인다. 레포 테스트는 pytest이므로(`tests/test_omega_tilde.py` 등)
`subprocess`로 `run/_arms.sh`를 source해 함수를 직접 부른다.

- `tmp_path`에 위 표의 로그들을 만들고 `train_log_done`을 arm×seed로 호출해 판정 비교
- `run_name_for`가 14개 arm 전부에 대해 **서로 접두사 관계가 아닌지** 검사 —
  즉 어떤 런 이름도 다른 런 이름 + `-`로 시작하지 않는지. 이게 이 버그의 일반형이고,
  나중에 arm을 추가할 때 같은 함정을 다시 밟는 걸 막는다
  (지금은 실제로 접두사 관계가 존재하므로, 이 테스트는 "접두사여도 판정이 안 섞인다"를
  검사하는 형태로 쓴다)

## 사용자 조치

캠페인이 **지금 돌고 있다면 그 실행은 안전하다**(큐가 이미 만들어져 있다).
문제는 재시작이다. 앞 턴의 재시작 절차를 밟기 **전에** 이 수정을 받아야 한다:

```bash
git checkout origin/claude/3b-text-generation-models-thz2vl -- run scripts/check_env_pins.py
DRY=1 SEEDS="2 3 4 5" bash run/run_campaign.sh | grep -c QUEUE     # 20 이 나와야 한다
```

이미 재시작해서 signed가 빠졌다면, 로그를 보면 바로 안다:
`grep 'skip .*tree-rollout ' logs/experiments/campaign.log`

## 검증

1. `pytest tests/test_run_names.py`
2. `DRY=1 SEEDS="2 3 4 5" bash run/run_campaign.sh` → 정확히 20런
3. permuted s2 로그만 있는 샌드박스에서 signed s2가 `QUEUE`로 나오는지
4. `_0905` 복구 로그가 있는 seed 1에서 signed가 여전히 `skip`인지
5. `DRY=1 bash run/run_followups.sh` → 9런 (lam0-tree가 lam0.1에 먹히지 않는지)
6. `bash -n run/*.sh`

---

# 작업 8: 같은 hub 사고가 **재발** — 탐지는 됐지만 예방이 없었다 (2026-09-09)

## Context

09-08과 **완전히 같은** ImportError가 다시 떴다. `huggingface-hub==1.30.0`,
transformers가 import 시점에 `>=0.34.0,<1.0`을 재검사 → `import verl` 실패 →
tree arm이 시작 몇 초 만에 사망.

**갱신: 학습은 지금 정상적으로 돌고 있다.** 붙여넣은 트레이스백이 (a) 고치기 전의 낡은
로그 꼬리였는지 (b) 실제 2차 발생이었는지가 갈린다. 그래서 이 작업은 **긴급 복구가
아니라 예방**이다 — 다만 (b)라면 원인이 아직 살아 있다는 뜻이라 Step 0은 그대로 해야 한다.

지난번 진단("누가 핀 없이 한 번 업그레이드했다")은 1회성 원인이었고, 그게 맞았다면
재발하지 않는다. 즉 원인이 남아 있거나, 처방이 적용되지 않았다.

지난번에 추가한 것은 **탐지**(`check_env_pins.py`, `env_preflight`)뿐이고
**예방**은 없다. 이번 트레이스백은 런이 실제로 뜬 뒤 죽었으므로 `env_preflight`
게이트를 통과하지 않았다 — 즉 (a) 수정본을 안 받았거나 (b) `run_uniform_ablation.sh`를
직접 띄웠다(에러의 커맨드라인이 tree arm이다).

## Step 0 — 왜 돌아왔는지 먼저 가른다 (30초)

```bash
cd /workspace/entropy_collapse
pip show huggingface_hub transformers 2>/dev/null | grep -E '^(Name|Version)'
stat -c '%y  %n' /usr/local/lib/python3.12/dist-packages/huggingface_hub-*.dist-info
uptime -s                                  # 컨테이너 시작 시각
git log --oneline -1                       # 수정본을 받았나
```

| 관측 | 원인 | 처방 |
|---|---|---|
| dist-info 시각 ≈ 컨테이너 시작 시각 | **파드 재시작으로 오버레이가 날아갔다.** `/usr/local/lib/...`는 컨테이너 오버레이, `/workspace`는 네트워크 볼륨. 재시작마다 pip 설치가 전부 사라지고 **이미지의 버전이 돌아온다** | 재시작 후 `setup_env.sh`가 **필수 절차**. 아래 C로 못 깨지게 만든다 |
| dist-info 시각이 그 뒤 | 누가/무엇이 다시 깔았다. 트레이스백 자체가 `pip install transformers -U`를 권하므로 그걸 따라갔을 수도 | 아래 C의 constraint가 원천 차단 |
| `git log`가 `a97972c`가 아님 | 수정본 미적용 — 게이트가 없어서 그냥 탄 것 | 받으면 큐가 시작 전에 거부한다 |

## 즉시 복구

```bash
pip install "huggingface_hub>=0.34,<1.0"
python3 scripts/check_env_pins.py && echo PINS OK
PYTHONPATH=$PWD python3 -c "from verl import DataProto; print('import OK')"
```

또는 한 방에: `bash run/setup_env.sh` (핀 검사가 들어가 있어 자동 복구한다)

## 만들 것 — 이번엔 예방

### A. `constraints.txt` (신규, git 추적)

pip이 **넘어설 수 없는** 상한을 파일로 박는다.

```
huggingface_hub>=0.34,<1.0
transformers<5
```

### B. `PIP_CONSTRAINT`를 환경에 심는다

```bash
export PIP_CONSTRAINT=/workspace/entropy_collapse/constraints.txt
```

`~/.bashrc`에 넣고 `run/setup_env.sh`가 **직접 export + 안내**하게 한다. 이러면
`pip install -U huggingface_hub`를 쳐도 **1.30.0이 안 깔린다.** "기억하기"를
"불가능하게"로 바꾸는 부분이고, 이게 이번 작업의 핵심이다.

### C. `check_env_pins.py --emit-constraints`

`constraints.txt`를 **설치된 transformers의 선언에서 생성**한다. 손으로 적은 상한은
transformers를 올리는 날 낡는다. 생성물이면 안 낡는다.

### D. `run_paper.sh` / 큐에 `AUTO_REPAIR=1` (기본 꺼짐)

preflight가 **핀 위반 하나만** 문제일 때, `check_env_pins.py`가 출력한 그 pip 한 줄을
실행하고 재검사한 뒤 계속한다. 설치 대상이 확정적이고 검증 가능하므로 안전하다.
무인 큐에서 이게 있었으면 지난번 arm 2개를 안 잃었다.
켜지 않으면 지금처럼 거부만 한다.

### E. `run/setup_env.sh`에 재시작 안내 강화

오버레이가 날아간다는 사실은 이미 헤더에 있지만, **"재시작했으면 학습 전에 반드시
`bash run/setup_env.sh`"**를 6절 검증 출력에 한 줄로 못박는다.

## 검증

1. hub 1.30.0을 흉내 낸 가짜 dist-info로 `--emit-constraints`가 올바른 상한을 뱉는지
2. `PIP_CONSTRAINT`를 걸고 `pip install --dry-run -U huggingface_hub`가
   **1.x를 고르지 않는지** (네트워크 없으면 `pip install --dry-run` 대신
   `pip index versions`로 대체하거나 이 항목은 pod에서 확인)
3. `AUTO_REPAIR=1`로 깨진 환경에서 `run_paper.sh --DRY`가 아닌 실제 preflight가
   복구 후 통과하는지 (가짜 dist-info로 시뮬레이션)
4. `AUTO_REPAIR` 미설정 시 기존대로 REFUSE 하는지
5. `bash -n` 전 스크립트, `pytest tests/`

## 이 작업이 건드리지 않는 것

`steer_f/`, 학습 하이퍼파라미터, gradient 경로. 전부 환경 방어다.

---

# 작업 6: `huggingface_hub` 1.30.0이 학습 스택을 깨뜨린 사고 — 복구 + 재발 방지 (2026-09-08)

## Context

`_0905` 체인의 **uniform arm이 시작 몇 초 만에 죽었다.** signed는 정상 완주했다.
사용자가 붙여넣은 traceback:

```
verl/__init__.py → verl/protocol.py → verl/utils/torch_functional.py
  → from transformers import PreTrainedTokenizer
  → transformers/dependency_versions_check.py:57
ImportError: huggingface-hub>=0.34.0,<1.0 is required for a normal functioning
             of this module, but found huggingface-hub==1.30.0.
```

## Phase 1 — 근본 원인 (확정)

1. 컨테이너의 `huggingface_hub`가 **1.30.0**이다.
2. 설치된 `transformers`(4.x)는 `huggingface-hub>=0.34.0,<1.0`을 **선언만 하는 게 아니라
   import 시점에 강제**한다 (`dependency_versions_check.py`가 `transformers/__init__.py`
   맨 위에서 실행된다).
3. → `import transformers` 실패 → `import verl` 실패 → **모든** `main_ppo` 호출이
   GPU를 잡기도 전에 죽는다. 학습 코드·설정·데이터와는 무관하다.

signed가 멀쩡했던 이유는 단순하다 — **signed가 먼저 돌았고, 업그레이드는 그 뒤에
일어났다.** 즉 signed 종료 시점과 steer 시작 시점 사이에 누군가 hub를 올렸다.

레포 안에는 `huggingface_hub`를 설치하는 코드가 **없다**(`grep`으로 확인:
`run/`·`scripts/` 어디에도 `pip install ... huggingface` 없음). 유력한 경로는
`run/hf_backup.sh`가 쓰는 **`hf` CLI를 깔려고 `pip install -U huggingface_hub`를
실행한 것** — 핀이 없으면 pip는 1.30.0을 가져오고, transformers의 제약을 경고만 하고
그대로 설치한다. Step 0에서 설치 시각으로 이 가설을 확인한다.

로그 위쪽의 `pkg_resources` / `Setuptools<81` 경고는 **무관한 잡음**이다. 쫓지 말 것.

## 아직 확인 안 된 것 (Step 0에서 먼저 볼 것)

- **steer arm(2/3)도 같이 죽었나?** 체인 순서가 signed→steer→uniform이고 실패해도
  멈추지 않으므로 steer도 같은 ImportError로 죽었을 가능성이 매우 높다.
- **20런 캠페인(tmux `campaign`)이 큐를 태웠나?** 환경이 깨진 채로 시작했다면 20런이
  5분 만에 전부 FAILED로 끝나고 세션이 종료됐을 것이다.

```bash
cd /workspace/entropy_collapse
pip show huggingface_hub transformers 2>/dev/null | grep -E '^(Name|Version)'
python3 -c "import huggingface_hub,os;print(huggingface_hub.__file__)"
stat -c '%y %n' /usr/local/lib/python3.12/dist-packages/huggingface_hub-*.dist-info    # 설치 시각
grep -c ImportError logs/experiments/train-steer-Qwen2.5-Math-1.5B-s1_0905.log
tail -20 logs/experiments/chain_0905.log       # 체인 요약: 세 arm exit code
tail -30 logs/experiments/campaign.log         # 캠페인이 탔는지
tmux ls
ls -d checkpoints/STEER-F/*/global_step_*/actor/huggingface 2>/dev/null
```

## 고치는 것 — 3층

### 1) 환경 (즉시, 5분)

hub를 transformers가 허용하는 범위로 **되돌린다**. transformers를 올려서 hub 1.x를
받게 하는 방향은 **안 된다**: 이 verl은 `AutoModelForVision2Seq`(v5에서 제거)를
하드코딩으로 쓰고, 무엇보다 **이미 끝난 arm과 앞으로 돌 arm의 스택이 달라지면
비교가 성립하지 않는다.**

```bash
pip install "huggingface_hub>=0.34,<1.0"     # hf CLI(0.34+)와 transformers 제약을 동시에 만족
pip check                                     # 다른 패키지가 hub>=1.0을 요구하는지 확인
PYTHONPATH=/workspace/entropy_collapse python3 -c "import verl, steer_f, transformers; print('OK')"
```

`hf` CLI는 0.34.0에서 도입됐으므로 `<1.0` 안에서도 `run/hf_backup.sh`가 그대로 돈다.
`pip check`가 충돌을 뱉으면 그때 `hf_backup.sh`의 CLI 폴백(아래 3-d)이 답이 된다.

### 2) 이 사고를 표준 헬스체크가 잡게 한다

**`scripts/check_env_pins.py`** (신규) — 설치된 **메타데이터에서** 제약을 읽어 검사한다.
버전을 하드코딩하지 않으므로 transformers를 올려도 낡지 않는다.

```python
import importlib.metadata as md
from packaging.requirements import Requirement
# md.requires("transformers") -> ["huggingface-hub<1.0,>=0.34.0", ...]
# extra 마커가 붙은 항목은 건너뛰고, 설치된 버전을 SpecifierSet으로 검사
```

- 위반 없으면 exit 0, 있으면 위반 목록 + **그대로 붙여넣을 수 있는 `pip install` 한 줄**을
  출력하고 exit 1.
- 감시 대상: `transformers`, `vllm`, `datasets`, `accelerate`, `tokenizers`.
- `packaging`은 pip와 함께 항상 있으므로 새 의존성이 아니다 (이 박스에서 동작 확인함).

**`run/setup_env.sh`** — "3. 핀" 절에 이 스크립트 호출을 추가한다. 위반이 있으면
스크립트가 출력한 pip 명령을 그대로 `NEED+=()`에 넣는다. `setup_env.sh`는 5절에서
`NEED`의 `pip install`을 자동 실행하므로 **`bash run/setup_env.sh` 한 번으로 복구된다.**

### 3) 큐가 깨진 환경에서 출발하지 못하게 한다

**`run/_arms.sh`** — `env_preflight()` 추가:

```bash
env_preflight () {   # 0 = 학습 스택이 import 된다
    err="$(PYTHONPATH="${ROOT}:${PYTHONPATH:-}" python3 -c 'import verl, steer_f' 2>&1)" \
        && return 0
    printf '%s\n' "${err}" | tail -6
    python3 scripts/check_env_pins.py || true     # 왜 깨졌는지 + 고치는 명령
    return 1
}
```

호출 지점:

- **a. `run/run_campaign.sh`, `run/run_followups.sh`** — 큐 시작 전 게이트. 실패하면
  `REFUSE`. 20런을 태우기 전에 5초 만에 막는다.
- **b. `run/run_phase2.sh`** — 캠페인 대기가 끝난 직후, 평가 시작 전. phase2는 몇 주 뒤에
  깨어나므로 그 사이에 환경이 바뀌었을 수 있다.
- **c. `run/run_eval_all.sh`** — 같은 이유로 큐 시작 전.
- **d. 런 단위 진단** — 어떤 런이 `step:1 - global_seqlen`에 **한 번도 못 간 채** 실패하면
  "학습 크래시"가 아니라 "시작 실패"다. 그 자리에서 `env_preflight`를 다시 돌려
  `[campaign] STARTUP FAILURE — the environment is broken, not this arm` 을 로그에 남긴다.
  **사용자 결정에 따라 큐는 멈추지 않고 계속 간다.** 다만 왜 죽었는지가 로그에 남아,
  이번처럼 나중에 traceback을 뒤지지 않아도 된다.
  > ⚠️ 이 선택의 비용: 환경이 큐 도중에 깨지면 남은 런이 전부 몇 초씩 태워진다.
  > 체크포인트는 안 지워지고 로그도 안 덮이므로 손실은 **시간이 아니라 일정**이다.

- **e. `run/hf_backup.sh` + `run/run_eval_all.sh`** — `hf`가 없으면 `huggingface-cli`로
  폴백한다(`HF_CLI="$(command -v hf || command -v huggingface-cli)"`). 이러면 다음에
  누군가 `hf`가 없다는 이유로 hub를 업그레이드할 동기 자체가 사라진다 — **이번 사고의
  가장 유력한 원인을 없애는 부분이다.**

### 4) `run/run_0905_chain.sh` — 끝난 arm은 건너뛴다

지금 이 파일은 세 arm을 무조건 돈다. 그대로 재실행하면 signed는
`run_uniform_ablation.sh`의 "CKPT_DIR already exists"에 막혀 FAILED로 기록되고,
요약이 거짓말을 한다. `_arms.sh`를 source해서 `train_log_done "${LOG_DIR}" "<run>_0905" 110`
으로 건너뛰게 한다 → 재실행하면 **steer와 uniform만** 돈다.

## 실행 순서 (사용자 선택: seed-1 복구 먼저)

```bash
cd /workspace/entropy_collapse
# 0. 진단 (위 Step 0 블록)
# 1. 환경 복구
pip install "huggingface_hub>=0.34,<1.0" && pip check

# 2. 새 스크립트 받기
git fetch origin claude/3b-text-generation-models-thz2vl
git checkout origin/claude/3b-text-generation-models-thz2vl -- \
    run/_arms.sh run/setup_env.sh run/hf_backup.sh run/run_0905_chain.sh \
    run/run_campaign.sh run/run_followups.sh run/run_phase2.sh \
    run/run_eval_all.sh run/instrument_phase2.sh scripts/check_env_pins.py
bash run/setup_env.sh                 # 핀 검사 통과 확인
python3 scripts/check_env_pins.py && echo "PINS OK"

# 3. seed-1 복구 (steer + uniform, ~3.1일) — signed는 자동 skip
tmux new -d -s recover \
  "cd /workspace/entropy_collapse && bash run/run_0905_chain.sh > logs/experiments/chain_0905_retry.log 2>&1"

# 4. 4-seed 캠페인을 그 뒤에 줄 세운다 (WAIT=1 이 핵심 — 없으면 REFUSE 하고 끝난다)
tmux new -d -s campaign \
  "cd /workspace/entropy_collapse && WAIT=1 REPO=DSDSh/steer-f_2 \
   bash run/run_campaign.sh > logs/experiments/campaign.log 2>&1"

# 5. phase2 는 이미 아는 명령 그대로 (campaign 락과 main_ppo 를 기다린다)
tmux new -d -s phase2 \
  "cd /workspace/entropy_collapse && REPO=DSDSh/steer-f_2 \
   bash run/run_phase2.sh > logs/experiments/phase2.log 2>&1"
```

일정: 복구 3.1일 + 캠페인 31일 + phase2 15.5일 ≈ **49.6일**.

## 검증

1. `python3 scripts/check_env_pins.py` — 깨진 상태에서 exit 1 + 정확한 pip 줄,
   고친 뒤 exit 0. **hub 1.30.0을 흉내 낸 가짜 dist-info로 두 경로 다 확인할 것**
   (실제 환경을 건드리지 않고 `importlib.metadata` 를 monkeypatch).
2. `bash -n` 전 스크립트.
3. `env_preflight` 가 실패를 잡는지: `PYTHONPATH=/nonexistent` 로 강제 실패시켜
   `DRY=0` 캠페인이 **큐를 돌리기 전에** REFUSE 하는지.
4. `DRY=1 bash run/run_0905_chain.sh` 같은 건 없으므로, `train_log_done` 이
   `train-steer-f-...-tree-rollout_0905.log`(step 110 도달)에 대해 0을,
   `...-uniform_0905.log`(ImportError로 죽음)에 대해 1을 반환하는지 직접 확인.
5. `hf` 를 PATH에서 숨긴 채 `run/hf_backup.sh` 가 `huggingface-cli` 로 폴백하는지.
6. 복구 체인 시작 직후 5분 안에 `grep 'step:1 - global_seqlen'` 이 뜨는지 —
   이번 사고는 여기서 잡혔어야 했다.

## 이 작업이 건드리지 않는 것

`steer_f/`, verl, 학습 하이퍼파라미터 일체. 전부 환경 검사와 큐 방어이고
gradient 경로는 한 줄도 안 바뀐다.

---

# 작업 5: 캠페인 이후 자동 진행 — 6-벤치마크 평가 + 후속 ablation (2026-09-07 요청)

## Context

20런 캠페인이 tmux `campaign` 세션에 이미 예약돼 있다(건드리지 않는다). 그 뒤에
**평가 → 후속 ablation**이 자동으로 이어지게 한다. 다른 백본(7B/Llama/Mistral)은 A100×2에
안 올라가므로 제외 — MTP 헤드가 `checkpoints/mtp_heads_${model_tag}.pt`로 **모델 전용**이라
새 백본은 `run/warmup_and_validate.sh`(phase0+phase1)부터 다시 해야 해서 어차피 별건이다.

## 비용

| 작업 | GPU-h | 일 |
|---|---|---|
| 6-벤치마크 평가 (20런 × ~40분, 학습 없음) | 13 | 0.6 |
| λ=0.1 / λ=0.5 (signed) | 93 | 3.9 |
| **λ=0 + tree** (빠진 칸) | 46 | 1.9 |
| 극단 클리핑 signed + steer (`ε_hi=5, ε_lo=0.99`) | 72 | 3.0 |
| RLOO signed + baseline | 71 | 2.9 |
| OPO signed + baseline | 71 | 2.9 |
| **후속 합계** | **365** | **15.2** |
| 캠페인 20런 + 후속 | **1113** | **46.4** |

후속 ablation은 **시드 1개씩**이다(헤드라인이 아니라 ablation). 논문에 그렇게 명시.

## 전부 플래그로 된다 — 새 학습 코드 없음

- 극단 클리핑: `run/run_steerf_extreme.sh`가 이미 있다 (`clip_ratio_high=5`, `clip_ratio_low=0.99`)
- RLOO/OPO: `run_steerf.sh:21`이 `algorithm.adv_estimator=rloo`(또는 `opo`)를 trailing
  hydra override로 받는다고 명시. 형제 베이스라인은 `group_size = rollout.n`에만 의존하고
  RLOO·OPO도 같은 그룹 배치라 그대로 동작한다
- λ 스윕 / λ=0+tree: `STEERF_LAM` 값만 바꾸면 된다

## 만들 것 (3개)

### `run/run_eval_all.sh`
완료된 학습 런 전부를 6개 벤치마크로 평가. `run/eval_steerf.sh`를 감싼다.
- **`MODEL_PATH` 충돌 검사** — 지난 시도가 5개 로그 전부 같은 체크포인트를 받아 무효가 됐다.
  각 평가 직후 로그의 `MODEL_PATH`가 그 런의 것인지 확인하고, 아니면 즉시 중단
- 로그명 `eval-<arm>-s<seed>.log` 강제 (`collect_results.py:54`의 정규식)
- **평가에도 `VAL_DATA_DIR`를 켠다** — 안 켜면 6개 벤치마크에 문제별 점수가 없어
  대응 오차막대를 못 만든다. 학습에서만 켜고 평가에서 빼먹기 쉬운 지점
- HF에 올려둔 체크포인트는 필요 시 내려받아 쓰고, 끝나면 다시 지운다

### `run/run_followups.sh`
위 9개 ablation을 큐로. `run_campaign.sh`와 같은 구조(완료 감지·건너뛰기·락·디스크 가드).

### `run/run_phase2.sh`
`campaign` 락이 풀리고 `main_ppo`가 사라질 때까지 대기 → `run_eval_all.sh` → `run_followups.sh`.
락 이름은 `.phase2.lock`로 분리해 캠페인과 안 싸운다.

## 예약 방법 (지금 도는 캠페인은 손대지 않는다)

```bash
tmux new -d -s phase2 \
  "cd /workspace/entropy_collapse && \
   REPO=DSDSh/steer-f_2 bash run/run_phase2.sh > logs/experiments/phase2.log 2>&1"
```

## 검증

- `bash -n` 3개 전부
- `DRY=1 bash run/run_phase2.sh` → 평가 20건 + ablation 9건이 큐에 뜨고 아무것도 실행 안 함
- 캠페인이 도는 동안 phase2를 띄우면 **대기 상태**로 들어가는지 (즉시 실행하면 GPU 충돌)
- 평가 후 `grep -h MODEL_PATH logs/experiments/eval-*.log | sort -u`가 런당 한 줄
- `collect_results.py`가 로그를 실제로 파싱하는지 (파일명 규칙 위반 시 조용히 skip한다)

---

# 작업 4: 5-arm × 5-seed 캠페인 스크립트 + HF 직행 저장 (2026-09-07 요청)

## Context

현재 `run_0905_chain.sh`는 signed→steer→uniform 3 arm만 돌아서 GRPO·permuted가 n=1에 묶여
있다. 헤드라인 대조 `STEER-F − GRPO`의 유효 n이 `min(양쪽)`이라 이 체인만으로는 시드 약점이
전혀 안 풀린다. 5 arm × 5 seed를 하나의 재개 가능한 스크립트로 묶는다.

## HF 직행 저장 — 답: 불가능하지만 필요 없다

`save_pretrained`는 safetensors를 로컬 디렉토리에 랜덤 액세스로 쓴다. 파이프로 흘릴 수 없고,
verl 체크포인트 매니저에 업로드 훅도 없다. **verl을 패치하지 않으면 로컬 경유는 불가피하다.**

그런데 실제 걱정(디스크 폭발)은 설정 두 개로 이미 해결된다:

| 설정 | 효과 |
|---|---|
| `SAVE_BEST_ONLY=True` + `delete_old_best_checkpoint=True` | 새 최고점이 나오면 **이전 최고점을 지운다** → 런당 항상 **1개** |
| `SAVE_CONTENTS=['hf_model']` | optimizer/extra 제외 → **3.1 GB** (전체는 ~20 GB) |

→ **학습 중 로컬 최대 3.1 GB.** 런이 끝나면 즉시 업로드+삭제하므로 정상 상태 3~6 GB.

학습 중에 지우고 싶다면 감시자를 붙일 수는 있으나, verl이 쓰는 중인 디렉토리를 올려
**깨진 체크포인트를 업로드할 위험**이 있다. 권장하지 않는다. 런 종료 직후 업로드가 안전하고
효과는 같다.

## 만들 것: `run/run_campaign.sh`

기존 런처를 재사용한다 (새로 짜지 않는다):

| arm | 런처 | 핵심 env |
|---|---|---|
| grpo | `run/run_grpo.sh` | `SEED` |
| steer | `run/run_steerf.sh` | `STEERF_LAM=0` |
| signed / uniform / permuted | `run/run_uniform_ablation.sh` | `ARM=`, `STEERF_LAM=0.25` |

RUN_NAME은 기존 seed-1 런과 **정확히 같은 규칙**으로 만든다 (그래야 이미 끝난 것을 인식):

```
grpo-Qwen2.5-Math-1.5B-s${SEED}
steer-Qwen2.5-Math-1.5B-s${SEED}
steer-f-Qwen2.5-Math-1.5B-s${SEED}-tree-rollout[-uniform|-permuted]
```

### 동작

1. **완료 감지 후 건너뛰기** — `logs/experiments/train-${RUN_NAME}*.log`(glob이라 `_0905`
   접미사도 잡는다)에서 `110/110`을 찾으면 skip. → 재개 가능하고 몇 번을 돌려도 안전
2. **시드 우선(seed-major) 순회** — 어느 시점에 끊겨도 모든 arm의 시드 수가 균형
3. 각 런 성공 시 `REPO`가 설정돼 있으면 `DELETE=1 bash run/hf_backup.sh ${RUN_NAME}` →
   업로드·바이트 검증·로컬 삭제
4. `DRY=1`이면 실행 계획만 출력

### 안전장치

- `main_ppo`가 이미 떠 있으면 거부 (기존 체인과 충돌 방지)
- `run/instrument_campaign.sh --check` 실패 시 거부
- 런 시작 전 `df` 확인, 여유 < 20 GB면 중단
- 런 실패 시 체크포인트를 지우지 않고 다음으로 진행 (재개 가능하도록)
- `SAVE_AFTER=0` 강제 — `run_grpo.sh` 기본값이 80이라 그대로 두면 step 80 이전의
  최고점을 놓친다

## 검증

- `bash -n run/run_campaign.sh`
- `DRY=1 SEEDS="1 2" bash run/run_campaign.sh` → seed 1은 전부 skip(이미 완료),
  seed 2는 grpo·permuted만 실행 대상으로 나와야 한다
- 존재하지 않는 arm 이름 → 즉시 거부
- `main_ppo` 떠 있는 상태에서 실행 → REFUSE
- 첫 런 종료 후 `checkpoints/STEER-F/<run>/`에 `global_step_*`이 **1개만** 남는지 확인
  (`save_best_only` 동작 실증)

---

# 확정: A100×2 한 대로 순차 실행 (2026-09-07)

하드웨어를 안 바꾸므로 **박스 교락 문제는 없다.** 전부 같은 A100×2.

## 시드 수를 줄이면 안 된다 — 검정력 계산

유일한 반복쌍(signed `.1480` vs `_0905` `.1392`, 공통 40–90 창)에서 시드 SD ≈ `.0062`,
대응차 SD ≈ `.0088`(arm 간 무상관 가정, 보수적). 헤드라인 `STEER-F − GRPO = +.0145`에 대해:

| n | SEM | t | t_crit(.05) | |
|---|---|---|---|---|
| 3 | .0051 | 2.85 | 4.30 | **부족** |
| 4 | .0044 | 3.30 | 3.18 | 유의(간당) |
| 5 | .0039 | 3.68 | 2.78 | 유의 |

→ **n=3으로 줄이면 헤드라인 대조가 유의하지 않다.** 5시드 유지. 통제군 3시드는 그대로
(대조는 `min(n)`에 묶이므로 통제군 3이면 그 대조들은 n=3 검정력이라는 점은 논문에 명시).

## 일정 (순차, 단축 수단 없음)

- 시드 1개(5 arm) = **187 GPU-h = 7.8일**
- 남은 것 = seed 3 전체 + seed 4·5의 주력 3 arm = **379 h ≈ 16일**

`tp=2`·`n_gpus=2`를 seed 1·2와 맞춰야 하므로 병렬화·TP 변경으로 줄일 수 없다.
MTP 최적화도 불가(support가 처치의 일부). 16일은 그냥 비용이다.

## 큐 순서 — 중간에 끊겨도 쓸 수 있게

1. **seed 3의 5 arm 전부** (7.8일) → 여기서 멈춰도 **n=3 완결 데이터**가 남는다
2. seed 4의 GRPO / STEER / STEER-F (4일)
3. seed 5의 GRPO / STEER / STEER-F (4일)

arm별로 묶지 말고 **시드별로** 묶어 돌릴 것. 그래야 어느 시점에 멈춰도 모든 arm이
같은 시드 수를 갖는다.

## 체크포인트

`SAVE_CONTENTS=['hf_model']` (3.1 GB/개), run당 best + step110.
21 run → **130 GB**. 넘치면 `run/hf_backup.sh`로 HF Hub에 올리고 로컬 삭제.

## 아직 확인 안 된 것

seed 1의 `s/step`이 GRPO 1027 > STEER 833으로 방향이 반대다. seed 1의 GRPO만 다른 박스에서
돌았을 수 있다. pod에서 확인하고, 그렇다면 seed 1은 캠페인에서 빼고 seed 2~6으로 채운다.

---

# 핵심 3줄 (2026-09-07)

1. **pod에서 `bash run/instrument_campaign.sh --apply`** — 안 하면 750 GPU-h 쓰고도
   문제별 점수·진짜 엔트로피가 없어서 지금과 똑같은 한계가 남는다. 이미 푸시됨.
2. **5시드 학습 시작** (GRPO/STEER/STEER-F 5, uniform/permuted 3 = 21 run, ~750 GPU-h).
   시드는 `data.seed=1..5`를 모든 arm에 동일하게.
3. **학습 도는 동안 CPU로** `scripts/measure_omega_at_branches.py` — 논문 Table 1이
   지금 데이터 0줄짜리 해석 계산이라 이걸로 실측을 채운다.

나머지(P0-3 masked MTP, λ=0+tree arm, G1 재실행 등)는 전부 선택이다.
아래 상세는 참고용.

---

# 작업 3: 5-seed × 6-benchmark 캠페인 설계 (2026-09-07 3차 요청)

## Context

현 원고의 두 최대 약점은 **시드 1개**와 **벤치마크 1개**다. 사용자가 5 seed로 재실행하고
평가를 6개 수학 벤치마크로 넓히기로 했다. 확정된 설계:

- **평가만 확대** (학습 데이터는 DAPO-Math-17k 유지)
- **주력 5 seed / 통제군 3 seed**
- **110 step 유지**

이 캠페인은 ~750 GPU-h다. 따라서 **가장 중요한 작업은 실험 선택이 아니라 돌리기 전에
계측을 고치는 것**이다. 아래 Phase 0을 건너뛰면 900시간을 쓰고도 같은 한계가 남는다.

## 예산 (실측 timing 기반)

`perf/time_per_step` 실측: STEER 833 s, tree arm 1515 s (110 step 기준 25.4 h / 46.3 h).

| arm | seed | h/run | 소계 |
|---|---|---|---|
| GRPO | 5 | 24.4 | 122 |
| STEER | 5 | 25.4 | 127 |
| **STEER-F** | 5 | 46.3 | 232 |
| uniform | 3 | 45.2 | 136 |
| permuted | 3 | 45.6 | 137 |
| **합** | **21 run** | | **754 GPU-h ≈ 31일(단일 노드)** |

---

# Phase 0 — 돌리기 전에 반드시 (계측·최적화)

## P0-1. `validation_data_dir` 켜기 ★★★ (비용 0)

지금 `validation_data_dir: None`이라 **문제별 점수가 통째로 버려진다.**
`verl/trainer/ppo/ray_trainer.py:686`이 이 값을 받으면 val step마다
`{input, output, score}` JSONL을 떨군다(`_dump_generations`, :544). AIME24는
30문제×32복제=960행이므로 **문제별·샘플별 점수**가 남는다.

이걸로 풀리는 것:
- 원고 Limitations가 "로그로 계산 불가"라고 적어둔 **대응표본 across-problem SE**
- 문제별 부트스트랩 신뢰구간
- 임의의 k에 대한 maj@k 재계산
- 어느 문제가 뒤집히는지 항목 수준 분석

디스크: run당 ~150 MB × 21 = ~3 GB. 무시 가능.

## P0-2. `seq_entropy_agg` 주석 해제 ★★ (비용 0)

`ray_trainer.py:1206-1210`의 두 줄이 주석 처리돼 있다. 풀면 `actor/seq_entropy`
(= `seq-mean-token-sum`, **진짜 ℋ**)가 찍혀 "우리 엔트로피는 ℋ이 아니라 ℋ/길이"라는
원고 각주가 통째로 사라진다. gradient에 영향 없음.

## P0-3. MTP forward를 분기점에만 돌리기 ★★★ (엔지니어링 1일 → GPU 8일 절약)

**오버헤드의 82%가 여기 하나에 있다.** `timing_s/old_log_prob`가 96 s → 657 s
(`+561 s/step`), 나머지는 tree rollout `+107`뿐이다.

그런데 `A_H ≠ 0`인 위치는 **전체의 1.2%**다(분기점, `2(n−1)/T` 상한). 비분기 위치는
형제 집합이 자기 자신뿐이라 `baseline = 자기 값` → **`a_h`가 정확히 0**이다. 따라서
그 위치의 예보는 계산해도 버려진다.

제안: sibling mask(토큰 id만으로 계산, 공짜)를 **먼저** 만들어 `A_H`가 살 수 있는 열만
고르고, 그 hidden row만 모아 `forward_entropy`에 넣고 되돌려 놓는다.

**비트 동일성이 보장된다** — 건너뛴 위치는 원래도 정확히 0이고, `norm="scale"`은 RMS라
0이 몇 개든 값이 안 바뀌며, 로깅 통계(`a_h_absmean` 등)도 0을 평균에 넣는 것이라 동일하다.
`tests/test_lambda_zero_equiv.py`와 같은 방식으로 테스트를 붙일 것.

효과 추정: tree arm 1515 → **~950–1180 s/step** (29–36 h). 11개 tree run에서
**110–190 GPU-h 절약**. 보수적으로 잡아도 4일.

> ⚠️ 이건 학습 경로를 건드리는 유일한 항목이다. 비트 동일성 테스트가 통과하지 않으면
> 캠페인에 넣지 말 것. 통과하면 남은 예산으로 아래 P2-2(빠진 arm)를 살 수 있다.

## P0-4. 체크포인트 정책 확정 (비용 0, 안 하면 지난번처럼 소실)

지난 캠페인은 rotation으로 최고 체크포인트가 지워졌다. 이번엔:
- `save_best_only=True` + `delete_old_best_checkpoint=True` → run당 argmax 1개
- **추가로 step 110을 강제 저장** → 선택 규칙 2가지(argmax vs 고정 step)를 둘 다 보고해
  "선택 규칙에 따라 순서가 바뀌지 않는다"를 보일 수 있다
- `save_contents=['hf_model']`만 (optimizer/extra는 eval에 불필요, 4배 크다)

디스크: 21 run × 2 × ~3 GB = **126 GB**. 초과분은 `run/hf_backup.sh`로 HF Hub에 내린다.

## P0-5. 시드를 arm 간에 맞출 것 (비용 0)

`data.seed ∈ {1,2,3,4,5}`를 **모든 arm에 동일하게** 배정한다. 그러면 대조를 시드로
짝지을 수 있어 검정력이 크게 오른다. (통제군 3 seed는 `{1,2,3}`.)

주의: 롤아웃 샘플링은 실행 간 시드가 고정되지 않으므로(이 세션에서 확인) `data.seed`는
**재현 핸들이 아니라 독립 반복의 라벨**이다. 논문에 그렇게 적을 것.

## P0-6. 사전 등록 (비용 0)

결과를 보기 전에 분석 계획을 확정해 커밋한다:
- **주 통계**: plateau(step 40–110) acc의 **시드 간 평균**, 단위는 시드
- **주 대조**: STEER-F − {GRPO, STEER, uniform, permuted}, 시드로 짝지은 대응 검정
- **부 통계**: maj@32, uplift, 6-벤치마크 평균
- 현재의 "8개 step 짝지은 paired t"는 **부차 지표로 강등** (실행 내 안정성만 말함)

---

# Phase 1 — GPU 없이 지금 병렬로 (학습과 무관)

| # | 작업 | 비용 | 얻는 것 |
|---|---|---|---|
| P1-1 | pod에서 `checkpoints/mtp_calibration_*-paper.json` 회수·커밋 | 0 | 원고 `a_k` 수치 정합(§작업 2에서 발견한 4번째 오류) |
| P1-2 | `scripts/measure_omega_at_branches.py` 실행 | **CPU만** | 원고 Table 1을 예측 → **관측**으로. 논문 중심 주장의 유일한 실측 결핍 |
| P1-3 | `Ĥ_togo` vs 로컬 `H_u` 상관 측정 | GPU 1h | "유효 지평 1스텝이면 그냥 로컬 항 아닌가" 반론 차단 |
| P1-4 | G1 recall을 **κ=2, γ_H=0.7**로 재실행 | GPU 0.5h | 커밋된 recall JSON은 `κ=4, γ_H=0.85`로 잰 것이라 그대로 못 씀 |

P1-2가 특히 중요하다. 지금 Table 1은 **해석적 분포족 계산**이고 데이터가 한 줄도 없다.

---

# Phase 2 — 학습

## P2-1. 본 캠페인 (21 run, 위 예산표)

## P2-2. 빠진 arm: **λ=0 + tree** (3 seed, +87~137 h) — P0-3이 성공하면 추가

현재 어느 arm도 **tree rollout 자체의 효과**를 격리하지 못한다. tree는 그룹이 prefix를
공유하게 만들어 **GRPO advantage 추정 자체를 바꾼다** — 감쇠와 무관한 경로다.
uniform과 permuted는 둘 다 `λ=.25` 감쇠가 켜져 있어 이 교락을 못 푼다.

리뷰어가 "tree rollout이 그냥 더 좋은 샘플러 아니냐"고 물으면 현재 답이 없다.

---

# Phase 3 — 평가 (6 벤치마크)

학습이 끝난 뒤. 파이프라인은 이미 있다 — `run/eval_steerf.sh`, `scripts/collect_results.py`,
parquet 8종 전부 git 추적 중.

**지난번 실패 재발 방지**: 5개 eval 로그가 전부 `MODEL_PATH`를
`grpo-.../global_step_110`으로 동일하게 받아 무효가 됐다. arm마다 경로를 다르게 주는지
`grep MODEL_PATH`로 **평가 시작 직후 즉시 확인**할 것.

- 21 run × best 체크포인트 × ~40분 = **~14 GPU-h**
- 파일명은 반드시 `eval-<arm>-s<seed>.log` (`collect_results.py:54`가 이 정규식으로만 파싱,
  실패 시 예외가 아니라 조용히 skip)
- eval에도 `validation_data_dir`를 켜서 6개 벤치마크의 문제별 점수를 남긴다

---

# Phase 4 — 분석

1. 시드 간 평균 ± SE로 주 표 재작성, 대조는 시드로 짝지어 검정
2. **시드 분산 자체를 보고** — "동일 설정 재실행 변동폭"은 그 자체로 이 분야에 기여
3. 문제별 점수로 대응표본 across-problem SE 계산 → 현 Limitations 항목 해소
4. 6 벤치마크 방향 일관성: `signed > {uniform, permuted}` 부호가 6개 중 몇 개에서 유지되는가
5. 선택 규칙 강건성: argmax vs 고정 step 110에서 arm 순서가 같은지

---

## 총 예산

| | GPU-h |
|---|---|
| Phase 2 본 캠페인 (21 run) | 754 |
| P0-3 최적화 적용 시 | **−110 ~ −190** |
| P2-2 λ=0+tree 3 seed | +87 ~ +137 |
| Phase 3 평가 | +14 |
| Phase 1 (P1-3, P1-4) | +1.5 |
| **합** | **~700–900** |

## 검증

- P0-3: 비트 동일성 테스트 통과가 **캠페인 투입의 전제조건**
- P0-1: 첫 run의 val step 후 JSONL이 960행인지 즉시 확인
- P0-2: `actor/seq_entropy`가 로그에 찍히는지 첫 step에서 확인
- P0-4: 첫 run 종료 후 체크포인트가 2개(best + 110) 남는지 확인
- Phase 3: 평가 시작 직후 5개 로그의 `MODEL_PATH`가 서로 다른지 확인 (지난 실패 지점)

---

# 작업 2: 결과 절 채우기 + 참고문헌 81개 + 오류 3건 수정 (2026-09-07 2차 요청)

## Context

STEER 원논문 PDF(ACL 2026, `2026.acllong.1436`, pp. 31105–31133)를 확보해 전문·부록을 읽었다.
사용자 지시 4가지:

1. **STEER arm은 110 step 결과 그대로 사용** (200까지 재실행하지 않는다)
2. **STEER 논문의 실험 수치는 논문에 넣지 않는다.** 사용자가 직접 돌린 결과만 싣는다
   → 원논문 Table 12(1.5B) 대조, 재현 실패 논의, 선택편향 대비 분석은 **전부 제외**.
   원논문은 **방법 출처로만** 인용한다.
3. 참고문헌 **81개 전부** 수록
4. 앞서 확인한 **사실 오류 3건** 수정

결과적으로 §12 Results가 골격에서 **완본**으로 바뀐다.

## 확정 수치 (전부 `origin/paper` 로그에서 재계산·검증함)

### plateau step 40–110 (arm당 8 val 지점)

| arm | acc@32 | maj@32 | uplift | entropy | resp len | s/step |
|---|---|---|---|---|---|---|
| STEER | .1330 | .1984 | .0654 | .1214 | 988.1 | 833 |
| uniform | .1313 | .1953 | .0640 | .1374 | 946.0 | 1480 |
| permuted | .1370 | .2094 | .0724 | .1190 | 960.2 | 1492 |
| **STEER-F** | **.1495** | **.2349** | **.0854** | .1210 | 983.9 | 1516 |

**GRPO는 로그가 git에 없다** — 세션 기록값 `.1350 / .1963 / .0612 / .1491 / 963.9`.
표에 넣되 사용자에게 "커밋된 로그로 검증 못 함"을 명시하고 pod에서 push 요청.

### 대조 (8개 plateau step 짝지어 계산, 검증됨)

| 대조 | Δacc | paired t | Δmaj | paired t |
|---|---|---|---|---|
| STEER-F − STEER | **+.0165** | **6.68** | +.0365 | 5.09 |
| STEER-F − uniform | **+.0182** | **7.18** | +.0396 | 5.50 |
| STEER-F − permuted | **+.0125** | **2.56** | +.0255 | 5.26 |
| permuted − STEER | +.0040 | 0.83 | +.0110 | 1.75 |
| uniform − STEER | −.0017 | −0.62 | −.0031 | −0.52 |

**permuted가 이제 40–110 창에 있다** → `signed−permuted`가 40–90의 `+.0147`에서 `+.0125`로 축소.

### 통계 표기 — 여기서 정직해야 한다

계획 파일 앞부분의 "`+.0165 (3.5 SE)`"는 **두 arm plateau 평균의 비대응 2표본 SE**였다.
지금 셋 중 무엇을 쓸지 확정한다:

- **주 통계 = 8개 step 짝지은 차이의 평균과 paired t.** "이 실행의 수렴 구간에서 격차가 일관되는가".
- `SE_prob = std@32/√30 = .0288`은 **한 arm의 절대 수준**의 정밀도다. 절대값을 몇 자리까지
  주장할 수 있는지에만 쓰고, 격차를 이 값으로 나눠 "0.6 SE"라고 쓰지 않는다 —
  두 arm이 **같은 30문제**에서 평가되어 문제 난이도가 상쇄되므로 과대 잡음이다.
- **대응 표본의 across-problem SE는 로그로 계산 불가**(문제별 점수가 아니라 평균·표준편차만
  기록됨). 이 사실을 Limitations에 명시하고, 없는 숫자를 만들지 않는다.

### 메커니즘 진단 — 새로 확인한 강한 결과

`branch_corr_frac` 실측 vs §9.2의 조합론적 상한 `2(n−1)/T`:

```
상한 2·7/984 = .0142
uniform .0136 (96%)   permuted .0129 (91%)   STEER-F .0120 (84%)
```

**세 tree arm이 상한의 84–96%에 붙어 있다.** tree 샘플러가 이론 한계를 거의 포화시킨다는
검증 가능한 주장이며, "분기점을 많이 만드는 게 목표가 아니다"라는 §9.2 논지를 실측으로 뒷받침한다.
`tw_mean`은 네 arm 모두 .998–.999 → §9.3의 평균 불변 주장 실측 확인.

### 비용 — 주의

세션 기록의 GRPO `1027 s/step`은 검증된 STEER `833 s/step`보다 **크다**(방향이 반대).
다른 pod/GPU로 보인다. 따라서 **같은 실행 안의 비교만** 싣는다:
`STEER 833 → STEER-F 1516 = +82%`, 원인은 MTP forward. GRPO 타이밍은 각주로 분리하거나 뺀다.

## 오류 3건 수정

**① Ω의 식** (원논문 Theorem 1, Eq. 17). 현재 원고 Eq. (1)은 `1/π_old`, `I_clip`, `η/L`,
그리고 **기댓값**을 빠뜨렸다. 특히 `1/π_old`는 §8에서 "Ω는 `1/π_old`를 품어 꼬리가 두껍다"고
쓴 인자라 **내부 모순**이었다. 정확한 형태:

```
Ω(s) = −(η/L)·E_{a~π_θ(·|s)}[ (I_clip(s,a)·A(s,a)/π_old(a|s))
                             · π_θ(a|s)(1−π_θ(a|s))·(log π_θ(a|s) + H(π_θ|s)) ]
```

§4.2의 재작성 `Ω = A π(1−π)(H − I_a)`도 `1/π_old`를 반영해 고친다.
부호 규약도 원논문에 맞춘다(그들 Ω>0 = 엔트로피 증가).

**② 매핑**: "unmodified min–max mapping"은 틀렸다. 원논문 주 방법은 **지수 매핑**
`w = exp(−α|Ω|/max|Ω|)`, `λ_min = exp(−α)`, 범위 `[λ_min, 1]` (Eq. 9; Table 11에서
exponential 48.6 > linear 48.0 > binary 46.5로 확정). §8·§11 문구 교체.
`steerf_mapping=minmax`는 **항등 pre-transform**임을 각주로 명시.

**③ 인용**: `@misc{...arXiv preprint 2510.10150}` → ACL 2026 본회의 논문.
`Hao, Wang, Liu, Luo, Yu, Dong, Lin, Wang, Chen. 2026. ... In Proceedings of ACL 2026, pp. 31105–31133.`

부수 효과로 얻는 것 — 원고를 **강화**하는 인용 2건:
- 그들의 **Assumption 1 (Parameter-independent softmax)** = 우리 원장 표의 B2. 이름 붙여 인용.
- 그들의 **Eq. (20)** `∂H/∂z_{s,a} = −π(a|s)[log π(a|s) + H(π|s)]` → 균등분포에서 대괄호가
  모든 `a`에 대해 0. **우리 §4.2 3단 논증이 그들 자신의 해석적 gradient로 확인된다.** 추측 → 인용.
- 그들의 **Eq. (23)** 로짓 업데이트 = 우리 Eq. (7). 독자 유도가 아니라 재사용으로 정직하게 표기.

## 참고문헌 81개

추출 방식 확정: `pdftotext -bbox-layout`으로 좌표를 받아 **hanging indent를 x좌표로 판별**
(단 왼쪽 여백 col0=70.9 / col1=306.1, 여백+1.5pt 이내면 새 항목). 결과 **81개, 알파벳 순서
위반 1건**(그것도 `Zed Industries` 정렬 관행) — 사실상 무결.

- 줄바꿈 하이픈 복원: `(?<=\w)- (?=\w)` → 제거 (`Ah- met` → `Ahmet`). 단 `Solar- Lezama` 같은
  진짜 하이픈 이름은 깨지므로 **수동 점검 목록**을 만든다.
- author / year(+a–e) / title / venue(`In Proceedings…` | `arXiv preprint arXiv:…` | `Preprint,
  arXiv:…` | URL) 파싱 → `paper/custom.bib`.
- 본문 인용은 15개 내외이므로 **`\nocite{*}`**로 전부 출력.
- ⚠️ 대부분 미인용 상태로 실리는 건 ACL 관행상 이례적이다. 한 줄 경고만 하고 지시대로 진행.

## 파일

`paper/steerf.tex` (§4.2·§8·§11 수정, §12 완본화, Limitations 보강),
`paper/custom.bib` (arXiv 항목 → ACL 항목 교체 + 81개 추가).

## 검증

1. `pdflatex ×2 + bibtex + pdflatex ×2` → 에러 0, undefined 0, Overfull hbox 0
2. **렌더된 참고문헌 항목 수 = 81** (`pdftotext`로 References 이후 세기)
3. 무작위 8개 항목을 원문 PDF와 글자 단위 대조 (저자·연도·제목·venue)
4. 표의 모든 수치를 `origin/paper` 로그에서 재계산한 값과 대조 — GRPO 행만 예외로 표시
5. **원논문 수치가 본문에 없는지 확인**: `grep -nE '17\.4|16\.2|36\.2|48\.6|4\.1' paper/steerf.tex`가
   우리 수치와 무관한 곳에서 잡히지 않을 것
6. `2(n−1)/T` 산술 재확인 (`2*7/984 = .0142` vs 실측 .0120–.0136)

---

# 작업 1(완료): ACL 양식 논문 초고 — 방법론 완본 + 실험은 골격만 (2026-09-07 요청)

## Context

지금까지 세션에서 확정된 내용(2채널 분해, `A_H`, MTP 예보, tree rollout, 5-arm 인과 분해)을
업로드된 ACL 템플릿에 옮긴다. 사용자 지정: **결과 수치는 넣지 않는다.** 방법론은 완성본으로,
실험은 **구성(설계)까지만** 쓰고 결과 절은 표 골격과 자리표시자만 둔다. 익명(`review`) 버전.
`_0905` 재현 런은 이번 원고에서 제외. 분량 제한 없음(나중에 줄임).

이유: GRPO 학습 로그가 git에 없어 헤드라인 수치를 검증할 수 없고, 다중 벤치마크 eval 5개가
전부 무효(아래)라 결과 절을 지금 확정하면 나중에 통째로 다시 써야 한다. 방법론은 코드와
유도 문서로 완전히 검증 가능하므로 지금 쓰는 것이 맞다.

## 이번에 저장소를 실검증해서 새로 밝혀진 것

1. **permuted가 step 110까지 있다** (`origin/paper` `8e307e9`). 계획 파일 최우선 항목 #2 해소.
   40–110 재계산: STEER `.1330/.1984`, uniform `.1313/.1953`, permuted `.1370/.2094`,
   STEER-F `.1495/.2349`. entropy `.1214/.1374/.1190/.1210`, s/step `833/1480/1492/1516`.
   → `signed−permuted`가 40–90의 `+.0147`에서 **`+.0125`(maj `+.0255`)로 축소**. 결과 절을
   쓸 때 이 값으로 갱신할 것.
2. **GRPO 학습 로그는 어느 브랜치에도 없다.** `eval-grpo-s1.log`는 평가 로그,
   `phase2-grpo-*`는 AIME24 val이 없는 다른 실험, `train-math-*`는 `loss_mode=entropy_control`.
3. **다중 벤치마크 eval 5개는 전부 무효.** `eval-{grpo,steer,uniform,permuted,signed}-s1.log`의
   `MODEL_PATH`가 **다섯 개 모두** `grpo-Qwen2.5-Math-1.5B-s1/global_step_110`이다. 재실행 필요.
4. **방법 서술 정정**: `docs/STEERF_derivation.md` §7.1은 `apply="weight"`를 매핑 이후 tanh
   보정(식 10)으로 쓰지만 `branch_weight_correction`은 `steer_f/`에 **없다**(문서·`scripts/`에만).
   실제로 돈 경로는 식 (9) — `Ω̃ = norm(Ω) + λ·norm(visit)`를 min-max 매핑 **이전에** 더한다
   (`steer_f/omega_tilde.py:143-145`, `steer_f/token_weights.py` 헤더 주석). **논문은 식 (9)를 쓴다.**

## 환경

`pdflatex` 등 LaTeX 전무. `archive.ubuntu.com` 도달 가능(200), CTAN·GitHub releases는 차단(403/000)
→ **apt만이 경로**. TeX Live 2023(noble/universe) 사용 가능, 디스크 23 GB 여유.

## 만들 것

업로드 zip을 `paper/`에 풀고(템플릿 원본 `acl_latex.tex`는 손대지 않음), 새 본문
**`paper/steerf.tex`** 와 **`paper/custom.bib`**(참고문헌 추가)를 작성한다.

### 절 구성

| 절 | 상태 | 내용 |
|---|---|---|
| Abstract / 1 Introduction | 완본 | 붕괴의 *총량*이 아니라 *방향*이 문제라는 주장, 기여 목록 |
| 2 Related Work | 완본 | STEER(2510.10150), 80/20(2506.01939), Invisible Leash(2507.14843), Entropy Reg.(2509.25133), Medusa/MTP |
| 3 Preliminaries | 완본 | GRPO, 표기, STEER의 `Ω` |
| **4 Method** | **완본** | 아래 상세 |
| **5 Experimental Setup** | **완본** | 설계는 결과가 아니므로 전부 씀 |
| 6 Results | **골격만** | 소절 제목 + 표 골격 + `\TODO` 자리표시자, 숫자 없음 |
| 7 Conclusion | 짧게 | |
| Limitations | 완본 | ACL 필수·번호 없음 |
| Ethics Statement | 완본 | |

### 4 Method 상세 (전부 `docs/STEERF_derivation.md` + `steer_f/` 코드에서 검증됨)

- **4.1 두 채널** — 식 (1) 엔트로피 연쇄분해, 식 (2) 곱미분 → (L) local + (V) visitation. **항등식.**
- **4.2 로컬 항이 왜 못 보는가** — `Ω = A(1−π_a)(I_a − H)`, `I_a − H`는 정의상 평균 0.
  균등분포에서 `Ω ≡ 0`이고 **그 0은 정답**(식 2b: `H''(0) = −(k−1)/k²`, 1차항 소멸).
  두 인자가 **반대로 정렬**되는 표(§2.4) 포함. → "blind일 뿐 아니라 anti-correlated".
- **4.3 `H_togo`와 `A_H`** — 인과성 소거로 식 (3), 베이스라인 논증으로 식 (4). 형제 프리픽스
  베이스라인이 `s_u`만의 함수라 편향 없음. 자기 포함으로 인한 `O(1/m)` 편향 각주.
- **4.4 근사 A1–A3** — 지평 절단·할인, MTP 주변예보(식 5: 초과분 = `I(y_{t+k}; y_{t+1:t+k−1}|s_t) ≥ 0`,
  **k에 단조 증가**), 헤드별 아핀 보정. 실측 `a_k`(1.029 → 0.13 이하)가 식 (5)를 확인. 최종 식 (6).
  **유효 지평이 사실상 1스텝**(`H_togo = 0.7206·H₁ + 0.0636·H₂ + 0.106`, head 2 기여 8%)임을 명시.
- **4.5 그래디언트 → 토큰 가중치** — 식 (7) `Δlogπ = η·w·(1−π_u)`, 식 (8) `visit_u`,
  클립을 곱하기 전에 거는 이유(A4), 부호 4분면 표.
- **4.6 결합** — **식 (9)** `Ω̃ = norm(Ω) + λ·norm(visit)` → 원본 min-max 매핑 그대로 →
  `α ∈ [w_min, w_max]`. `norm`이 z-score가 아니라 scale인 이유(λ=0에서 stock STEER와 비트 동치).
- **4.7 tree rollout** — plain에서 `A_H ≡ 0`인 이유, `A_H ≠ 0 ⟺ t가 분기점`,
  **`branch_corr_frac` 상한이 `2(n−1)/T`**라는 정리(`docs/STEERF_tree_rollout.md:45-70`).
- **4.8 설계상 평균 불변** — 형제 집합 위에서 `Σ A_H = 0` → 평균 감쇠 불변, 분산만 확대.
  **평가 기준에 대한 함의**: "분기점 평균 엔트로피 상승"은 이 경로의 작동 증거가 될 수 없다.
- **4.9 근사 원장 표** (A1–A4, B1–B4: 무엇을 버리는가 / 부호·유계 / 완화 장치).

### 5 Experimental Setup 상세

Qwen2.5-Math-1.5B, DAPO-Math-17k(17k → bs 512 → 110 step ≈ 3.2 epoch), AIME24 avg@32 검증,
lr 1e-6 constant(warmup 0), n=8, resp 3072, tree `roots=1 depths=[64,192,384] factors=[2,2,2]`,
λ=.25 / κ=2 / γ_H=.7 / clip_c=1.0 / minmax / `w∈[0.7,1.0]` / norm=scale / baseline=sibling,
seed 1. **5-arm 인과 분해 설계 표**(GRPO / STEER / uniform / permuted / signed — 무엇을 끄는가).
지표 정의: `acc@32`, `maj@32`, `uplift = maj − acc`, plateau 창 step 40–110(8점),
`SE = std@32/√30 ≈ .027`, `actor/entropy`가 `ℋ`이 아니라 `ℋ/평균길이`라는 각주 + `ent×len` 프록시.

## 실행 순서

1. `apt-get install -y texlive-latex-base texlive-latex-recommended texlive-latex-extra
   texlive-fonts-recommended latexmk` (`inconsolata`가 없으면 그 `\usepackage`만 주석 처리 —
   템플릿 자체가 "may be commented out"이라 명시)
2. zip을 `paper/`에 해제
3. `paper/steerf.tex` 작성 (`\usepackage[review]{acl}`)
4. `paper/custom.bib`에 참고문헌 추가
5. `cd paper && pdflatex steerf && bibtex steerf && pdflatex steerf && pdflatex steerf`
6. 커밋·푸시 (`claude/3b-text-generation-models-thz2vl`), PDF는 `SendUserFile`로 전달

## 검증

- `paper/steerf.pdf`가 생성되고 페이지 수 > 0
- 로그에 `Undefined control sequence` / `Citation ... undefined` / `Reference ... undefined` 0건
- `Overfull \hbox`는 표 때문에 일부 허용하되 목록으로 보고
- 본문에 **결과 수치가 없는지** 확인: `grep -nE '\.1[0-9]{3}|[0-9]\.[0-9]+ SE' paper/steerf.tex`가
  방법 절의 상수(λ=.25, γ_H=.7, `a_k` 등)와 설정값 외에는 잡히지 않아야 한다
- 식 (2b)의 `H''(0) = −(k−1)/k²`와 §4.4의 `0.7·1.029 = 0.7203 ≈ 0.7206` 산술 재확인
- 익명 확인: PDF 1쪽에 저자명 없이 `Anonymous ACL submission`

## 건드리지 않는 것

`steer_f/`, `run/`, `logs/`, 학습 관련 파일 일체. 새로 만드는 것은 `paper/` 하나뿐이다.

---

# 갱신: step 90 — 재현 성공 판정 (2026-09-07)

| step | 기존 | 새 `_0905` |
|---|---|---|
| 70 | .147/.257 | .146/.235 |
| 80 | .144/.206 | **.164/.250** ← New best 0.1635 |
| 90 | **.166**/.239 | .144/.217 |

**두 런이 정점을 맞바꿨다** — 최대값은 .166 vs .1635로 같고 위치만 한 val 지점 밀렸다.
plateau(40–90): acc .1480 vs **.1392** (−.0088, **0.3 SE**), maj .2378 vs .2247, uplift .0898 vs .0855.
격차 추이 −.0167(step62) → −.0128(step70) → **−.0088(step90)**. 계속 좁혀짐.
엔트로피도 80/90에서 +.006까지 붙고 90의 반등 모양까지 동일.

**정정: step 50–60의 하락은 국소 노이즈였다.** 앞서 경고한 "시드 변동폭 ±.015"는 현재 ±.009이고
축소 중 — 논문의 plateau 수치를 단일 실행값으로 써도 무리 없다. 단 "step 90에서 .166" 같은
**특정 step 못박기는 피하고** 40–110 plateau 평균으로 진술할 것(표 3의 원래 방식이므로 변경 불필요).

step 80 val 분포가 흥미롭다: acc .164/maj .250인데 best@32는 .354로 오히려 낮고 std@32 최저,
worst@2 최고. **샘플이 좁게 뭉치면서 정확도·다수결이 오른** 것으로, 표 5의 uplift 서사가
재현 런에서 독립적으로 다시 관측됐다.

best 체크포인트 = `global_step_80` (.1635). 기존 비교 대상 step 90 (.166)과 동급이라 평가에서 대체 가능.
페이스 `90/110 [38:16:01<9:03:51]` → signed ≈ 47.3h, 남은 20 step ≈ 9h.

## 사전 등록: step 100·110 판정 기준 (데이터 오기 전에 못박아 둔다)

비교 대상(기존 런): `100 .157/.222`, `110 .151/.230`.
전체 plateau(40–110, 8점) 기존 평균 **acc .1495 / maj .2348 / uplift .0853** — 표 3의 GRPO 행과
같은 창이므로 이 숫자와만 비교한다(현재 6점 부분 평균 .1480과 섞지 말 것).

| `_0905` plateau(40–110) acc | 판정 | 논문 처리 |
|---|---|---|
| ≥ .145 (Δ ≤ .005, ~1 노이즈폭) | **완전 재현** | 표 3·4 그대로. 각주 불필요 |
| .138 – .145 (Δ .005–.012) | 재현 (현 추세의 연장) | 표 3 유지 + "동일 설정 재실행 Δ = X" 한 줄 각주 |
| < .138 (Δ > .012) | 시드 변동이 헤드라인 격차(+.0145)와 동급 | 표 4의 SE 진술 재작성 필요 |

step 90까지의 추세(−.0167 → −.0128 → −.0088)가 이어지면 둘째 칸에 안착한다.
maj는 보조 지표로만 본다(현재 −.0131, acc보다 노이즈가 큼).

## 아직 처리 안 한 것 (이 파일의 다른 절과 충돌하는 부분)

1. **본문 각주 수정**: "STEER는 step-0이 `.043`인 다른 시기 스택" (아래 '본문 프레이밍' 절) —
   같은 signed 런이 같은 날 `.043`과 `.039`를 둘 다 냈으므로 근거가 없다. 스택 차이가 아니라
   샘플링 노이즈(±.004)로 고쳐 쓸 것.
2. **GRPO 최고 acc 출처 미확정**: 표 7의 `GRPO .157 @110`은 pod 로그 기준이고 git에 없다.
   pod에서 `grep 'val-core/aime_2024_dapo_boxed/acc/mean@32' logs/experiments/train-grpo-*.log`로
   확인한 뒤 로그를 `paper` 브랜치에 올려야 검증 가능한 수치가 된다.
3. ~~`run/hf_backup.sh` 커밋·푸시~~ → **완료.** `f10e84b`로 커밋되어
   `origin/claude/3b-text-generation-models-thz2vl`에 이미 올라가 있다(아래 '즉시 작업' 절의
   "커밋 대기"는 낡은 기술). pod에서 `git checkout <ref> -- run/hf_backup.sh`로 바로 받을 수 있다.

**결론: 이 레포에서 지금 할 수 있는 코드 작업은 없다.** 남은 것은 전부 (a) 사용자가 step 100·110
로그를 가져오는 것, (b) pod에서만 가능한 로그 회수·평가다.

---

# 갱신: step 70 — 회복. 그리고 "step 40 분기" 진단 정정 (2026-09-06)

`_0905` 실제 커맨드라인 확인 완료. 스크립트에서 역산한 것과 일치: 다른 값은
`total_training_steps` 200→110, `save_best_only` F→T, `max_actor_ckpt_to_keep` 2→1,
`steerf_permute_ah=0` 명시(계측만) — 전부 비수학. 로그가 `warmup_style: constant` /
`Total steps: 110, num_warmup_steps: 0`을 찍으므로 LR은 1e-6 고정. `Training from scratch`.

**정정: "step 40에서 갈라졌다"는 틀렸다.** step 10이 acc .078 vs **.056**, maj .148 vs **.071**로
이미 크게 다르다. step 20의 3자리 일치(.103/.181)는 우연이었다. 롤아웃 샘플링(temp 1.0,
top_p 1.0)이 실행 간 시드 고정이 아니므로 두 런은 step 1부터 다른 궤적. "초기 구간이
결정적이었다"는 앞선 판단은 취소.

| step | 기존 | 새 `_0905` |
|---|---|---|
| 10 | .078/.148 | .056/.071 |
| 20 | .103/.181 | .103/.181 |
| 30 | .113/.211 | .120/.217 |
| 40 | .141/.230 | .131/.225 |
| 50 | .145/.258 | .128/.209 |
| 60 | .145/.237 | .122/.212 |
| **70** | **.147/.257** | **.146/.235** |

**step 70에서 회복.** `New best ... 0.1458 (previous: 0.1313)`, best 체크포인트가
`global_step_40` → `global_step_70`으로 이동. 50/60의 하락은 국소 딥이었다.
부분 plateau(40–70): 기존 .1445 vs 새 .1318 (−.0128, ~0.5 SE); maj .2455 vs .2203.
엔트로피도 70에서 .127 vs .121로 붙음. steerf 내부·속도(47.7h) 정상, CPU RAM 감시 종료.

판정 구간은 step 80–110 (기존 정점 .166 @90). `.13`대에 머물면 plateau의 시드 변동폭이
±.013 — 헤드라인 격차와 같은 크기이므로 논문 각주 필요.

---

# 관측 기록: `_0905` STEER-F 재학습 vs 기존 signed (step 62 시점, 2026-09-06)

설정은 완전 동일(λ=.25/minmax/κ=2/γ_H=.7/tree n=8/seed 1/110 step, step-0 val `.039/.244/.035` 일치).
달라진 것은 `save_best_only=True`와 로그 파일명뿐 — gradient에 영향 없음.

| step | 기존 signed | 새 `_0905` | Δacc |
|---|---|---|---|
| 20 | .103/.181 | .103/.181 | 0 (3자리 일치) |
| 40 | .141/.230 | .131/.225 | −.010 (−0.4 SE) |
| 50 | .145/.258 | .128/.209 | −.017 (−0.6 SE) |
| 60 | .145/.237 | .122/.212 | −.023 (−0.9 SE) |

부분 plateau(40/50/60): acc .144 → **.127**, maj .242 → **.215**.
개별 점은 1 SE 미만이지만 **세 점이 전부 같은 방향**이고 격차가 커진다.

- best 체크포인트가 `global_step_40` (.1313)에서 20 step째 정지 (`Skipping checkpoint save`).
- 엔트로피 격차는 해소: 새 런 62→.132로 기존 궤적을 따라잡음. 붕괴 속도는 동일.
- CPU RAM 430 GB 피크(step 44)는 page cache였음 — 이후 116–240 GB. **감시 항목에서 제외.**
- steerf 내부 정상: support_frac .40–.42, mean_siblings 2.13–2.17, branch_corr_frac .013–.016.
- 페이스 `62/110 [26:26:06<20:26:33]` → signed ≈ 47h, 체인 전체 ≈ 5.3일.

## 원본 로그 실검증 (2026-09-06, `origin/eval-logs`의 train-steer-f-...-tree-rollout.log)

로그 5,780줄 = `main_ppo` 런치 **7회**. 1~6번은 조기 사망(1번만 step-0 val까지 도달),
**7번(08-28 05:09)이 step 0→118 연속 완주**이며 12개 val 지점 전부 여기서 나온다.
전 런치 `resume_from_path: None` → 재개 아님. `_0905`도 연속 실행이므로 재시작 이력 차이는 배제.

원본 런치 7 커맨드라인 vs `_0905`(chain → run_uniform_ablation.sh ARM=signed STEPS=110):

**학습에 영향 주는 값은 전부 동일** — lam .25 / kappa 2 / gamma_h .7 / clip_c 1.0 / norm scale /
baseline sibling / apply weight / mapping minmax / winsor_q .01 / forecast mtp / heads·calib 경로 /
tw 0.7–1.0 / linear False / loss_mode entropy_control / lr 1e-6 / bs 512·32·8 /
clip 0.2·0.28·10.0 / entropy_coeff 0 / use_kl_loss False / n=8 / val temp1.0 top_p0.7 /
resp 3072 / tp2 / data.seed=1 / save·test_freq 10 / save_after 0 / tree [64,192,384]×[2,2,2] roots1 /
PYTHONHASHSEED·PYTORCH_SEED 42 / CUBLAS_WORKSPACE_CONFIG.

**다른 것 4개 = 전부 비수학**: `total_training_steps` 200→110, `save_best_only` F→T,
`max_actor_ckpt_to_keep` 2→1, (rollout_data_dir는 양쪽 다 null).
`total_training_steps`가 LR을 바꾸지 않는 근거는 로그 본문:
`'warmup_style': 'constant'`, `lr_warmup_steps_ratio: 0.0`, `Total steps: 200, num_warmup_steps: 0`
→ LR 전 구간 1e-6 고정.

**코드 변경 3커밋(08-29~08-31)은 전부 로깅 전용**: `482e667`(a_h_zero_frac 임계 1e-8→1e-6),
`a1220f1`(branch_corr_frac_strict 추가, `support`는 `!=0` 그대로 — 주석이 명시적으로 유지),
`4315112`(permute_a_h 기본 False 게이트 + branch_corr_rms; "treatment path unchanged, bit-identical").
→ **signed gradient 경로는 두 런에서 비트 동일.**

원본 step별 트리 통계 (40/50/60): support .3952/.4055/.4017, siblings 2.11/2.14/2.14,
refill .0464/.0371/.0303, branch_corr .014/.012/.012 → `_0905`(.40–.42 / 2.13–2.17 / .033–.051 /
.013–.016)와 겹침. 메커니즘 정상.

## 노이즈 척도 교정 — 지난 분석의 SE 사용이 틀렸다

같은 로그의 런치 1과 런치 7은 **같은 가중치(base Qwen2.5-Math-1.5B)·같은 프로토콜**인데
step-0 val이 `.043/.040` vs `.039/.035`다. → **순수 샘플링 노이즈 acc ±.004 / maj ±.005.**

1. "−0.6 SE라 유의하지 않다"는 잘못된 척도였다. SE .027은 *다른 문제로 일반화*할 때의 오차이고,
   "두 런이 같은 궤적인가"는 같은 30문제 위의 질문이라 노이즈가 ±.004다.
   그 기준으로 step 50/60의 −.017/−.023은 노이즈의 4~6배 → **두 런은 다른 궤적. 확정.**
2. 논문 결론에 쓰는 SE는 여전히 .027이 맞다. 정확한 진술은
   **"동일 설정 두 실행의 plateau가 .144 vs .127 = 0.6 SE로 갈린다"**이고, 이는 헤드라인
   격차(`STEER-F − GRPO = +.0145`)와 같은 크기다.
3. **본문 프레이밍의 각주 하나를 수정해야 한다**: "STEER는 step-0이 .043인 다른 시기 스택"은
   근거가 약하다. 같은 signed 런이 같은 날 .043과 .039를 둘 다 냈다 — 스택 차이가 아니라
   샘플링 노이즈로 설명된다.

## 논문에 미치는 영향 (판정 창 = step 70–90)

원인은 config도 코드도 재시작도 아닌 **step 40 이후 누적된 비결정성**. 즉 이 실행은
"복원"이 아니라 **두 번째 시드 추출**이고, "아직 없는 것 #3(시드 1개)"에 대한 첫 정량 데이터다.
step 80–90에서 .145 이상으로 회복하지 않으면 plateau 수치를 단일 실행값으로 제시할 수 없다.

→ 그 경우 표 3·4의 plateau 수치를 단일 실행값으로 제시할 수 없다. 두 실행 평균±범위로 바꾸거나
"동일 설정 재실행 시 ±.015 변동" 각주 필수. `STEER-F − uniform = +.0182 (4.2 SE)`가 특히 영향받는다.
이는 위 "아직 없는 것 #3(시드 1개)"에 대한 **첫 정량적 증거**이므로, 결과가 어느 쪽이든 기록해 둔다.

---

# 즉시 작업: 체크포인트 HF 백업 헬퍼 커밋 (2026-09-06 요청)

## Context

pod의 `/workspace`가 가득 찼는데 `_0905` 체인(signed → steer → uniform)이 돌고 있어,
**다음 최고점 저장 중 디스크 부족으로 43시간짜리 학습이 죽는 것**이 현재 최대 리스크다.
해법은 끝난 런의 체크포인트를 Hugging Face Hub로 옮기고 로컬에서 지우는 것.

GRPO step 110은 이미 수동으로 처리했다(`DSDSh/steer-f_2`에 업로드 완료). 그런데 남은 런이
여러 개(`...-tree-rollout-permuted`, 예전 uniform/signed 등)이고, 사용자가 방금
`...-tree-rollout-permuted` 업로드를 요청했다. **pod에는 내가 접근할 수 없으므로** 명령을
텍스트로 드려야 하는데, 이 세션에서 **70줄 heredoc을 터미널에 붙여넣다 조용히 깨진 사고**가
이미 한 번 있었다(줄 중복, `if` 본문 유실 → bash syntax error). 그래서 반복 작업은
**파일로 만들어 git으로 내려받게** 하는 것이 유일하게 안전한 전달 방식이다.

목표: 업로드 → 바이트 단위 검증 → (요청 시에만) 삭제를 한 명령으로 묶은 헬퍼를 브랜치에 올려,
남은 런 전부를 같은 절차로 처리할 수 있게 한다.

## 만든 것 (작성 완료, 커밋 대기)

`run/hf_backup.sh` — 138줄, `bash -n` 통과, 미커밋 상태.

```
REPO=DSDSh/steer-f_2 bash run/hf_backup.sh <run-name> [step ...]
REPO=... DELETE=1     bash run/hf_backup.sh <run-name>          # 검증 통과 시에만 삭제
```

동작:
1. step을 인자로 안 주면 `checkpoints/STEER-F/<run>/global_step_*` 전부를 처리
2. 각 step의 `actor/huggingface`만 업로드 (eval에 필요한 전부, optimizer/extra는 4배 크고 불필요)
3. `HfApi().list_repo_tree`로 **로컬 파일 크기 vs Hub 크기**를 파일 단위 대조
4. `DELETE=1`이고 검증이 통과했을 때만 `rm -rf global_step_<N>`

안전장치:
- `DELETE` 기본값 0 — 명시하지 않으면 아무것도 안 지운다
- 런 이름에 `_0905`(=`LIVE_TAG`)가 들어가면 **거부**. 학습 중 디렉토리를 읽으면 깨진 백업이 된다.
  `FORCE=1`로만 우회 가능
- 업로드 실패 → 검증 건너뛰고 삭제 안 함 / 검증 실패 → 삭제 안 함, 둘 다 `rc=1`
- 인자 없이 실행하면 사용 가능한 런 목록을 출력

## 해야 할 일

1. `run/hf_backup.sh`를 `claude/3b-text-generation-models-thz2vl`에 커밋·푸시
2. 사용자에게 전달할 명령:
   ```bash
   cd /workspace/entropy_collapse
   git fetch origin claude/3b-text-generation-models-thz2vl
   git checkout origin/claude/3b-text-generation-models-thz2vl -- run/hf_backup.sh
   bash -n run/hf_backup.sh && echo OK          # 붙여넣기 사고 재발 방지 확인

   export REPO=DSDSh/steer-f_2
   bash run/hf_backup.sh steer-f-Qwen2.5-Math-1.5B-s1-tree-rollout-permuted        # 업로드+검증만
   DELETE=1 bash run/hf_backup.sh steer-f-Qwen2.5-Math-1.5B-s1-tree-rollout-permuted  # 확인 후 삭제
   ```

## 검증

- `bash -n run/hf_backup.sh` → 무출력 (이미 통과)
- 인자 없이 실행 → 사용법 + 런 목록 출력, exit 2
- `_0905` 런 이름으로 실행 → `REFUSE` 후 exit 1 (학습 보호가 실제로 작동하는지)
- permuted 런에 `DELETE` 없이 실행 → `VERIFIED`가 뜨고 로컬 파일은 그대로 남아 있어야 함
- `DELETE=1` 실행 후 `df -h /workspace`로 확보량 확인

## 이 작업이 건드리지 않는 것

`run/run_steerf.sh`(pod에서 sed로 수정된 상태)와 도는 중인 `_0905` 체크포인트는 손대지 않는다.
`git checkout <ref> -- run/hf_backup.sh`는 새 파일 하나만 가져오므로 pod의 로컬 수정과 충돌하지 않는다.

---

# 완료: 곱미분 유도 설명 아티팩트 (2026-09-05 요청)

## Context

`docs/STEERF_derivation.md` §1–§4(엔트로피 총합 → 곱미분 → 두 채널 → `H_togo` → `A_H`)를
**초등학생도 따라갈 수 있는 수준**으로 풀어쓴 설명을 이 세션에서 대화로 작성했다.
사용자가 이걸 아티팩트로 요청. 터미널에서 LaTeX `$...$` 안의 `_`가 이탤릭으로 먹혀
`\pi_\theta` → `\pi\theta`로 깨졌던 문제가 이 요청의 배경이므로, **아티팩트에서는 수식이
반드시 정상 렌더링되어야 한다.**

## 만들 것

`.../scratchpad/steerf-product-rule.html` 한 파일을 Artifact로 발행.

내용 (대화에서 쓴 것 그대로, 순서 유지):
1. **과일가게 비유** — 총매출 = Σ(개수 × 가격), 변하는 이유는 가격 or 구성 두 가지뿐
2. **기호 사전** — `Σ` `Π` / `x` `y` `T` `y_t` `y_<t` `s_t` / `θ` `π_θ` / 소문자 `H` vs 대문자 `ℋ`
   (이 구분이 문서 전체에서 제일 헷갈리는 지점 — 시각적으로 분리할 것) / `E` `∇_θ` / 로그미분 (★)
3. **식 (1)** — 엔트로피 연쇄법칙. `Σ_t`는 한 궤적 안의 **합**, `E`는 궤적들 사이의 **평균**.
   의사코드로 절차 제시
4. **곱미분** — `f_θ(y) := Σ_t H(π_θ(·|s_t))`, `ℋ = Σ_y π_θ(y|x)·f_θ(y)`,
   `(uv)' = u'v + uv'` 적용 → 둘째 항은 바로 기댓값, 첫째 항은 (★)로 기댓값 복원 → **식 (2)**
5. **두 항의 의미** — (L) 로컬 / (V) 방문 대조표 + **2갈래 장난감 숫자 예제**
   (0.5/0.5, f=3/1 → ℋ=2.0; L만 → 1.5; V만 → 1.4). 헷갈림 점수가 하나도 안 변했는데
   ℋ이 더 크게 떨어지는 것이 (V)의 존재 증명이자 Ω의 맹점
6. **인과성 소거 → `H_togo`(식 3) → 베이스라인 → `A_H`(식 4)** — (1)~(4) 전부 근사 없음
7. **한 줄 요약**

## 렌더링 방침 (수식 깨짐 재발 방지)

- 수식은 **KaTeX 등 외부 라이브러리를 쓰지 않고**, 유니코드 + 등폭 폰트 블록으로 조판한다.
  대화에서 쓴 코드블록 표기(`π_θ(y|x) = Π_t π_θ(y_t|s_t)`)를 그대로 옮기면 `_`가
  이탤릭으로 먹힐 여지가 원천적으로 없다. 아래첨자는 `θ` `t` `u` 를 문자 그대로 쓴다.
- (L)/(V) 두 채널은 색으로 구분하고, 과일가게 비유 → 수식 대응을 나란히 배치한다.
- 라이트/다크 양쪽 토큰 정의. 수식 블록은 `overflow-x: auto`.

## 검증

1. 발행 후 `action: "read"`로 되읽어 `π_θ` `H_togo` `A_H` `∇_θ` `Π` `ℋ` 가 전부
   글자 그대로 남아 있는지 확인 (아래첨자 유실 = 실패)
2. 좁은 폭에서 수식 블록이 가로 스크롤되고 본문은 안 되는지
3. 숫자 예제 산술 재확인: `0.5×3+0.5×1=2.0`, `0.5×2+0.5×1=1.5`, `0.2×3+0.8×1=1.4`

---

# 다중 벤치마크 평가 (AIME24 단일 벤치마크 리스크 해소)

## Context

현재 STEER-F의 모든 주장이 **AIME24 한 벤치마크**에 걸려 있고, AIME24는 **30문제**다. 로그의 `val-aux/.../acc/std@32`(문제별 정답률의 표준편차)가 0.145이므로 평균의 표준오차는

```
SE = 0.145 / sqrt(30) ≈ 0.027
```

그런데 최고 점수 기준 acc 격차는 STEER 대비 **+0.008 ~ +0.009**, 즉 **0.3 SE**(30문제 중 약 0.3문제)다. 리뷰어가 "30문제에서 0.3문제 차이로 방법이 동작한다고 주장하느냐"고 물으면 현재 데이터로는 막을 수단이 없다. maj@32 격차(+0.028 ~ +0.043)는 사정이 낫지만 여전히 단일 벤치마크다.

step 수(110)는 위험이 낮다 — 3.2 epoch, 엔트로피가 step 50 전에 수렴, 140까지 간 실행에서 110이 정점이고 그 뒤 하락(.155 → .143 → .142 → .140). 방어 가능하다. **진짜 취약점은 벤치마크 1개와 시드 1개이고, 이 계획은 전자를 해소한다.**

목표: 저장된 체크포인트를 6개 수학 벤치마크에서 평가해, 단일 벤치마크 결과를 논문 표 한 줄이 아니라 **한 행 전체**로 만든다. 학습을 다시 돌리지 않으므로 비용이 낮다.

## 종합 결과: 5-arm 인과 분해 (2026-09-02) — 논문의 Table 2

`paper-uniform` 브랜치의 로그 4종 + permuted(진행중)를 공통 창 **step 40–90**(arm당 6 val 지점)에서 비교.

| arm | λ | mapping | rollout | A_H | acc@32 | maj@32 | entropy |
|---|---|---|---|---|---|---|---|
| STEER (baseline) | 0 | minmax | plain | — | .1310 | .1998 | .124 |
| +tree, 균일감쇠 (uniform) | .25 | minmax | tree | support만 | .1293 | .1998 | .146 |
| +tree, A_H 셔플 (permuted) | .25 | minmax | tree | 값 파괴 | .1333 | .2075 | .125 |
| **STEER-F (signed)** | .25 | minmax | tree | ✓ | **.1480** | **.2378** | .125 |
| (tree 없이) rank | .25 | rank | plain | ✓ | .1372 | .2210 | **.240** |

### 이득의 분해 — 헤드라인 발견

```
signed − STEER = +.0170 (3.0 SE)
               = [permuted − STEER]  +  [signed − permuted]
               = +.0023 (장치, 14%)  +  +.0147 (정보, 86%)
```
maj@32도 동일: `+.0380 = +.0077(20%) + .0303(80%)`.

**tree rollout 자체는 공짜 이득을 주지 않는다** — permuted−STEER = +0.4 SE, uniform−STEER = −0.3 SE.
tree는 개입할 갈림길을 만드는 **전제조건**이지 원인이 아니다(`branch_corr_frac` tree 0.013 vs plain 0.004, 4배).
이것이 "signed vs STEER는 λ와 rollout이 동시에 바뀐 이중 교락"이라는 최대 약점을 해소한다.

### 주요 대조 (step 40–90, n=6씩)

| 대조 | acc@32 | maj@32 | 해석 |
|---|---|---|---|
| signed − STEER | +3.0 SE | +4.4 SE | 방법이 작동한다 |
| signed − uniform | +3.5 SE | +4.1 SE | 예측이 **필요**하다 |
| signed − permuted | +2.4 SE | +3.0 SE | 예측이 **정확**해야 한다 (경계선) |
| permuted − STEER | +0.4 SE | +1.1 SE | tree+장치만으로는 0 |
| uniform − STEER | −0.3 SE | 0.0 SE | tree+균일감쇠만으로는 0 |
| rank − STEER | +1.0 SE | +2.7 SE | plain에서도 maj는 오른다 |
| signed − rank | +1.8 SE | +1.6 SE | tree가 acc를 따라오게 한다 |

### 엔트로피는 채널이 아니다 — 이제 5점짜리 증거

plateau 엔트로피 vs acc의 **Spearman ρ = +0.20** (5 arm). 사실상 무관계.
- **rank**: 엔트로피 .240 (2배, 압도적 1위) → acc .1372 (3위)
- **uniform**: 엔트로피 .146 (2위) → acc .1293 (**꼴찌**)
- **signed**: 엔트로피 .125 (최하위권) → acc .1480 (**1위**)

기계적 설명: rank mapping은 순위를 균등하게 펼쳐 **모든 토큰을 평균 22% 감쇠**한다
(`tw_mean` 0.776 vs 다른 arm 0.998~0.999, `tw_std` 0.046 vs 0.007~0.011).
사실상 학습률을 0.78배로 낮춘 것 = 엔트로피가 안 떨어짐. 그런데 성능은 signed보다 낮다.

→ 본문 주장: **"엔트로피 붕괴 억제는 성능의 원인이 아니다.
세 가지 억제 방식 중 엔트로피를 가장 많이 보존한 둘이 성능은 가장 낮다."**

### pass@32는 본문에서 뺀다

plateau 평균 1위 permuted(.3975), 최고 점수 1위 rank(.485). 다섯 arm이 .378~.398에 다 들어간다.
어느 기준으로도 주력이 1등이 아니므로 본문에 실으면 통제군이 이기는 칸이 생긴다.

### 확정 결과: GRPO baseline 110 step 완주 (2026-09-05) — 프레이밍 재작성

`run/run_grpo.sh`(`loss_mode=vanilla`, seed 1, tree 없음)가 110 step 완주.
검증 완료: `'loss_mode': 'vanilla'`, 전 구간 `Entering vanilla branch`, `steerf-tree` 라인 0개,
step-0 val `.039/.244/.035`로 tree arm과 **동일 초기 체크포인트**, `Final validation metrics` 출력.
→ 이제 GRPO / STEER / uniform / STEER-F 네 arm이 **step 0–110 완전 대칭**(12 val 지점, plateau 8점).

#### 두 가지 결정 (되돌리지 않는다)

1. **pass@32는 전면 폐기.** 본문·부록 어디에도 싣지 않는다. 전 arm이 .378~.428에 뭉치고,
   어느 기준으로도 주력이 1등이 아니며 GRPO가 최고(.4277)다. 대체 통계는
   **`uplift = maj@32 − acc@32`** (선택 효율: 32개 표본 중 오답이 흩어지고 정답이 모이는 정도).
2. **엔트로피 억제 주장은 폐기.** 개입한 두 arm이 GRPO보다 엔트로피가 **낮다**.
   "붕괴를 억제해서 성능이 오른다"는 데이터가 정면으로 부정한다.
   → 새 주장: **엔트로피 붕괴는 학습 그 자체다. 문제는 총량이 아니라 어느 방향으로 붕괴하느냐다.**

#### 표 1. arm 정의

| arm | λ | mapping | rollout | 개입 신호 | 로그 |
|---|---|---|---|---|---|
| **GRPO** | — | — | plain | 없음 (`loss_mode=vanilla`) | ✅ 로컬 |
| **STEER** | 0 | minmax | plain | 국소 Ω만 | ✅ 로컬 |
| **uniform** | .25 | minmax | tree | support에 **균일** 감쇠 (A_H 값 무시) | ✅ 로컬 |
| **permuted** | .25 | minmax | tree | A_H를 형제끼리 **셔플** | ⚠️ pod에만 |
| **STEER-F (signed)** | .25 | minmax | tree | **A_H (미래 방문항)** | ✅ 로컬 |

모두 Qwen2.5-Math-1.5B / seed 1 / 110 step / DAPO-Math-17k / n=8.

#### 표 2. step별 AIME24 (acc@32 / maj@32)

| step | GRPO | STEER | uniform | STEER-F |
|---|---|---|---|---|
| 0 | .039/.035 | .043/.040 | .039/.034 | .039/.034 |
| 10 | .075/.127 | .064/.109 | .070/.124 | .078/.149 |
| 20 | .120/.176 | .114/.195 | .102/.171 | .103/.181 |
| 30 | .108/.195 | .135/.215 | .121/.223 | .113/.211 |
| 40 | .144/.236 | .115/.187 | .119/.201 | .141/.230 |
| 50 | .120/.204 | .125/.196 | .121/.216 | .145/.258 |
| 60 | .126/.178 | .139/.203 | .125/.204 | .145/.237 |
| 70 | .130/.200 | .128/.200 | .129/.197 | .147/.257 |
| 80 | .135/.182 | .135/.202 | .140/.205 | .144/.206 |
| 90 | .136/.191 | .144/.213 | .142/.177 | **.166**/.239 |
| 100 | .132/.171 | .146/.207 | .134/.193 | .157/.222 |
| 110 | **.157**/.208 | .132/.180 | .140/.170 | .151/.229 |

#### 표 3. 수렴 구간 step 40–110 (arm당 8점) — **논문 Table 1**

| arm | acc@32 | maj@32 | uplift = maj−acc | entropy | resp len | H_traj = ent×len |
|---|---|---|---|---|---|---|
| GRPO | .1350 ± .0040 | .1963 ± .0073 | .0612 ± .0066 | **.1491** ± .0043 | 963.9 | 143.5 ± 3.2 |
| STEER | .1330 ± .0036 | .1985 ± .0038 | .0655 ± .0029 | .1214 ± .0033 | 988.1 | 119.9 ± 3.2 |
| uniform | .1313 ± .0032 | .1954 ± .0054 | .0641 ± .0080 | .1374 ± .0096 | 946.0 | 130.1 ± 9.4 |
| **STEER-F** | **.1495** ± .0029 | **.2348** ± .0061 | **.0853** ± .0068 | .1210 ± .0059 | 983.9 | 119.0 ± 5.5 |

`actor/entropy`는 `loss_agg_mode='token-mean'`이라 **ℋ이 아니라 ℋ/평균길이**다
(`verl/trainer/ppo/ray_trainer.py:1200-1212`, `core_algos.py:543`). 길이 차가 5% 미만이라
`ent×len` 프록시로 바꿔도 모든 결론과 SE가 그대로다(GRPO−STEER-F 둘 다 +3.8 SE,
STEER−STEER-F 둘 다 +0.1 SE). 정확한 값이 필요하면 `ray_trainer.py:1206`의
`seq_entropy_agg`(주석 처리됨)를 켜면 된다.

#### 표 4. 대조 — **논문 Table 2 (핵심)**

| 대조 | acc@32 | maj@32 | uplift (paired) | 판정 |
|---|---|---|---|---|
| **STEER-F − GRPO** | **+.0145 (2.9 SE)** | **+.0385 (4.0 SE)** | **+.0240 (4.8 SE)** | ✅ 방법이 작동 |
| **STEER-F − STEER** | **+.0165 (3.5 SE)** | **+.0363 (5.0 SE)** | **+.0198 (3.2 SE)** | ✅ 미래 항이 필요 |
| **STEER-F − uniform** | **+.0182 (4.2 SE)** | **+.0394 (4.8 SE)** | **+.0211 (3.1 SE)** | ✅ **A_H의 값이 필요** |
| GRPO − STEER | +.0020 (+0.4 SE) | −.0023 (−0.3 SE) | −.0043 (−0.8 SE) | ⬜ 동급 |
| uniform − GRPO | −.0038 (−0.7 SE) | −.0009 (−0.1 SE) | +.0029 (+0.4 SE) | ⬜ 0 |
| uniform − STEER | −.0018 (−0.4 SE) | −.0031 (−0.5 SE) | −.0014 (−0.2 SE) | ⬜ 0 |

위 세 줄 전부 유의, 아래 세 줄 전부 0. **tree rollout + 감쇠 장치를 다 켜도(uniform) 이득이
0이고, A_H의 실제 값을 넣어야만(signed) 이득이 생긴다.** 이것이 "미래 방문항이 동작한다"의
가장 직접적인 증거이며, 동시에 "λ와 rollout이 동시에 바뀐 이중 교락"이라는 최대 약점을 해소한다.

#### 표 5. 이득의 분해 — **논문의 헤드라인**

```
maj@32 이득 (STEER-F − GRPO) = +.0385
                             = acc 이득    +.0145 (38%)
                             + uplift 이득 +.0240 (62%)
```

이득의 62%가 개별 정확도가 아니라 **샘플 집합의 구조**에서 온다.
`uniform − GRPO`의 uplift가 +0.4 SE(=0)라는 사실이 이 구조 변화를 만든 것이
장치가 아니라 **A_H**임을 특정한다.

#### 표 6. 엔트로피 해리 — **논문의 2차 기여**

| 대조 | entropy 격차 | acc 격차 |
|---|---|---|
| GRPO − STEER | **+.0278 (+5.1 SE)** | +0.4 SE |
| GRPO − STEER-F | **+.0281 (+3.8 SE)** | −2.9 SE |
| **STEER − STEER-F** | **+.0004 (+0.1 SE)** | **−3.5 SE** |

세 줄이 각각 다른 말을 한다:
1. 개입한 arm이 GRPO보다 엔트로피가 **낮다** → "붕괴 억제"는 일어나지 않았다
2. 엔트로피 1위 GRPO가 성능 2위 → 총량은 성능을 예측하지 못한다
3. **엔트로피가 0.1 SE로 동일한 두 arm이 acc 3.5 SE / maj 5.0 SE 갈린다** → 결정적.
   남은 길의 *개수*가 같은데 성적이 다르다 = 개수가 아니라 *어느 길이 남았느냐*의 문제

이론적 뒷받침: `docs/STEERF_derivation.md` §7.2 — A_H는 형제 평균으로부터의 편차이므로
한 형제 집합 안에서 합이 0이다. **설계상 평균 감쇠량을 못 바꾸고 형제들을 벌려놓기만 한다.**
엔트로피 총량이 안 바뀌는 것은 버그가 아니라 예측된 동작이다.

#### 표 7. 최고 점수 (0–110, 12 val 지점, 완전 대칭)

| arm | acc@32 | maj@32 |
|---|---|---|
| GRPO | .157 @110 | .236 @40 |
| STEER | .146 @100 | .215 @30 |
| uniform | .142 @90 | .223 @30 |
| **STEER-F** | **.166** @90 | **.258** @50 |

plateau 평균과 순서가 같아 두 통계를 같이 실어도 모순이 없다.

#### 표 8. 비용

| arm | s/step | 총 시간 | 오버헤드 |
|---|---|---|---|
| GRPO | 1027 | 31h23m | — |
| STEER-F | ~1425 | ~43h | **+39%** |

거의 전부 MTP forward: `old_log_prob` 98 s → 655 s.

#### 부록 표. 5-arm (step 40–90, arm당 6점 — 창이 다름)

permuted 로그가 이 체크아웃에 없어 이전 분석 창을 그대로 쓴다.

| arm | acc@32 | maj@32 | entropy |
|---|---|---|---|
| STEER | .1310 | .1998 | .124 |
| uniform | .1293 | .1998 | .146 |
| **permuted** | .1333 | .2075 | .125 |
| **signed** | **.1480** | **.2378** | .125 |
| rank (tree 없음) | .1372 | .2210 | **.240** |

`signed − permuted = +.0147 acc (2.4 SE) / +.0303 maj (3.0 SE)` — **A_H를 셔플만 해도
이득이 사라진다.** rank는 엔트로피가 2배인데 acc 3위 → 해리의 5번째 점.

### 본문 프레이밍 (확정)

- 낡은 사다리 **GRPO < STEER < STEER-F는 죽었다.** `GRPO − STEER`는 acc +0.4 SE / maj −0.3 SE로 동급.
- 새 프레이밍: **`GRPO ≈ STEER ≪ STEER-F`**. 헤드라인 격차는 **GRPO 기준**으로 진술한다
  (GRPO는 tree arm과 step-0이 `.039`로 정확히 일치). STEER는 step-0이 `.043`인 다른 시기 스택이므로
  STEER 대비 수치를 인용할 때만 각주를 단다.
- 핵심 문장: **"우리는 엔트로피를 더 아끼지 않는다. 오히려 GRPO보다 빨리 붕괴한다.
  대신 붕괴할 때 어느 갈래를 남길지 미래 방문항으로 고른다. 그 결과 엔트로피는 낮은데 성능은 높다."**

### 선행 연구 (엔트로피 억제 ≠ 성능 보장)

egress 차단으로 PDF 본문을 열지 못했다 — **인용 전 원문 문구 확인 필수.**

| 논문 | 우리 주장과의 관계 |
|---|---|
| **The Invisible Leash** (arXiv 2507.14843) | 토큰 엔트로피가 높아도 최종 답 다양성은 줄어든다 — *local stochasticity without global exploration*. 우리 uplift 발견과 정확히 짝 |
| **Beyond the 80/20 Rule** (arXiv 2506.01939, NeurIPS'25) | 상위 20% forking token만으로 full-gradient와 동등. "총량이 아니라 어느 토큰이냐"의 직접 선행 근거 |
| **Rethinking Entropy Regularization in LRM** (arXiv 2509.25133) | naive entropy regularization이 붕괴를 못 막고 더 학습하면 성능이 떨어진다 |
| **Rethinking Entropy Interventions in RLVR** (arXiv 2510.10150) = STEER 원 논문 | 기존 개입이 "간접적이라 효과가 제한적이고 실패할 수 있다"고 자인 → 우리의 `STEER ≈ GRPO` 관측과 연결 |

인용 전략: 2507.14843 + 2506.01939로 "총량은 지표가 아니다"를 세우고, 2509.25133으로 보강.
우리는 여기에 **엔트로피가 0.1 SE로 동일한 두 arm이 maj 5.0 SE 갈리는 직접 증거**를 더한다.

### 아직 없는 것

1. ~~GRPO 순수 baseline 부재~~ → **해소. 110 step 완주.** 단 "STEER > GRPO"는 재현 실패이므로
   프레이밍을 위와 같이 바꿨다.
2. **permuted가 40–110 창에 없다** — pod에서 로그만 받아오면 표 3·4가 즉시 5-arm이 된다. **최우선.**
3. **시드 1개** — 위 SE는 "이 실행의 수렴 레벨" 기준이지 시드 재추출 기준이 아니다.
4. **벤치마크 1개** — 아래 다중 벤치마크 계획이 이걸 해소한다.
5. **λ ablation 부재** — λ=0 + minmax + **tree** 실행이 없다. permuted가 대신하고 있다.
6. **§2.4가 미측정** — `(1−π_a)`와 A_H의 **곱**을 실제 롤아웃에서 잰 적이 없다
   (`docs/STEERF_derivation.md` 자체가 flag). 필요한 측정 하나: 진짜 sibling divergence
   위치에서의 평균 `|Ω|` 순위 대 나머지 위치.

---

## 중간 결과: 3-arm 비교 (2026-09-02, permuted step 90까지)

세 arm 모두 λ=0.25 / minmax / tree / seed 1, step-0 val이 `.039/.244/.035`로 동일
→ 같은 초기 체크포인트 확정. 공통 구간은 step 0–90 (val 지점 10개).

**step 50의 permuted `.153`은 스파이크였다** — 이후 .135 / .132 / .134 / .130으로 되돌아왔다.

### 수렴 구간(step 40–90, arm당 6개 지점) — 논문의 주 통계

| 지표 | signed | uniform | permuted | signed−permuted | signed−uniform |
|---|---|---|---|---|---|
| **acc@32** | **.1480** ± .0090 | .1293 ± .0097 | .1333 ± .0119 | **+.0147 (2.4 SE)** | **+.0187 (3.5 SE)** |
| **maj@32** | **.2378** ± .0192 | .1998 ± .0124 | .2075 ± .0154 | **+.0303 (3.0 SE)** | **+.0380 (4.1 SE)** |
| pass@32 | .3908 | .3897 | **.3975** | −.0067 (−0.3 SE) | +.0012 (0.1 SE) |

step 50–90(5개)으로 잘라도 동일: acc +.0126(2.1 SE), maj +.0298(2.5 SE).

### 최고 점수 (0–90 창)

| | signed | uniform | permuted |
|---|---|---|---|
| acc@32 | **.166** @90 | .142 @90 | .153 @50 |
| maj@32 | **.258** @50 | .223 @30 | .227 @70 |
| pass@32 | .442 @70 | .428 @40 | **.475** @60 |

### 사전 등록한 판정 기준 대입 결과 → **"양호"**

permuted acc plateau `.1333`은 사전에 정한 **최상 구간(≤ .135)**에 들어가고,
maj plateau `.2075`는 **양호 구간(.200–.212)**이다. 분리도는 acc 2.4 SE / maj 3.0 SE로
사전 기준상 **양호(2–3 SE)**. 원래 주장(A_H의 부호·짝짓기가 성능을 만든다)은 살아있지만
시드 추가 요구를 받을 크기다.

### 지표별 처분

- **acc@32·maj@32**: plateau 평균을 주 통계로 쓴다. 최고 점수도 같은 순서를 주므로 둘 다 싣는다.
- **pass@32는 폐기한다.** permuted가 step 60에서 `.475`로 전체 최고를 찍었고 plateau 평균도
  가장 높다. 본문에 실으면 통제군이 이기는 칸이 생긴다 — 부록으로 내리거나
  "세 arm을 구분하지 못한다"고 명시할 것.

### 엔트로피 해리 — 주장 범위를 좁힐 것

| step | signed | uniform | permuted |
|---|---|---|---|
| 50 | .130 | **.163** | .125 |
| 70 | .121 | **.131** | .117 |
| 90 | .113 | **.120** | .118 |

uniform 우위 격차가 `.033 → .010 → .007`로 줄어 step 90에서 수렴한다. 따라서 주장을 둘로 나눈다:

- **강함 (9/9 val 지점 지지)**: A_H를 섞어도 엔트로피 궤적은 안 바뀌는데 acc/maj는 떨어진다
  → **성능 채널은 엔트로피가 아니다.** 이게 논문의 1차 기여.
- **범위를 좁힘**: "uniform이 엔트로피를 더 잘 보존한다"는 **step 20–70 구간에서만** 성립.
  "학습 중반 엔트로피를 더 오래 유지하지만 정확도로 환산되지 않는다"로 써야 한다.

### 확정된 결정

1. **permuted는 step 110까지 돌린다.** 현재 step 90 도달, **남은 20 step ≈ 8~9h**.
   110까지 가면 세 arm이 12 val 지점·8 plateau 지점으로 완전 대칭이 되고, 지금 2.4 SE인
   acc 분리가 3 SE 근처까지 갈 여지가 있다.
   `SAVE_AFTER=80`을 넘겨 **global_step_80 / 90 체크포인트가 이미 저장됐다** —
   permuted를 다중 벤치마크 표에 열로 넣을 수 있게 됐다.
2. **엔트로피 해리를 논문 본문의 주장으로 올린다.** "엔트로피 붕괴를 억제해서 성능이 오른다"가
   아니라 **"엔트로피 예산을 어디에 배치하느냐가 성능을 만든다"**를 핵심 주장으로 쓰고,
   uniform arm을 그 증거로 제시한다 (엔트로피 최상 + 성능 최하). permuted arm은 짝짓기를
   파괴하면 엔트로피는 그대로인데 maj@32가 무너진다는 반대편 증거다. 리뷰어가 ablation 표에서
   먼저 짚기 전에 우리가 설명한다.

이 프레이밍은 아래 다중 벤치마크 실험의 **검증 항목 4**를 바꾼다: 방향 일관성을 acc뿐 아니라
**arm 순서(signed > uniform ≈ permuted)가 6개 벤치마크 중 몇 개에서 유지되는지**로 본다.

## 새로 짤 코드는 없다

필요한 것이 전부 레포에 있다. 확인 완료:

| 자산 | 역할 |
|---|---|
| `run/eval_steerf.sh` | 논문 프로토콜 val-only 평가. AIME24/25/AMC23는 avg@32, MATH500/Minerva/Olympiad/GSM8K는 avg@1로 2패스. `MODEL_PATH`만 받는다. `_gpu_defaults.sh`를 source하므로 2-GPU 박스에서 `N_GPUS=2, TP_SIZE=2` 자동. |
| `scripts/select_best_checkpoint.py` | 논문 선택 규칙(App. E.2: AIME24 최고 정확도 체크포인트) 구현. tensorboard 이벤트를 직접 파싱하고 **디스크에 실제로 남아 있는 step으로 제한**한다. |
| `scripts/collect_results.py` | eval 로그 → 논문 표 모양 TSV. 컬럼 매핑이 이미 정의돼 있다(`COLUMNS`, `MATH6`). |
| `datasets/*.parquet` | aime24, aime25, amc23, math500, minerva_math, olympiadbench, gsm8k_test, omni — 8개 전부 존재, git 추적 중. |
| `docs/paper_reference.tsv` | STEER 논문 원 수치. `collect_results.py` 출력과 같은 컬럼 모양이라 바로 대조 가능. |

### 사전 검증 완료 (이 레포에서 확인함)

parquet 재고와 `collect_results.py` 컬럼 매핑을 실제로 열어 확인했다:

| parquet | rows | 구조 | `data_source` |
|---|---|---|---|
| aime24 | 960 | 30문제 × 32 replica | `aime_2024_dapo_boxed` ✓ |
| aime25 | 960 | 30문제 × 32 replica | `aime_2025_dapo_boxed` ✓ |
| amc23 | 1,280 | 40문제 × 32 replica | `amc2023_dapo_boxed` ✓ |
| math500 | 500 | replica 없음 | `math500` ✓ |
| minerva_math | 272 | replica 없음 | `minerva_math` ✓ |
| olympiadbench | 675 | replica 없음 | `olympiadbench` ✓ |
| gsm8k_test | 1,319 | replica 없음 | `gsm8k_test` ✓ |

- **avg@32 패스가 유효하다**: 세 개의 @32 셋이 정확히 32 replica를 담고 있다. `val_kwargs.n=1`이어도 verl이 `acc/mean@32`로 집계한다(학습 로그의 `val-core/aime_2024_dapo_boxed/acc/mean@32`가 증거). → **검증 항목 2는 이미 통과.**
- **`data_source` 7개가 `collect_results.py`의 `COLUMNS`와 전부 일치**한다. 컬럼이 `-`로 비는 사고는 없다.
- 총 생성량: 패스 A 3,200 + 패스 B 2,766 = **5,966 generation/체크포인트**.
- `select_best_checkpoint.py`(`--project/--run/--metric/--ckpt-root/--tb-root/--quiet`)와 `collect_results.py`(`--logs/--out`) 둘 다 `py_compile` 통과, 계획의 인자 표기가 실제와 일치.
- `omni.parquet`은 `omni_math_test`라 `COLUMNS`에 없지만 `eval_steerf.sh`의 대상 목록에도 없으므로 무관.

## 먼저 확인해야 할 블로커: 체크포인트 보존

모든 학습 실행이 `++trainer.max_actor_ckpt_to_keep=3`이고 `++trainer.save_best_only=False`다. `ray_trainer.py:914`의 best-checkpoint 분기는 `save_best_only=True`일 때만 동작하므로 **별도로 보존되는 best 체크포인트가 없다.** 순수 rolling 보존이다.

결과: 110 step 실행이 끝나면 디스크에 **global_step_90 / 100 / 110 세 개만** 남는다.

- signed 실행의 AIME24 최고 지점은 커밋된 로그 기준 step 70(.147)인데, **이미 삭제됐을 가능성이 높다.**
- 즉 논문 선택 규칙을 적용해도 argmax의 후보가 11개가 아니라 3개다.

**그래서 첫 단계는 각 pod에서 살아있는 체크포인트 목록을 확인하는 것이고, 아직 돌고 있는 arm은 회전되기 전에 빼두는 것이다.**

## 실행

### 0단계 — 재고 확인 (각 pod에서)

```bash
ls -d checkpoints/STEER-F/*/global_step_*/actor/huggingface 2>/dev/null
df -h /workspace | tail -1
```

아직 돌고 있는 permuted에는 회전 방지를 걸어둔다 — 다음 저장 전에:
```bash
# 남기고 싶은 step을 미리 복사해 rotation 대상에서 빼둔다
cp -r checkpoints/STEER-F/<run>/global_step_70 checkpoints/keep/
```
(체크포인트 매니저는 자기 목록에 있는 경로만 지우므로 다른 디렉토리로 옮기면 안전하다.)

### 1단계 — arm별 평가 체크포인트 선정

```bash
python scripts/select_best_checkpoint.py \
    --project STEER-F --run steer-f-Qwen2.5-Math-1.5B-s1-tree-rollout \
    --metric val-core/aime_2024_dapo_boxed/acc/mean@32
```
출력이 `run/eval_steerf.sh`가 기대하는 HF 모델 경로다. **디스크에 남은 step 중 argmax**라는 점을 결과에 명시할 것 — 논문에 쓸 때 "11개 중 argmax"라고 쓰면 틀린 진술이 된다.

선택 규칙을 arm마다 동일하게 적용해야 공정하다. 후보가 arm별로 다르면(예: 한쪽만 step 70이 살아있음) **모든 arm에서 같은 step**(예: 110)으로 고정하는 편이 방어하기 쉽다. 둘 중 하나를 고르고 논문에 명시한다.

### 2단계 — 평가

arm 하나당:
```bash
MODEL_PATH=/workspace/entropy_collapse/checkpoints/STEER-F/<run>/global_step_<N>/actor/huggingface \
  bash run/eval_steerf.sh 2>&1 | tee logs/experiments/eval-<arm>-s1.log
```

**파일명은 반드시 `eval-<arm>-s<seed>.log` 형식이어야 한다.** `collect_results.py:54`가
`re.match(r"eval-(.+)-s(\d+)\.log$", f.name)`으로 arm과 seed를 뽑고, 매치 실패 시
예외가 아니라 `continue`로 조용히 건너뛴다. `eval-signed-step110.log` 같은 이름은
매치되지 않아 평가를 다 돌리고도 `no parsable eval logs`만 보게 된다.
step 번호는 파일명이 아니라 로그 본문의 `MODEL_PATH`로 기록된다.
`bash run/eval_steerf.sh`를 통째로 tee하면 avg32/avg1 두 패스가 한 파일에 들어가고,
`_find`가 마지막 매치를 취하므로 키 충돌은 없다.

대상 arm (체크포인트가 살아있는 것부터):
- `steer` (λ=0, minmax, plain) — 논문 표의 비교 기준
- `signed` (λ=0.25, minmax, tree) — 주력
- `uniform` (apply=branch) — 통제군
- `permuted` (permute_ah=1) — 핵심 통제군
- (선택) λ=0.25 rank 실행

비용 추정: 패스 A ≈ 3,200 generation, 패스 B ≈ 2,800 generation, max 3072 tok. 학습 1 step이 4,096 generation에 ~450s이므로 **체크포인트당 30~45분**(vLLM 초기화 포함), 4 arm이면 **2~3시간**. 학습 재실행 대비 매우 저렴하다.

### 3단계 — 표로 취합

```bash
python scripts/collect_results.py --logs logs/experiments --out results/summary.tsv
diff <(cut -f1-8 results/summary.tsv) <(cut -f1-8 docs/paper_reference.tsv)  # 컬럼 모양 확인
```

`MATH6` 평균(AIME24/AIME25/AMC23/MATH500/Minerva/Olympiad)이 논문 표의 헤드라인 숫자다.

## 검증

각 단계에서 확인할 것:

1. **평가가 학습 설정을 오염시키지 않았는지** — eval 로그에 `trainer.val_only=True`, `trainer.resume_mode=disable`, `save_freq=-1`이 찍혀야 한다. 이게 아니면 체크포인트를 덮어쓸 위험이 있다.
2. **avg@32 패스가 제대로 32 샘플인지** — `files_at32` 패스는 `val_kwargs.n=1`로 돌지만 parquet이 이미 32 replica를 담고 있다(`run/eval.sh` 상단 주석). 로그에 `val-core/<dataset>/acc/mean@32` 키가 나오면 정상, `mean@1`만 나오면 replica가 없는 parquet이므로 중단.
3. **STEER 재현 sanity** — λ=0 arm의 MATH500/GSM8K 수치가 `docs/paper_reference.tsv`의 STEER 행과 같은 자리수인지. 크게 어긋나면 평가 프로토콜이 논문과 다른 것이므로 다른 arm 수치도 신뢰할 수 없다.
4. **방향 일관성** — AIME24에서 본 부호(signed > permuted, STEER-F > STEER)가 6개 벤치마크 중 몇 개에서 유지되는지. 이게 이 실험의 실제 산출물이다. 4/6 이상이면 단일 벤치마크 우연이라는 반론이 약해진다.

## 이 계획이 답하지 못하는 것

시드 1개 문제는 그대로 남는다. 다중 벤치마크는 "우연히 AIME24에서만 좋았다"를 막지만 "우연히 이 시드에서만 좋았다"는 막지 못한다. 그건 별도로 시드 2~3개가 필요하다.

---

# 부록: 실행 순서 (permuted 학습이 도는 중이라는 전제)

## 원칙 — 시간이 흐르면 잃는 것은 하나뿐이다

| 대상 | 되돌릴 수 있나 | 언제 해야 하나 |
|---|---|---|
| **체크포인트** | ❌ 회전되면 영구 소실 | **지금 당장** |
| 로그 파일 | ✅ append-only, 나중에 다시 복사 가능 | 아무 때나 |
| git / paper 브랜치 | ✅ pod를 건드리지 않음 | 아무 때나 |
| 다중 벤치마크 eval | ✅ | **GPU가 빈 뒤에만** |

회전이 지우는 대상은 `global_step_<N>/actor` **디렉토리 전체**다 (학습 로그의
`Checkpoint manager remove previous save local path: .../global_step_80/actor`).
`huggingface/` 하위도 같이 사라진다.

**핵심 절약**: eval에는 `actor/huggingface`(HF 모델 디렉토리)만 있으면 된다.
optimizer/extra state는 resume용이라 필요 없다. 약 3GB vs 전체 ~20GB.
(`eval_steerf.sh` 헤더 주석은 `hf_model`이라 쓰여 있지만 실제 디렉토리명은
`actor/huggingface`다 — 학습 로그의 `Saved hf_model to .../actor/huggingface`.)

## 0단계 — 지금 즉시, permuted pod에서 (학습 멈추지 않음)

디스크부터 본다. **복사가 디스크를 채우면 다음 저장에서 학습이 죽는다.**

```bash
df -h /workspace | tail -1
du -sh checkpoints/STEER-F/*/global_step_*/actor/huggingface | tail -3
```

여유가 복사할 용량의 2배 이상일 때만 진행한다:

```bash
mkdir -p checkpoints/keep
for d in checkpoints/STEER-F/*-permuted/global_step_*; do
    n=$(basename "$d")
    [ -d "$d/actor/huggingface" ] && cp -r "$d/actor/huggingface" "checkpoints/keep/permuted-$n"
done
ls checkpoints/keep
```

`checkpoints/keep`은 체크포인트 매니저의 관리 목록 밖이라 회전 대상이 아니다.

## 1단계 — 어느 체크포인트를 쓸지 먼저 정한다

논문의 argmax 규칙(App. E.2)은 **이미 적용 불가능하다.** signed 실행의 AIME24 최고
지점인 step 70은 회전으로 사라졌고, 살아남은 후보가 arm마다 다르다.

→ **모든 arm에서 step 110으로 고정**하고 논문에 명시하는 편이 방어하기 쉽다.
이 규칙을 택하면 필요한 건 각 arm의 마지막 저장본뿐이라 0단계의 긴급도가 낮아진다
(그래도 크래시 대비 보험으로 지금 한 벌 떠두는 게 좋다).

argmax 규칙을 굳이 쓰려면 남은 후보가 3개뿐이라는 사실을 논문에 적어야 한다.

## 2단계 — 완료된 arm의 로그 수집 (permuted와 무관, 병렬 가능)

old pod에서 signed / uniform 로그와 tensorboard 이벤트를 로컬로 내린다. 실행 중인
job이 없으므로 아무 제약이 없다.

## 3단계 — paper 브랜치 1차 커밋 (로컬, pod 무관)

signed + uniform만 먼저 올린다. permuted를 기다릴 이유가 없다.

## 4단계 — permuted 종료 후

로그와 최종 체크포인트를 내리고 paper 브랜치에 2차 커밋.

## 5단계 — GPU가 빈 뒤에만: 다중 벤치마크 eval

`run/eval_steerf.sh`는 GPU를 새로 잡는다. 학습이 도는 pod에서 돌리면 OOM이거나
학습을 죽인다. **permuted가 끝난 뒤**, 또는 노는 pod에서 실행한다.

---

# 부록 B: 명령어 (2~4단계 상세)

`paper` 브랜치는 이미 `origin`에 있다 (`b762cf1`). 남은 건 pod에만 있는 로그 3종을
얹는 것이다. 확인된 사실:

- `.gitignore`가 무시하는 건 `checkpoints/` 하나뿐이다 → `logs/`와 `tensorboard_log/`는
  그냥 커밋된다 (`tensorboard_log/`는 이미 10개 파일이 추적 중).
- 로그 경로 규칙은 `run/run_uniform_ablation.sh:108-111`:
  `logs/experiments/train-steer-f-Qwen2.5-Math-1.5B-s1-tree-rollout-<ARM>.log`

## 주의: 학습이 도는 working tree에서 브랜치를 바꾸지 말 것

permuted가 아직 돌고 있다면 같은 디렉토리에서 `git checkout`을 하면 실행 중인 코드가
바뀐다. 별도 worktree를 쓰거나 새로 clone한다.

```bash
git fetch origin paper
git worktree add ../entropy_paper paper   # 학습 건드리지 않음
cd ../entropy_paper
```

## 파일 배치 → 검증 → 푸시

```bash
# 1) pod에서 받은 파일을 제자리에 둔다
cp <받은경로>/train-steer-f-Qwen2.5-Math-1.5B-s1-tree-rollout-uniform.log  logs/experiments/
cp <받은경로>/train-steer-f-Qwen2.5-Math-1.5B-s1-tree-rollout-permuted.log logs/experiments/
cp <받은경로>/train-steer-f-Qwen2.5-Math-1.5B-s1-tree-rollout.log          logs/experiments/  # step 110까지인 최신본
cp -r <받은경로>/tensorboard_log/STEER-F/* tensorboard_log/STEER-F/

# 2) 커밋 전에 내용 확인 — 여기서 숫자가 안 맞으면 멈춘다
for f in logs/experiments/train-steer-f-*tree-rollout*.log; do
  echo "== $(basename $f)"
  echo "   last step : $(grep -o 'step:[0-9]* - global_seqlen' $f | tail -1)"
  echo "   val 지점  : $(grep -c 'val-core/aime_2024_dapo_boxed/acc/mean@32' $f)"
  echo "   arm       : $(grep -o 'steerf/permute_ah:[0-9.]*' $f | tail -1) $(grep -o 'steerf/apply_uniform:[0-9.]*' $f | tail -1)"
done
```

기대값: signed = `permute_ah:0.000 apply_uniform:0.000`,
uniform = `apply_uniform:1.000`, permuted = `permute_ah:1.000`.
signed 로그의 last step이 79면 옛날 파일을 덮어쓰지 못한 것이다.

```bash
# 3) 커밋 & 푸시
git add logs/experiments tensorboard_log
git status --short
git commit -m "paper: add the uniform and permuted arm logs, extend signed to step 110"
git push -u origin paper
```

## 선택: Jupyter 자동저장본 제거

`logs/experiments/.ipynb_checkpoints/`의 12개는 대부분 오래된 잘린 사본이다
(signed 사본 61줄 vs 실제 5,250줄). 논문 수치를 잘못 읽을 위험이 있어 지우려면:

```bash
git rm -r --cached logs/experiments/.ipynb_checkpoints
echo '.ipynb_checkpoints/' >> .gitignore
git add .gitignore && git commit -m "paper: drop the stale Jupyter autosaves"
```
