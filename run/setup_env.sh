#!/usr/bin/env bash
# Rebuild the container-side environment after a pod restart.
#
# On RunPod, /workspace is a network volume and / is the container overlay.
# A pod restart (or `rm -rf /*`) wipes the overlay and keeps /workspace, so
# the repo, datasets and checkpoints survive while every pip install, and the
# HuggingFace model cache under /root, do not.  This script rebuilds exactly
# that side and nothing else.
#
#   bash run/setup_env.sh              # check, install what is missing, verify
#   CHECK_ONLY=1 bash run/setup_env.sh # report only, install nothing
#
# It is idempotent: run it as often as you like.
#
# What it deliberately does NOT install: torch, vLLM, flash-attn, ray.  Those
# come from the pod image and are matched to its CUDA build; reinstalling them
# from PyPI is how a working pod turns into a broken one.  If they are missing
# the script says so and stops, because that means the image itself is wrong.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CHECK_ONLY=${CHECK_ONLY:-0}
FAIL=0
NEED=()

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mOK\033[0m    %s\n' "$*"; }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; FAIL=1; }

pyver() { python3 -c "import $1,sys; print(getattr($1,'__version__','?'))" 2>/dev/null; }

# ---------------------------------------------------------------- 0. 기본
say "0. 기본 도구"
for c in python3 pip git; do
    if command -v "$c" >/dev/null 2>&1; then ok "$c"; else bad "$c 없음"; fi
done
if [ "${FAIL}" = "1" ]; then
    echo; echo "python3/pip/git 이 없습니다. 오버레이가 아직 안 올라왔습니다 — 파드를 재시작하세요."
    exit 1
fi
ok "python $(python3 -c 'import sys;print(".".join(map(str,sys.version_info[:3])))')"
# nvidia-smi 는 여기서 멈추지 않고 기록만 합니다. 한 번에 전체 그림을 보는 편이
# 낫고, GPU 없는 곳(랩톱, CI)에서도 나머지 진단은 유효합니다.
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  GPU   /'
else
    bad "nvidia-smi 없음 — 학습은 불가. 나머지 진단은 계속합니다"
fi

# ---------------------------------------------------------------- 1. 캐시
# HF 캐시가 /root 아래(오버레이)면 파드 재시작마다 모델을 다시 받습니다.
# /workspace 로 옮기면 살아남습니다.
say "1. HuggingFace 캐시를 /workspace 로"
HF_TARGET="${STEER_ROOT}/.cache/huggingface"
if [ "${HF_HOME:-}" = "${HF_TARGET}" ]; then
    ok "HF_HOME=${HF_HOME}"
else
    warn "HF_HOME 이 ${HF_TARGET} 가 아닙니다 (현재: ${HF_HOME:-unset})"
    NEED+=("export HF_HOME=${HF_TARGET}   # ~/.bashrc 에 추가")
fi
mkdir -p "${HF_TARGET}"
[ -d "${HF_TARGET}/hub" ] && ok "모델 캐시 있음 — 재다운로드 불필요" \
                          || warn "모델 캐시 없음 — Qwen2.5-Math-1.5B 를 다시 받습니다 (~3GB, 첫 실행 시 자동)"

# ---------------------------------------------------------------- 2. GPU 스택
# torch / vLLM / ray / flash-attn 은 서로 ABI 로 묶여 있습니다. vLLM 은 자기
# torch 핀을 강제하고, flash-attn 은 설치 시점의 torch 에 대해 컴파일됩니다.
# 그래서 순서가 있고, 아무 버전이나 깔면 멀쩡하던 파드가 깨집니다:
#     vllm  ->  ray  ->  transformers 재핀  ->  flash-attn
#
# vllm==0.8.4 인 근거: requirements.txt 의 `# vllm==0.8.4` 주석과, 이전 학습
# 로그의 `NCCL version 2.21.5+cuda12.4` 가 그 버전이 요구하는 torch 2.6.0+cu124
# 와 일치한다는 것. 최신 vLLM 은 torch 2.7+ 로 올려 verl 0.4.1.x 와 어긋납니다.
VLLM_PIN=${VLLM_PIN:-0.8.4}
say "2. GPU 스택"
GPU_MISSING=()
for m in torch vllm ray flash_attn; do
    v=$(pyver "$m")
    if [ -n "$v" ]; then ok "$m $v"; continue; fi
    # "없음" 과 "있는데 못 불러옴" 은 처방이 다릅니다. flash-attn 은 설치 시점의
    # torch 에 대해 컴파일된 .so 를 싣기 때문에, torch 가 그 뒤에 바뀌면
    #   undefined symbol: _ZN3c105ErrorC2E...   (= c10::Error, libtorch 심볼)
    # 로 죽습니다. 파일은 멀쩡히 있으니 "없음" 이라고 하면 오진입니다.
    err=$(python3 -c "import $m" 2>&1 | tail -1)
    case "$err" in
        *"undefined symbol"*|*"cannot open shared object"*|*"ABI"*)
            bad "$m ABI 불일치 — 지금 torch 와 다른 버전으로 빌드됐습니다"
            printf '        %s\n' "$err"
            printf '        고치기: pip uninstall -y %s && pip install --no-cache-dir %s\n' "$m" "$m"
            printf '        (--no-cache-dir 없으면 예전 torch 로 빌드된 캐시 wheel 을 다시 씁니다)\n'
            ;;
        *) warn "$m 없음"; GPU_MISSING+=("$m") ;;
    esac
done
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    ok "torch.cuda 사용 가능 (built for CUDA $(python3 -c 'import torch;print(torch.version.cuda)'))"
elif [ -n "$(pyver torch)" ]; then
    bad "torch 가 GPU 를 못 봅니다 — CPU 빌드가 깔렸을 수 있습니다"
fi

if [ ${#GPU_MISSING[@]} -gt 0 ]; then
    cat <<EOT

  없는 것: ${GPU_MISSING[*]}
  아래 순서로 설치하세요. 순서를 바꾸면 깨집니다.

    pip install vllm==${VLLM_PIN}          # torch 를 자기 핀에 맞춰 함께 설치
    pip install "ray[default]"
    pip install "transformers<5"           # vllm 이 올렸을 수 있으니 재핀
    pip install flash-attn --no-build-isolation

  또는:  INSTALL_GPU_STACK=1 bash run/setup_env.sh

  flash-attn 이 소스 빌드로 넘어가 오래 걸리면 건너뛰어도 학습은 됩니다
  (transformers 가 sdpa 로 폴백). 다만 속도가 떨어지고 수치가 미세하게 달라지니
  비교하려는 네 팔은 반드시 같은 상태로 맞추세요.
EOT
    if [ "${INSTALL_GPU_STACK:-0}" = "1" ] && [ "${CHECK_ONLY}" != "1" ]; then
        say "2b. GPU 스택 설치"
        pip install "vllm==${VLLM_PIN}"        || bad "vllm 설치 실패"
        pip install "ray[default]"             || bad "ray 설치 실패"
        pip install "transformers<5"           || bad "transformers 재핀 실패"
        pip install flash-attn --no-build-isolation || warn "flash-attn 실패 — sdpa 폴백으로 진행 가능"
        python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null \
            && ok "설치 후 torch.cuda 정상 ($(pyver torch))" \
            || bad "설치 후 torch 가 GPU 를 못 봅니다"
    else
        FAIL=1
    fi
fi

# ---------------------------------------------------------------- 3. 핀
say "3. 버전이 고정돼야 하는 것"
# transformers 5 는 AutoModelForVision2Seq 를 제거했고 이 verl 은 그것을 하드코딩으로
# 씁니다 (fsdp_workers.py:193 등). 핀이 없으면 학습이 import 에서 죽습니다.
TV=$(pyver transformers)
if [ -z "$TV" ]; then
    warn "transformers 없음"; NEED+=('pip install "transformers<5"')
elif [ "${TV%%.*}" -ge 5 ] 2>/dev/null; then
    bad "transformers ${TV} — v5 는 AutoModelForVision2Seq 를 제거했습니다"
    NEED+=('pip install "transformers<5"')
else
    ok "transformers ${TV} (<5)"
fi


# ---------------------------------------------------------------- 4. 순수 파이썬
# 성공적인 `import X` 는 X 가 설치돼 있다는 뜻이 아닙니다. __init__.py 가 없는
# 디렉터리는 네임스페이스 패키지가 되어 import 가 통과합니다. 이 레포에는
# datasets/ 폴더가 있고 PYTHONPATH 에 레포 루트가 들어가므로, HF datasets 가
# 없으면 그 폴더가 `datasets` 모듈이 되어
#     AttributeError: module 'datasets' has no attribute 'load_dataset'
# 로 학습 도중에 죽습니다. 그러니 심볼까지 확인합니다.
say "4. 순수 파이썬 의존성 (심볼까지 확인)"
cd "${STEER_ROOT}" || exit 1   # 그림자를 재현하려면 레포 루트에서 확인해야 함
MISSING=$(python3 "${SCRIPT_DIR}/_check_deps.py")
[ -n "${MISSING}" ] && NEED+=("pip install ${MISSING}")

# sentence-transformers 는 Part A 측정(--embed-model)에만 쓰이고 학습에는 불필요.
# 최신판이 transformers>=5 를 요구하므로 반드시 <5 로 맞춰야 합니다.
if python3 -c "import sentence_transformers" >/dev/null 2>&1; then
    ok "sentence-transformers $(pyver sentence_transformers) (Part A 측정용)"
else
    warn "sentence-transformers 없음 — Part A 재측정 시에만 필요"
    NEED+=('pip install "sentence-transformers<5"   # 선택: Part A 측정용')
fi

# ---------------------------------------------------------------- 5. 설치
if [ ${#NEED[@]} -gt 0 ]; then
    say "5. 실행할 명령"
    printf '  %s\n' "${NEED[@]}"
    if [ "${CHECK_ONLY}" = "1" ]; then
        echo; echo "CHECK_ONLY=1 이라 설치하지 않았습니다."
    else
        echo
        for cmd in "${NEED[@]}"; do
            # export 안내와 주석이 달린 선택 항목은 자동 실행하지 않습니다.
            case "$cmd" in export*|*'#'*) continue;; esac
            case "$cmd" in pip\ install*) echo "  \$ $cmd"; eval "$cmd" || bad "실패: $cmd";; esac
        done
    fi
else
    say "5. 설치할 것 없음"
    ok "모든 의존성 충족"
fi

# ---------------------------------------------------------------- 6. 검증
say "6. 예전에 깨졌던 import 검증"
export PYTHONPATH="${STEER_ROOT}:${PYTHONPATH:-}"
python3 - <<'PY'
import sys
checks = [
    ("transformers.AutoModelForVision2Seq", "from transformers import AutoModelForVision2Seq"),
    ("verl (레포 내장)",                     "import verl"),
    ("steer_f (레포 내장)",                  "import steer_f"),
    ("steer_f.tree_rollout",                "import steer_f.tree_rollout"),
    ("vllm",                                "import vllm"),
]
bad = 0
for name, stmt in checks:
    try:
        exec(stmt); print(f"  \033[32mOK\033[0m    {name}")
    except Exception as e:
        print(f"  \033[31mFAIL\033[0m  {name}: {type(e).__name__}: {e}"); bad = 1
for mod in ("verl", "steer_f"):
    try:
        print(f"        {mod:8s}-> {__import__(mod).__file__}")
    except Exception:
        pass
sys.exit(bad)
PY
[ $? -ne 0 ] && FAIL=1

# ---------------------------------------------------------------- 7. 레포 상태
say "7. /workspace 쪽 (파드 재시작에도 살아남는 것)"
cd "${STEER_ROOT}" || exit 1
for p in datasets/DAPO-Math-17k.parquet datasets/aime24.parquet \
         steer_f/tree_rollout.py run/run_steerf.sh; do
    [ -e "$p" ] && ok "$p" || bad "$p 없음"
done
if grep -q "steerf_tree_depths" verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py 2>/dev/null; then
    ok "트리 롤아웃 패치 적용됨"
else
    warn "트리 롤아웃 패치 미적용"
    NEED+=("git apply patches/steerf_tree_rollout.patch")
fi
for f in checkpoints/mtp_heads_Qwen2.5-Math-1.5B-paper.pt \
         checkpoints/mtp_calibration_Qwen2.5-Math-1.5B-paper.json; do
    [ -f "$f" ] && ok "$f" || warn "$f 없음 — lam>0 학습에 필요"
done
[ -f rollout_data/warmup/Qwen2.5-Math-1.5B-paper/rollouts.jsonl ] \
    && ok "워밍업 롤아웃 있음 — 헤드 재학습 시 수집 단계 생략 가능" \
    || warn "워밍업 롤아웃 없음 — 헤드가 없다면 collect_warmup_rollouts.sh 부터"

# ---------------------------------------------------------------- 8. 스모크
say "8. CPU 스모크 (GPU 불필요)"
if [ -f scripts/smoke_tree_rollout_cpu.py ]; then
    if python3 scripts/smoke_tree_rollout_cpu.py >/tmp/smoke.log 2>&1; then
        ok "트리 롤아웃 스모크 통과 ($(grep -c PASS /tmp/smoke.log) checks)"
    else
        bad "스모크 실패 — /tmp/smoke.log 확인"; tail -5 /tmp/smoke.log
    fi
else
    warn "scripts/smoke_tree_rollout_cpu.py 없음"
fi

# ---------------------------------------------------------------- 결과
echo
if [ "${FAIL}" = "1" ]; then
    printf '\033[31m환경이 아직 준비되지 않았습니다.\033[0m 위 FAIL 항목을 해결하세요.\n'
    exit 1
fi
printf '\033[32m환경 준비 완료.\033[0m\n'
echo
echo "다음:"
echo "  1) export PYTHONPATH=${STEER_ROOT}:\$PYTHONPATH   (~/.bashrc 에 추가)"
echo "  2) export HF_HOME=${HF_TARGET}                     (~/.bashrc 에 추가)"
echo "  3) MTP 헤드가 없다면:"
echo "       SCALE=paper MODEL_PATH=Qwen/Qwen2.5-Math-1.5B N_GPUS=1 bash run/warmup_and_validate.sh"
echo "  4) 학습 재개 (트리 롤아웃, lam=0.25):"
echo "       bash run/run_tree_2x2.sh   또는 개별 run_steerf.sh 명령"
