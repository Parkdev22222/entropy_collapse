#!/usr/bin/env bash
# Phase 1 파이프라인 CPU 스모크 — GPU/transformers/vLLM 없이 배관만 검증한다.
#
#   bash run/run_smoke_cpu.sh [WORKDIR]
#
# 하는 일: 합성 문제 생성 → 롤아웃 생성 → MTP 헤드 워밍업 → 예보 검증 → 리포트.
# 필요한 것은 torch(CPU)와 numpy 뿐이다.
#
# **결과에는 아무 의미도 없다.** 합성 모델은 난수 가중치라 예보가 실측 엔트로피와
# 상관이 없고, 따라서 게이트 G1 은 거의 항상 실패한다(종료코드 2). 이 스크립트가
# 확인하는 것은 "판정이 내려지고 리포트가 쓰이는가"이지 "통과했는가"가 아니다.
#
# 종료코드:
#   0 = 파이프라인 완주 (게이트 판정 자체는 리포트 참조)
#   3 = 빈 풀 등 설정/데이터 오류 (phase1_validate.EMPTY_POOL_EXIT)
#   그 외 = 배관이 깨졌다 — GPU 노드에 올리기 전에 고칠 것

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK="${1:-${ROOT}/artifacts/smoke}"
MODEL="${SMOKE_MODEL:-smoke:h16,l2,seed0}"

cd "${ROOT}"
rm -rf "${WORK}"
mkdir -p "${WORK}"

echo "=== [1/4] 합성 문제 생성 ==="
python scripts/make_smoke_data.py --out "${WORK}/problems.jsonl" --n-problems 6 --seed 0

echo "=== [2/4] 롤아웃 생성 ==="
python scripts/phase1_warmup_heads.py generate \
    --model "${MODEL}" --prompts "${WORK}/problems.jsonl" --out "${WORK}/rollouts.jsonl" \
    --n-prompts 6 --n-samples 2 --max-response-length 48 --gen-batch 4 --no-vllm --seed 0

echo "=== [3/4] MTP 헤드 워밍업 ==="
python scripts/phase1_warmup_heads.py train \
    --model "${MODEL}" --rollouts "${WORK}/rollouts.jsonl" --out "${WORK}/mtp_heads.pt" \
    --num-heads 3 --head-hidden 16 --batch-size 4 --max-len 128 \
    --log-every 1 --dtype float32 --device cpu

echo "=== [4/4] 예보 검증 + 게이트 판정 ==="
set +e
python scripts/phase1_validate.py \
    --model "${MODEL}" --heads "${WORK}/mtp_heads.pt" --problems "${WORK}/problems.jsonl" \
    --workdir "${WORK}/phase1" --report "${WORK}/phase1_report.md" \
    --n-problems 3 --problem-pool 6 --n-trajectories 4 --n-mc 2 \
    --min-pass-rate 0.0 --max-pass-rate 1.0 \
    --max-response-length 48 --mc-max-tokens 16 --gt-horizon 8 \
    --kappa-grid 1,2,3 --gamma-grid 0.85,1.0 \
    --branch-group-size 3 --branch-max-len 24 \
    --embed-model "" --dtype float32 --device cpu --no-vllm --calibrate --seed 0
CODE=$?
set -e

echo
echo "=== 스모크 결과 ==="
echo "게이트 종료코드: ${CODE}  (0=G1 통과, 2=G1 실패, 3=설정/데이터 오류)"
echo "리포트: ${WORK}/phase1_report.md"

if [ "${CODE}" -eq 3 ]; then
    echo "설정/데이터 오류로 중단됐다 — 위 [fatal] 메시지를 볼 것." >&2
    exit 3
fi
if [ "${CODE}" -ne 0 ] && [ "${CODE}" -ne 2 ]; then
    echo "배관이 깨졌다 (예상치 못한 종료코드 ${CODE}) — GPU 노드에 올리기 전에 고칠 것." >&2
    exit "${CODE}"
fi
echo "파이프라인 완주 — 배관 정상."
