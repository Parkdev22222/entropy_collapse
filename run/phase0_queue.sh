#!/usr/bin/env bash
# Serialise the Phase 0 arms on the single GPU, then derisk the STEER-F path.
#
# Nothing here changes the experiment; it only stops the GPU idling between
# stages. Run it detached:  nohup ./run/phase0_queue.sh &
#
# Order and why:
#   1. ARM=steer      the other half of G0 (GRPO is assumed already done)
#   2. steerf smoke   2 steps at lambda>0 with throwaway heads.
#                     docs/experiment_log.md "Phase 2 미해결 리스크" #1 records
#                     that _steerf_unembedding()'s FSDP path has never been
#                     executed. It costs 3 minutes to find out here instead of
#                     ~12 hours from now at the top of Phase 2, and it is not a
#                     result — the heads are random, so only the plumbing is
#                     under test.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEERF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${STEERF_ROOT}"

log() { echo "[queue $(date +%H:%M:%S)] $*"; }

# --- wait for whatever is on the GPU now -----------------------------------
while pgrep -f "verl.trainer.main_ppo" >/dev/null; do sleep 30; done
log "GPU free"

# --- 1. Phase 0 STEER arm ---------------------------------------------------
log "starting ARM=steer"
ARM=steer "${SCRIPT_DIR}/phase0_1gpu.sh"
log "ARM=steer exited $?"

while pgrep -f "verl.trainer.main_ppo" >/dev/null; do sleep 20; done

# --- 2. STEER-F plumbing smoke test ----------------------------------------
log "building throwaway heads for the lambda>0 smoke test"
python scripts/make_dummy_heads.py \
    --model "${MODEL_PATH:-/workspace/hf_cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}" \
    --out checkpoints/mtp_heads_smoke.pt --num-heads 8 || exit 1

log "starting steerf smoke (2 steps, lambda=0.5, random heads)"
PROJECT_NAME=STEERF-smoke RUN_TAG=smoke-steerf \
STEERF_HEADS="${STEERF_ROOT}/checkpoints/mtp_heads_smoke.pt" \
STEERF_CALIB="" \
ARM=steerf STEERF_LAM=0.5 STEPS=2 \
    "${SCRIPT_DIR}/phase0_1gpu.sh" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    > logs/smoke_steerf.log 2>&1
log "steerf smoke exited $? -> logs/smoke_steerf.log"
