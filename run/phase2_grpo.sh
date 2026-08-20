#!/usr/bin/env bash
# Plain GRPO under Phase 2's exact evaluation protocol.
#
# Phase 0 ran GRPO, but with val_kwargs.n=1 and no checkpoint, so its pass@16 is
# unavailable and unrecoverable. Without this arm the headline table compares
# STEER-F against STEER only, and the paper cannot say what either buys over no
# reweighting at all on the metric the gate is judged on.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEERF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SEED=${SEED:-1}
STEPS=${STEPS:-100}
export PROJECT_NAME=${PROJECT_NAME:-STEER-F-phase2}
export RUN_TAG=${RUN_TAG:-"phase2-grpo-s${SEED}"}
source "${SCRIPT_DIR}/env_1gpu.sh"
export TOTAL_EPOCHS=1
export TEST_PATH=${TEST_PATH:-${STEER_ROOT}/datasets/math500.parquet}
LOG="${STEERF_ROOT}/logs/${RUN_TAG}.log"
mkdir -p "$(dirname "$LOG")"
echo "[phase2] ARM=grpo seed=${SEED} steps=${STEPS} -> ${LOG}"
ARM=grpo "${SCRIPT_DIR}/run_steerf_small.sh" \
    data.max_response_length=1024 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    +data.seed="${SEED}" \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    trainer.logger="['console','tensorboard']" \
    trainer.val_before_train=False \
    trainer.test_freq=100 \
    trainer.save_freq=100 \
    trainer.total_training_steps="${STEPS}" \
    "$@" > "$LOG" 2>&1
