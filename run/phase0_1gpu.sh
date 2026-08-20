#!/usr/bin/env bash
# Phase 0 (gate G0) on a single A100-80GB.
#
#   ARM=grpo  ./run/phase0_1gpu.sh
#   ARM=steer ./run/phase0_1gpu.sh
#
# Everything that differs from run/run_steerf_small.sh is a machine budget
# decision, applied IDENTICALLY to every arm so the comparison stays controlled.
# Recorded here as a file rather than as shell history — plan §10 asks for the
# hyper-parameters to be committed, not typed.
#
#   n_gpus            4 -> 1        only one GPU on this host
#   max_response_len  2048 -> 1024  time budget; see docs/experiment_log.md
#   total steps       ~265 -> 100   plan §2.3 asks for 100-150
#   val set           aime24 -> math500
#                       a 1.5B-Instruct scores ~0% on AIME24 inside a 1024-token
#                       budget, so that curve carries no signal. math500 is in
#                       the same parquet format and the same reward function.
#   save_freq         25 -> -1      G0 needs curves, not checkpoints
#
# PINNED and identical across every arm and every later phase (the value is part
# of STEER's method, not a memory knob — docs/steer_code_map.md §8.2):
#   ppo_micro_batch_size_per_gpu = 8
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEERF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARM=${ARM:-grpo}
export PROJECT_NAME=${PROJECT_NAME:-STEER-F-small}
# VARIANT distinguishes two runs of the same arm that differ in the weighting
# configuration (e.g. symmetric vs asymmetric). Without it the second run would
# overwrite the first one's log and tensorboard directory, which is how a
# failed-gate result gets silently lost.
VARIANT=${VARIANT:-}
# Fixed per arm, not per invocation: a rerun of the same arm overwrites its own
# tensorboard run instead of littering the UI with timestamped near-duplicates.
export RUN_TAG="phase0-${ARM}${VARIANT:+-$VARIANT}"
source "${SCRIPT_DIR}/env_1gpu.sh"

export TOTAL_EPOCHS=1
export TEST_PATH=${TEST_PATH:-${STEER_ROOT}/datasets/math500.parquet}

STEPS=${STEPS:-100}
LOG="${STEERF_ROOT}/logs/phase0_${ARM}${VARIANT:+_$VARIANT}.log"
mkdir -p "$(dirname "$LOG")"

echo "[phase0] ARM=${ARM} steps=${STEPS} -> ${LOG}"
ARM="${ARM}" "${SCRIPT_DIR}/run_steerf_small.sh" \
    data.max_response_length=1024 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    trainer.logger="['console','tensorboard']" \
    trainer.val_before_train=True \
    trainer.test_freq=25 \
    trainer.save_freq=-1 \
    trainer.total_training_steps="${STEPS}" \
    "$@" > "$LOG" 2>&1
