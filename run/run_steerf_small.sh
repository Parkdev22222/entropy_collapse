#!/usr/bin/env bash
# Small-model arm (plan §2.3 reproduction, §4.2 E2-1). Scaled down from
# run_steerf.sh: 1.5B-1.7B policy, single node, short run.
#
#   ARM=grpo   ./run/run_steerf_small.sh    # Phase 0 baseline
#   ARM=steer  ./run/run_steerf_small.sh    # Phase 0 STEER reproduction (G0)
#   ARM=steerf STEERF_LAM=0.25 ./run/run_steerf_small.sh
#
# Plan §2.3 asks for 100-150 steps. train_batch_size=64 over DAPO-Math-17k
# gives ~265 steps/epoch, so TOTAL_EPOCHS=1 with an early stop covers it.
#
# NOTE (docs/steer_code_map.md §8.2): STEER's token-weight thresholds are
# computed per micro-batch, so ppo_micro_batch_size_per_gpu is part of the
# method's configuration, not just a memory knob. It is pinned here and must
# match across every arm being compared.
set -x

export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-1.5B-Instruct"}
export PROJECT_NAME=${PROJECT_NAME:-"STEER-F-small"}
export N_GPUS=${N_GPUS:-4}
export TP_SIZE=${TP_SIZE:-1}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_BATCH_SIZE=64 \
MINI_BATCH_SIZE=16 \
MAX_RESPONSE_LENGTH=2048 \
  "${SCRIPT_DIR}/run_steerf.sh" \
    data.train_batch_size=64 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    data.max_response_length=2048 \
    trainer.save_freq=25 \
    trainer.test_freq=5 \
    "$@"
# Forward this wrapper's own arguments too, so a caller can override anything
# set above (hydra takes the last assignment for a key).
