#!/usr/bin/env bash
# Phase 2 (gate G2), one arm, on a single A100-80GB.
#
#   LAM=0.5 SEED=2 ./run/phase2_1gpu.sh
#
# G2 asks for "평균 pass@16 +1.0pp 이상, pass@1 비열화". Two things follow from
# that sentence and neither is the default:
#
#   1. pass@16 needs 16 validation samples per problem.  verl's default
#      val_kwargs.n=1 only ever yields acc/mean@1, so the gate's primary metric
#      would silently not exist.  n=16 makes verl emit best@16/mean, which is
#      pass@16, alongside mean@16.
#
#   2. +1.0pp is smaller than the sampling noise of a single run.  MATH500 has
#      500 problems, so the binomial SE of mean@1 is ~2.2pp — a one-seed
#      comparison cannot resolve the gate's own threshold.  Every arm is
#      therefore repeated over SEEDS, and the gate is judged on the seed mean.
#      Plan §10 lists this as "시드 3개 이상으로 각 팔 반복".
#
# SEED is applied to the dataloader ONLY, deliberately.
#
# `+actor_rollout_ref.rollout.seed` looks like the natural companion, and it was
# set here at first, but it lands in vLLM's SamplingParams -- and verl validates
# by repeating each prompt val_kwargs.n times as separate requests
# (ray_trainer.py:608), not by asking for n samples in one request. A fixed
# per-request seed therefore makes all 16 "samples" byte-identical, and the run
# reported
#
#     mean@16 = best@16 = maj@16 = 0.552,  best@16/std = 0.0
#
# i.e. pass@16 silently degenerated into pass@1 -- destroying the metric G2 is
# judged on. Replicates still differ, because a different data order gives a
# different training trajectory; what is given up is only sampler-level
# determinism within a run.
#
# save_freq=100 keeps the final checkpoint. Phase 0 ran with save_freq=-1 and
# consequently could not be re-scored when this exact class of bug was found;
# ~3 GB per run is cheap insurance against re-training instead of re-validating.
#
# Everything else is inherited from run/phase0_1gpu.sh's budget decisions so
# that Phase 0 and Phase 2 remain comparable, including the pinned
# ppo_micro_batch_size_per_gpu=8 (docs/steer_code_map.md §8.2).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEERF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LAM=${LAM:-0.0}
SEED=${SEED:-1}
STEPS=${STEPS:-100}

# lambda=0 goes through the identical code path and is bit-identical to stock
# STEER (tests/test_lambda_zero_equiv.py), so it is the baseline arm rather
# than a separate implementation.
if [ "${LAM}" = "0.0" ] || [ "${LAM}" = "0" ]; then
  ARM=steer
else
  ARM=steerf
fi

export PROJECT_NAME=${PROJECT_NAME:-STEER-F-phase2}
FORECAST=${FORECAST:-mtp}
export STEERF_FORECAST="${FORECAST}"
# The oracle control shares the lambda grid, so its runs need distinct names.
#
# A caller-supplied RUN_TAG wins. Naming by (lambda, seed) alone is not unique
# once several formulations share the grid: queue_e's apply=weight lam=0.25
# seed 1 wrote to phase2-lam0.25-s1.log, the name the main grid's lam=0.25
# seed 1 had already used, then renamed it on exit. That destroyed one finished
# run's log outright -- 337 GPU-minutes with no recoverable result -- and
# overwrote another's entropy trajectory. The formulation must be in the name,
# and only the caller knows which formulation it is running.
if [ -n "${RUN_TAG:-}" ]; then
  export RUN_TAG
elif [ "${FORECAST}" = "oracle" ]; then
  export RUN_TAG="phase2-oracle-lam${LAM}-s${SEED}"
else
  export RUN_TAG="phase2-lam${LAM}-s${SEED}"
fi
source "${SCRIPT_DIR}/env_1gpu.sh"

export TOTAL_EPOCHS=1
export TEST_PATH=${TEST_PATH:-${STEER_ROOT}/datasets/math500.parquet}

# Weighting configuration, pinned identically across every arm of the sweep.
#
# NOT the defaults of run/run_steerf.sh. Gate G0 failed under the symmetric
# [0.8, 1.2] configuration that run_linear.sh uses: the weights degenerated to a
# near-constant 1.07 (tw_std 0.005, and the attenuating third of the range was
# empty at every one of 100 steps), so the baseline acted as a uniform ~7%
# learning-rate multiplier and collapsed entropy *faster* than plain GRPO
# (0.168 vs 0.314 at step 100). The asymmetric configuration below is STEER's
# own run_asymmetric.sh, and under it G0 passes: 0.630 at step 100, with the
# attenuating range populated (tw_frac_low 0.126) and mean weight below one.
#
# Evaluating STEER-F against a baseline configuration that does not clear G0
# would make the comparison meaningless, so these are not overridable per arm.
export ENTROPY_CONTROL_MODE=asymmetric
export TOKEN_WEIGHT_MIN=0.9
export TOKEN_WEIGHT_MAX=1.0
export LINEAR=True
export STEERF_LAM="${LAM}"
export STEERF_HEADS=${STEERF_HEADS:-${STEERF_ROOT}/checkpoints/mtp_heads.pt}
export STEERF_CALIB=${STEERF_CALIB:-${STEERF_ROOT}/checkpoints/mtp_calibration.json}
export STEERF_KAPPA=${STEERF_KAPPA:-4}
export STEERF_GAMMA_H=${STEERF_GAMMA_H:-0.85}

LOG="${STEERF_ROOT}/logs/${RUN_TAG}.log"
mkdir -p "$(dirname "$LOG")"

if [ "${ARM}" = "steerf" ] && [ "${FORECAST}" != "oracle" ]; then
  if [ ! -f "${STEERF_HEADS}" ]; then
    echo "[phase2] no head checkpoint at ${STEERF_HEADS} — Phase 1 must run first"
    exit 1
  fi
  # Refuse to turn a plumbing test into a result.
  if python -c "
import sys, torch
sys.exit(1 if torch.load('${STEERF_HEADS}', map_location='cpu', weights_only=False).get('untrained') else 0)
"; then :; else
    echo "[phase2] ${STEERF_HEADS} is an UNTRAINED (dummy) checkpoint — refusing"
    exit 1
  fi
fi

echo "[phase2] ARM=${ARM} lambda=${LAM} seed=${SEED} steps=${STEPS} -> ${LOG}"
ARM="${ARM}" "${SCRIPT_DIR}/run_steerf_small.sh" \
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
