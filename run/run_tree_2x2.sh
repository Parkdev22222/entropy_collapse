#!/usr/bin/env bash
# The 2x2 that makes a tree-rollout result attributable.
#
#                    | tree off        | tree on
#     ---------------+-----------------+-----------------
#     steerf_lam=0   | A1 stock STEER  | A2
#     steerf_lam=L   | A3              | A4
#
# A2-A1 is what the tree does on its own.  A3-A1 is what the forecast does
# without a support to compute A_H on.  A4 is the method.  With only A1 and A4
# -- which is what the runs so far amount to -- a win is unattributable, and
# that is the first thing a reviewer asks.
#
# Every arm here differs from every other in exactly two bits: steerf_lam and
# the tree overrides.  Nothing else moves.  In particular STEERF_MAPPING is
# held FIXED across all four; the existing pair of runs varies it with lam
# (rank at lam=0.25, minmax at lam=0) and that alone makes their difference
# uninterpretable.  So does their hardware -- 1 GPU/TP=1 against 2 GPU/TP=2 --
# which is why N_GPUS is set once here and not per arm.
#
#   bash run/run_tree_2x2.sh              # all four, in order
#   ARMS="A2 A4" bash run/run_tree_2x2.sh # just the tree arms
#
# Arms that finished are skipped on a re-run (state in experiments_state/), so
# this is safe to restart after a crash.  Set FORCE=1 to redo them.
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${STEER_ROOT}" || exit 1

if [ ! -f run/run_steerf.sh ]; then
    echo "FATAL: run/run_steerf.sh not found. Run this from the training tree."
    exit 1
fi

# ---- held identical across all four arms ------------------------------------
export SCALE=${SCALE:-paper}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-Math-1.5B}
SEED=${SEED:-1}
# Fix the horizon BEFORE looking at any curve, and use the same one everywhere.
# Unequal horizons make "best checkpoint" a max-of-N statistic that rewards
# whichever arm was evaluated more often, independently of which is better.
# 120 is deliberately not 110: the existing STEER-F run's global maximum sits
# exactly at step 110, and stopping on your own peak reads as cherry-picking
# whether or not it was.
STEPS=${STEPS:-120}
MAPPING=${MAPPING:-minmax}   # ONE value for all four arms. minmax keeps A1
                             # bit-identical to stock STEER; set rank here to
                             # move the whole 2x2, never a single arm.
LAM=${LAM:-0.25}
TREE_DEPTHS=${TREE_DEPTHS:-64,192,384}
TREE_FACTORS=${TREE_FACTORS:-2,2,2}
TREE_ROOTS=${TREE_ROOTS:-1}
export N_GPUS=${N_GPUS:-1}   # same for every arm; _gpu_defaults.sh derives
                             # TP_SIZE and GPU_MEM_UTIL from it
export LOGP_MBS=${LOGP_MBS:-4}
export SAVE_CONTENTS=${SAVE_CONTENTS:-"['hf_model','model','optimizer','extra']"}
export RESUME_MODE=${RESUME_MODE:-auto}
export SAVE_AFTER_OVERRIDE=${SAVE_AFTER_OVERRIDE:-0}
export MAX_CKPT_KEEP=${MAX_CKPT_KEEP:-2}

TAG=${TAG:-tree2x2}
model_tag=$(basename "${MODEL_PATH}")
LOG_DIR=${LOG_DIR:-${STEER_ROOT}/logs/experiments}
STATE_DIR=${STATE_DIR:-${STEER_ROOT}/experiments_state}
mkdir -p "${LOG_DIR}" "${STATE_DIR}"

heads="${STEER_ROOT}/checkpoints/mtp_heads_${model_tag}-${SCALE}.pt"
ARMS=${ARMS:-"A1 A2 A3 A4"}
FAILED=()

run_arm () {   # <arm-id> <lam> <tree 0|1>
    local arm="$1" lam="$2" tree="$3"
    local kind="flat"; [ "${tree}" = "1" ] && kind="tree"
    local lam_tag; lam_tag=$(echo "lam${lam}" | tr -d '.')
    # MAPPING is in the name so a second 2x2 at the other mapping cannot
    # silently overwrite this one's tensorboard directory and state files.
    local run_name="${TAG}-${model_tag}-${MAPPING}-${lam_tag}-${kind}-s${SEED}"
    local name="train-${run_name}"

    if [ -f "${STATE_DIR}/${name}.done" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "[2x2] SKIP  ${arm} ${name} (already done; FORCE=1 to redo)"
        return 0
    fi
    # lam=0 never loads the heads, so only the treatment arms need them.
    if [ "${lam}" != "0" ] && [ ! -f "${heads}" ]; then
        echo "[2x2] SKIP  ${arm} ${name}: no MTP heads at ${heads}"
        echo "[2x2]       run/collect_warmup_rollouts.sh then run/warmup_and_validate.sh,"
        echo "[2x2]       or run ARMS=\"A1 A2\" to do the lam=0 half meanwhile."
        FAILED+=("${arm}")
        return 1
    fi

    # A run name that already has checkpoints is the trap that cost a
    # baseline: default_local_dir is checkpoints/<project>/<experiment_name>,
    # so reusing a name plus RESUME_MODE=auto makes verl RESUME that run --
    # the new arm silently continues the old one's weights and optimizer, and
    # max_actor_ckpt_to_keep then evicts the old checkpoints as it writes.
    # Nothing warns; the log just says "Resuming from ...".
    local ckpt_dir="${STEER_ROOT}/checkpoints/STEER-F/${run_name}"
    if [ -d "${ckpt_dir}" ] && [ "${RESUME:-0}" != "1" ]; then
        echo "[2x2] REFUSE ${arm}: ${ckpt_dir} already exists."
        echo "[2x2]        RESUME_MODE=${RESUME_MODE} would CONTINUE whatever is in there"
        echo "[2x2]        instead of starting this arm, and overwrite its checkpoints."
        echo "[2x2]        Move it aside, pick a different TAG, or set RESUME=1 if you"
        echo "[2x2]        really are continuing this exact arm after a crash."
        FAILED+=("${arm}")
        return 1
    fi

    local extra=()
    if [ "${tree}" = "1" ]; then
        extra+=("++actor_rollout_ref.rollout.steerf_tree_depths=[${TREE_DEPTHS}]")
        extra+=("++actor_rollout_ref.rollout.steerf_tree_factors=[${TREE_FACTORS}]")
        extra+=("++actor_rollout_ref.rollout.steerf_tree_roots=${TREE_ROOTS}")
    fi

    ray stop --force >/dev/null 2>&1
    pkill -f main_ppo >/dev/null 2>&1
    sleep 5

    echo "[2x2] START ${arm} ${name}  ($(date '+%F %T'))"
    if env MODEL_PATH="${MODEL_PATH}" SEED="${SEED}" RUN_NAME="${run_name}" \
           STEERF_LAM="${lam}" STEERF_MAPPING="${MAPPING}" \
       bash run/run_steerf.sh \
           "trainer.logger=['console','tensorboard']" \
           "++trainer.total_training_steps=${STEPS}" \
           "${extra[@]}" \
           > "${LOG_DIR}/${name}.log" 2>&1
    then
        touch "${STATE_DIR}/${name}.done"
        echo "[2x2] DONE  ${arm} ${name}"
        [ "${tree}" = "1" ] && grep -m3 "steerf-tree" "${LOG_DIR}/${name}.log"
        grep -o "steerf/branch_corr_frac:[0-9.]*" "${LOG_DIR}/${name}.log" | head -2
        return 0
    fi
    touch "${STATE_DIR}/${name}.failed"
    FAILED+=("${arm}")
    echo "[2x2] FAIL  ${arm} ${name} -> ${LOG_DIR}/${name}.log (queue continues)"
    return 1
}

for arm in ${ARMS}; do
    case "${arm}" in
        A1) run_arm A1 0       0 || true ;;
        A2) run_arm A2 0       1 || true ;;
        A3) run_arm A3 "${LAM}" 0 || true ;;
        A4) run_arm A4 "${LAM}" 1 || true ;;
        *)  echo "[2x2] unknown arm '${arm}' (expected A1..A4)" ;;
    esac
done

set +x
echo
echo "[2x2] logs in ${LOG_DIR}/train-${TAG}-${model_tag}-*.log"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "[2x2] arms that did not finish: ${FAILED[*]}"
    exit 1
fi
echo "[2x2] all requested arms finished"
