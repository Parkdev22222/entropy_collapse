#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 0905 re-training chain:  signed (STEER-F)  ->  steer  ->  uniform
#
# All three arms: 110 steps, seed 1, minmax, save_best_only=True, and a log
# named train-<run>_0905.log.  Runs sequentially -- a single trainer at a time,
# because each one takes both GPUs.
#
# Why this file exists rather than a pasted heredoc: a ~70-line heredoc pasted
# into a terminal is silently corrupted by bracketed-paste/echo races (lines
# duplicated, `if` bodies truncated), and the corruption is only visible as a
# bash syntax error at launch.  Fetch this file with git instead of pasting it.
#
#   git fetch origin claude/3b-text-generation-models-thz2vl
#   git checkout origin/claude/3b-text-generation-models-thz2vl -- run/run_0905_chain.sh
#   bash -n run/run_0905_chain.sh            # must print nothing
#   nohup bash run/run_0905_chain.sh > logs/experiments/chain_0905.log 2>&1 &
#
# Prerequisite, applied once on the training box (run_steerf.sh hardcodes
# save_best_only=False; this makes it an env knob and defaults to the old
# value, so nothing else changes):
#
#   sed -i 's|++trainer.save_best_only=False|++trainer.save_best_only=${SAVE_BEST_ONLY:-False}|' run/run_steerf.sh
#
# The chain refuses to start if that edit is missing -- otherwise all three
# arms would quietly save every checkpoint and fill the disk.
#
# Roughly 122 h end to end (signed 43.5 + steer 32 + uniform 43.5, plus
# validation and vLLM restarts).  Budget six days.
# ---------------------------------------------------------------------------
set -uo pipefail

STEER_ROOT=${STEER_ROOT:-/workspace/entropy_collapse}
cd "${STEER_ROOT}" || { echo "FATAL: no ${STEER_ROOT}"; exit 1; }
# shellcheck source=run/_arms.sh
. "${STEER_ROOT}/run/_arms.sh"

TAG=${TAG:-0905}
SEED=${SEED:-1}
STEPS=${STEPS:-110}
LOG_DIR="${STEER_ROOT}/logs/experiments"
CKPT_ROOT="${STEER_ROOT}/checkpoints/STEER-F"
mkdir -p "${LOG_DIR}"

# ---- held identical across all three arms ---------------------------------
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-Math-1.5B}
export N_GPUS=${N_GPUS:-2}
export TP_SIZE=${TP_SIZE:-2}
export SEED
export STEERF_KAPPA=${STEERF_KAPPA:-2}
export STEERF_GAMMA_H=${STEERF_GAMMA_H:-0.7}
export STEERF_MAPPING=${STEERF_MAPPING:-minmax}
export SAVE_BEST_ONLY=${SAVE_BEST_ONLY:-True}
# Full contents so a crash is resumable; trim_ckpt drops the optimizer state
# once an arm has finished, which takes a run from 25 GiB down to 6.7 GiB.
export SAVE_CONTENTS=${SAVE_CONTENTS:-"['hf_model','model','optimizer','extra']"}
export RESUME_MODE=${RESUME_MODE:-auto}
export SAVE_AFTER_OVERRIDE=${SAVE_AFTER_OVERRIDE:-0}
export MAX_CKPT_KEEP=${MAX_CKPT_KEEP:-1}

banner () {
    echo
    echo "=========================================================="
    echo " $*   |   $(date '+%F %T')"
    echo "=========================================================="
}

# Re-running this chain must not redo an arm that already finished, and left
# alone it would: run_uniform_ablation.sh refuses when the checkpoint directory
# exists, so a completed signed would be recorded as FAILED and the summary
# would lie about it. This is not hypothetical -- on 2026-09-08 a huggingface_hub
# upgrade killed steer and uniform at import while signed had already completed,
# and the fix is to re-run the chain for the two that died.
arm_done () { train_log_done "${LOG_DIR}" "$1" "${STEPS}"; }

# eval only needs actor/huggingface; optimizer/extra exist for resume.
trim_ckpt () {
    local run="$1" d
    for d in "${CKPT_ROOT}/${run}"/global_step_*; do
        [ -d "${d}/actor/huggingface" ] || continue
        find "${d}/actor" -mindepth 1 -maxdepth 1 ! -name huggingface -exec rm -rf {} +
    done
    echo "[trim] ${run} -> $(du -sh "${CKPT_ROOT}/${run}" 2>/dev/null | cut -f1)"
}

# Trim only on success. A failed arm keeps its optimizer state so RESUME=1 can
# pick it up instead of restarting 40 hours of training.
finish () {   # <exit-status> <run-name> <arm-label>
    if [ "$1" = "0" ]; then
        trim_ckpt "$2"
    else
        echo "[chain] $3 FAILED (exit $1) -- keeping the checkpoint so it can be resumed"
    fi
}

# ---- preflight ------------------------------------------------------------
if ! grep -q 'save_best_only=\${SAVE_BEST_ONLY' run/run_steerf.sh; then
    echo "REFUSE: run/run_steerf.sh still hardcodes save_best_only."
    echo "        Apply the sed in this file's header first."
    exit 2
fi
for f in checkpoints/mtp_heads_Qwen2.5-Math-1.5B-paper.pt \
         checkpoints/mtp_calibration_Qwen2.5-Math-1.5B-paper.json; do
    [ -f "$f" ] || { echo "REFUSE: missing ${f} (needed by signed and uniform)"; exit 2; }
done
if pgrep -f main_ppo >/dev/null 2>&1; then
    echo "REFUSE: a trainer is already running (pgrep main_ppo). Stop it first."
    exit 2
fi
if ! env_preflight "${STEER_ROOT}"; then
    echo "REFUSE: the training environment is broken -- nothing would train."
    exit 2
fi

banner "0905 chain: signed -> steer -> uniform   steps=${STEPS} seed=${SEED}"
df -h /workspace | tail -1

# ---------------------------------------------------------------- 1/3 signed
RUN_SIGNED="steer-f-Qwen2.5-Math-1.5B-s${SEED}-tree-rollout_${TAG}"
if arm_done "${RUN_SIGNED}"; then
    banner "1/3  signed -- already at step ${STEPS}, skipping"
    st_signed=0
else
    banner "1/3  signed (STEER-F, lam=0.25, tree)  ->  ${RUN_SIGNED}"
    ray stop --force >/dev/null 2>&1 || true
    sleep 5
    ARM=signed RUN_NAME="${RUN_SIGNED}" STEPS="${STEPS}" STEERF_LAM=0.25 \
        bash run/run_uniform_ablation.sh
    st_signed=$?
    echo "[chain] signed exit ${st_signed}"
    finish "${st_signed}" "${RUN_SIGNED}" signed
fi

# ---------------------------------------------------------------- 2/3 steer
# lam=0, no tree overrides. run_uniform_ablation.sh only accepts
# uniform/signed/permuted, so this arm calls run_steerf.sh directly and has to
# repeat the two overrides that script passes for every arm: the tensorboard
# logger (select_best_checkpoint.py parses the event files) and
# rollout_data_dir=null (writing rollouts leaked ~300 GB of host RAM).
RUN_STEER="steer-Qwen2.5-Math-1.5B-s${SEED}_${TAG}"
if arm_done "${RUN_STEER}"; then
    banner "2/3  steer -- already at step ${STEPS}, skipping"
    st_steer=0
else
    banner "2/3  steer (lam=0, plain rollout)  ->  ${RUN_STEER}"
    ray stop --force >/dev/null 2>&1 || true
    sleep 5
    STEERF_LAM=0 STEERF_APPLY=weight STEERF_PERMUTE_AH=0 RUN_NAME="${RUN_STEER}" \
        bash run/run_steerf.sh \
            "trainer.logger=['console','tensorboard']" \
            "++trainer.total_training_steps=${STEPS}" \
            "++trainer.rollout_data_dir=null" \
        > "${LOG_DIR}/train-${RUN_STEER}.log" 2>&1
    st_steer=$?
    echo "[chain] steer exit ${st_steer}"
    finish "${st_steer}" "${RUN_STEER}" steer
fi

# -------------------------------------------------------------- 3/3 uniform
RUN_UNIFORM="steer-f-Qwen2.5-Math-1.5B-s${SEED}-tree-rollout-uniform_${TAG}"
if arm_done "${RUN_UNIFORM}"; then
    banner "3/3  uniform -- already at step ${STEPS}, skipping"
    st_uniform=0
else
    banner "3/3  uniform (lam=0.25, tree, apply=branch)  ->  ${RUN_UNIFORM}"
    ray stop --force >/dev/null 2>&1 || true
    sleep 5
    ARM=uniform RUN_NAME="${RUN_UNIFORM}" STEPS="${STEPS}" STEERF_LAM=0.25 \
        bash run/run_uniform_ablation.sh
    st_uniform=$?
    echo "[chain] uniform exit ${st_uniform}"
    finish "${st_uniform}" "${RUN_UNIFORM}" uniform
fi

# ---------------------------------------------------------------- summary
banner "chain finished"
printf '  signed   exit %-3s  logs/experiments/train-%s.log\n' "${st_signed}"  "${RUN_SIGNED}"
printf '  steer    exit %-3s  logs/experiments/train-%s.log\n' "${st_steer}"   "${RUN_STEER}"
printf '  uniform  exit %-3s  logs/experiments/train-%s.log\n' "${st_uniform}" "${RUN_UNIFORM}"
echo
echo "  best step per arm:"
for r in "${RUN_SIGNED}" "${RUN_STEER}" "${RUN_UNIFORM}"; do
    printf '    %-58s %s\n' "$r" "$(cat "${CKPT_ROOT}/${r}/best_checkpoint_info.json" 2>/dev/null || echo '(none)')"
done
echo
df -h /workspace | tail -1

# Non-zero if any arm failed, so `echo $?` after the chain is meaningful.
[ "${st_signed}" = "0" ] && [ "${st_steer}" = "0" ] && [ "${st_uniform}" = "0" ]
