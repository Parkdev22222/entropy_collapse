#!/usr/bin/env bash
# Copyright 2026 STEER-F authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ---------------------------------------------------------------------------
# The one experiment that decides whether the forecast is doing anything.
#
# `branch_weight_correction` has two modes and the difference is the whole
# claim:
#
#   signed  (STEERF_APPLY=weight)  delta = lam*band*tanh(visit/rms)
#                                  uses A_H's VALUE to rank siblings
#   uniform (STEERF_APPLY=branch)  delta = -lam*band  on the same support
#                                  uses only WHERE A_H is defined, never its value
#
# Both compute the MTP forecast, so `support` and the wall-clock are identical;
# the only thing that changes is whether the forecast's numbers are read. From
# omega_tilde.py's own docstring:
#
#     "it needs no forecast, no MTP heads and no Phase 1, because 'does this
#      rollout still have siblings here' is answered by the rollout group
#      alone. If it matches `signed`, the forecast is not carrying the effect."
#
# Why this is the next run and not seeds or baseline=group: no logged metric
# bears on the question. `branch_recall` is pinned at ~0.5 by the sign of a
# zero-centred score (pure noise scores 0.49 -- scripts/measure_ah_support.py),
# and `branch_entropy_gap` selects the same decile. Adding seeds now measures
# an arm whose mechanism is unidentified.
#
# Reading it:
#   uniform ~= signed  -> the forecast contributes nothing. The MTP heads,
#                         the calibration and Phase 1 are all unjustified, and
#                         the method reduces to "attenuate branch points",
#                         which is a cleaner and much cheaper claim.
#   signed  >  uniform -> the forecast ranks siblings usefully. That is the
#                         first direct evidence for it.
#   uniform >  signed  -> A_H's values are actively misleading.
#
# Every knob below is pinned to the in-flight signed run
# (steer-f-Qwen2.5-Math-1.5B-s1-tree-rollout, read off its log) so the two arms
# differ in STEERF_APPLY and nothing else.
#
# Usage
#   bash run/run_uniform_ablation.sh                 # the uniform arm
#   ARM=signed bash run/run_uniform_ablation.sh      # re-run the control
#   DRY_RUN=1 bash run/run_uniform_ablation.sh       # print the command only
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ARM=${ARM:-uniform}
case "${ARM}" in
    uniform) APPLY=branch ;;
    signed)  APPLY=weight ;;
    *) echo "ARM must be 'uniform' or 'signed', got '${ARM}'" >&2; exit 2 ;;
esac

# ---- matched to the in-flight signed run; change these only in pairs --------
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-Math-1.5B}
export SEED=${SEED:-1}
export N_GPUS=${N_GPUS:-2}
export TP_SIZE=${TP_SIZE:-2}

export STEERF_LAM=${STEERF_LAM:-0.25}
export STEERF_KAPPA=${STEERF_KAPPA:-2}
export STEERF_GAMMA_H=${STEERF_GAMMA_H:-0.7}
export STEERF_CLIP_C=${STEERF_CLIP_C:-1.0}
export STEERF_NORM=${STEERF_NORM:-scale}
export STEERF_BASELINE=${STEERF_BASELINE:-sibling}
export STEERF_MAPPING=${STEERF_MAPPING:-minmax}
export STEERF_WINSOR_Q=${STEERF_WINSOR_Q:-0.01}
export STEERF_FORECAST=${STEERF_FORECAST:-mtp}
export STEERF_HEADS=${STEERF_HEADS:-${STEER_ROOT}/checkpoints/mtp_heads_Qwen2.5-Math-1.5B-paper.pt}
export STEERF_CALIB=${STEERF_CALIB:-${STEER_ROOT}/checkpoints/mtp_calibration_Qwen2.5-Math-1.5B-paper.json}
export RESUME_MODE=${RESUME_MODE:-auto}

# the one line under test
export STEERF_APPLY=${APPLY}

TREE_DEPTHS=${TREE_DEPTHS:-64,192,384}
TREE_FACTORS=${TREE_FACTORS:-2,2,2}
TREE_ROOTS=${TREE_ROOTS:-1}
STEPS=${STEPS:-200}

RUN_NAME=${RUN_NAME:-"steer-f-Qwen2.5-Math-1.5B-s${SEED}-tree-rollout-${ARM}"}
export RUN_NAME
LOG_DIR=${LOG_DIR:-${STEER_ROOT}/logs/experiments}
LOG="${LOG_DIR}/train-${RUN_NAME}.log"

# ---- guards ----------------------------------------------------------------
# A second trainer on the same GPUs will OOM both. The in-flight signed run is
# the control for this arm, so waiting for it is the point, not an obstacle.
# `pgrep -f main_ppo` matches any command line that merely mentions the string,
# including a shell running this very check, so confirm the match is a real
# interpreter (python / a ray worker) rather than a shell.
trainer_running() {
    local pid comm
    for pid in $(pgrep -f main_ppo 2>/dev/null); do
        [ "${pid}" = "$$" ] && continue
        comm=$(cat "/proc/${pid}/comm" 2>/dev/null) || continue
        case "${comm}" in
            python*|pt_main_thread*|ray*) return 0 ;;
        esac
    done
    return 1
}

if [ "${FORCE_CONCURRENT:-0}" != "1" ] && trainer_running; then
    echo "REFUSE: a trainer is already running (python ... main_ppo)."
    echo "        This arm needs the GPUs the in-flight run is using, and that"
    echo "        run is this arm's control. Wait for it to reach step ${STEPS},"
    echo "        then start this. FORCE_CONCURRENT=1 overrides, only if you"
    echo "        know the GPUs are free."
    exit 1
fi

CKPT_DIR="${STEER_ROOT}/checkpoints/STEER-F/${RUN_NAME}"
if [ -d "${CKPT_DIR}" ] && [ "${RESUME:-0}" != "1" ]; then
    echo "REFUSE: ${CKPT_DIR} already exists."
    echo "        RESUME_MODE=${RESUME_MODE} would continue it rather than start"
    echo "        this arm. Move it aside, pick a different RUN_NAME, or set"
    echo "        RESUME=1 if you really are continuing this arm after a crash."
    exit 1
fi

if [ "${STEERF_FORECAST}" = "mtp" ] && [ ! -f "${STEERF_HEADS}" ]; then
    echo "REFUSE: MTP heads not found at ${STEERF_HEADS}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

echo "=============================================================="
echo " arm            ${ARM}   (STEERF_APPLY=${STEERF_APPLY})"
echo " run name       ${RUN_NAME}"
echo " control        steer-f-Qwen2.5-Math-1.5B-s${SEED}-tree-rollout"
echo " lam/kappa/g_H  ${STEERF_LAM} / ${STEERF_KAPPA} / ${STEERF_GAMMA_H}"
echo " tree           roots=${TREE_ROOTS} depths=[${TREE_DEPTHS}] factors=[${TREE_FACTORS}]"
echo " seed / gpus    ${SEED} / ${N_GPUS} (tp=${TP_SIZE})"
echo " steps          ${STEPS}"
echo " log            ${LOG}"
echo "=============================================================="

# rollout_data_dir=null: writing rollouts leaked to ~300 GB of host RAM and
# killed the first tree run at step 10. The in-flight run has it nulled too.
ARGS=(
    "trainer.logger=['console','tensorboard']"
    "++trainer.total_training_steps=${STEPS}"
    "++trainer.rollout_data_dir=null"
    "++actor_rollout_ref.rollout.steerf_tree_depths=[${TREE_DEPTHS}]"
    "++actor_rollout_ref.rollout.steerf_tree_factors=[${TREE_FACTORS}]"
    "++actor_rollout_ref.rollout.steerf_tree_roots=${TREE_ROOTS}"
)

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY RUN — would execute:"
    printf '  STEERF_APPLY=%s RUN_NAME=%s bash run/run_steerf.sh \\\n' "${STEERF_APPLY}" "${RUN_NAME}"
    printf '      %s \\\n' "${ARGS[@]}"
    echo "  > ${LOG} 2>&1"
    exit 0
fi

ray stop --force >/dev/null 2>&1 || true
sleep 5

# `set -e` would abort before the summary below, so capture instead of trap.
status=0
bash "${STEER_ROOT}/run/run_steerf.sh" "${ARGS[@]}" > "${LOG}" 2>&1 || status=$?

echo "[${ARM}] exit ${status} -> ${LOG}"
grep -m2 "steerf-tree" "${LOG}" || true
for k in branch_corr_frac branch_corr_frac_strict branch_corr_mean_abs tw_std twg_std; do
    printf '  %-24s %s\n' "${k}" "$(grep -o "steerf/${k}:[0-9.]*" "${LOG}" | tail -1)"
done
echo
echo "Compare against the signed control once both reach step ${STEPS}:"
echo "  for f in train-steer-f-Qwen2.5-Math-1.5B-s${SEED}-tree-rollout.log \\"
echo "           train-${RUN_NAME}.log; do"
echo "    echo \"== \$f\"; grep -o 'acc/mean@32:[0-9.]*\\|maj@32:[0-9.]*' ${LOG_DIR}/\$f | tail -6"
echo "  done"
exit ${status}
