#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The headline contrast on a second, third, fourth backbone.
#
#   BACKBONES="Qwen2.5-Math-7B" bash run/run_backbones.sh
#   DRY=1 bash run/run_backbones.sh              # print the queue, run nothing
#   ARMS="grpo signed" bash run/run_backbones.sh # a subset
#   REPO=user/repo bash run/run_backbones.sh     # upload + delete each checkpoint
#
# WHY ONLY THREE ARMS
#   GRPO / STEER / STEER-F. The five-arm decomposition answers "is it the
#   apparatus or the value of the forecast", and that question is about the
#   method, not about the backbone -- it was answered once on the 1.5B. What a
#   second backbone answers is narrower and cheaper: does the SIGN survive.
#
# WHY THE VALIDATION SET CHANGES PER BACKBONE
#   AIME24 is 30 problems, so SE = std@32/sqrt(30) ~ 2.7 points. A backbone that
#   scores near zero there puts all three arms inside one standard error, and --
#   worse -- save_best_only picks the checkpoint by that same noisy number. The
#   non-mathematics backbones therefore validate on MATH500 (500 problems, no
#   replicas, mean@1), which is about four times tighter. backbone_profile() in
#   _arms.sh carries the pairing; this queue only passes it through.
#
#   The cost is a per-backbone protocol difference. It is stated in the paper,
#   and it is why the analysis compares arms WITHIN a backbone and never
#   absolute numbers ACROSS backbones.
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}" || { echo "FATAL: cannot cd to ${ROOT}" >&2; exit 1; }

# shellcheck source=run/_arms.sh
. "${ROOT}/run/_arms.sh"

BACKBONES=${BACKBONES:-"Qwen2.5-Math-7B Llama-3.1-8B Mistral-7B-v0.3"}
ARMS=${ARMS:-"grpo steer signed"}
SEED=${SEED:-1}
STEPS=${STEPS:-110}
LOG_DIR="${ROOT}/logs/experiments"
MIN_FREE_GB=${MIN_FREE_GB:-20}
DRY=${DRY:-0}
REPO=${REPO:-}

banner () { printf '\n========================================\n%s\n========================================\n' "$*"; }
free_gb () { df -BG --output=avail "${ROOT}" 2>/dev/null | tail -1 | tr -dc '0-9'; }

# --------------------------------------------------------------- validation
for a in ${ARMS}; do
    case "${a}" in
        grpo|steer|signed) ;;
        *) echo "FATAL: '${a}' is not one of grpo steer signed" >&2; exit 2 ;;
    esac
done

# ------------------------------------------------------------------- queue
# Backbone-major: every arm of one backbone, then the next. Stop after any
# backbone and that backbone's three-way contrast is complete, which is the
# unit the paper reports. Arm-major would leave every backbone half-finished.
declare -a QUEUE=()
for b in ${BACKBONES}; do
    ( export MODEL_TAG="${b}"
      unset MODEL_PATH VAL_PARQUET BEST_METRIC_KEY
      backbone_profile "${b}" >/dev/null 2>&1 ) \
        || { echo "FATAL: unknown backbone '${b}' -- add it to backbone_profile()" >&2; exit 2; }
    for a in ${ARMS}; do
        rn="$(MODEL_TAG="${b}" run_name_for "${a}" "${SEED}")"
        if train_log_done "${LOG_DIR}" "${rn}" "${STEPS}"; then
            printf '  skip  %-52s (done)\n' "${rn}"
        else
            printf '  QUEUE %-52s\n' "${rn}"
            QUEUE+=("${b}:${a}:${rn}")
        fi
    done
done
printf '\n%s run(s) queued, %s backbone(s) x %s arm(s)\n' \
    "${#QUEUE[@]}" "$(echo ${BACKBONES} | wc -w)" "$(echo ${ARMS} | wc -w)"
[ "${#QUEUE[@]}" -eq 0 ] && { echo "Nothing to do."; exit 0; }
[ "${DRY}" = "1" ] && { echo "(DRY=1, stopping here)"; exit 0; }

# ------------------------------------------------------------------ guards
LOCK="${ROOT}/.backbones.lock"
if ! mkdir "${LOCK}" 2>/dev/null; then
    holder="$(cat "${LOCK}/pid" 2>/dev/null || echo '?')"
    if [ "${holder}" != "?" ] && kill -0 "${holder}" 2>/dev/null; then
        echo "REFUSE: another backbone queue is running (pid ${holder})." >&2
        exit 2
    fi
    rm -rf "${LOCK}"; mkdir "${LOCK}" || { echo "FATAL: cannot take ${LOCK}" >&2; exit 2; }
fi
echo $$ > "${LOCK}/pid"
trap 'rm -rf "${LOCK}"' EXIT INT TERM

is_busy && { echo "REFUSE: a training process is already running." >&2; exit 2; }
env_preflight "${ROOT}" || { echo "REFUSE: the training environment is broken." >&2; exit 2; }
topology_guard || exit 2

# ---------------------------------------------------------- shared settings
export SAVE_BEST_ONLY=True
export SAVE_CONTENTS="['hf_model']"
export SAVE_AFTER=0
export SAVE_AFTER_OVERRIDE=0
export N_GPUS TP_SIZE
export VAL_DATA_DIR="${ROOT}/validation_data"
export STEPS
mkdir -p "${LOG_DIR}" "${VAL_DATA_DIR}"

banner "backbones: ${#QUEUE[@]} run(s), seed ${SEED}, steps=${STEPS}"

for item in "${QUEUE[@]}"; do
    bb="${item%%:*}"; rest="${item#*:}"; arm="${rest%%:*}"; rn="${rest##*:}"
    log="${LOG_DIR}/train-${rn}.log"

    # Everything the backbone decides, decided together and exported, so no
    # launcher default is reachable -- including run_uniform_ablation.sh's
    # hardcoded 1.5B heads.
    export MODEL_TAG="${bb}"
    unset MODEL_PATH VAL_PARQUET BEST_METRIC_KEY
    backbone_profile "${bb}" || exit 2
    export STEERF_HEADS="${ROOT}/checkpoints/mtp_heads_${bb}.pt"
    export STEERF_CALIB="${ROOT}/checkpoints/mtp_calibration_${bb}.json"

    if ! model_guard; then exit 2; fi

    # Only the STEER-F arm opens the forecaster. GRPO does not, and STEER runs
    # at lambda=0, where run_steerf.sh:76 never loads it.
    if [ "${arm}" = "signed" ] && ! heads_guard; then
        echo "[backbones] SKIP ${rn} -- warm up ${bb}'s heads first"
        continue
    fi

    avail="$(free_gb)"
    if [ -n "${avail}" ] && [ "${avail}" -lt "${MIN_FREE_GB}" ]; then
        echo "STOP: only ${avail} GB free, need ${MIN_FREE_GB}." >&2
        exit 1
    fi

    banner "${bb}  ${arm}  ->  ${rn}   (${avail:-?} GB free)"
    echo "  model  ${MODEL_PATH}"
    echo "  val    ${VAL_PARQUET}  ->  ${BEST_METRIC_KEY}"
    start=$(date +%s)
    ray stop --force >/dev/null 2>&1 || true
    sleep 5
    await_gpus || true

    # The launchers hardcode aime24 and the AIME24 selection key, so both go in
    # as trailing hydra overrides, which land last and win.
    val_over=( "data.val_files=['${ROOT}/${VAL_PARQUET}']"
               "++trainer.best_metric_key=${BEST_METRIC_KEY}" )

    case "${arm}" in
        grpo)
            SEED="${SEED}" RUN_NAME="${rn}" LOG="${log}" \
                bash run/run_grpo.sh "${val_over[@]}"
            st=$? ;;
        steer)
            SEED="${SEED}" RUN_NAME="${rn}" STEERF_LAM=0 \
                bash run/run_steerf.sh $(steer_plain_args "${STEPS}") "${val_over[@]}" 2>&1 | tee "${log}"
            st=${PIPESTATUS[0]} ;;
        signed)
            ARM=signed SEED="${SEED}" RUN_NAME="${rn}" STEERF_LAM=0.25 \
                bash run/run_uniform_ablation.sh "${val_over[@]}"
            st=$? ;;
    esac
    printf '[backbones] %s %s exit %s after %s min\n' \
        "${bb}" "${arm}" "${st}" "$(( ($(date +%s) - start) / 60 ))"

    if [ "${st}" -ne 0 ]; then
        echo "[backbones] FAILED -- keeping the checkpoint, moving on"
        diagnose_startup_failure "${log}" "${bb} ${arm}" || true
        continue
    fi
    if ! train_log_done "${LOG_DIR}" "${rn}" "${STEPS}"; then
        echo "[backbones] WARNING: exit 0 but the log never reached step ${STEPS}"
        continue
    fi
    if [ -n "${REPO}" ]; then
        REPO="${REPO}" DELETE=1 bash run/hf_backup.sh "${rn}" \
            || echo "[backbones] upload/verify failed -- the local copy is kept"
    fi
done

banner "backbones finished"
for b in ${BACKBONES}; do
    for a in ${ARMS}; do
        rn="$(MODEL_TAG="${b}" run_name_for "${a}" "${SEED}")"
        train_log_done "${LOG_DIR}" "${rn}" "${STEPS}" && r=done || r=MISSING
        printf '  %-52s %s\n' "${rn}" "${r}"
    done
done
