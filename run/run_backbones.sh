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

# Mistral-7B-v0.3 is the BASE checkpoint, and the base Llama that used to sit in
# this list scored .020 on MATH500 -- see the 2026-09-17 note in _arms.sh. Gate a
# base backbone on the training-set group pass rate before spending a run on it:
#     python3 scripts/phase3_port_model.py passrate --model <path> \
#         --prompts datasets/DAPO-Math-17k.parquet --min-informative <frac>
BACKBONES=${BACKBONES:-"Qwen2.5-Math-7B Llama-3.2-3B-Instruct Mistral-7B-v0.3"}
ARMS=${ARMS:-"grpo steer signed"}
SEED=${SEED:-1}
STEPS=${STEPS:-110}
LOG_DIR="${ROOT}/logs/experiments"
# 30, not 20: a resumable checkpoint carries the optimizer state, so it is
# ~25 GiB rather than 3.1 GiB. See the shared settings below.
MIN_FREE_GB=${MIN_FREE_GB:-30}
CKPT_ROOT="${ROOT}/checkpoints/STEER-F"
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
# One definition of how a backbone arm is launched, so the OOM retry re-runs
# what failed rather than a second copy of it.
launch_backbone () {   # <arm> <run-name> <log> [hydra overrides...]
    local arm="$1" rn="$2" log="$3"
    shift 3
    local resume=""
    # The launchers refuse an existing checkpoint dir on purpose; continuing is
    # what this queue means, since the arm is here because it is unfinished.
    [ -d "${CKPT_ROOT}/${rn}" ] && resume=1
    case "${arm}" in
        grpo)
            SEED="${SEED}" RUN_NAME="${rn}" LOG="${log}" ${resume:+RESUME=1} \
                bash run/run_grpo.sh "$@"
            return $? ;;
        steer)
            SEED="${SEED}" RUN_NAME="${rn}" STEERF_LAM=0 ${resume:+RESUME=1} \
                bash run/run_steerf.sh $(steer_plain_args "${STEPS}") "$@" 2>&1 | tee "${log}"
            return "${PIPESTATUS[0]}" ;;
        signed)
            ARM=signed SEED="${SEED}" RUN_NAME="${rn}" STEERF_LAM=0.25 ${resume:+RESUME=1} \
                bash run/run_uniform_ablation.sh "$@"
            return $? ;;
    esac
    echo "[backbones] unknown arm '${arm}'" >&2
    return 3
}

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
# A crash used to cost the whole run: ['hf_model'] writes no optimizer or rng
# state, so trainer.resume_mode had nothing to resume FROM (run_steerf.sh:154).
# On 2026-09-15 a CUDA OOM at step 47 cost all 25 h of the seed-2 steer arm.
# The optimizer state turns that into at most SAVE_FREQ steps. It costs disk --
# ~25 GiB per checkpoint against 3.1 GiB -- which MAX_CKPT_KEEP=1 holds flat and
# `PRUNE=1 bash run/hf_backup.sh <run>` gives back once a run has finished.
export SAVE_CONTENTS="['hf_model','model','optimizer','extra']"
export RESUME_MODE=auto
export MAX_CKPT_KEEP=1
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
    # ${SCALE} or the file will not be there. warmup_and_validate.sh:24
    # defaults SCALE to "paper" and names its outputs
    # mtp_heads_<tag>-paper.pt, run_steerf.sh:73 derives the same name from
    # the same variable, and run_uniform_ablation.sh:95 -- the launcher the
    # signed arm actually goes through -- hardcodes the -paper variant. This
    # line used to build the bare mtp_heads_<tag>.pt, which nothing writes.
    #
    # The cost was quiet, which is the point: heads_guard gates only the
    # signed arm, so GRPO and STEER would train for ~80 H200-hours and the
    # one arm the backbone row exists for would be skipped with a line that
    # scrolls past in a log.
    scale="${SCALE-paper}"
    export STEERF_HEADS="${ROOT}/checkpoints/mtp_heads_${bb}${scale:+-${scale}}.pt"
    export STEERF_CALIB="${ROOT}/checkpoints/mtp_calibration_${bb}${scale:+-${scale}}.json"
    # Fall back to the un-suffixed pair when that is what the box has: a
    # backbone warmed up with SCALE= set empty writes those names, and
    # refusing a forecaster that exists would be a worse failure than either.
    if [ ! -f "${STEERF_HEADS}" ] && [ -f "${ROOT}/checkpoints/mtp_heads_${bb}.pt" ]; then
        export STEERF_HEADS="${ROOT}/checkpoints/mtp_heads_${bb}.pt"
        export STEERF_CALIB="${ROOT}/checkpoints/mtp_calibration_${bb}.json"
    fi

    if ! model_guard; then exit 2; fi

    # Only the STEER-F arm opens the forecaster. GRPO does not, and STEER runs
    # at lambda=0, where run_steerf.sh:76 never loads it.
    if [ "${arm}" = "signed" ] && ! heads_guard; then
        # Loud, because this is the arm the row exists for. Skipping it
        # quietly leaves GRPO and STEER to run for days and produces a table
        # whose decisive column is empty.
        echo "[backbones] ================================================" >&2
        echo "[backbones] SKIP ${rn} -- ${bb} has no forecaster." >&2
        echo "[backbones] This is the STEER-F arm. GRPO and STEER will still" >&2
        echo "[backbones] run and the row will come back WITHOUT its result." >&2
        echo "[backbones] Warm it up first:" >&2
        echo "[backbones]   MODEL_PATH=${MODEL_PATH} bash run/collect_warmup_rollouts.sh" >&2
        echo "[backbones]   MODEL_PATH=${MODEL_PATH} bash run/warmup_and_validate.sh" >&2
        echo "[backbones] ================================================" >&2
        SKIPPED_SIGNED="${SKIPPED_SIGNED:-} ${bb}"
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

    launch_backbone "${arm}" "${rn}" "${log}" "${val_over[@]}"
    st=$?
    printf '[backbones] %s %s exit %s after %s min\n' \
        "${bb}" "${arm}" "${st}" "$(( ($(date +%s) - start) / 60 ))"

    if [ "${st}" -ne 0 ]; then
        diagnose_run_failure "${log}" "${bb} ${arm}" "${STEPS}"
        why=$?
        if [ "${why}" = "2" ] && [ "${OOM_RETRY:-1}" = "1" ]; then
            echo "[backbones] retrying ${rn} with OFFLOAD=1 after the CUDA OOM."
            echo "[backbones] NOTE: OFFLOAD runs the AdamW update on CPU fp32, so"
            echo "[backbones]       this arm is not bit-identical to one that did not."
            ray stop --force >/dev/null 2>&1 || true
            sleep 5
            await_gpus || true
            start=$(date +%s)
            OFFLOAD=1 launch_backbone "${arm}" "${rn}" "${log}" "${val_over[@]}"
            st=$?
            printf '[backbones] %s %s retry exit %s after %s min\n' \
                "${bb}" "${arm}" "${st}" "$(( ($(date +%s) - start) / 60 ))"
        fi
    fi
    if [ "${st}" -ne 0 ]; then
        echo "[backbones] FAILED -- keeping the checkpoint, moving on"
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

# A row whose STEER-F arm never ran is not a partial result, it is no result:
# the backbone section exists to say whether STEER-F beats GRPO on another
# family. Say so once more at the end, where the operator actually looks.
if [ -n "${SKIPPED_SIGNED:-}" ]; then
    echo
    echo "[backbones] NO FORECASTER, so no STEER-F arm, for:${SKIPPED_SIGNED}"
    echo "[backbones] Those rows cannot answer the question they are for."
    echo "[backbones] Heads are looked for at checkpoints/mtp_heads_<tag>${SCALE-paper:+-${SCALE-paper}}.pt"
fi
