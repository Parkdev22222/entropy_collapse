#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Six-benchmark evaluation of one LoRA run at one checkpoint.
#
#   bash run/eval_lora.sh <arm> <seed> final     # the primary endpoint
#   bash run/eval_lora.sh <arm> <seed> best      # the secondary one
#   DRY=1 bash run/eval_lora.sh signed 1 final   # resolve and print, run nothing
#
# POINTS (fixed before any LoRA result existed; docs/preregistration_lora.md)
#   final  global_step_<last step of the arm>: no selection, so AIME24 is as
#          held-out as every other benchmark.
#   best   the step named in best_checkpoint_info.json -- selected on AIME24,
#          so AIME24 is in-sample there and the paper says so.
#
# WHAT IT DOES
#   1. resolves <run>/global_step_N/actor/lora_adapter (never "the highest
#      step": a LoRA run keeps 50, 100, the last and the best);
#   2. merges it into the base model (scripts/merge_lora.py, which refuses an
#      adapter that did not load or was never trained);
#   3. runs run/eval_steerf.sh on the merged model, both passes at
#      LORA_EVAL_N samples per problem (32);
#   4. reads the model path back out of the log and runs
#      scripts/check_eval_logs.py on it; either failing fails the eval.
# Logs: logs/lora/eval-<point>-k<N>/eval-<arm>-s<seed>.log, the name
# scripts/collect_results.py reads. Per-problem dumps:
# validation_data/lora/eval-<point>-k<N>/<arm>-s<seed>/.
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}" || exit 1
# shellcheck source=run/_arms.sh
. run/_arms.sh
# shellcheck source=run/_lora_arms.sh
. run/_lora_arms.sh

ARM=${1:?usage: eval_lora.sh <arm> <seed> <final|best>}
SEED=${2:?usage: eval_lora.sh <arm> <seed> <final|best>}
POINT=${3:?usage: eval_lora.sh <arm> <seed> <final|best>}
N=${LORA_EVAL_N:-32}
DRY=${DRY:-0}
FORCE=${FORCE:-0}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-Math-1.5B}
CKPT_ROOT=${CKPT_ROOT:-${ROOT}/checkpoints/STEER-F}
LOG_ROOT=${LOG_ROOT:-${ROOT}/logs/lora}

RUN="$(lora_run_name "${ARM}" "${SEED}")" || { echo "FATAL: unknown arm '${ARM}'" >&2; exit 2; }
RUN_DIR="${CKPT_ROOT}/${RUN}"
case "${POINT}" in
    final) STEP="$(steps_for_arm "${ARM}")" ;;
    best)
        INFO="${RUN_DIR}/best_checkpoint_info.json"
        [ -f "${INFO}" ] || { echo "FAIL: ${INFO} missing -- was the run trained with KEEP_EVERY?" >&2; exit 1; }
        STEP="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint_step"])' "${INFO}")"
        ;;
    *) echo "FATAL: point must be final or best, got '${POINT}'" >&2; exit 2 ;;
esac
case "${STEP}" in ''|None|*[!0-9]*) echo "FAIL: no step for ${RUN} at '${POINT}' (got '${STEP}')" >&2; exit 1 ;; esac

ADAPTER="${RUN_DIR}/global_step_${STEP}/actor/lora_adapter"
MERGED="${RUN_DIR}/global_step_${STEP}/actor/merged"
EVAL_DIR="${LOG_ROOT}/eval-${POINT}-k${N}"
LOG="${EVAL_DIR}/eval-${ARM}-s${SEED}.log"
VAL_ROOT=${VAL_ROOT:-${ROOT}/validation_data/lora}
VAL_DIR="${VAL_ROOT}/eval-${POINT}-k${N}/${ARM}-s${SEED}"

eval_done () {   # both passes' headline keys are in the log, and it is coherent
    [ -f "${LOG}" ] || return 1
    grep -q "val-core/aime_2024_dapo_boxed/acc/mean@32" "${LOG}" || return 1
    grep -q "val-core/math500/acc/mean@${N}" "${LOG}" || return 1
    python3 scripts/check_eval_logs.py "${LOG}" >/dev/null 2>&1
}

echo "[eval-lora] ${ARM} s${SEED} ${POINT} -> ${RUN} step ${STEP}"
echo "[eval-lora]   adapter ${ADAPTER}"
echo "[eval-lora]   log     ${LOG}"
if [ "${FORCE}" != 1 ] && eval_done; then
    echo "[eval-lora]   done already (FORCE=1 re-runs)"
    exit 0
fi
[ -f "${ADAPTER}/adapter_model.safetensors" ] || { echo "FAIL: no adapter at ${ADAPTER}" >&2; exit 1; }
[ "${DRY}" = 1 ] && exit 0

# The best is often the last step itself. Then the two points are one
# checkpoint, and three hours of sampling would measure it twice; the best's
# log is the final's, marked as such. (Sampling is not seeded per eval, so two
# evals of one checkpoint would even disagree slightly -- one record is right.)
if [ "${POINT}" = best ] && [ "${STEP}" = "$(steps_for_arm "${ARM}")" ]; then
    FINAL_LOG="${LOG_ROOT}/eval-final-k${N}/eval-${ARM}-s${SEED}.log"
    if [ -f "${FINAL_LOG}" ] && grep -q "val-core/math500/acc/mean@${N}" "${FINAL_LOG}" \
            && python3 scripts/check_eval_logs.py "${FINAL_LOG}" >/dev/null 2>&1; then
        mkdir -p "${EVAL_DIR}"
        { echo "### eval_lora.sh: best = final (step ${STEP}); copied from ${FINAL_LOG}"
          cat "${FINAL_LOG}"; } > "${LOG}"
        if [ -d "${VAL_ROOT}/eval-final-k${N}/${ARM}-s${SEED}" ]; then
            mkdir -p "$(dirname "${VAL_DIR}")"
            rm -rf "${VAL_DIR}"
            cp -r "${VAL_ROOT}/eval-final-k${N}/${ARM}-s${SEED}" "${VAL_DIR}"
        fi
        echo "[eval-lora]   best is the final step; reused ${FINAL_LOG}"
        exit 0
    fi
fi

python3 scripts/merge_lora.py --base "${BASE_MODEL}" --adapter "${ADAPTER}" --out "${MERGED}" \
    || { echo "FAIL: merge of ${ADAPTER}" >&2; exit 1; }

mkdir -p "${EVAL_DIR}" "${VAL_DIR}"
{
    echo "### eval_lora.sh arm=${ARM} seed=${SEED} point=${POINT} step=${STEP} n=${N}"
    echo "### run=${RUN} adapter=${ADAPTER} merged=${MERGED} base=${BASE_MODEL}"
} > "${LOG}"
MODEL_PATH="${MERGED}" AVG1_N="${N}" VAL_DATA_DIR="${VAL_DIR}" \
    bash run/eval_steerf.sh >> "${LOG}" 2>&1
st=$?

# The first six-benchmark attempt scored one checkpoint five times over; the
# path the trainer actually loaded is the only proof of which one this is.
got="$(grep -oE "actor_rollout_ref\.model\.path=[^ ]+" "${LOG}" | sort -u)"
if [ "${got}" != "actor_rollout_ref.model.path=${MERGED}" ]; then
    echo "FAIL: ${LOG} loaded '${got:-nothing}', expected ${MERGED}" >&2
    exit 1
fi
if [ "${st}" -ne 0 ] || ! eval_done; then
    echo "FAIL: ${LOG} (exit ${st}); see scripts/check_eval_logs.py ${LOG}" >&2
    exit 1
fi
echo "[eval-lora]   OK"
