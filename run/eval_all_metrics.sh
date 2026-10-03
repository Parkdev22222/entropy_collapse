#!/usr/bin/env bash
# Every metric STEER reports for a math checkpoint, from one command.
#
#   RUN_NAME=steerf-r2-1p5b-s1-200 POINT=best  ARM=r2 SEED=1 bash run/eval_all_metrics.sh
#   RUN_NAME=steerf-r2-1p5b-s1-200 POINT=final ARM=r2 SEED=1 bash run/eval_all_metrics.sh
#   MODEL_PATH=Qwen/Qwen2.5-Math-1.5B POINT=base ARM=base-1p5b SEED=0 bash run/eval_all_metrics.sh
#
# What it produces, under results/steer_metrics/<POINT>/ (one table per point,
# one row per ARM/SEED, a seed-averaged row once a second seed lands):
#   bench.tsv            Tables 3/12: AIME24/AIME25/AMC23 avg@32, MATH500/
#                        Minerva/Olympiad avg@1, Avg(math6)   (+ GSM8K, extra)
#   passk_verl.tsv       Figure 6: verl best@{256,512,1024}/mean on AIME24/25
#   passk_unbiased.tsv   the same samples as unbiased pass@k, k = 1..1024
# and results/steer_metrics/curves/<ARM>-s<SEED>.tsv when the run's training
# log is found: Figure 7 (AIME24 avg@32 during training) and Figure 8
# (actor/entropy), with reward and response length beside them.
#
# Which checkpoint. POINT=best reads best_checkpoint_info.json -- the paper's
# rule, best AIME24 on the 10-step validation grid. POINT=final is the run's
# last step (STEPS, default 200). POINT=base takes MODEL_PATH as given.
# Pass@k is EXPENSIVE (2 x 30 problems x 1024 samples) and runs by default
# only on best and base, the two points the paper's figure is about; set
# PASSK=1/0 to override.
#
# Each pass is skipped when its log already holds the result; FORCE=1 redoes it.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${STEER_ROOT}"

POINT=${POINT:?set POINT to best, final or base}
ARM=${ARM:?set ARM, the row label (e.g. r2, base-1p5b)}
SEED=${SEED:?set SEED (0 for a base model)}
STEPS=${STEPS:-200}
case "${POINT}" in best|base) PASSK=${PASSK:-1} ;; final) PASSK=${PASSK:-0} ;;
    *) echo "POINT must be best, final or base, got '${POINT}'" >&2; exit 2 ;; esac

weights_in () { [ -n "$(find "$1" -maxdepth 1 -type f \( -name '*.safetensors' \
                 -o -name 'pytorch_model*.bin' \) -print -quit 2>/dev/null)" ]; }

if [ "${POINT}" != base ]; then
    RUN_NAME=${RUN_NAME:?set RUN_NAME, the training run whose checkpoint to evaluate}
    ckroot=${CKPT_ROOT:-checkpoints/STEER-F}/${RUN_NAME}
    if [ "${POINT}" = best ]; then
        step=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['best_checkpoint_step'])" \
               "${ckroot}/best_checkpoint_info.json") \
            || { echo "REFUSE: no readable ${ckroot}/best_checkpoint_info.json" >&2; exit 2; }
    else
        step=${STEPS}
    fi
    MODEL_PATH=${ckroot}/global_step_${step}/actor/huggingface
    weights_in "${MODEL_PATH}" || { echo "REFUSE: no model weights in ${MODEL_PATH}" >&2; exit 2; }
    TRAIN_LOG=${TRAIN_LOG:-logs/experiments/train-${RUN_NAME}.log}
else
    MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the base model}
    step=0
    TRAIN_LOG=${TRAIN_LOG:-}
fi
export MODEL_PATH

out=results/steer_metrics/${POINT}
val=validation_data/steer_metrics/${POINT}/${ARM}-s${SEED}
name=eval-${ARM}-s${SEED}.log
mkdir -p "${out}/logs" "${out}/logs-passk" "${val}"
echo "[metrics] ${ARM} s${SEED} ${POINT} step ${step}: ${MODEL_PATH}"
echo "[metrics] passk=${PASSK}  out=${out}  dumps=${val}"

fail=0
# ---- Tables 3/12: the six benchmarks ----------------------------------------
log="${out}/logs/${name}"
if [ "${FORCE:-0}" != 1 ] && grep -q 'val-core/aime_2024_dapo_boxed/acc/mean@32' "${log}" 2>/dev/null \
        && grep -q 'val-core/math500/acc/mean@' "${log}" 2>/dev/null; then
    echo "[metrics] six-benchmark log present, skipping: ${log}"
else
    VAL_DATA_DIR="${val}" bash run/eval_steerf.sh > "${log}" 2>&1
fi
python3 scripts/check_eval_logs.py "${log}" || fail=1
python3 scripts/collect_results.py --logs "${out}/logs" --out "${out}/bench.tsv" || fail=1

# ---- Figure 6: Pass@k ---------------------------------------------------------
if [ "${PASSK}" = 1 ]; then
    plog="${out}/logs-passk/${name}"
    if [ "${FORCE:-0}" != 1 ] && grep -q 'val-core/aime_2025_dapo_boxed/acc/best@1024/mean' "${plog}" 2>/dev/null; then
        echo "[metrics] Pass@k log present, skipping: ${plog}"
    else
        PASSK=1 VAL_DATA_DIR="${val}" bash run/eval_steerf.sh > "${plog}" 2>&1
    fi
    python3 scripts/collect_results.py --passk --logs "${out}/logs-passk" --out "${out}/passk_verl.tsv" || fail=1
    python3 scripts/passk_from_dump.py "${val}/passk" --label "${ARM}-s${SEED}" \
        --out "${out}/passk_unbiased.tsv" || fail=1
fi

# ---- Figures 7/8: training curves ----------------------------------------------
if [ -n "${TRAIN_LOG}" ]; then
    if [ -f "${TRAIN_LOG}" ]; then
        python3 scripts/export_curves.py "${TRAIN_LOG}" \
            --out "results/steer_metrics/curves/${ARM}-s${SEED}.tsv" || fail=1
    else
        echo "[metrics] no training log at ${TRAIN_LOG}; curves skipped (set TRAIN_LOG)"
    fi
fi

[ "${fail}" = 0 ] && echo "[metrics] done: ${ARM} s${SEED} ${POINT}" \
                  || echo "[metrics] FINISHED WITH ERRORS: ${ARM} s${SEED} ${POINT} -- read the lines above"
exit "${fail}"
