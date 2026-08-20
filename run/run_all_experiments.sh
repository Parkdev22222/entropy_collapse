#!/usr/bin/env bash
# One-shot driver for the full paper-parity experiment suite.
#
# Runs every experiment this repo can add a STEER-F row for (see STEERF.md's
# table mapping) as a sequential queue on one 8-GPU node:
#
#   Table 3   math, Qwen2.5-Math-7B            2 seeds
#   Table 12  math, Qwen2.5-Math-1.5B          2 seeds
#   Table 4   math, Qwen2.5-14B                2 seeds
#   Fig 20a   math, Llama-3.2-3B-Instruct      2 seeds
#   Table 6   extreme scenario (7B)            1 run
#   App F.3   RLOO / OPO arms (7B)             1 run each
#   Table 5   code-gen, Qwen2.5-Coder-3B/7B/14B  2 seeds  (needs HF access once)
#   Fig 6     Pass@256/512/1024 (7B, seed 1)   opt-in: RUN_PASSK=1
#
# Every stage is resumable: a stage that finished leaves a marker in
# ${STATE_DIR} and is skipped on re-run; a failed stage is recorded, its
# dependents are skipped, and the queue moves on. Just re-run this script
# after fixing whatever broke. Delete a stage's .done file to force a redo.
#
# Toggles (env):
#   MATH_MODELS   default "Qwen/Qwen2.5-Math-7B Qwen/Qwen2.5-Math-1.5B Qwen/Qwen2.5-14B meta-llama/Llama-3.2-3B-Instruct"
#   CODE_MODELS   default "Qwen/Qwen2.5-Coder-3B Qwen/Qwen2.5-Coder-7B Qwen/Qwen2.5-Coder-14B"
#   SEEDS         default "1 2"      (paper: main tables average two runs)
#   RUN_MATH / RUN_CODE / RUN_EXTREME / RUN_RL_ALGOS   default 1
#   RUN_PASSK     default 0          (2 x 30 problems x 1024 samples — expensive)
#   STEERF_LAM / STEERF_APPLY / STEERF_MAPPING ...     forwarded to run_steerf.sh
#
# Rough budget: a 7B math run is ~200 steps on 8xH20; the full default queue
# is 8 math + 3 aux + 6 code trainings. Prune via the toggles if you don't
# have the node-weeks.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${STEER_ROOT}"
export PYTHONPATH="${STEER_ROOT}:${PYTHONPATH:-}"

MATH_MODELS=${MATH_MODELS:-"Qwen/Qwen2.5-Math-7B Qwen/Qwen2.5-Math-1.5B Qwen/Qwen2.5-14B meta-llama/Llama-3.2-3B-Instruct"}
CODE_MODELS=${CODE_MODELS:-"Qwen/Qwen2.5-Coder-3B Qwen/Qwen2.5-Coder-7B Qwen/Qwen2.5-Coder-14B"}
SEEDS=${SEEDS:-"1 2"}
RUN_MATH=${RUN_MATH:-1}
RUN_CODE=${RUN_CODE:-1}
RUN_EXTREME=${RUN_EXTREME:-1}
RUN_RL_ALGOS=${RUN_RL_ALGOS:-1}
RUN_PASSK=${RUN_PASSK:-0}
AUX_MODEL=${AUX_MODEL:-"Qwen/Qwen2.5-Math-7B"}   # extreme / RLOO / OPO / Pass@k arms

STATE_DIR=${STATE_DIR:-${STEER_ROOT}/experiments_state}
LOG_DIR=${LOG_DIR:-${STEER_ROOT}/logs/experiments}
mkdir -p "${STATE_DIR}" "${LOG_DIR}" "${STEER_ROOT}/results"
FAILED=()

# Fail fast on missing deps / wrong GPU count BEFORE queueing anything —
# every check in preflight.py is a failure mode that has actually happened.
# Skip with PREFLIGHT=0 (not recommended).
if [ "${PREFLIGHT:-1}" = "1" ]; then
    python3 "${STEER_ROOT}/scripts/preflight.py" || exit 1
fi

# ---------------------------------------------------------------- stage
# stage <name> <cmd...>   — run once, log to LOG_DIR/<name>.log, resume-safe.
stage () {
    local name="$1"; shift
    if [ -f "${STATE_DIR}/${name}.done" ]; then
        echo "[all] SKIP  ${name} (done)"; return 0
    fi
    if [ -f "${STATE_DIR}/${name}.failed" ]; then rm -f "${STATE_DIR}/${name}.failed"; fi
    echo "[all] START ${name}  ($(date '+%F %T'))"
    if "$@" >"${LOG_DIR}/${name}.log" 2>&1; then
        touch "${STATE_DIR}/${name}.done"
        echo "[all] DONE  ${name}"
        return 0
    else
        touch "${STATE_DIR}/${name}.failed"
        FAILED+=("${name}")
        echo "[all] FAIL  ${name}  -> ${LOG_DIR}/${name}.log (queue continues)"
        return 1
    fi
}

need () {  # need <stage-name> : true iff dependency finished
    [ -f "${STATE_DIR}/$1.done" ]
}

phase1_kappa_gamma () {  # <model_tag> -> exports SEL_KAPPA / SEL_GAMMA
    local tag="$1"
    read -r SEL_KAPPA SEL_GAMMA < <(python3 - "$tag" <<'PY'
import json, sys
d = json.load(open(f"docs/phase1_results_{sys.argv[1]}.json"))
print(d["chosen"]["kappa"], d["chosen"]["gamma_h"])
if d["chosen"]["rho_mean"] < 0.2:
    print(f"[all] WARNING: gate-G1 rho={d['chosen']['rho_mean']:.3f} < 0.2 for "
          f"{sys.argv[1]} — lam>0 runs are not justified by the forecast",
          file=sys.stderr)
PY
    ) || true
    SEL_KAPPA=${SEL_KAPPA:-2}
    SEL_GAMMA=${SEL_GAMMA:-0.7}
}

# ---------------------------------------------------------------- math
train_and_eval () {
    # <arm> <model> <seed> <train_script> [extra hydra overrides...]
    local arm="$1" model="$2" seed="$3" script="$4"; shift 4
    local tag; tag=$(basename "${model}")
    local run_name="${arm}-s${seed}"
    local project="STEER-F"
    case "${script}" in *code*) project="STEER-F-code";; esac

    if ! need "phase1-${tag}"; then
        echo "[all] SKIP  train-${arm}-s${seed} (phase1-${tag} not done)"; return 1
    fi
    phase1_kappa_gamma "${tag}"

    stage "train-${arm}-s${seed}" \
        env MODEL_PATH="${model}" SEED="${seed}" RUN_NAME="${run_name}" \
            STEERF_KAPPA="${SEL_KAPPA}" STEERF_GAMMA_H="${SEL_GAMMA}" \
        bash "${script}" "trainer.logger=['console','tensorboard']" "$@" \
        || return 1

    local metric="val-core/aime_2024_dapo_boxed/acc/mean@32"
    case "${script}" in *code*) metric="val-core/codecontests/acc/mean@4";; esac
    stage "eval-${arm}-s${seed}" \
        bash -c "
            best=\$(python3 scripts/select_best_checkpoint.py \
                       --project '${project}' --run '${run_name}' --metric '${metric}' --quiet) &&
            echo \"[all] selected checkpoint: \$best\" &&
            MODEL_PATH=\"\$best\" ${EVAL_PREFIX:-} bash run/eval_steerf.sh"
}

if [ "${RUN_MATH}" = "1" ]; then
    for model in ${MATH_MODELS}; do
        tag=$(basename "${model}")
        stage "warmup-rollouts-${tag}" env MODEL_PATH="${model}" bash run/collect_warmup_rollouts.sh
        need "warmup-rollouts-${tag}" && \
            stage "phase1-${tag}" env MODEL_PATH="${model}" bash run/warmup_and_validate.sh
        for seed in ${SEEDS}; do
            train_and_eval "math-${tag}" "${model}" "${seed}" run/run_steerf.sh || true
        done
    done
fi

# ---------------------------------------------------------------- aux arms (7B)
AUX_TAG=$(basename "${AUX_MODEL}")
if [ "${RUN_EXTREME}" = "1" ]; then
    train_and_eval "extreme-${AUX_TAG}" "${AUX_MODEL}" 1 run/run_steerf_extreme.sh || true
fi
if [ "${RUN_RL_ALGOS}" = "1" ]; then
    train_and_eval "rloo-${AUX_TAG}" "${AUX_MODEL}" 1 run/run_steerf.sh \
        algorithm.adv_estimator=rloo || true
    train_and_eval "opo-${AUX_TAG}" "${AUX_MODEL}" 1 run/run_steerf.sh \
        algorithm.adv_estimator=opo || true
fi
if [ "${RUN_PASSK}" = "1" ] && need "train-math-${AUX_TAG}-s1"; then
    stage "passk-${AUX_TAG}-s1" \
        bash -c "
            best=\$(python3 scripts/select_best_checkpoint.py \
                       --project STEER-F --run 'math-${AUX_TAG}-s1' --quiet) &&
            MODEL_PATH=\"\$best\" PASSK=1 bash run/eval_steerf.sh"
fi

# ---------------------------------------------------------------- code
if [ "${RUN_CODE}" = "1" ]; then
    stage "prepare-code-data" \
        python3 scripts/prepare_code_data.py --archer --lcb --out "${STEER_ROOT}/datasets"
    if need "prepare-code-data"; then
        for model in ${CODE_MODELS}; do
            tag=$(basename "${model}")
            stage "warmup-rollouts-${tag}" env MODEL_PATH="${model}" \
                TRAIN_PATH="${STEER_ROOT}/datasets/archercoder.parquet" \
                TEST_PATH="${STEER_ROOT}/datasets/livecodebench_v5.parquet" \
                bash run/collect_warmup_rollouts.sh
            need "warmup-rollouts-${tag}" && \
                stage "phase1-${tag}" env MODEL_PATH="${model}" \
                    PROBLEMS="${STEER_ROOT}/datasets/archercoder.parquet" \
                    bash run/warmup_and_validate.sh
            for seed in ${SEEDS}; do
                EVAL_PREFIX="CODE=1" \
                train_and_eval "code-${tag}" "${model}" "${seed}" run/run_steerf_code.sh || true
            done
        done
    fi
fi

# ---------------------------------------------------------------- summary
echo
echo "================ experiment queue summary ================"
for f in "${STATE_DIR}"/*.done;   do [ -e "$f" ] && echo "DONE   $(basename "${f%.done}")";   done
for f in "${STATE_DIR}"/*.failed; do [ -e "$f" ] && echo "FAILED $(basename "${f%.failed}")"; done
python3 scripts/collect_results.py --logs "${LOG_DIR}" --out "${STEER_ROOT}/results/summary.tsv" || true
echo "=========================================================="
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "[all] ${#FAILED[@]} stage(s) failed this pass: ${FAILED[*]}"
    echo "[all] fix and re-run this script — finished stages are skipped."
    exit 1
fi
echo "[all] queue complete. Paper-table numbers: results/summary.tsv"
