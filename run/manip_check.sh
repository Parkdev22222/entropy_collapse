#!/usr/bin/env bash
# Manipulation check for the A_H correction (docs/manipulation_check.md).
#
#   DRY=1 EVAL_REPOS="<repo> [<repo>...]" bash run/manip_check.sh   # where each checkpoint is
#         EVAL_REPOS="<repo> [<repo>...]" bash run/manip_check.sh   # run everything
#
# One GPU. Stages are skipped when their output exists, so a rerun resumes:
#   build  -> results/manip/probe.jsonl   (vLLM, base model)
#   togo   -> results/manip/togo.json     (HF, base model)
#   score  -> results/manip/scores/*.json (HF fp32, base + every checkpoint)
#   analyze-> results/manip_check.json
# A checkpoint that is not on disk is fetched from EVAL_REPOS with the same
# layout run_eval_all.sh reads, scored, and deleted again to keep the disk free.
#
# Amendment 1 (docs/manipulation_check.md): the registered checkpoints do not
# exist, so the checkpoints can instead be named explicitly, one label=dir each:
#   CKPTS="signed:1=/path/a permuted:1=/path/b grpo:1=/path/c" bash run/manip_check.sh
# When CKPTS is set, SEEDS/ARMS/EVAL_REPOS are not used.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
# shellcheck source=run/_arms.sh
source run/_arms.sh

BASE=${BASE_MODEL:-Qwen/Qwen2.5-Math-1.5B}
DATA=${DATA:-datasets/DAPO-Math-17k.parquet}
PROBE=${PROBE_DIR:-results/manip}
OUT=${OUT:-results/manip_check.json}
SEEDS=${SEEDS:-"1 3 4"}
ARMS=${ARMS:-"signed uniform permuted steer grpo"}
CKPT_ROOT=${CKPT_ROOT:-"${ROOT}/checkpoints/STEER-F"}
STAGE=${STAGE:-"${ROOT}/checkpoints/_manip_stage"}
EVAL_REPOS=${EVAL_REPOS:-}
N_PROMPTS=${N_PROMPTS:-512}
CKPTS=${CKPTS:-}
DRY=${DRY:-0}
PY="python3 scripts/manip_check.py"

# A score file counts as done only if it was made from THIS checkpoint. Keyed
# on the label alone, a rerun that pointed permuted:1 at step 100 kept the
# step-110 scores of an earlier invocation and compared them with step 100.
scored () {   # <score-file> <model-dir>
    [ -s "$1" ] && python3 -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["model"] != sys.argv[2])' "$1" "$2"
}

local_ckpt () {   # <arm> <seed> -> HF dir with weights, or nothing
    local rn cand d best=""
    rn="$(run_name_for "$1" "$2")"
    for cand in "${rn}" $(ckpt_alias_for "$1" "$2"); do
        for d in $(ls -d "${CKPT_ROOT}/${cand}"/global_step_* 2>/dev/null |
                   sed 's/.*global_step_//' | sort -n); do
            hf_weights_present "${CKPT_ROOT}/${cand}/global_step_${d}/actor/huggingface" \
                && best="${CKPT_ROOT}/${cand}/global_step_${d}/actor/huggingface"
        done
        [ -n "${best}" ] && { echo "${best}"; return 0; }
    done
    return 0
}

echo "base=${BASE}  probe=${PROBE}  seeds=${SEEDS}  arms=${ARMS}"
echo "checkpoints:"
missing=0
if [ -n "${CKPTS}" ]; then
    for e in ${CKPTS}; do
        label="${e%%=*}"; p="${e#*=}"
        if hf_weights_present "${p}"; then
            printf '  %-12s %s\n' "${label}" "${p}"
        else
            printf '  %-12s MISSING (no weights in %s)\n' "${label}" "${p}"
            missing=$((missing + 1))
        fi
    done
    SEEDS=""   # the explicit list replaces the arm x seed loops below
fi
for seed in ${SEEDS}; do
    for arm in ${ARMS}; do
        rn="$(run_name_for "${arm}" "${seed}")"
        p="$(local_ckpt "${arm}" "${seed}")"
        if [ -z "${p}" ] && [ -n "${EVAL_REPOS}" ]; then
            p="$(${PY} fetch --repos "${EVAL_REPOS}" --run "${rn}" --dest "${STAGE}" --dry || true)"
        fi
        printf '  %-9s s%s  %s\n' "${arm}" "${seed}" "${p:-MISSING}"
        [ -n "${p}" ] || missing=$((missing + 1))
    done
done
[ "${missing}" -eq 0 ] || echo "WARNING: ${missing} checkpoint(s) missing; their contrasts will be absent."
if [ "${DRY}" = "1" ]; then exit 0; fi

mkdir -p "${PROBE}/scores"
[ -s "${PROBE}/probe.jsonl" ] || ${PY} build --base "${BASE}" --data "${DATA}" \
    --probe "${PROBE}" --n-prompts "${N_PROMPTS}"
[ -s "${PROBE}/togo.json" ] || ${PY} togo --base "${BASE}" --probe "${PROBE}"
[ -s "${PROBE}/scores/base.json" ] || ${PY} score --model "${BASE}" --label base --probe "${PROBE}"

for e in ${CKPTS}; do
    label="${e%%=*}"; p="${e#*=}"
    if scored "${PROBE}/scores/${label/:/-s}.json" "${p}"; then continue; fi
    if ! hf_weights_present "${p}"; then
        echo "[manip] MISSING ${label} (${p}) -- skipped"; continue
    fi
    ${PY} score --model "${p}" --label "${label}" --probe "${PROBE}"
done

for seed in ${SEEDS}; do
    for arm in ${ARMS}; do
        if [ -s "${PROBE}/scores/${arm}-s${seed}.json" ]; then continue; fi
        rn="$(run_name_for "${arm}" "${seed}")"
        p="$(local_ckpt "${arm}" "${seed}")"
        fetched=""
        if [ -z "${p}" ] && [ -n "${EVAL_REPOS}" ]; then
            p="$(${PY} fetch --repos "${EVAL_REPOS}" --run "${rn}" --dest "${STAGE}" | tail -1 || true)"
            fetched="${p}"
        fi
        if [ -z "${p}" ] || ! hf_weights_present "${p}"; then
            echo "[manip] MISSING ${arm} s${seed} (${rn}) -- skipped"
            if [ -n "${fetched}" ]; then rm -rf "${STAGE}"; fi
            continue
        fi
        ${PY} score --model "${p}" --label "${arm}:${seed}" --probe "${PROBE}"
        if [ -n "${fetched}" ]; then rm -rf "${STAGE}"; fi
    done
done

${PY} analyze --probe "${PROBE}" --out "${OUT}"
