#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Every experiment in the paper, from one command.
#
#   REPO=DSDSh/steer-f_2 bash run/run_paper.sh          # the whole thing
#   DRY=1 bash run/run_paper.sh                         # print the plan only
#   STAGES=analysis bash run/run_paper.sh               # re-run one stage
#   STAGES="measure analysis" bash run/run_paper.sh     # a subset, in order
#
# Arm it and walk away:
#   tmux new -d -s paper "cd /workspace/entropy_collapse && \
#     REPO=DSDSh/steer-f_2 bash run/run_paper.sh > logs/experiments/paper.log 2>&1"
#
# Every stage is resumable: each queue skips what is already finished, so
# killing this and re-arming it costs nothing but the run in flight.
#
# STAGES, in execution order
#   preflight  environment, instrumentation, disk. Refuses rather than burning
#              a queue on a box that cannot import verl.
#   recover    seed 1's steer and uniform, whose checkpoints were deleted.
#              signed/grpo/permuted at seed 1 survive and are skipped.
#   campaign   seeds 2-5 x 5 arms = 20 runs. The main table.
#   measure    the GPU-cheap measurements that answer reviewers rather than
#              reviewers' questions about the table. Runs AFTER training so it
#              never competes for the GPUs; none of it takes more than ~2 h.
#   eval       the 25 finished runs on six benchmarks.
#   followups  the 10 ablations at seed 1, grpo-long (the compute-matched
#              control) included.
#   eval2      those 10 on six benchmarks.
#   analysis   every table the paper prints, into results/.
#
# WHAT THE MEASURE STAGE IS FOR
#   Three of the four scripts answer an objection that the accuracy table
#   cannot:
#     locality  "your future term has an effective horizon of one step, so it
#               is a local signal in disguise" -- the sharpest attack on the
#               paper. ~1 GPU-h.
#     omega     "is |Omega| really small exactly where rollouts fork?" Turns
#               manuscript Table 1 from an analytic prediction into a
#               measurement on the model's own rollouts.
#     recall    gate G1's branch recall, re-run at the kappa=2 / gamma_H=0.7
#               the training runs actually used. The committed JSON was
#               measured at kappa=4 / gamma_H=0.85 and cannot be quoted.
#     support   the combinatorial ceiling 2(n-1)/T against the measured
#               branch_corr_frac. CPU only.
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}" || { echo "FATAL: cannot cd to ${ROOT}" >&2; exit 1; }
# shellcheck source=run/_arms.sh
. "${ROOT}/run/_arms.sh"

DRY=${DRY:-0}
REPO=${REPO:-}
SEEDS=${SEEDS:-"2 3 4 5"}
STEPS=${STEPS:-110}
LONG_STEPS=${LONG_STEPS:-200}
FOLLOWUP_ARMS=${FOLLOWUP_ARMS:-}   # empty = every arm run_followups.sh defines
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-Math-1.5B}
KAPPA=${STEERF_KAPPA:-2}
GAMMA_H=${STEERF_GAMMA_H:-0.7}
STAGES=${STAGES:-"preflight recover campaign measure eval followups eval2 analysis"}

LOG_DIR="${ROOT}/logs/experiments"
DOC_DIR="${ROOT}/docs"
RES_DIR="${ROOT}/results"
HEADS="${ROOT}/checkpoints/mtp_heads_${MODEL_TAG}-paper.pt"
CALIB="${ROOT}/checkpoints/mtp_calibration_${MODEL_TAG}-paper.json"
ROLLOUTS=${ROLLOUTS:-${ROOT}/rollout_data/warmup/${MODEL_TAG}-paper/rollouts.jsonl}

mkdir -p "${LOG_DIR}" "${DOC_DIR}" "${RES_DIR}"

stage () { printf '\n################################################################\n# %s  %s\n################################################################\n' "$(date -Is)" "$*"; }
note ()  { printf '[paper] %s\n' "$*"; }
have ()  { case " ${STAGES} " in *" $1 "*) return 0 ;; esac; return 1; }

# A stage that fails does not stop the rest: the analysis of what DID finish is
# more useful than an empty results/ directory. Failures are collected and
# printed at the end, and the exit status reflects them.
declare -a FAILED=()
guard () {   # <label> <command...>
    local label="$1"; shift
    if [ "${DRY}" = "1" ]; then printf '  would run: %s\n' "$*"; return 0; fi
    "$@" || { FAILED+=("${label}"); note "STAGE FAILED: ${label} (continuing)"; return 1; }
}

# Same, for a command whose output IS the artefact. Redirecting the guard call
# itself would send the DRY-mode "would run" line into the artefact file.
guard_to () {   # <file> <label> <command...>
    local out="$1" label="$2"; shift 2
    if [ "${DRY}" = "1" ]; then printf '  would run: %s > %s\n' "$*" "${out}"; return 0; fi
    "$@" > "${out}" || { FAILED+=("${label}"); note "STAGE FAILED: ${label} (continuing)"; return 1; }
}

# ============================================================= 0. preflight
if have preflight; then
    stage "preflight"
    if [ "${DRY}" != "1" ]; then
        if ! env_preflight "${ROOT}"; then
            echo "REFUSE: the training stack does not import. Nothing below would run." >&2
            exit 2
        fi
        note "verl + steer_f import"
        bash run/instrument_campaign.sh --apply || {
            echo "REFUSE: instrument_campaign.sh --apply failed" >&2; exit 2; }
        bash run/instrument_phase2.sh --apply || {
            echo "REFUSE: instrument_phase2.sh --apply failed" >&2; exit 2; }
        # Only the measure stage reads these. The training arms use the heads
        # without the -paper suffix (run_steerf.sh's own default), so refusing
        # here on a box that has never run the paper warm-up would block a
        # 31-day campaign on two files it never opens. Fatal when measure is
        # actually queued, a warning otherwise.
        missing=()
        for f in "${HEADS}" "${CALIB}"; do
            [ -f "${f}" ] || missing+=("${f}")
        done
        if [ "${#missing[@]}" -eq 0 ]; then
            note "MTP heads and calibration present"
        elif have measure; then
            printf 'REFUSE: measure is queued but these are missing:\n' >&2
            printf '  %s\n' "${missing[@]}" >&2
            printf 'Run run/warmup_and_validate.sh, or drop measure from STAGES.\n' >&2
            exit 2
        else
            note "measure not queued; missing (not needed by the other stages):"
            printf '[paper]   %s\n' "${missing[@]}"
        fi
        avail="$(df -BG --output=avail "${ROOT}" 2>/dev/null | tail -1 | tr -dc '0-9')"
        note "${avail:-?} GB free"
        [ -n "${REPO}" ] || note "REPO unset: checkpoints stay local (~3.1 GB per finished run)"
    else
        echo "  would run: env_preflight, instrument_campaign --apply, instrument_phase2 --apply"
    fi
fi

# ============================================================== 1. recover
# Seed 1's steer and uniform checkpoints were deleted; their logs exist but the
# weights do not, and the six-benchmark table needs the weights. run_0905_chain
# skips the arms whose _0905 log already reached the final step, so re-running
# it does exactly the two that are missing.
if have recover; then
    stage "recover seed 1 (steer + uniform)"
    guard recover bash run/run_0905_chain.sh
fi

# ============================================================= 2. campaign
if have campaign; then
    stage "campaign: seeds ${SEEDS} x 5 arms"
    guard campaign env SEEDS="${SEEDS}" STEPS="${STEPS}" REPO="${REPO}" WAIT=1 \
        bash run/run_campaign.sh
fi

# ============================================================== 3. measure
if have measure; then
    stage "measurements (reviewer defences)"

    # CPU only -- the combinatorial ceiling against the measured branch fraction.
    guard_to "${DOC_DIR}/ah_support.txt" measure-support \
        python3 scripts/measure_ah_support.py

    # Is |Omega| small exactly at the forks? Generates its own rollouts.
    guard measure-omega python3 scripts/measure_omega_at_branches.py \
        --model "${MODEL_PATH}" --out "${DOC_DIR}/omega_at_branches.json"

    if [ -f "${ROLLOUTS}" ]; then
        # THE one that answers "your future term is a local signal".
        guard measure-locality python3 scripts/measure_forecast_locality.py \
            --model "${MODEL_PATH}" --heads "${HEADS}" --calib "${CALIB}" \
            --rollouts "${ROLLOUTS}" --kappa "${KAPPA}" --gamma-h "${GAMMA_H}" \
            --out "${DOC_DIR}/forecast_locality.json"

        # G1 recall at the kappa/gamma_H the runs actually used, plus the
        # untrained control -- recall without its chance floor says nothing.
        guard measure-recall python3 scripts/phase1_branch_recall.py \
            --model "${MODEL_PATH}" --heads "${HEADS}" --calib "${CALIB}" \
            --rollouts "${ROLLOUTS}" --kappa "${KAPPA}" --gamma-h "${GAMMA_H}" \
            --out "${DOC_DIR}/phase1_recall_k${KAPPA}_g${GAMMA_H}.json"
        CONTROL="${ROOT}/checkpoints/mtp_heads_control_${MODEL_TAG}.pt"
        if [ -f "${CONTROL}" ]; then
            guard measure-recall-control python3 scripts/phase1_branch_recall.py \
                --model "${MODEL_PATH}" --heads "${CONTROL}" --allow-untrained \
                --rollouts "${ROLLOUTS}" --kappa "${KAPPA}" --gamma-h "${GAMMA_H}" \
                --out "${DOC_DIR}/phase1_recall_control_k${KAPPA}_g${GAMMA_H}.json"
        else
            note "no untrained control heads at ${CONTROL}; recall has no chance floor"
        fi
    else
        note "SKIP locality + recall: no rollouts at ${ROLLOUTS}"
        note "  make them with: bash run/collect_warmup_rollouts.sh"
        FAILED+=("measure-locality:no-rollouts")
    fi
fi

# ================================================================= 4. eval
if have eval; then
    stage "six-benchmark evaluation of the campaign runs"
    guard eval env REPO="${REPO}" SEEDS="1 ${SEEDS}" bash run/run_eval_all.sh
fi

# ============================================================ 5. followups
if have followups; then
    stage "follow-up ablations (seed 1, grpo-long included)"
    # FOLLOWUP_ARMS lets one box take a subset. Splitting the work across two
    # pods needs this: grpo-long has to stay wherever its STEER-F reference
    # trained, because analyze_seeds.py integrates its per-step wall clock and
    # compares it against what STEER-F spent -- seconds measured on a different
    # box are not the same unit.
    if [ -n "${FOLLOWUP_ARMS}" ]; then
        note "followup arms: ${FOLLOWUP_ARMS}"
        guard followups env REPO="${REPO}" WAIT=1 LONG_STEPS="${LONG_STEPS}" \
            ARMS="${FOLLOWUP_ARMS}" bash run/run_followups.sh
    else
        guard followups env REPO="${REPO}" WAIT=1 LONG_STEPS="${LONG_STEPS}" \
            bash run/run_followups.sh
    fi
fi

if have eval2; then
    stage "six-benchmark evaluation of the follow-ups"
    # run_eval_all.sh names this variable too, and defaults it to all ten arms.
    # On a box that trained a subset, letting it default would queue evals for
    # runs that trained on the other pod.
    if [ -n "${FOLLOWUP_ARMS}" ]; then
        guard eval2 env REPO="${REPO}" EVAL_SET=followups \
            FOLLOWUP_ARMS="${FOLLOWUP_ARMS}" bash run/run_eval_all.sh
    else
        guard eval2 env REPO="${REPO}" EVAL_SET=followups bash run/run_eval_all.sh
    fi
fi

# ============================================================= 6. analysis
if have analysis; then
    stage "analysis"
    guard analysis-seeds python3 scripts/analyze_seeds.py \
        --logs "${LOG_DIR}" --out "${RES_DIR}" --steps "${STEPS}"
    guard analysis-bench python3 scripts/collect_results.py \
        --logs "${LOG_DIR}" --out "${RES_DIR}/benchmarks.tsv"
fi

# ================================================================ summary
stage "done"
if [ "${DRY}" = "1" ]; then
    echo "(DRY=1: nothing ran)"
    exit 0
fi
echo "Tables:"
for f in per_seed arm_means contrasts compute_match benchmarks; do
    p="${RES_DIR}/${f}.tsv"
    [ -f "${p}" ] && printf '  %-28s %s rows\n' "${p##*/}" "$(( $(wc -l < "${p}") - 1 ))"
done
[ -f "${RES_DIR}/tables.tex" ] && echo "LaTeX bodies:  ${RES_DIR}/tables.tex"
echo "Measurements:"
for f in ah_support.txt omega_at_branches.json forecast_locality.json \
         "phase1_recall_k${KAPPA}_g${GAMMA_H}.json"; do
    [ -f "${DOC_DIR}/${f}" ] && printf '  %s\n' "${DOC_DIR}/${f}"
done
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo
    echo "${#FAILED[@]} step(s) did not complete:"
    printf '  %s\n' "${FAILED[@]}"
    echo "Re-run just those with STAGES=..., everything finished is skipped."
    exit 1
fi
echo
echo "Everything completed."
