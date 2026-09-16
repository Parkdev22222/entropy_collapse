#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The nine follow-up ablations, as one resumable queue. Seed 1 only.
#
#   bash run/run_followups.sh                       # all nine
#   DRY=1 bash run/run_followups.sh                 # print the queue
#   ARMS="lam0.1 lam0.5" bash run/run_followups.sh  # a subset
#   REPO=user/repo bash run/run_followups.sh        # upload + delete each ckpt
#   WAIT=1 bash run/run_followups.sh                # queue behind a running job
#
# These are ablations, not headline arms, so they get ONE seed each. The paper
# must say so: every contrast drawn from this table is n=1 and describes the
# mechanism, not the effect size.
#
#   lam0.1 / lam0.5   lambda sweep around the 0.25 the campaign uses. Answers
#                     "is 0.25 a tuned number or an arbitrary one".
#   lam0-tree         THE MISSING CELL. Tree rollout with the future channel
#                     switched off entirely. No existing arm isolates the
#                     sampler: uniform and permuted both run lambda=.25, and
#                     tree rollout changes the GRPO advantage estimate itself
#                     by making a group share prefixes. Without this cell the
#                     answer to "isn't tree rollout just a better sampler?" is
#                     an argument rather than a measurement.
#   xclip-*           The paper's extreme entropy-control setting
#                     (clip_ratio_low=0.99, clip_ratio_high=5): ratio clipping
#                     all but removed, which is where entropy interventions
#                     are supposed to matter most.
#   rloo-* / opo-*    RL-algorithm generalization. The sibling baseline depends
#                     only on group_size = rollout.n, which RLOO and OPO share,
#                     so the method transfers unchanged.
#
# WHY NOT run/run_steerf_extreme.sh
#   It exists, but it is a stale fork of run_steerf.sh: log_prob micro batch 8
#   (which OOMs on an A100 -- run_steerf.sh dropped it to 4 for exactly that
#   reason), save_after hardcoded to 80, save_best_only hardcoded to False, and
#   best_metric_key pointing at mean@1 instead of mean@32. Reaching the same
#   two clip values as trailing overrides on run_steerf.sh keeps every other
#   knob at campaign parity, which is the whole point of an ablation.
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}" || { echo "FATAL: cannot cd to ${ROOT}" >&2; exit 1; }
# shellcheck source=run/_arms.sh
. "${ROOT}/run/_arms.sh"

ARMS=${ARMS:-"lam0-tree oracle grpo-long lam0.1 lam0.5 xclip-signed xclip-steer rloo-signed rloo-steer opo-signed opo-steer"}
SEED=${SEED:-1}
STEPS=${STEPS:-110}
LONG_STEPS=${LONG_STEPS:-200}      # grpo-long only: run past the wall-clock crossing
LOG_DIR="${ROOT}/logs/experiments"
# 30, not 20: a resumable checkpoint carries the optimizer state, so it is
# ~25 GiB rather than 3.1 GiB. See the shared settings below.
MIN_FREE_GB=${MIN_FREE_GB:-30}
CKPT_ROOT="${ROOT}/checkpoints/STEER-F"
DRY=${DRY:-0}
REPO=${REPO:-}
WAIT=${WAIT:-0}
WAIT_POLL=${WAIT_POLL:-120}

banner () { printf '\n========================================\n%s\n========================================\n' "$*"; }
free_gb () { df -BG --output=avail "${ROOT}" 2>/dev/null | tail -1 | tr -dc '0-9'; }


# ------------------------------------------------------- the arm definitions
# Each arm prints, on stdout, one line per token: the launcher, then the env
# assignments, then "--", then the trailing hydra overrides.
arm_spec () {   # <arm>
    case "$1" in
        lam0.1)       echo "tree STEERF_LAM=0.1"  ;;
        lam0.5)       echo "tree STEERF_LAM=0.5"  ;;
        lam0-tree)    echo "tree STEERF_LAM=0"    ;;
        oracle)       echo "tree STEERF_LAM=0.25 STEERF_FORECAST=oracle" ;;
        xclip-signed) echo "tree STEERF_LAM=0.25 -- ${XCLIP}" ;;
        rloo-signed)  echo "tree STEERF_LAM=0.25 -- algorithm.adv_estimator=rloo" ;;
        opo-signed)   echo "tree STEERF_LAM=0.25 -- algorithm.adv_estimator=opo"  ;;
        grpo-long)    echo "grpo STEERF_LAM=0" ;;
        xclip-steer)  echo "plain STEERF_LAM=0 -- ${XCLIP}" ;;
        rloo-steer)   echo "plain STEERF_LAM=0 -- algorithm.adv_estimator=rloo" ;;
        opo-steer)    echo "plain STEERF_LAM=0 -- algorithm.adv_estimator=opo"  ;;
        *) return 1 ;;
    esac
}
XCLIP="actor_rollout_ref.actor.clip_ratio_high=5 actor_rollout_ref.actor.clip_ratio_low=0.99"

# --------------------------------------------------------------- validation
for a in ${ARMS}; do
    arm_spec "${a}" >/dev/null || { echo "FATAL: unknown arm '${a}'" >&2; exit 2; }
    run_name_for "${a}" "${SEED}" >/dev/null || { echo "FATAL: no run name for '${a}'" >&2; exit 2; }
done

# ------------------------------------------------------------------- queue
declare -a QUEUE=()
needs_passthrough=0
for a in ${ARMS}; do
    rn="$(run_name_for "${a}" "${SEED}")"
    spec="$(arm_spec "${a}")"
    if train_log_done "${LOG_DIR}" "${rn}" "$(steps_for_arm "${a}")"; then
        printf '  skip  %-14s %-52s (done)\n' "${a}" "${rn}"
        continue
    fi
    printf '  QUEUE %-14s %-52s  %s\n' "${a}" "${rn}" "${spec}"
    QUEUE+=("${a}")
    case "${spec}" in tree*--*) needs_passthrough=1 ;; esac
done
printf '\n%s run(s) queued at seed %s, %s steps (grpo-long: %s)\n' \
    "${#QUEUE[@]}" "${SEED}" "${STEPS}" "${LONG_STEPS}"
[ "${#QUEUE[@]}" -eq 0 ] && { echo "Nothing to do."; exit 0; }
[ "${DRY}" = "1" ] && { echo "(DRY=1, stopping here)"; exit 0; }

# ------------------------------------------------------------------ guards
LOCK="${ROOT}/.followups.lock"
if ! mkdir "${LOCK}" 2>/dev/null; then
    holder="$(cat "${LOCK}/pid" 2>/dev/null || echo '?')"
    if [ "${holder}" != "?" ] && kill -0 "${holder}" 2>/dev/null; then
        echo "REFUSE: another follow-up queue is running (pid ${holder})." >&2
        exit 2
    fi
    echo "[followups] reclaiming a stale lock from pid ${holder}"
    rm -rf "${LOCK}"; mkdir "${LOCK}" || { echo "FATAL: cannot take ${LOCK}" >&2; exit 2; }
fi
echo $$ > "${LOCK}/pid"
trap 'rm -rf "${LOCK}"' EXIT INT TERM

if [ "${WAIT}" = "1" ]; then
    if is_busy; then
        echo "[followups] waiting for the running job (polling every ${WAIT_POLL}s)"
        while is_busy; do sleep "${WAIT_POLL}"; done
        echo "[followups] $(date -Is)  the box is free, starting"
        sleep 30
    fi
elif is_busy; then
    echo "REFUSE: a training process is already running. Re-run with WAIT=1 to queue behind it." >&2
    exit 2
fi
if ! bash run/instrument_campaign.sh --check >/dev/null 2>&1; then
    echo "REFUSE: instrumentation is not in place. run: bash run/instrument_campaign.sh --apply" >&2
    exit 2
fi
# The tree arms that carry a trailing override reach run_steerf.sh only through
# run_uniform_ablation.sh, which forwards "$@" ONLY after the phase-2 patch.
# Without it the override is silently dropped and the run is a duplicate of the
# signed arm under a different name -- the worst possible failure, because it
# produces plausible numbers.
if [ "${needs_passthrough}" = "1" ] && ! bash run/instrument_phase2.sh --check >/dev/null 2>&1; then
    echo "REFUSE: run_uniform_ablation.sh does not forward trailing overrides yet." >&2
    echo "        Without it xclip/rloo/opo would silently rerun the signed arm." >&2
    echo "        run: bash run/instrument_phase2.sh --apply" >&2
    exit 2
fi

if ! env_preflight "${ROOT}"; then
    echo "REFUSE: the training environment is broken -- nothing would train." >&2
    exit 2
fi
model_guard || exit 2
# Same idea one level out: run_uniform_ablation.sh:83-84 hardcodes N_GPUS=2 and
# EXPORTS it, so the tree arms silently take two cards on a four-card box while
# the arms that go through run_steerf.sh take four. Settling it here means no
# launcher default is ever reached and every arm in this queue matches.
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
export MODEL_PATH                # never let a launcher's own default win
export N_GPUS TP_SIZE             # settled by topology_guard, above
export VAL_DATA_DIR="${ROOT}/validation_data"

banner "follow-ups: ${#QUEUE[@]} run(s), seed ${SEED}, steps=${STEPS}"
mkdir -p "${LOG_DIR}" "${VAL_DATA_DIR}"

# One definition of how a followup arm is launched, so the OOM retry re-runs
# what failed instead of a second copy that drifts away from it.
# shellcheck disable=SC2086  -- the trailing overrides are a deliberate word list
launch_followup () {   # <arm> <kind> <lam> <run-name> <steps> <envs> [hydra ...]
    # <envs> is arm_spec's assignment list verbatim. The loop used to pull only
    # STEERF_LAM out of it and drop the rest, so an arm whose spec set anything
    # else -- STEERF_FORECAST=oracle is the first -- ran under its own name with
    # the default value. Unquoted on purpose: these are K=V words.
    local a="$1" kind="$2" lam="$3" rn="$4" steps="$5" envs="$6"
    shift 6
    local resume=""
    # The launchers refuse an existing checkpoint dir on purpose; continuing is
    # exactly what this queue means, since the arm is here because it is unfinished.
    [ -d "${CKPT_ROOT}/${rn}" ] && resume=1
    if [ "${kind}" = "grpo" ]; then
        # run_grpo.sh honours STEPS and writes ${LOG_DIR}/train-<run>.log itself.
        env ${envs} SEED="${SEED}" RUN_NAME="${rn}" LOG="${LOG_DIR}/train-${rn}.log" \
            STEPS="${steps}" ${resume:+RESUME=1} bash run/run_grpo.sh
        return $?
    elif [ "${kind}" = "tree" ]; then
        # run_uniform_ablation.sh writes ${LOG_DIR}/train-${RUN_NAME}.log itself.
        env ${envs} ARM=signed SEED="${SEED}" RUN_NAME="${rn}" STEERF_LAM="${lam}" \
            STEPS="${steps}" ${resume:+RESUME=1} bash run/run_uniform_ablation.sh "$@"
        return $?
    fi
    env ${envs} SEED="${SEED}" RUN_NAME="${rn}" STEERF_LAM="${lam}" ${resume:+RESUME=1} \
        bash run/run_steerf.sh $(steer_plain_args "${steps}") "$@" \
        > "${LOG_DIR}/train-${rn}.log" 2>&1
    return $?
}

for a in "${QUEUE[@]}"; do
    rn="$(run_name_for "${a}" "${SEED}")"
    spec="$(arm_spec "${a}")"
    kind="${spec%% *}"; rest="${spec#* }"
    if [ "${rest}" = "${spec}" ]; then rest=""; fi
    env_part="${rest%%--*}"; extra_part=""
    case "${rest}" in *--*) extra_part="${rest#*-- }" ;; esac
    lam="${env_part#*STEERF_LAM=}"; lam="${lam%% *}"

    avail="$(free_gb)"
    if [ -n "${avail}" ] && [ "${avail}" -lt "${MIN_FREE_GB}" ]; then
        echo "STOP: only ${avail} GB free, need ${MIN_FREE_GB}." >&2
        exit 1
    fi

    banner "${a}  ->  ${rn}   (lam=${lam}, ${avail:-?} GB free)"
    # A run whose ray.init() died leaves gcs_server, raylet and a
    # /tmp/ray session behind, and the next run inherits the wreck and dies
    # in the same place. run_0905_chain.sh:126 has always done this; the
    # queues did not, which is why one broken start took the whole queue
    # down with it. Safe here: the queue already refuses to begin while
    # another trainer is running.
    ray stop --force >/dev/null 2>&1 || true
    sleep 5
    # ray is down; VRAM is not necessarily back yet, and NCCL fails at init
    # rather than waiting for it.
    await_gpus || true
    [ -n "${extra_part}" ] && echo "  extra overrides: ${extra_part}"
    start=$(date +%s)

    launch_followup "${a}" "${kind}" "${lam}" "${rn}" "${STEPS}" "${env_part}" ${extra_part}
    st=$?
    printf '[followups] %s exit %s after %s min\n' "${a}" "${st}" "$(( ($(date +%s) - start) / 60 ))"

    if [ "${st}" -ne 0 ]; then
        diagnose_run_failure "${LOG_DIR}/train-${rn}.log" "${a}" "${STEPS}"
        why=$?
        if [ "${why}" = "2" ] && [ "${OOM_RETRY:-1}" = "1" ]; then
            echo "[followups] retrying ${rn} with OFFLOAD=1 after the CUDA OOM."
            echo "[followups] NOTE: OFFLOAD runs the AdamW update on CPU fp32, so"
            echo "[followups]       this arm is not bit-identical to one that did not."
            ray stop --force >/dev/null 2>&1 || true
            sleep 5
            await_gpus || true
            start=$(date +%s)
            OFFLOAD=1 launch_followup "${a}" "${kind}" "${lam}" "${rn}" "${STEPS}" \
                "${env_part}" ${extra_part}
            st=$?
            printf '[followups] %s retry exit %s after %s min\n' \
                "${a}" "${st}" "$(( ($(date +%s) - start) / 60 ))"
        fi
    fi
    if [ "${st}" -ne 0 ]; then
        echo "[followups] FAILED -- keeping the checkpoint, moving on"
        continue
    fi
    if ! train_log_done "${LOG_DIR}" "${rn}" "$(steps_for_arm "${a}")"; then
        echo "[followups] WARNING: exit 0 but the log never reached step $(steps_for_arm "${a}"); not uploading"
        continue
    fi
    if [ -n "${REPO}" ]; then
        echo "[followups] uploading ${rn} to ${REPO} and freeing the disk"
        REPO="${REPO}" DELETE=1 bash run/hf_backup.sh "${rn}" \
            || echo "[followups] upload/verify failed -- the local copy is kept"
    fi
done

banner "follow-ups finished"
for a in ${ARMS}; do
    rn="$(run_name_for "${a}" "${SEED}")"
    train_log_done "${LOG_DIR}" "${rn}" "$(steps_for_arm "${a}")" && r=done || r=MISSING
    printf '  %-14s %-52s %s\n' "${a}" "${rn}" "${r}"
done
