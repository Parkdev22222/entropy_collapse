#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The 5-arm x N-seed campaign, as one resumable queue.
#
#   bash run/run_campaign.sh                  # seeds 2 3 4 5, all five arms
#   DRY=1 bash run/run_campaign.sh            # print the queue, run nothing
#   SEEDS="2 3" bash run/run_campaign.sh      # a subset
#   REPO=user/repo bash run/run_campaign.sh   # upload + delete each checkpoint
#   WAIT=1 bash run/run_campaign.sh           # start once the current job ends
#
# Seed 1 already exists for all five arms, so the default picks up at 2.
#
# WHY THIS SCRIPT EXISTS
#   run_0905_chain.sh covers only signed/steer/uniform, which leaves GRPO and
#   permuted at one seed. A paired contrast is limited by min(n) on its two
#   arms, so the headline STEER-F - GRPO gap stays at n=1 no matter how many
#   seeds the other three accumulate.
#
# ORDER
#   Seed-major: every arm of seed 2, then every arm of seed 3, and so on. Stop
#   the queue at any point and all five arms have the same number of seeds,
#   which is what the paired analysis needs. Arm-major would leave STEER-F at
#   five seeds and GRPO at one.
#
# RESUMABLE
#   A run whose log already reached the final step is skipped, so re-running
#   this script after a crash, a reboot or a manual run costs nothing. The
#   check reads train-<run>.log and train-<run>_*.log, so a recovery run with a
#   tag suffix (the _0905 chain, say) also counts as done. The underscore is
#   load-bearing: a bare train-<run>* would also match train-<run>-permuted.log
#   and mark the signed arm done because a sibling arm finished.
#
# DISK
#   SAVE_BEST_ONLY=True makes verl delete the previous best when a new one
#   appears, and SAVE_CONTENTS=['hf_model'] drops the optimizer and extra state
#   that only a resume would need. Together that is one ~3.1 GB directory per
#   run instead of three ~20 GB ones. With REPO set, each finished run is
#   uploaded, verified byte for byte and deleted locally before the next one
#   starts, so the campaign's steady-state footprint is a single checkpoint.
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}" || { echo "FATAL: cannot cd to ${ROOT}" >&2; exit 1; }

# shellcheck source=run/_arms.sh
. "${ROOT}/run/_arms.sh"

SEEDS=${SEEDS:-"2 3 4 5"}
ARMS=${ARMS:-${CAMPAIGN_ARMS}}
STEPS=${STEPS:-110}
LOG_DIR="${ROOT}/logs/experiments"
CKPT_ROOT="${ROOT}/checkpoints/STEER-F"
# 30, not 20: a resumable checkpoint carries the optimizer and rng state, so
# it is ~25 GiB rather than the 3.1 GiB of an hf_model-only one. See the
# shared settings below for why that trade is worth the disk.
MIN_FREE_GB=${MIN_FREE_GB:-30}
DRY=${DRY:-0}
REPO=${REPO:-}
WAIT=${WAIT:-0}            # 1 = queue behind a running job instead of refusing
WAIT_POLL=${WAIT_POLL:-120}

banner () { printf '\n========================================\n%s\n========================================\n' "$*"; }

is_done () { train_log_done "${LOG_DIR}" "$1" "${STEPS}"; }

free_gb () { df -BG --output=avail "${ROOT}" 2>/dev/null | tail -1 | tr -dc '0-9'; }

# Is a training job running? pgrep matches on the whole command line, so the
# shell that launched this script also matches whenever the launch command
# mentions main_ppo. Excluding our own ancestors removes exactly those without
# hiding a real trainer.

# --------------------------------------------------------------- validation
for a in ${ARMS}; do
    run_name_for "${a}" 0 >/dev/null || { echo "FATAL: unknown arm '${a}'" >&2; exit 2; }
done
for s in ${SEEDS}; do
    case "${s}" in ''|*[!0-9]*) echo "FATAL: seed '${s}' is not a number" >&2; exit 2 ;; esac
done

# ------------------------------------------------------------------- queue
declare -a QUEUE=()
for s in ${SEEDS}; do
    for a in ${ARMS}; do
        rn="$(run_name_for "${a}" "${s}")"
        if is_done "${rn}"; then
            printf '  skip  %-52s (done)\n' "${rn}"
        else
            printf '  QUEUE %-52s\n' "${rn}"
            QUEUE+=("${a}:${s}:${rn}")
        fi
    done
done
printf '\n%s run(s) queued, %s arm(s) x %s seed(s)\n' \
    "${#QUEUE[@]}" "$(echo ${ARMS} | wc -w)" "$(echo ${SEEDS} | wc -w)"
[ "${#QUEUE[@]}" -eq 0 ] && { echo "Nothing to do."; exit 0; }
[ "${DRY}" = "1" ] && { echo "(DRY=1, stopping here)"; exit 0; }

# ------------------------------------------------------------------ guards
# Two campaigns racing would interleave runs on the same GPUs. A directory is
# an atomic create, so this works without flock; a lock whose PID is gone is
# stale and reclaimed.
LOCK="${ROOT}/.campaign.lock"
if ! mkdir "${LOCK}" 2>/dev/null; then
    holder="$(cat "${LOCK}/pid" 2>/dev/null || echo '?')"
    if [ "${holder}" != "?" ] && kill -0 "${holder}" 2>/dev/null; then
        echo "REFUSE: another campaign is running (pid ${holder}). Remove ${LOCK} if that is wrong." >&2
        exit 2
    fi
    echo "[campaign] reclaiming a stale lock from pid ${holder}"
    rm -rf "${LOCK}"; mkdir "${LOCK}" || { echo "FATAL: cannot take ${LOCK}" >&2; exit 2; }
fi
echo $$ > "${LOCK}/pid"
trap 'rm -rf "${LOCK}"' EXIT INT TERM

if [ "${WAIT}" = "1" ]; then
    # Queue behind whatever is training now. Matching on the command line
    # rather than a PID means a recycled PID cannot end the wait early.
    if is_busy; then
        echo "[campaign] waiting for the running job to finish (polling every ${WAIT_POLL}s)"
        busy_pids | head -3 | while read -r h; do
            printf '[campaign]   holder: %s %s\n' "${h}" "$(tr '\0' ' ' < "/proc/${h}/cmdline" 2>/dev/null | cut -c1-80)"
        done
        while is_busy; do sleep "${WAIT_POLL}"; done
        echo "[campaign] $(date -Is)  the box is free, starting"
        sleep 30    # let the GPUs actually release before vLLM grabs them
    fi
elif is_busy; then
    echo "REFUSE: a training process is already running. Wait for it, stop it," >&2
    echo "        or re-run with WAIT=1 to queue behind it." >&2
    exit 2
fi
if ! bash run/instrument_campaign.sh --check >/dev/null 2>&1; then
    echo "REFUSE: instrumentation is not in place." >&2
    echo "        run: bash run/instrument_campaign.sh --apply" >&2
    exit 2
fi
if [ -n "${REPO}" ] && ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "REFUSE: REPO is set but huggingface_hub is not importable." >&2
    exit 2
fi
# Five seconds in front of a 40-hour run. On 2026-09-08 an unpinned
# huggingface_hub upgrade made every `import verl` raise, and a queued chain
# burned two arms in seconds before anyone noticed.
if ! env_preflight "${ROOT}"; then
    echo "REFUSE: the training environment is broken -- nothing would train." >&2
    exit 2
fi
# run_steerf.sh defaults to a 7B model and the steer arm is the one arm that
# calls it directly, so the run name and the network can disagree without
# anything erroring. Refuse before a single step rather than discover it in the
# table.
model_guard || exit 2
# Same idea one level out: run_uniform_ablation.sh:83-84 hardcodes N_GPUS=2 and
# EXPORTS it, so the tree arms silently take two cards on a four-card box while
# the arms that go through run_steerf.sh take four. Settling it here means no
# launcher default is ever reached and every arm in this queue matches.
topology_guard || exit 2

# ---------------------------------------------------------- shared settings
export SAVE_BEST_ONLY=True                 # keep only the best checkpoint
# A crash used to cost the whole run. verl's checkpoint_contents defaults to
# ['model','optimizer','extra'] and naming save_contents REPLACES that list, so
# ['hf_model'] wrote no optimizer or rng state and trainer.resume_mode had
# nothing to resume FROM (run_steerf.sh:154). This queue said "keeping the
# checkpoint so it can be resumed" while making resuming impossible; on
# 2026-09-15 a CUDA OOM at step 47 of the seed-2 steer arm therefore cost all
# 25 hours of it. Writing the optimizer turns that into at most SAVE_FREQ steps.
#
# The cost is disk: a full 1.5B checkpoint is ~25 GiB against 3.1 GiB for
# hf_model alone. MAX_CKPT_KEEP=1 holds the peak flat (verl deletes the old one
# BEFORE writing the new one), MIN_FREE_GB above is raised to match, and
# `PRUNE=1 bash run/hf_backup.sh <run>` gives the difference back once a run
# has finished -- only actor/huggingface is needed to evaluate or publish.
export SAVE_CONTENTS="['hf_model','model','optimizer','extra']"
export RESUME_MODE=auto                    # run_steerf.sh defaults to disable
export MAX_CKPT_KEEP=1                     # peak stays at one checkpoint
export SAVE_AFTER=0                        # run_grpo.sh
export SAVE_AFTER_OVERRIDE=0               # run_steerf.sh
export MODEL_PATH                          # never let a launcher's own default win
export N_GPUS TP_SIZE                       # settled by topology_guard, above
export VAL_DATA_DIR="${ROOT}/validation_data"
export STEPS

banner "campaign: ${#QUEUE[@]} run(s), steps=${STEPS}, best-only, resumable"
mkdir -p "${LOG_DIR}" "${VAL_DATA_DIR}"

# One place that knows how each arm is launched, so the OOM retry below runs
# exactly what failed rather than a second, drifting copy of it.
#
# RESUME=1 is passed whenever a checkpoint directory is already there.
# run_grpo.sh:137 and run_uniform_ablation.sh:143 refuse in that case on
# purpose -- resume_mode=auto would otherwise continue an old run when someone
# meant to start a fresh one. Continuing IS what this queue means: the run is
# in the queue because train_log_done says it never reached ${STEPS}.
launch_arm () {   # <arm> <seed> <run-name> <log>
    local arm="$1" seed="$2" rn="$3" log="$4" resume=""
    [ -d "${CKPT_ROOT}/${rn}" ] && resume=1

    # Only run_steerf.sh writes the trainer's output to stdout. run_grpo.sh
    # and run_uniform_ablation.sh both redirect their child into exactly
    # ${LOG_DIR}/train-<run>.log themselves, so teeing there would put two
    # writers at independent offsets on one file and shred it.
    case "${arm}" in
        grpo)
            SEED="${seed}" RUN_NAME="${rn}" LOG="${log}" ${resume:+RESUME=1} \
                bash run/run_grpo.sh
            return $?
            ;;
        steer)
            # run_steerf.sh hardcodes STEPS=200 inside its SCALE case, so an
            # exported STEPS never reaches it -- without the trailing override
            # this arm trains to 200 and the queue stalls for an extra 20 h.
            SEED="${seed}" RUN_NAME="${rn}" STEERF_LAM=0 ${resume:+RESUME=1} \
                bash run/run_steerf.sh $(steer_plain_args "${STEPS}") 2>&1 | tee "${log}"
            return "${PIPESTATUS[0]}"
            ;;
        signed|uniform|permuted)
            ARM="${arm}" SEED="${seed}" RUN_NAME="${rn}" STEERF_LAM=0.25 ${resume:+RESUME=1} \
                bash run/run_uniform_ablation.sh
            return $?
            ;;
    esac
    echo "[campaign] unknown arm '${arm}'" >&2
    return 3
}

for item in "${QUEUE[@]}"; do
    arm="${item%%:*}"; rest="${item#*:}"; seed="${rest%%:*}"; rn="${rest##*:}"
    log="${LOG_DIR}/train-${rn}.log"

    avail="$(free_gb)"
    if [ -n "${avail}" ] && [ "${avail}" -lt "${MIN_FREE_GB}" ]; then
        echo "STOP: only ${avail} GB free, need ${MIN_FREE_GB}. Upload or delete checkpoints first." >&2
        exit 1
    fi

    banner "${arm}  seed ${seed}  ->  ${rn}   (${avail:-?} GB free)"
    start=$(date +%s)
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

    launch_arm "${arm}" "${seed}" "${rn}" "${log}"
    st=$?
    printf '[campaign] %s seed %s exit %s after %s min\n' \
        "${arm}" "${seed}" "${st}" "$(( ($(date +%s) - start) / 60 ))"

    if [ "${st}" -ne 0 ]; then
        diagnose_run_failure "${log}" "${arm} seed ${seed}" "${STEPS}"
        why=$?
        # rc 2 means CUDA OOM. Offloading the optimizer and params to CPU is the
        # one large lever that leaves the treatment alone -- unlike the micro
        # batch, which IS the treatment. Once, then move on: a second OOM at the
        # same setting is a ceiling, not a tail event.
        if [ "${why}" = "2" ] && [ "${OOM_RETRY:-1}" = "1" ]; then
            echo "[campaign] retrying ${rn} with OFFLOAD=1 after the CUDA OOM."
            echo "[campaign] NOTE: OFFLOAD moves the AdamW update to CPU fp32, so"
            echo "[campaign]       this seed's arithmetic is not bit-identical to"
            echo "[campaign]       seeds that ran without it. analyze_seeds.py"
            echo "[campaign]       reads param_offload out of the log and reports it."
            ray stop --force >/dev/null 2>&1 || true
            sleep 5
            await_gpus || true
            start=$(date +%s)
            OFFLOAD=1 launch_arm "${arm}" "${seed}" "${rn}" "${log}"
            st=$?
            printf '[campaign] %s seed %s retry exit %s after %s min\n' \
                "${arm}" "${seed}" "${st}" "$(( ($(date +%s) - start) / 60 ))"
            [ "${st}" -ne 0 ] && diagnose_run_failure "${log}" \
                "${arm} seed ${seed} (OFFLOAD=1 retry)" "${STEPS}" || true
        fi
    fi
    if [ "${st}" -ne 0 ]; then
        echo "[campaign] FAILED -- the checkpoint carries the optimizer state, so"
        echo "[campaign] the next pass over this queue resumes rather than restarts."
        continue
    fi
    if ! is_done "${rn}"; then
        echo "[campaign] WARNING: exit 0 but the log never reached step ${STEPS}; not uploading"
        continue
    fi
    if [ -n "${REPO}" ]; then
        echo "[campaign] uploading ${rn} to ${REPO} and freeing the disk"
        REPO="${REPO}" DELETE=1 bash run/hf_backup.sh "${rn}" \
            || echo "[campaign] upload/verify failed -- the local copy is kept"
    fi
done

banner "campaign finished"
for s in ${SEEDS}; do
    for a in ${ARMS}; do
        rn="$(run_name_for "${a}" "${s}")"
        is_done "${rn}" && r=done || r=MISSING
        printf '  %-52s %s\n' "${rn}" "${r}"
    done
done
