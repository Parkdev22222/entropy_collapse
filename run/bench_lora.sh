#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Does LoRA make the campaign finish sooner?  Measure before rewriting it.
#
#   bash run/bench_lora.sh                          # all cases, 4 steps each
#   BENCH_CASES="lora_1gpu_x4 full_tp4" bash run/bench_lora.sh
#   DRY=1 bash run/bench_lora.sh                    # print the commands only
#   BENCH_ARM=signed BENCH_CASES="lora_1gpu_x4_graph" \
#     BENCH_BASELINE_LOG=<a campaign STEER-V log> bash run/bench_lora.sh
#
# WHY
#   The step-time breakdown of the full fine-tuning runs (4x H100, seed 3):
#
#       arm      gen    old_log_prob  update_actor  step
#       STEER-F  259 s  150 s          60 s          473 s
#       GRPO     402 s   29 s          61 s          498 s
#
#   LoRA only shrinks update_actor and the checkpoint write. Generation, which
#   is 55-80% of a step, is unchanged or slower (vLLM runs the adapter through
#   its LoRA kernels). So LoRA by itself should move a step by about +-10%.
#   What it can buy is MEMORY: without full optimizer state a 1.5B run fits on
#   one GPU next to a large vLLM pool, and a 4-GPU box runs four trainers side
#   by side. Whether that beats one tp=4 trainer is an empirical question, and
#   this script answers it in box-steps per hour.
#
# CASES (every one is the paper's GRPO configuration -- batch 512 x 8,
# 3072-token responses, same data -- only the listed knobs differ)
#   full_tp4       full fine-tuning, 4 GPUs, tp=4          (the paper's setup)
#   lora_tp4       LoRA, 4 GPUs, tp=4
#   lora_1gpu_x4   LoRA, four one-GPU trainers at once
#   full_1gpu_x4   full fine-tuning, four one-GPU trainers at once -- separates
#                  "running four at once" from "LoRA"; may OOM, which is itself
#                  the answer
#   lora_tp4_graph      lora_tp4 with vLLM CUDA graphs (ROLLOUT_EAGER=0)
#   lora_1gpu_x4_graph  lora_1gpu_x4 with vLLM CUDA graphs
#   LoRA = rank 64, alpha 32, all-linear, lr 1e-5, load_format=safetensors.
#
# BENCH_ARM=signed times the paper's method instead of GRPO: the tree arm
# (run/run_uniform_ablation.sh) with the environment _lora_arms.sh gives it,
# so the number is the campaign's own step, not a proxy. BENCH_BASELINE names
# the case the speedup is taken against (default full_tp4); with
# BENCH_BASELINE_LOG=<log> an existing tp=4 training log -- e.g. the campaign
# run that was stopped to make room for this -- is summarised as lora_tp4 and
# becomes the baseline, so it need not be re-timed.
#
# COST: a step is ~8 min at tp=4 and several times that on one GPU, so four
# steps per case is roughly 35 min (tp=4) to 1.5 h (x4) per case. Run the
# cases you need; lora_1gpu_x4 against full_tp4 is the decision.
#
# Nothing here is a result for the paper. Validation, checkpointing and the
# pre-training validation pass are all off, so only the training step is timed.
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}" || exit 1
# shellcheck source=run/_arms.sh
. run/_arms.sh
# shellcheck source=run/_lora_arms.sh
. run/_lora_arms.sh

CASES=${BENCH_CASES:-"lora_1gpu_x4 full_tp4 lora_tp4 full_1gpu_x4"}
STEPS=${BENCH_STEPS:-4}
NGPU=${BENCH_GPUS:-4}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_LR=${LORA_LR:-1e-5}
LORA_1GPU_MEM=${LORA_1GPU_MEM:-0.6}
DRY=${DRY:-0}
TS=$(date +%Y%m%d_%H%M%S)
BENCH_DIR=${BENCH_DIR:-logs/bench/${TS}}
BENCH_ARM=${BENCH_ARM:-grpo}
BENCH_BASELINE=${BENCH_BASELINE:-full_tp4}
BENCH_BASELINE_LOG=${BENCH_BASELINE_LOG:-}
ARM_ENV=()
case "${BENCH_ARM}" in
    grpo)   LAUNCHER=run/run_grpo.sh ;;
    signed) LAUNCHER=run/run_uniform_ablation.sh
            # "tree ARM=signed STEERF_LAM=... STEERF_FORECAST=oracle": the words after the kind
            read -r -a ARM_ENV <<<"$(lora_arm_spec signed | cut -d' ' -f2-)" ;;
    *) echo "FATAL: BENCH_ARM must be grpo or signed, got '${BENCH_ARM}'" >&2; exit 2 ;;
esac
if [ -n "${BENCH_BASELINE_LOG}" ]; then
    [ -f "${BENCH_BASELINE_LOG}" ] || { echo "FATAL: BENCH_BASELINE_LOG ${BENCH_BASELINE_LOG} not found" >&2; exit 2; }
    BENCH_BASELINE=lora_tp4
fi

LORA_ARGS=(
    actor_rollout_ref.model.lora_rank="${LORA_RANK}"
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}"
    actor_rollout_ref.model.target_modules=all-linear
    actor_rollout_ref.actor.optim.lr="${LORA_LR}"
    actor_rollout_ref.rollout.load_format=safetensors
)
# test_freq/save_freq here too: the tree launcher's SCALE block fixes both at
# 10 and reads no env for them; a later hydra value wins.
COMMON_ARGS=( trainer.val_before_train=False trainer.test_freq=-1 trainer.save_freq=-1 )
# On the lora branch the launchers train LoRA unless FULL_FT=1, so the full
# fine-tuning cases pass it; an older launcher ignores it and the explicit
# LORA_ARGS below are what turn LoRA on there.

# ---- guards -----------------------------------------------------------------
if [ ! -f "${LAUNCHER}" ]; then
    echo "REFUSE: ${LAUNCHER} is not here. It comes from the donor branch;" >&2
    echo "        run 'bash run/bootstrap_pod.sh' first." >&2
    exit 1
fi
# A launcher that honours RAY_STOP (the lora branch) is used as it is; an older
# one gets a copy with its 'ray stop' line removed.
if grep -q 'RAY_STOP:-1' "${LAUNCHER}"; then
    HONOURS_RAY_STOP=1
elif [ "$(grep -c '^ray stop --force' "${LAUNCHER}")" != "1" ]; then
    echo "REFUSE: expected exactly one 'ray stop --force' line in ${LAUNCHER}." >&2
    echo "        The side-by-side cases remove it from a copy so the four trainers" >&2
    echo "        do not stop each other's Ray; with a different launcher that edit" >&2
    echo "        is not known to be safe." >&2
    exit 1
fi
if [ "${DRY}" != "1" ]; then
    if is_busy; then
        echo "REFUSE: a trainer or evaluation is running (pids: $(busy_pids | tr '\n' ' '))." >&2
        echo "        The benchmark needs the whole box; timings next to another job are meaningless." >&2
        exit 1
    fi
    if ! gpus_free && [ "${ALLOW_BUSY_GPUS:-0}" != "1" ]; then
        echo "REFUSE: GPUs still hold memory:" >&2; gpu_holders >&2
        echo "        ALLOW_BUSY_GPUS=1 overrides." >&2
        exit 1
    fi
    have=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "${have}" -lt "${NGPU}" ]; then
        echo "REFUSE: ${NGPU} GPUs needed, ${have} visible." >&2
        exit 1
    fi
fi
mkdir -p "${BENCH_DIR}"

NR_LAUNCHER="run/.bench_run_grpo.$$.sh"
cleanup() { rm -f "${NR_LAUNCHER}"; }
trap cleanup EXIT

# The run name decides the log file, the checkpoint dir and the tensorboard dir;
# a timestamped name keeps every bench run away from the campaign's.
base_env() {   # <run-name>
    echo "RUN_NAME=$1 LOG_DIR=${BENCH_DIR} SEED=1 STEPS=${STEPS} TEST_FREQ=-1 SAVE_FREQ=-1 SAVE_AFTER=0 RESUME_MODE=disable"
}

declare -A CASE_LOGS=()

run_serial() {   # <case> <lora:0|1> [eager:0|1]
    local c=$1 lora=$2 eager=${3:-1} name="bench-${TS}-$1" extra=("${COMMON_ARGS[@]}")
    [ "${lora}" = 1 ] && extra+=("${LORA_ARGS[@]}")
    [ "${eager}" = 0 ] && extra+=(actor_rollout_ref.rollout.enforce_eager=False)
    CASE_LOGS[$c]="${BENCH_DIR}/train-${name}.log"
    echo "[bench] ${c}: ${NGPU} GPUs, tp=${NGPU}  -> ${CASE_LOGS[$c]}"
    # shellcheck disable=SC2046
    env $(base_env "${name}") "${ARM_ENV[@]}" FULL_FT=$((1 - lora)) ROLLOUT_EAGER="${eager}" \
        N_GPUS="${NGPU}" TP_SIZE="${NGPU}" DRY_RUN="${DRY}" bash "${LAUNCHER}" "${extra[@]}"
}

run_side_by_side() {   # <case> <lora:0|1> [eager:0|1]
    local c=$1 lora=$2 eager=${3:-1} i name logs="" pids=() extra=("${COMMON_ARGS[@]}") mem_env=()
    [ "${lora}" = 1 ] && { extra+=("${LORA_ARGS[@]}"); mem_env=(GPU_MEM_UTIL="${LORA_1GPU_MEM}"); }
    [ "${eager}" = 0 ] && extra+=(actor_rollout_ref.rollout.enforce_eager=False)
    if [ "${HONOURS_RAY_STOP:-0}" = 1 ]; then
        cp "${LAUNCHER}" "${NR_LAUNCHER}"
    else
        sed -e '/^ray stop --force/d' "${LAUNCHER}" > "${NR_LAUNCHER}"
    fi
    [ "${DRY}" = 1 ] || { ray stop --force >/dev/null 2>&1 || true; sleep 5; }
    for ((i = 0; i < NGPU; i++)); do
        name="bench-${TS}-${c}-g${i}"
        logs+="${logs:+,}${BENCH_DIR}/train-${name}.log"
        echo "[bench] ${c}: GPU ${i} alone, tp=1  -> ${BENCH_DIR}/train-${name}.log"
        # Separate RAY_TMPDIR per trainer: each starts its own local Ray, and
        # a shared temp dir is how one of them would find and join another's.
        # shellcheck disable=SC2046
        env $(base_env "${name}") "${ARM_ENV[@]}" "${mem_env[@]}" FULL_FT=$((1 - lora)) \
            ROLLOUT_EAGER="${eager}" CUDA_VISIBLE_DEVICES="${i}" \
            N_GPUS=1 TP_SIZE=1 FORCE_CONCURRENT=1 RAY_STOP=0 RAY_TMPDIR="/tmp/rb${i}" \
            DRY_RUN="${DRY}" bash "${NR_LAUNCHER}" "${extra[@]}" &
        pids+=($!)
        [ "${DRY}" = 1 ] || sleep 20   # stagger start-up, not the timed steps
    done
    for i in "${pids[@]}"; do wait "${i}"; done
    CASE_LOGS[$c]="${logs}"
}

for c in ${CASES}; do
    case "${c}" in
        full_tp4)     run_serial "${c}" 0 ;;
        lora_tp4)     run_serial "${c}" 1 ;;
        lora_1gpu_x4) run_side_by_side "${c}" 1 ;;
        lora_tp4_graph)     run_serial "${c}" 1 0 ;;
        lora_1gpu_x4_graph) run_side_by_side "${c}" 1 0 ;;
        full_1gpu_x4) run_side_by_side "${c}" 0 ;;
        *) echo "[bench] unknown case '${c}', skipped" >&2 ;;
    esac
done

[ "${DRY}" = 1 ] && exit 0

specs=()
[ -n "${BENCH_BASELINE_LOG}" ] && specs+=(--case "lora_tp4=${BENCH_BASELINE_LOG}")
for c in ${CASES}; do [ -n "${CASE_LOGS[$c]:-}" ] && specs+=(--case "${c}=${CASE_LOGS[$c]}"); done
echo
echo "[bench] box throughput (median of steps 2..$((STEPS - 1)); speedup vs ${BENCH_BASELINE}):"
python3 scripts/bench_lora_summary.py --steps "${STEPS}" --baseline "${BENCH_BASELINE}" \
    "${specs[@]}" --out "${BENCH_DIR}/summary.tsv"
echo "[bench] table -> ${BENCH_DIR}/summary.tsv"
