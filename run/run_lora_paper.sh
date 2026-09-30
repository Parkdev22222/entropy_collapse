#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Every experiment of the LoRA paper, in order, from one command.
#
#   bash run/run_lora_paper.sh                     # everything, this box alone
#   DRY=1 bash run/run_lora_paper.sh               # print the plan, run nothing
#   SHARD=0/2 bash run/run_lora_paper.sh           # box 1 of 2 (the other: 1/2)
#   TOPOLOGY=1gpu bash run/run_lora_paper.sh       # one trainer per GPU, side by side
#   STAGES="core eval-core" bash run/run_lora_paper.sh
#
# STAGES (in this order; each skips what is already done, so re-running after a
# crash or a reboot continues where it stopped)
#   preflight      environment, data, the MTP heads if mtp is planned,
#                  no other trainer on the box
#   core           grpo steer uniform permuted signed mtp, seeds 1 2 3,
#                  seed by seed -- a queue stopped part-way leaves whole seeds
#   eval-core      six benchmarks at 32 samples/problem, at the last step
#                  (primary) and at the AIME24 best (secondary)
#   followups      ten ablations at seed 1 (run/_lora_arms.sh)
#   eval-followups the same evaluation at their last step
#   backbones      grpo steer signed at seed 1 on Qwen2.5-Math-7B and
#                  Llama-3.2-3B-Instruct, always on every GPU of the box
#   eval-backbones the six benchmarks at their last step
#   analysis       coherence check, results tables, macros, paired errors
#
# WHAT A RUN IS -- run/_lora_arms.sh (arms, names, seeds), run/_lora_defaults.sh
# (rank 64, alpha 32, lr 1e-5, all-linear; 150 steps), and here: checkpoints
# every 50 steps + the last + the best (KEEP_EVERY=50, verl/trainer/ppo/ckpt_policy.py),
# validation every 10 steps, resume on restart.
#
# TOPOLOGY -- one value for the whole campaign; the paper states it, and every
# arm of a table must share it.
#   tp4   one trainer at a time on all GPUs (N_GPUS/TP_SIZE detected)
#   1gpu  one trainer per GPU, up to that many at once. Each has its own Ray
#         (RAY_TMPDIR) and the launchers skip `ray stop` (RAY_STOP=0).
#         Choose it only after run/bench_lora.sh says it is faster.
# Evaluation always runs one checkpoint at a time on all GPUs.
# ROLLOUT_EAGER -- 1 (verl's default) or 0 (vLLM CUDA graphs); run/_lora_defaults.sh.
#
# Both are locked: the first training invocation writes them to
# logs/lora/campaign_settings, and a later one (a restart, another shard on
# this box) that asks for different values is refused, so a campaign cannot
# drift from one setting to another between arms. CAMPAIGN_SETTINGS_OVERRIDE=1
# lets it through. TOPOLOGY is locked only by the core and follow-up stages
# (backbones always use the whole box); ROLLOUT_EAGER by every training stage.
#
# SHARDS -- every box computes the same plan and takes lines k with
# k mod N == i. No coordination; the boxes need the same LORA_* settings and,
# for analysis, each other's logs and eval dumps (run the analysis stage on one
# box after gathering them).
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}" || exit 1
# shellcheck source=run/_arms.sh
. run/_arms.sh
# shellcheck source=run/_lora_arms.sh
. run/_lora_arms.sh

STAGES=${STAGES:-"preflight core eval-core followups eval-followups backbones eval-backbones analysis"}
TOPOLOGY=${TOPOLOGY:-tp4}
SHARD=${SHARD:-0/1}
DRY=${DRY:-0}
KEEP_EVERY=${KEEP_EVERY:-50}
LORA_1GPU_MEM=${LORA_1GPU_MEM:-0.6}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-Math-1.5B}
LOG_DIR=${LOG_DIR:-${ROOT}/logs/lora/train}
CKPT_ROOT=${CKPT_ROOT:-${ROOT}/checkpoints/STEER-F}
RES_DIR=${RES_DIR:-${ROOT}/results/lora}
STATE_DIR=${STATE_DIR:-${ROOT}/logs/lora/state}
LORA_EVAL_N=${LORA_EVAL_N:-32}
export LORA_EVAL_N BASE_MODEL CKPT_ROOT

SHARD_I=${SHARD%/*}; SHARD_N=${SHARD#*/}
case "${SHARD_I}:${SHARD_N}" in
    *[!0-9:]*|:*|*:) echo "FATAL: SHARD must be i/N, got '${SHARD}'" >&2; exit 2 ;;
esac
[ "${SHARD_I}" -lt "${SHARD_N}" ] || { echo "FATAL: SHARD ${SHARD}: i must be < N" >&2; exit 2; }
case "${TOPOLOGY}" in tp4|1gpu) ;; *) echo "FATAL: TOPOLOGY must be tp4 or 1gpu" >&2; exit 2 ;; esac
ROLLOUT_EAGER=${ROLLOUT_EAGER:-1}
case "${ROLLOUT_EAGER}" in 0|1) ;; *) echo "FATAL: ROLLOUT_EAGER must be 0 or 1" >&2; exit 2 ;; esac
export ROLLOUT_EAGER
SETTINGS_FILE=${SETTINGS_FILE:-${ROOT}/logs/lora/campaign_settings}

banner () { printf '\n========================================\n%s\n========================================\n' "$*"; }
has_stage () { case " ${STAGES} " in *" $1 "*) return 0 ;; esac; return 1; }

# ------------------------------------------------------------------ the plan
my_plan () {   # <core|followups>
    lora_plan "$1" | lora_shard "${SHARD_I}" "${SHARD_N}"
}

run_done () {   # <arm> <seed>: the log reached the last step and its adapter is on disk
    local a=$1 s=$2 rn steps
    rn="$(lora_run_name "${a}" "${s}")"; steps="$(steps_for_arm "${a}")"
    train_log_done "${LOG_DIR}" "${rn}" "${steps}" \
        && [ -f "${CKPT_ROOT}/${rn}/global_step_${steps}/actor/lora_adapter/adapter_model.safetensors" ]
}

# ---------------------------------------------------------------- launching
# One definition of how an arm is launched; the 1gpu scheduler and the tp4
# loop both call it, so the two cannot drift. Extra environment (the GPU pin)
# arrives as leading K=V words.
launch () {   # <arm> <seed> [K=V ...] [-- hydra overrides ...]
    local a=$1 s=$2; shift 2
    local kv=() hy=()
    while [ "$#" -gt 0 ]; do
        if [ "$1" = "--" ]; then shift; hy=("$@"); break; fi
        kv+=("$1"); shift
    done
    set -- "${kv[@]}"
    local rn steps spec kind rest envs extra resume=""
    rn="$(lora_run_name "${a}" "${s}")"; steps="$(steps_for_arm "${a}")"
    spec="$(lora_arm_spec "${a}")"; kind="${spec%% *}"; rest="${spec#"${kind}"}"
    envs="${rest%%--*}"; extra=""
    case "${rest}" in *--*) extra="${rest#*-- }" ;; esac
    [ -d "${CKPT_ROOT}/${rn}" ] && resume=1
    local common=(SEED="${s}" RUN_NAME="${rn}" STEPS="${steps}" LOG_DIR="${LOG_DIR}"
                  KEEP_EVERY="${KEEP_EVERY}" SAVE_FREQ=10 TEST_FREQ=10 SAVE_AFTER=0
                  SAVE_AFTER_OVERRIDE=0 SAVE_BEST_ONLY=False RESUME_MODE=auto
                  MODEL_PATH="${BASE_MODEL}" VAL_DATA_DIR="${ROOT}/validation_data/lora/train"
                  ${resume:+RESUME=1} "$@")
    # shellcheck disable=SC2086  -- envs/extra are deliberate word lists
    case "${kind}" in
        grpo)
            env "${common[@]}" ${envs} LOG="${LOG_DIR}/train-${rn}.log" \
                bash run/run_grpo.sh ${extra} "${hy[@]}" ;;
        tree)
            env "${common[@]}" ${envs} bash run/run_uniform_ablation.sh ${extra} "${hy[@]}" ;;
        plain)
            # run_steerf.sh fixes STEPS=200 in its SCALE case and writes to
            # stdout; steer_plain_args carries the step count, the logger and
            # rollout_data_dir=null.
            env "${common[@]}" ${envs} bash run/run_steerf.sh \
                $(steer_plain_args "${steps}") ${extra} "${hy[@]}" > "${LOG_DIR}/train-${rn}.log" 2>&1 ;;
        *) echo "[lora] unknown kind '${kind}' for ${a}" >&2; return 3 ;;
    esac
}

train_stage () {   # <core|followups>
    local stage=$1 item a s rn todo=()
    while read -r item; do
        [ -n "${item}" ] || continue
        a="$(cut -d: -f2 <<<"${item}")"; s="$(cut -d: -f3 <<<"${item}")"
        rn="$(lora_run_name "${a}" "${s}")"
        if run_done "${a}" "${s}"; then
            printf '  done  %-12s s%s  %s\n' "${a}" "${s}" "${rn}"
        else
            printf '  QUEUE %-12s s%s  %s   [%s]\n' "${a}" "${s}" "${rn}" "$(lora_arm_spec "${a}")"
            todo+=("${a}:${s}")
        fi
    done < <(my_plan "${stage}")
    [ "${DRY}" = 1 ] && return 0
    [ "${#todo[@]}" -gt 0 ] || return 0
    mkdir -p "${LOG_DIR}" "${STATE_DIR}"
    if [ "${TOPOLOGY}" = tp4 ]; then
        gpu_topology
        for item in "${todo[@]}"; do
            a="${item%%:*}"; s="${item#*:}"
            banner "${stage}: ${a} seed ${s} -> $(lora_run_name "${a}" "${s}")"
            await_gpus || true
            launch "${a}" "${s}" N_GPUS="${N_GPUS}" TP_SIZE="${TP_SIZE}"
            report_run "${a}" "${s}" $?
        done
    else
        schedule_1gpu "${todo[@]}"
    fi
}

report_run () {   # <arm> <seed> <exit>
    if [ "$3" -eq 0 ] && run_done "$1" "$2"; then
        echo "[lora] OK   $1 s$2"
    else
        echo "[lora] FAIL $1 s$2 (exit $3) -- $(lora_run_name "$1" "$2"); re-running this stage resumes it"
        diagnose_run_failure "${LOG_DIR}/train-$(lora_run_name "$1" "$2").log" "$1" \
            "$(steps_for_arm "$1")" || true
    fi
}

# Up to NGPU trainers at once, each pinned to one card. A card is handed to the
# next run the moment its trainer exits.
schedule_1gpu () {
    local ngpu i a s item pids=() arms=() seeds=() queue=("$@")
    ngpu="$(nvidia-smi -L 2>/dev/null | wc -l)"
    [ "${ngpu}" -gt 0 ] || { echo "FATAL: no GPUs visible" >&2; return 1; }
    for ((i = 0; i < ngpu; i++)); do pids[i]=""; done
    while [ "${#queue[@]}" -gt 0 ] || [ -n "$(printf '%s' "${pids[@]}")" ]; do
        for ((i = 0; i < ngpu; i++)); do
            if [ -n "${pids[i]}" ] && ! kill -0 "${pids[i]}" 2>/dev/null; then
                wait "${pids[i]}"; report_run "${arms[i]}" "${seeds[i]}" $?
                pids[i]=""
            fi
            if [ -z "${pids[i]}" ] && [ "${#queue[@]}" -gt 0 ]; then
                item="${queue[0]}"; queue=("${queue[@]:1}")
                a="${item%%:*}"; s="${item#*:}"
                echo "[lora] GPU ${i}: ${a} s${s} -> $(lora_run_name "${a}" "${s}")"
                launch "${a}" "${s}" CUDA_VISIBLE_DEVICES="${i}" N_GPUS=1 TP_SIZE=1 \
                    GPU_MEM_UTIL="${LORA_1GPU_MEM}" RAY_STOP=0 FORCE_CONCURRENT=1 \
                    RAY_TMPDIR="/tmp/rl${i}" &
                pids[i]=$!; arms[i]="${a}"; seeds[i]="${s}"
                sleep "${LORA_STAGGER:-30}"   # stagger start-up (model load, vLLM init)
            fi
        done
        sleep "${LORA_POLL:-20}"
    done
}

eval_stage () {   # <core|followups> <points...>
    local stage=$1 item a s p; shift
    while read -r item; do
        [ -n "${item}" ] || continue
        a="$(cut -d: -f2 <<<"${item}")"; s="$(cut -d: -f3 <<<"${item}")"
        for p in "$@"; do
            if [ "${DRY}" = 1 ]; then
                printf '  EVAL  %-12s s%s  %-5s  %s\n' "${a}" "${s}" "${p}" "$(lora_run_name "${a}" "${s}")"
                continue
            fi
            if ! run_done "${a}" "${s}"; then
                echo "  skip ${a} s${s} ${p}: training not finished"
                continue
            fi
            await_gpus || true
            bash run/eval_lora.sh "${a}" "${s}" "${p}" || echo "[lora] eval FAIL ${a} s${s} ${p}"
        done
    done < <(my_plan "${stage}")
}

# ---------------------------------------------------------------- backbones
# Each plan line carries the model tag; everything that depends on it --
# the run name, the model path, the validation set, the best key, the merge
# base -- is set inside a subshell by lora_enter_backbone, so nothing leaks
# into the next line or back into the 1.5B stages.
backbone_stage () {
    local item a s tag
    while read -r item; do
        [ -n "${item}" ] || continue
        IFS=: read -r _ a s tag <<<"${item}"
        (
            lora_enter_backbone "${tag}" >/dev/null || exit 2
            rn="$(lora_run_name "${a}" "${s}")"
            if run_done "${a}" "${s}"; then
                printf '  done  %-8s %-24s %s\n' "${a}" "${tag}" "${rn}"; exit 0
            fi
            printf '  QUEUE %-8s %-24s %s   [%s; val %s]\n' "${a}" "${tag}" "${rn}" \
                "$(lora_arm_spec "${a}")" "${VAL_PARQUET}"
            [ "${DRY}" = 1 ] && exit 0
            mkdir -p "${LOG_DIR}"
            # Always the whole box: a 7B does not fit one GPU, and a backbone
            # row is compared only with its own arms, which all run this way.
            unset N_GPUS TP_SIZE CUDA_VISIBLE_DEVICES
            gpu_topology
            banner "backbones: ${tag} ${a} -> ${rn}"
            ray stop --force >/dev/null 2>&1 || true
            await_gpus || true
            # The launchers hardcode AIME24 and its key; the backbone's own
            # set and key go in last, where hydra lets them win.
            hydra=( "data.val_files=['${ROOT}/${VAL_PARQUET}']"
                    "++trainer.best_metric_key=${BEST_METRIC_KEY}" )
            launch "${a}" "${s}" N_GPUS="${N_GPUS}" TP_SIZE="${TP_SIZE}" \
                MODEL_PATH="${MODEL_PATH}" -- "${hydra[@]}"
            st=$?
            log="${LOG_DIR}/train-${rn}.log"
            if [ "${st}" -ne 0 ] && log_says_oom "${log}" && [ "${OOM_RETRY:-1}" = 1 ]; then
                echo "[lora] ${tag} ${a}: CUDA OOM -- one retry with OFFLOAD=1 (optimizer on CPU;"
                echo "[lora]   the log records it, and the paper says which rows ran that way)"
                ray stop --force >/dev/null 2>&1 || true
                await_gpus || true
                launch "${a}" "${s}" N_GPUS="${N_GPUS}" TP_SIZE="${TP_SIZE}" \
                    MODEL_PATH="${MODEL_PATH}" OFFLOAD=1 -- "${hydra[@]}"
                st=$?
            fi
            report_run "${a}" "${s}" "${st}"
        )
    done < <(my_plan backbones)
}

backbone_eval_stage () {
    local item a s tag
    while read -r item; do
        [ -n "${item}" ] || continue
        IFS=: read -r _ a s tag <<<"${item}"
        (
            lora_enter_backbone "${tag}" >/dev/null || exit 2
            if [ "${DRY}" = 1 ]; then
                printf '  EVAL  %-8s %-24s final  %s\n' "${a}" "${tag}" "$(lora_run_name "${a}" "${s}")"
                exit 0
            fi
            run_done "${a}" "${s}" || { echo "  skip ${tag} ${a}: training not finished"; exit 0; }
            await_gpus || true
            # Per-backbone log and dump roots: the eval logs are named
            # eval-<arm>-s<seed>.log, which would collide with the 1.5B's.
            LOG_ROOT="${ROOT}/logs/lora/backbones/${tag}" \
            VAL_ROOT="${ROOT}/validation_data/lora/backbones/${tag}" \
            BASE_MODEL="${MODEL_PATH}" bash run/eval_lora.sh "${a}" "${s}" final \
                || echo "[lora] eval FAIL ${tag} ${a}"
        )
    done < <(my_plan backbones)
}

analysis_stage () {
    local p d
    mkdir -p "${RES_DIR}"
    for p in final best; do
        d="${ROOT}/logs/lora/eval-${p}-k${LORA_EVAL_N}"
        compgen -G "${d}/eval-*.log" >/dev/null || { echo "  no ${p} evals yet"; continue; }
        python3 scripts/check_eval_logs.py "${d}"/eval-*.log \
            || { echo "FAIL: incoherent eval logs in ${d}; not tabulating them" >&2; continue; }
        python3 scripts/collect_results.py --logs "${d}" --out "${RES_DIR}/summary_${p}.tsv"
        python3 scripts/eval_paired_se.py \
            --val-data "${ROOT}/validation_data/lora/eval-${p}-k${LORA_EVAL_N}" \
            --passes "avg1x${LORA_EVAL_N}" \
            --out "${RES_DIR}/paired_se_${p}.json" \
            --tex "${RES_DIR}/numbers-pairedse-${p}.tex" \
            --macro-prefix "$( [ "${p}" = best ] && echo Best )"
    done
    # Training curves, plateau table, contrasts, follow-up rows: from the
    # training logs, over the campaign's seeds. The six-benchmark table is the
    # primary endpoint's (the last step).
    local seeds; seeds="$(tr ' ' ',' <<<"${LORA_SEEDS}")"
    python3 scripts/analyze_seeds.py --lora --logs "${LOG_DIR}" --run-prefix "${LORA_PREFIX}" \
        --seeds "${seeds}" --balanced --out "${RES_DIR}" \
        --steps "${STEPS}" --plateau "40:${STEPS}" --eval-point final \
        --eval-table "${RES_DIR}/summary_final.tsv"
    # Each backbone through the same analysis under its own macro prefix
    # (B<stem>), reading the accuracy key its validation set logs.
    local tag stem d acc
    for tag in ${LORA_BACKBONES}; do
        stem="$(lora_backbone_stem "${tag}")" || continue
        d="${ROOT}/logs/lora/backbones/${tag}/eval-final-k${LORA_EVAL_N}"
        mkdir -p "${RES_DIR}/${stem}"
        if compgen -G "${d}/eval-*.log" >/dev/null \
                && python3 scripts/check_eval_logs.py "${d}"/eval-*.log; then
            python3 scripts/collect_results.py --logs "${d}" --out "${RES_DIR}/${stem}/summary_final.tsv"
        fi
        acc="$( (lora_enter_backbone "${tag}" >/dev/null && echo "${BEST_METRIC_KEY}") )"
        python3 scripts/analyze_seeds.py --model-tag "${tag}" --logs "${LOG_DIR}" \
            --run-prefix "${LORA_PREFIX}" --steps "${STEPS}" --plateau "40:${STEPS}" \
            --acc-key "${acc}" --eval-point final \
            --eval-table "${RES_DIR}/${stem}/summary_final.tsv" --out "${RES_DIR}/${stem}" \
            --macro-prefix "B${stem}" --tex-macros "${RES_DIR}/numbers-${stem}.tex" \
            || echo "[lora] analysis of ${tag} failed (no logs yet?)"
    done
    # The secondary endpoint (AIME24-best checkpoint) through the same
    # analysis, every macro prefixed Best so the two cannot be confused.
    if [ -f "${RES_DIR}/summary_best.tsv" ]; then
        python3 scripts/analyze_seeds.py --lora --logs "${LOG_DIR}" --run-prefix "${LORA_PREFIX}" \
            --seeds "${seeds}" --balanced --out "${RES_DIR}/best" \
            --steps "${STEPS}" --plateau "40:${STEPS}" --eval-point best \
            --eval-table "${RES_DIR}/summary_best.tsv" \
            --macro-prefix Best --tex-macros "${RES_DIR}/numbers-best.tex"
    fi
}

preflight () {
    local a need_heads=0 rc=0
    banner "preflight"
    if ! env_preflight "${ROOT}"; then
        echo "FAIL: training stack does not import (bash run/setup_env.sh)"; rc=1
    fi
    # vLLM's LoRA cache calls a private cachetools method that cachetools 6
    # removed, so every LoRA run dies building the rollout engine with
    # "'LoRALRUCache' object has no attribute '_LRUCache__update'" -- a path
    # full fine-tuning never takes. Exercise that exact call rather than
    # checking a version number.
    if ! python3 - <<'PY' 2>/dev/null
import sys
try:
    from vllm.utils import LRUCache
except Exception:
    sys.exit(0)            # no vllm here; env_preflight reports that
c = LRUCache(2)
c.put(1, 1)
try:
    c.touch(1)
except AttributeError:
    sys.exit(1)
PY
    then
        echo "FAIL: vLLM's LoRA cache is broken by cachetools>=6 (the '_LRUCache__update' error)."
        echo "      Fix: pip install \"cachetools<6\""
        rc=1
    fi
    for f in datasets/DAPO-Math-17k.parquet datasets/aime24.parquet; do
        [ -f "${f}" ] || { echo "FAIL: ${f} missing"; rc=1; }
    done
    for a in ${LORA_CORE_ARMS} ${LORA_FOLLOWUP_ARMS}; do lora_needs_heads "${a}" && need_heads=1; done
    if [ "${need_heads}" = 1 ]; then
        for f in "${STEERF_HEADS:-checkpoints/mtp_heads_Qwen2.5-Math-1.5B-paper.pt}" \
                 "${STEERF_CALIB:-checkpoints/mtp_calibration_Qwen2.5-Math-1.5B-paper.json}"; do
            [ -f "${f}" ] || { echo "FAIL: ${f} missing -- mtp needs it" \
                "(REPO=<hub repo> bash run/migrate_pod.sh --import, or drop mtp" \
                "from LORA_CORE_ARMS)"; rc=1; }
        done
    fi
    if is_busy; then
        echo "FAIL: another trainer or eval is running: $(busy_pids | tr '\n' ' ')"; rc=1
    fi
    [ "${rc}" = 0 ] && echo "[preflight] OK (topology ${TOPOLOGY}, shard ${SHARD})"
    return "${rc}"
}

# ---------------------------------------------------------- settings lock
# One "key=value" per line. A key this invocation depends on is compared with
# the recorded value, or recorded if absent. Nothing is written on DRY.
lock_settings () {
    local kv key want have bad=0 keys=()
    if has_stage core || has_stage followups; then keys+=("topology=${TOPOLOGY}"); fi
    if has_stage core || has_stage followups || has_stage backbones; then
        keys+=("rollout_eager=${ROLLOUT_EAGER}")
    fi
    for kv in "${keys[@]}"; do
        key="${kv%%=*}"; want="${kv#*=}"
        have="$(grep -m1 "^${key}=" "${SETTINGS_FILE}" 2>/dev/null | cut -d= -f2-)"
        if [ -z "${have}" ]; then
            [ "${DRY}" = 1 ] && continue
            mkdir -p "$(dirname "${SETTINGS_FILE}")"
            echo "${kv}" >> "${SETTINGS_FILE}"
            echo "[lora] campaign setting recorded: ${kv} -> ${SETTINGS_FILE}"
        elif [ "${have}" != "${want}" ]; then
            echo "REFUSE: ${key}=${want}, but this campaign runs ${key}=${have} (${SETTINGS_FILE})." >&2
            bad=1
        fi
    done
    if [ "${bad}" = 1 ]; then
        if [ "${CAMPAIGN_SETTINGS_OVERRIDE:-0}" = 1 ]; then
            echo "[lora] CAMPAIGN_SETTINGS_OVERRIDE=1: continuing with mixed settings." >&2
            return 0
        fi
        echo "        Every arm of a table must share these. Start from the recorded values," >&2
        echo "        or archive the campaign's runs and that file to start a new one." >&2
        return 1
    fi
    return 0
}

# -------------------------------------------------------------------- main
banner "LoRA campaign: stages [${STAGES}]  topology ${TOPOLOGY}  rollout_eager ${ROLLOUT_EAGER}  shard ${SHARD}  keep_every ${KEEP_EVERY}"
lock_settings || exit 2
if has_stage preflight && [ "${DRY}" != 1 ]; then
    preflight || { echo "STOP: preflight failed" >&2; exit 1; }
fi
has_stage core           && { banner "core";           train_stage core; }
has_stage eval-core      && { banner "eval-core";      eval_stage core final best; }
has_stage followups      && { banner "followups";      train_stage followups; }
has_stage eval-followups && { banner "eval-followups"; eval_stage followups final; }
has_stage backbones      && { banner "backbones";      backbone_stage; }
has_stage eval-backbones && { banner "eval-backbones"; backbone_eval_stage; }
if has_stage analysis && [ "${DRY}" != 1 ]; then
    banner "analysis"; analysis_stage
fi
echo "[lora] finished stages: ${STAGES}"
