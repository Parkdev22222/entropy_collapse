#!/usr/bin/env bash
# Copyright 2026 STEER-F authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ---------------------------------------------------------------------------
# The cheap cut of run/eval_steerf.sh: AIME25, AMC23, MATH500 only.
#
# Why these three. Every claim we have rests on AIME24, which is 30 problems
# (SE ~= 0.027 on the mean), so no gap we have measured reaches conventional
# significance on it. These three add 30 + 40 + 500 = 570 problems for about a
# third of the full protocol's cost, and they span the range that matters:
#
#     AIME25    30 problems, avg@32   same difficulty as AIME24, held out
#     AMC23     40 problems, avg@32   easier, catches "only helps at the top"
#     MATH500  500 problems, avg@1    the one set large enough to move an SE
#
# MATH500 alone is worth more statistically than AIME24 and AIME25 combined.
# Run run/eval_steerf.sh instead when you need the full paper row (adds
# AIME24, Minerva, OlympiadBench, GSM8K) -- collect_results.py only fills
# Avg(math6) when all six are present, so this script always leaves it "-".
#
# Two passes, because verl names the metric after samples-per-prompt and the
# avg@32 parquets already hold 32 replicas of every problem (aime25: 30 -> 960
# rows, amc23: 40 -> 1280). val_kwargs.n=1 over a 32x parquet is what yields
# acc/mean@32; raising n would multiply the cost by n AND rename the metric out
# from under collect_results.py.
#
#     pass A   aime25 + amc23   val_kwargs.n=1  ->  val-core/<ds>/acc/mean@32
#     pass B   math500          val_kwargs.n=1  ->  val-core/math500/acc/mean@1
#
# Sampling matches the paper and our training-time validation exactly:
# temperature 1.0, top_p 0.7, max response 3072.
#
# Usage
#   MODEL_PATH=checkpoints/STEER-F/grpo-Qwen2.5-Math-1.5B-s1/global_step_110/actor/huggingface \
#     ARM=grpo bash run/eval_subset.sh
#
#   ARM=signed   MODEL_PATH=.../steer-f-...-tree-rollout/global_step_110/actor/huggingface  ...
#   ARM=permuted MODEL_PATH=.../-permuted/global_step_110/actor/huggingface                 ...
#
#   WITH_AIME24=1 ...   also run aime24 in pass A (comparable to training-time val)
#   DRY_RUN=1 ...       print the two commands, run nothing
#
# The log name is load-bearing: collect_results.py:54 parses arm and seed out
# of `eval-<arm>-s<seed>.log` and SILENTLY SKIPS anything else -- a name like
# eval-grpo-step110.log costs you the whole evaluation and reports only
# "no parsable eval logs". Both passes tee into one file; _find takes the last
# match per key, and the two passes share no keys, so there is no collision.
#
# After every arm is evaluated:
#   python scripts/collect_results.py --logs logs/experiments --out results/summary.tsv
# ---------------------------------------------------------------------------
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=run/_gpu_defaults.sh
. "${SCRIPT_DIR}/_gpu_defaults.sh"

export PYTHONPATH="${STEER_ROOT}:${PYTHONPATH:-}"
export PYTHONHASHSEED=42
export PYTORCH_SEED=42
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

model_path=${MODEL_PATH:?set MODEL_PATH to the checkpoint to evaluate (the actor/huggingface dir)}
ARM=${ARM:?set ARM to the arm name that goes in the results table, e.g. grpo / steer / signed / uniform / permuted}
SEED=${SEED:-1}

# collect_results.py's regex is `eval-(.+)-s(\d+)\.log$`. Fail loudly here
# rather than after three hours of generation.
case "${ARM}" in
    *[!A-Za-z0-9_.-]*) echo "REFUSE: ARM='${ARM}' has characters that break the log-name parse" >&2; exit 2 ;;
esac
case "${SEED}" in
    ''|*[!0-9]*) echo "REFUSE: SEED='${SEED}' must be digits -- collect_results.py parses it as \\d+" >&2; exit 2 ;;
esac

d=${STEER_ROOT}/datasets
LOG_DIR=${LOG_DIR:-${STEER_ROOT}/logs/experiments}
LOG=${LOG:-${LOG_DIR}/eval-${ARM}-s${SEED}.log}
RESP_LEN=${RESP_LEN:-3072}
LOGP_MBS=${LOGP_MBS:-8}
# _gpu_defaults.sh clamps GPU_MEM_UTIL to 0.35 on a single card because a
# TRAINING run keeps weights + grads + AdamW state resident alongside vLLM's
# pool. val_only holds none of that, so the clamp would just starve the KV
# cache and make the eval several times slower. eval_steerf.sh:62 hardcodes
# 0.8 for the same reason; match it, and keep a knob for odd boxes.
EVAL_GPU_MEM_UTIL=${EVAL_GPU_MEM_UTIL:-0.8}

# DAPO-Math-17k is never read in val_only mode, but verl's config validation
# still wants a train_files entry.
train_files="['$d/DAPO-Math-17k.parquet']"

if [ "${WITH_AIME24:-0}" = "1" ]; then
    files_at32="['$d/aime24.parquet', '$d/aime25.parquet', '$d/amc23.parquet']"
    at32_label="aime24 + aime25 + amc23"
else
    files_at32="['$d/aime25.parquet', '$d/amc23.parquet']"
    at32_label="aime25 + amc23"
fi
files_at1="['$d/math500.parquet']"

# ---- guards ----------------------------------------------------------------
if [ ! -d "${model_path}" ]; then
    echo "REFUSE: MODEL_PATH is not a directory: ${model_path}" >&2
    echo "        It must be the HF model dir, i.e. <ckpt>/global_step_<N>/actor/huggingface" >&2
    exit 1
fi
if [ ! -f "${model_path}/config.json" ]; then
    echo "REFUSE: no config.json under ${model_path} -- that is not an HF model dir." >&2
    echo "        Checkpoints saved with save_contents=['hf_model'] put it in" >&2
    echo "        global_step_<N>/actor/huggingface, not in global_step_<N>." >&2
    exit 1
fi
for p in "$d/aime25.parquet" "$d/amc23.parquet" "$d/math500.parquet"; do
    [ -f "$p" ] || { echo "REFUSE: missing dataset ${p}" >&2; exit 1; }
done

# vLLM will take the GPUs a live trainer is using. Same check as run_grpo.sh.
trainer_running() {
    local pid comm
    for pid in $(pgrep -f main_ppo 2>/dev/null); do
        [ "${pid}" = "$$" ] && continue
        comm=$(cat "/proc/${pid}/comm" 2>/dev/null) || continue
        case "${comm}" in
            python*|pt_main_thread*|ray*) return 0 ;;
        esac
    done
    return 1
}
if [ "${FORCE_CONCURRENT:-0}" != "1" ] && trainer_running; then
    echo "REFUSE: a trainer is already running (python ... main_ppo)."
    echo "        eval_subset.sh grabs its own GPUs and will OOM it."
    echo "        FORCE_CONCURRENT=1 overrides, only if you know they are free."
    exit 1
fi

mkdir -p "${LOG_DIR}"

echo "=============================================================="
echo " arm / seed     ${ARM} / ${SEED}"
echo " model          ${model_path}"
echo " pass A @32     ${at32_label}"
echo " pass B @1      math500"
echo " gpus           ${N_GPUS} (tp=${TP_SIZE}, mem_util=${EVAL_GPU_MEM_UTIL})"
echo " log            ${LOG}"
echo "=============================================================="

run_eval () {
    local test_files="$1"; local val_n="$2"; local tag="$3"
    # val_only=True + resume_mode=disable + save_freq=-1 is what keeps this from
    # touching the checkpoint it is reading. Do not relax any of the three.
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files="$train_files" \
        data.val_files="$test_files" \
        data.train_batch_size=512 \
        data.max_prompt_length=1024 \
        data.max_response_length="${RESP_LEN}" \
        data.filter_overlong_prompts=True \
        data.truncation='left' \
        actor_rollout_ref.model.path="$model_path" \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOGP_MBS}" \
        actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE:-4}" \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization="${EVAL_GPU_MEM_UTIL}" \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.val_kwargs.n="${val_n}" \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOGP_MBS}" \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger="['console']" \
        trainer.project_name=STEERF-eval \
        trainer.experiment_name="eval-${ARM}-s${SEED}-${tag}" \
        trainer.n_gpus_per_node="${N_GPUS:-8}" \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=1 \
        trainer.total_epochs=1 \
        trainer.val_only=True \
        trainer.resume_mode=disable
}

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY RUN -- would run two passes into ${LOG}:"
    echo "  A) val_files=${files_at32}  val_kwargs.n=1  -> acc/mean@32"
    echo "  B) val_files=${files_at1}   val_kwargs.n=1  -> acc/mean@1"
    exit 0
fi

ray stop --force >/dev/null 2>&1 || true
sleep 5

# Each pass gets its own status. A single status for the whole block would be
# pass B's alone, so a failed pass A would be reported as success and the
# AIME25/AMC23 columns would silently come out empty in summary.tsv.
echo "### eval_subset.sh  arm=${ARM} seed=${SEED}  MODEL_PATH=${model_path}" > "${LOG}"

echo "### pass A (avg@32): ${at32_label}" >> "${LOG}"
run_eval "$files_at32" 1 "avg32" >> "${LOG}" 2>&1
status_a=$?
echo "[eval-${ARM}] pass A (avg@32) exit ${status_a}"

echo "### pass B (avg@1): math500" >> "${LOG}"
run_eval "$files_at1" 1 "avg1" >> "${LOG}" 2>&1
status_b=$?
echo "[eval-${ARM}] pass B (avg@1) exit ${status_b}"

status=$(( status_a > status_b ? status_a : status_b ))
echo "[eval-${ARM}] exit ${status} -> ${LOG}"
echo
echo "Sanity -- these three lines must all be present:"
grep -o "'val_only': True" "${LOG}" | tail -1
grep -o "'resume_mode': 'disable'" "${LOG}" | tail -1
grep -o "'save_freq': -1" "${LOG}" | tail -1
echo
echo "Numbers:"
for k in "aime_2025_dapo_boxed/acc/mean@32" "amc2023_dapo_boxed/acc/mean@32" "math500/acc/mean@1"; do
    printf '  %-36s %s\n' "${k}" "$(grep -o "val-core/${k}:[0-9.]*" "${LOG}" | tail -1)"
done
if [ "${WITH_AIME24:-0}" = "1" ]; then
    printf '  %-36s %s\n' "aime_2024_dapo_boxed/acc/mean@32" \
        "$(grep -o 'val-core/aime_2024_dapo_boxed/acc/mean@32:[0-9.]*' "${LOG}" | tail -1)"
fi
echo
echo "A mean@1 where mean@32 was expected means the parquet lost its 32 replicas -- stop and check."
echo
echo "Once every arm is done:"
echo "  python scripts/collect_results.py --logs ${LOG_DIR} --out results/summary.tsv"
echo "  (Avg(math6) stays '-' until Minerva/Olympiad/AIME24 are added by run/eval_steerf.sh)"
exit ${status}
