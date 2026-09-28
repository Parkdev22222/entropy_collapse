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
# Plain GRPO. The baseline every other arm is measured against and the one
# thing the project does not have.
#
# STEER's paper claims STEER > GRPO. Every arm we have run so far is a STEER
# variant (lambda=0 is stock STEER, not GRPO), so that claim is unverified in
# our setup and the ablation ladder has no floor:
#
#     GRPO            <- this script.  no token weighting at all
#     STEER           run_steerf.sh with STEERF_LAM=0
#     +tree, uniform  run_uniform_ablation.sh ARM=uniform
#     +tree, permuted run_uniform_ablation.sh ARM=permuted
#     STEER-F         run_uniform_ablation.sh ARM=signed
#
# What "plain GRPO" means here, exactly: policy_loss.loss_mode=vanilla, which
# routes to core_algos.compute_policy_loss (dual-clip PPO on GRPO advantages)
# instead of compute_policy_loss_with_entropy. No Omega, no [0.7, 1.0] band, no
# per-token weight. dp_actor.py:771 is the branch. `actor/entropy` is still
# logged: fsdp_workers.py:725 computes it in the old_log_prob pass regardless
# of loss_mode, so the entropy-vs-accuracy comparison stays complete.
#
# Rollout is the stock one. The tree rollout activates only when
# rollout.steerf_tree_depths is set (vllm_rollout_spmd.py:134) and this script
# never sets it -- a baseline must not inherit the treatment's data generation.
#
# PARITY -- every value below is pinned to the signed run
# (steer-f-Qwen2.5-Math-1.5B-s1-tree-rollout, read off its config dump) so the
# only differences are the three above. Do not "improve" any of them here
# without changing the other arms too:
#
#     512 / 32 / 8      train / mini / micro-per-gpu batch
#     1024 / 3072       max prompt / response
#     lr 1e-6           constant, no warmup
#     n=8               rollouts per prompt, temperature 1.0, top_p 1.0
#     clip 0.2 / 0.28   dual-clip c = 10.0
#     entropy_coeff 0   use_kl_loss False, use_kl_in_reward False
#     val n=1           on aime24's 32-replica parquet -> acc/mean@32
#     test/save 10      save_after 80
#
# 110 steps because that is where every other arm stopped: 3.3 epochs of
# DAPO-Math-17k (17,388 prompts after the overlong filter / 512 = 33 steps per
# epoch), and the 140-step rank run peaks at 110 and declines after.
#
# Usage
#   bash run/run_grpo.sh                       # 1.5B, seed 1, 110 steps
#   SEED=2 bash run/run_grpo.sh                # second seed
#   STEPS=200 bash run/run_grpo.sh             # longer
#   DRY_RUN=1 bash run/run_grpo.sh             # print the command, run nothing
#
# Then evaluate the checkpoint:
#   MODEL_PATH=checkpoints/STEER-F/grpo-Qwen2.5-Math-1.5B-s1/global_step_110/actor/huggingface \
#     ARM=grpo bash run/eval_subset.sh
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
# Not expandable_segments:True -- vLLM sleep mode asserts against it
# (pytorch#147851). See the same note in run_steerf.sh.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-garbage_collection_threshold:0.8}

project_name='STEER-F'
train_path=${STEER_ROOT}/datasets/DAPO-Math-17k.parquet
test_path=${STEER_ROOT}/datasets/aime24.parquet
model_path=${MODEL_PATH:-"Qwen/Qwen2.5-Math-1.5B"}
model_tag=$(basename "${model_path}")

SEED=${SEED:-1}
STEPS=${STEPS:-110}
TRAIN_BS=${TRAIN_BS:-512}
MINI_BS=${MINI_BS:-32}
RESP_LEN=${RESP_LEN:-3072}
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:-10}
# Matches the other arms: checkpoint only the back half. The multi-benchmark
# eval needs 80/90/100/110, and a full checkpoint is ~7 GiB of hf_model.
SAVE_AFTER=${SAVE_AFTER:-80}
LOGP_MBS=${LOGP_MBS:-4}
OFFLOAD=${OFFLOAD:-0}
if [ "${OFFLOAD}" = "1" ]; then PARAM_OFF=True; OPT_OFF=True; else PARAM_OFF=False; OPT_OFF=False; fi
save_contents=${SAVE_CONTENTS:-"['hf_model']"}
RESUME_MODE=${RESUME_MODE:-auto}

RUN_NAME=${RUN_NAME:-"grpo-${model_tag}-s${SEED}"}
LOG_DIR=${LOG_DIR:-${STEER_ROOT}/logs/experiments}
LOG=${LOG:-${LOG_DIR}/train-${RUN_NAME}.log}
CKPT_DIR="${STEER_ROOT}/checkpoints/${project_name}/${RUN_NAME}"

# ---- guards ----------------------------------------------------------------
# A second trainer on the same GPUs OOMs both. Same check as
# run_uniform_ablation.sh: pgrep -f matches any command line mentioning the
# string, so confirm the pid is a real interpreter and not a shell.
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
    echo "        Wait for it to finish -- this baseline needs the same GPUs."
    echo "        FORCE_CONCURRENT=1 overrides, only if you know they are free."
    exit 1
fi

if [ ! -f "${train_path}" ]; then
    echo "REFUSE: training data not found at ${train_path}" >&2
    exit 1
fi

# RESUME_MODE=auto would silently continue an existing run rather than start a
# fresh baseline. Make that an explicit choice.
if [ -d "${CKPT_DIR}" ] && [ "${RESUME:-0}" != "1" ]; then
    echo "REFUSE: ${CKPT_DIR} already exists."
    echo "        RESUME_MODE=${RESUME_MODE} would continue it. Move it aside,"
    echo "        pick a different RUN_NAME, or set RESUME=1 if you really are"
    echo "        continuing this run after a crash."
    exit 1
fi

mkdir -p "${LOG_DIR}"

echo "=============================================================="
echo " arm            grpo   (policy_loss.loss_mode=vanilla, no token weighting)"
echo " rollout        stock  (no steerf_tree_depths -> no tree)"
echo " run name       ${RUN_NAME}"
echo " model          ${model_path}"
echo " seed / gpus    ${SEED} / ${N_GPUS} (tp=${TP_SIZE}, mem_util=${GPU_MEM_UTIL})"
echo " batch          ${TRAIN_BS} / ${MINI_BS} / 8   resp_len=${RESP_LEN}  n=8"
echo " steps          ${STEPS}   (test every ${TEST_FREQ}, save every ${SAVE_FREQ} after ${SAVE_AFTER})"
echo " train data     ${train_path}"
echo " val data       ${test_path}"
echo " log            ${LOG}"
echo "=============================================================="

train_files="['$train_path']"
test_files="['$test_path']"

# rollout_data_dir=null: writing rollouts leaked ~300 GB of host RAM and killed
# the first tree run at step 10. The other arms have it nulled too.
ARGS=(
    algorithm.adv_estimator=grpo
    data.train_files="$train_files"
    data.val_files="$test_files"
    data.train_batch_size="${TRAIN_BS}"
    data.max_prompt_length=1024
    data.max_response_length="${RESP_LEN}"
    data.filter_overlong_prompts=True
    data.truncation=left
    ++data.seed="${SEED}"
    actor_rollout_ref.model.path="$model_path"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BS}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_c=10.0
    # THE line that makes this GRPO and not STEER. dp_actor.py:771.
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
    ++actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.fsdp_config.param_offload="${PARAM_OFF}"
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${OPT_OFF}"
    actor_rollout_ref.actor.checkpoint.save_contents="${save_contents}"
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE:-4}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL:-0.6}"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOGP_MBS}"
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOGP_MBS}"
    actor_rollout_ref.ref.fsdp_config.param_offload="${PARAM_OFF}"
    algorithm.use_kl_in_reward=False
    trainer.critic_warmup=0
    "trainer.logger=['console','tensorboard']"
    trainer.project_name="$project_name"
    trainer.experiment_name="$RUN_NAME"
    trainer.n_gpus_per_node="${N_GPUS:-8}"
    trainer.nnodes=1
    trainer.save_freq="${SAVE_FREQ}"
    trainer.test_freq="${TEST_FREQ}"
    trainer.total_epochs=10
    trainer.total_training_steps="${STEPS}"
    trainer.resume_mode="${RESUME_MODE}"
    ++trainer.rollout_data_dir=null
    ++trainer.save_best_only=False
    ++trainer.delete_old_best_checkpoint=True
    ++trainer.save_after="${SAVE_AFTER}"
    ++trainer.best_metric_key=val-core/aime_2024_dapo_boxed/acc/mean@32
)

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY RUN -- would execute:"
    printf '  python3 -m verl.trainer.main_ppo \\\n'
    printf '      %q \\\n' "${ARGS[@]}" "$@"
    echo "  > ${LOG} 2>&1"
    exit 0
fi

ray stop --force >/dev/null 2>&1 || true
sleep 5

# Write the exact invocation into the log before running. `set -x` would send
# the trace to this shell's stderr, i.e. the terminal, not to ${LOG} -- which
# is how run_steerf.sh gets it, because its caller redirects the whole script.
# Months later the recorded command line is the only thing that identifies an
# arm from its log, so it has to be inside the file.
{
    echo "### run_grpo.sh  arm=grpo  run_name=${RUN_NAME}  seed=${SEED}  steps=${STEPS}"
    echo "### model=${model_path}  gpus=${N_GPUS} tp=${TP_SIZE}"
    printf '+ python3 -m verl.trainer.main_ppo'
    printf ' %q' "${ARGS[@]}" "$@"
    printf '\n'
} > "${LOG}"

python3 -m verl.trainer.main_ppo "${ARGS[@]}" "$@" >> "${LOG}" 2>&1
status=$?

echo "[grpo] exit ${status} -> ${LOG}"
echo "  loss_mode : $(grep -o "'loss_mode': '[a-z_]*'" "${LOG}" | tail -1)   (expect 'vanilla')"
echo "  tree      : $(grep -c 'steerf-tree' "${LOG}") tree lines   (expect 0)"
echo "  last step : $(grep -o 'step:[0-9]* - global_seqlen' "${LOG}" | tail -1)"
echo "  val pts   : $(grep -c 'val-core/aime_2024_dapo_boxed/acc/mean@32:' "${LOG}")"
echo
echo "AIME24 curve:"
grep -o 'step:[0-9]* .*val-core/aime_2024_dapo_boxed/acc/mean@32:[0-9.]*' "${LOG}" |
    sed -E 's/^(step:[0-9]+).*acc\/mean@32:([0-9.]+)$/  \1  acc@32 \2/' || true
echo
echo "Next: evaluate the checkpoint on the comparison benchmarks --"
echo "  MODEL_PATH=${CKPT_DIR}/global_step_${STEPS}/actor/huggingface \\"
echo "    ARM=grpo SEED=${SEED} bash run/eval_subset.sh"
exit ${status}
