#!/usr/bin/env bash
# Collect base-policy rollouts for MTP head warm-up (STEER-F Phase 1, step 0).
#
# Runs plain GRPO for a handful of steps (default 8 -> 8 x 512 x 8 = 32k
# sequences) with trainer.rollout_data_dir enabled, then converts verl's dump
# format into the JSONL that scripts/phase1_warmup_heads.py expects. The
# policy barely moves in 8 steps at lr 1e-6, so these are effectively
# base-policy samples on the training distribution — exactly what the heads
# must model. Uses the same data/lengths/sampling as run_steerf.sh.
set -xe

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${STEER_ROOT}:$PYTHONPATH"

model_path=${MODEL_PATH:-"Qwen/Qwen2.5-Math-7B"}
model_tag=$(basename "${model_path}")
N_STEPS=${N_STEPS:-8}
dump_dir=${STEER_ROOT}/rollout_data/warmup/${model_tag}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="['${STEER_ROOT}/datasets/DAPO-Math-17k.parquet']" \
    data.val_files="['${STEER_ROOT}/datasets/aime24.parquet']" \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE:-4} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    algorithm.use_kl_in_reward=False \
    trainer.rollout_data_dir=${dump_dir} \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.project_name=STEERF-warmup \
    trainer.experiment_name=warmup-${model_tag} \
    trainer.n_gpus_per_node=${N_GPUS:-8} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_training_steps=${N_STEPS} \
    trainer.total_epochs=1 \
    trainer.resume_mode=disable

python3 ${STEER_ROOT}/scripts/phase0_collect_rollouts.py \
    "${dump_dir}" \
    --out ${STEER_ROOT}/rollout_data/warmup/${model_tag}/rollouts.jsonl \
    --min-chars 32

echo "warmup rollouts: ${STEER_ROOT}/rollout_data/warmup/${model_tag}/rollouts.jsonl"
