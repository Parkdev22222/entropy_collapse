#!/usr/bin/env bash
# Paper-protocol evaluation for a STEER-F checkpoint.
#
# The paper reports, per model:
#   avg@32  on AIME24, AIME25, AMC23         (small sets, 32 samples each)
#   avg@1   on MATH500, Minerva, OlympiadBench (+ GSM8K in the release)
# with sampling temperature 1.0, top_p 0.7, max response 3072 — the same
# values as the authors' eval.sh. This script therefore runs verl's val-only
# mode twice: pass A with val_kwargs.n=32 on the three avg@32 sets, pass B
# with val_kwargs.n=1 on the rest. Read the numbers from the log lines
#   val-core/<dataset>/acc/mean@32   (pass A)
#   val-core/<dataset>/acc/mean@1    (pass B)
# and paste them into the corresponding table column.
#
#   MODEL_PATH=/path/to/checkpoint/hf_model bash run/eval_steerf.sh
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${STEER_ROOT}:$PYTHONPATH"

export PYTHONHASHSEED=42
export PYTORCH_SEED=42
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

model_path=${MODEL_PATH:?set MODEL_PATH to the checkpoint to evaluate}
d=${STEER_ROOT}/datasets
train_files="['$d/DAPO-Math-17k.parquet']"

# pass A: avg@32 benchmarks       pass B: avg@1 benchmarks
files_at32="['$d/aime24.parquet', '$d/aime25.parquet', '$d/amc23.parquet']"
files_at1="['$d/math500.parquet', '$d/minerva_math.parquet', '$d/olympiadbench.parquet', '$d/gsm8k_test.parquet']"

run_eval () {
    local test_files="$1"; local val_n="$2"; local tag="$3"
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files="$train_files" \
        data.val_files="$test_files" \
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
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.val_kwargs.n=${val_n} \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger="['console']" \
        trainer.project_name=STEERF-eval \
        trainer.experiment_name=eval-${tag}-$(date +%Y%m%d_%H%M%S) \
        trainer.n_gpus_per_node=${N_GPUS:-8} \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=1 \
        trainer.total_epochs=1 \
        trainer.val_only=True \
        trainer.resume_mode=disable
}

run_eval "$files_at32" 32 "avg32"
run_eval "$files_at1"   1 "avg1"

echo "Done. Table numbers:"
echo "  avg@32 -> val-core/<aime24|aime25|amc23>/acc/mean@32 in the avg32 pass"
echo "  avg@1  -> val-core/<math500|minerva_math|olympiadbench|gsm8k>/acc/mean@1 in the avg1 pass"
