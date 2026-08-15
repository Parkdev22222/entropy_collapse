#!/usr/bin/env bash
# STEER-F training run. Derived from STEER's run/run_linear.sh; every line that
# differs from upstream is marked  # STEER-F.
#
# Arms (set ARM, or override STEERF_LAM directly):
#   ARM=grpo     -> loss_mode=vanilla                       (baseline)
#   ARM=steer    -> loss_mode=entropy_control, lambda=0     (== stock STEER, bit-identical)
#   ARM=steerf   -> loss_mode=entropy_control, lambda>0     (STEER-F)
#
#   ARM=steerf STEERF_LAM=0.5 MODEL_PATH=Qwen/Qwen2.5-Math-7B ./run/run_steerf.sh
#
# The lambda=0 arm goes through exactly the same code path as lambda>0 and is
# verified bit-identical to upstream by tests/test_lambda_zero_equiv.py, so the
# STEER-F delta is a controlled comparison rather than a cross-codebase one.
set -x

export PYTHONHASHSEED=42
export PYTORCH_SEED=42
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEERF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STEER_ROOT="${STEER_ROOT:-${STEERF_ROOT}/third_party/STEER}"

if [ ! -d "${STEER_ROOT}/verl" ]; then
  echo "STEER checkout not found at ${STEER_ROOT}."
  echo "Run: bash ${STEERF_ROOT}/scripts/setup_steer.sh"
  exit 1
fi

# STEER-F: steer_f must precede verl so `from steer_f...` resolves inside the
# patched core_algos.
export PYTHONPATH="${STEERF_ROOT}:${STEER_ROOT}:$PYTHONPATH"
echo "Current VERL path:"
python3 -c "import verl; print(verl.__file__)"
python3 -c "import steer_f; print('steer_f', steer_f.__version__)"

ARM=${ARM:-steerf}
project_name=${PROJECT_NAME:-'STEER-F'}
train_path=${TRAIN_PATH:-${STEER_ROOT}/datasets/DAPO-Math-17k.parquet}
test_path=${TEST_PATH:-${STEER_ROOT}/datasets/aime24.parquet}
model_path=${MODEL_PATH:-"Qwen/Qwen2.5-Math-7B"}

# ---- STEER-F hyper-parameters (Phase 1 outputs; defaults are the plan's) ----
STEERF_LAM=${STEERF_LAM:-0.5}       # lambda; 0.0 == stock STEER
STEERF_ETA=${STEERF_ETA:-1.0}
STEERF_CLIP_C=${STEERF_CLIP_C:-1.0}
STEERF_NORM=${STEERF_NORM:-scale}   # see docs/steer_code_map.md §3.3 for why not "z"
STEERF_KAPPA=${STEERF_KAPPA:-4}
STEERF_GAMMA_H=${STEERF_GAMMA_H:-0.85}
STEERF_BASELINE=${STEERF_BASELINE:-sibling}
STEERF_BETA_MTP=${STEERF_BETA_MTP:-0.05}
STEERF_HEADS=${STEERF_HEADS:-${STEERF_ROOT}/checkpoints/mtp_heads.pt}
STEERF_CALIB=${STEERF_CALIB:-${STEERF_ROOT}/checkpoints/mtp_calibration.json}

case "$ARM" in
  grpo)   loss_mode=vanilla;         STEERF_LAM=0.0 ;;
  steer)  loss_mode=entropy_control; STEERF_LAM=0.0 ;;
  steerf) loss_mode=entropy_control ;;
  *) echo "unknown ARM=$ARM (want grpo|steer|steerf)"; exit 1 ;;
esac

run_name="${ARM}"
[ "$ARM" = "steerf" ] && run_name="${run_name}-lam${STEERF_LAM}-k${STEERF_KAPPA}-g${STEERF_GAMMA_H}"
run_name="${run_name}_$(date +"%Y%m%d_%H%M%S")"

export WANDB_INIT_TIMEOUT=300
export WANDB_TIMEOUT=300
export WANDB_RETRY_DELAY=60
export WANDB_MAX_RETRIES=10

save_contents="['hf_model']"
train_files="['$train_path']"
test_files="['$test_path']"

# STEER-F: only pass the entropy_control / STEER-F knobs when that loss is in
# use — hydra rejects `+key=` overrides that the vanilla path never reads.
steer_args=()
if [ "$loss_mode" = "entropy_control" ]; then
  steer_args+=(
    +actor_rollout_ref.actor.policy_loss.token_weight_min=${TOKEN_WEIGHT_MIN:-0.8}
    +actor_rollout_ref.actor.policy_loss.token_weight_max=${TOKEN_WEIGHT_MAX:-1.2}
    +actor_rollout_ref.actor.policy_loss.linear=${LINEAR:-True}
    +actor_rollout_ref.actor.policy_loss.entropy_control_mode=${ENTROPY_CONTROL_MODE:-symmetric}
    # ---- STEER-F ----
    +actor_rollout_ref.actor.policy_loss.steerf_lam=${STEERF_LAM}
    +actor_rollout_ref.actor.policy_loss.steerf_eta=${STEERF_ETA}
    +actor_rollout_ref.actor.policy_loss.steerf_clip_c=${STEERF_CLIP_C}
    +actor_rollout_ref.actor.policy_loss.steerf_norm=${STEERF_NORM}
    +actor_rollout_ref.actor.policy_loss.steerf_kappa=${STEERF_KAPPA}
    +actor_rollout_ref.actor.policy_loss.steerf_gamma_h=${STEERF_GAMMA_H}
    +actor_rollout_ref.actor.policy_loss.steerf_baseline=${STEERF_BASELINE}
    +actor_rollout_ref.actor.policy_loss.steerf_beta_mtp=${STEERF_BETA_MTP}
    +actor_rollout_ref.actor.policy_loss.steerf_heads_path=${STEERF_HEADS}
    +actor_rollout_ref.actor.policy_loss.steerf_calib_path=${STEERF_CALIB}
  )
fi

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
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    "${steer_args[@]}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE:-4} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.actor.checkpoint.save_contents=${save_contents} \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.rollout_data_dir=${STEERF_ROOT}/rollout_data/$project_name/$run_name \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name=$project_name \
    trainer.experiment_name=$run_name \
    trainer.n_gpus_per_node=${N_GPUS:-8} \
    trainer.nnodes=${NNODES:-1} \
    trainer.save_freq=10 \
    trainer.test_freq=1 \
    trainer.total_epochs=${TOTAL_EPOCHS:-10} \
    trainer.resume_mode=disable \
    +trainer.save_best_only=False \
    +trainer.delete_old_best_checkpoint=True \
    +trainer.save_after=80 \
    +trainer.best_metric_key=val-core/math_dapo/acc/mean@32 \
    "$@"
# Trailing "$@" lets wrappers (run_steerf_small.sh) append hydra overrides;
# later assignments win in hydra, so these override the defaults above.
