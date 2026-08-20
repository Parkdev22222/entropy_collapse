#!/usr/bin/env bash
# STEER-F on the paper's CODE-GENERATION track (Table 5, LCB-v5 rows).
#
# Paper protocol (arXiv:2510.10150v4 §6.1, Appendix E):
#   models    : Qwen2.5-Coder-3B / 7B / 14B  (one run per model = one Table 5 cell)
#   training  : ArcherCodeR (6,753 tasks; DeepCoder/CodeContests/CodeForces
#               with regenerated tests) — build with scripts/prepare_code_data.py
#   recipe    : identical to math (512/32/8, n=8, lr 1e-6, prompt 1024 /
#               response 3072, no KL, temp 1.0 / top-p 1.0, <=200 steps,
#               checkpoint every 10 steps)
#   evaluation: LiveCodeBench v5 (279 problems), avg@4, temp 1.0 / top_p 0.7
#   reward    : prime_code (APPS-style stdin/stdout tests) via
#               data_source="codecontests"; set SANDBOX_FUSION_URL to route
#               through a sandbox instead of local execution.
#
# The paper's code-EDITING track (internal 51k corpus + Zeta) is not
# reproducible from public data and is out of scope.
set -x

export PYTHONHASHSEED=42
export PYTORCH_SEED=42
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${STEER_ROOT}:$PYTHONPATH"
echo "Current VERL path:"
python3 -c "import verl; print(verl.__file__)"
python3 -c "import steer_f; print(steer_f.__file__)"

project_name='STEER-F-code'
train_path=${STEER_ROOT}/datasets/archercoder.parquet
test_path=${STEER_ROOT}/datasets/livecodebench_v5.parquet
model_path=${MODEL_PATH:-"Qwen/Qwen2.5-Coder-7B"}
if [ ! -f "${train_path}" ] || [ ! -f "${test_path}" ]; then
    echo "FATAL: code parquets missing. Run scripts/prepare_code_data.py --archer --lcb first."
    exit 1
fi
model_tag=$(basename "${model_path}")

# ---- STEER-F knobs (everything else stays at the paper's values) ----
STEERF_LAM=${STEERF_LAM:-0.25}            # 0 = stock STEER, bit-identical
STEERF_KAPPA=${STEERF_KAPPA:-2}           # take from warmup_and_validate.sh output
STEERF_GAMMA_H=${STEERF_GAMMA_H:-0.7}
STEERF_CLIP_C=${STEERF_CLIP_C:-1.0}
STEERF_NORM=${STEERF_NORM:-scale}
STEERF_BASELINE=${STEERF_BASELINE:-sibling}   # or: group   (ablation A5)
STEERF_APPLY=${STEERF_APPLY:-weight}          # metric | weight | branch
STEERF_MAPPING=${STEERF_MAPPING:-minmax}      # minmax | winsor | rank
STEERF_WINSOR_Q=${STEERF_WINSOR_Q:-0.01}
STEERF_FORECAST=${STEERF_FORECAST:-mtp}       # mtp | oracle (free control arm)
STEERF_HEADS=${STEERF_HEADS:-${STEER_ROOT}/checkpoints/mtp_heads_${model_tag}.pt}
STEERF_CALIB=${STEERF_CALIB:-${STEER_ROOT}/checkpoints/mtp_calibration_${model_tag}.json}

if [ "${STEERF_LAM}" != "0" ] && [ "${STEERF_FORECAST}" = "mtp" ] && [ ! -f "${STEERF_HEADS}" ]; then
    echo "FATAL: steerf_lam=${STEERF_LAM} needs MTP heads at ${STEERF_HEADS}."
    echo "       Run run/collect_warmup_rollouts.sh then run/warmup_and_validate.sh first,"
    echo "       or set STEERF_FORECAST=oracle for the head-free control arm."
    exit 1
fi

export WANDB_INIT_TIMEOUT=300
export WANDB_TIMEOUT=300
export WANDB_RETRY_DELAY=60
export WANDB_MAX_RETRIES=10

save_contents="['hf_model']"
current_datetime=$(date +"%Y%m%d_%H%M%S")
run_name="STEERF-code-${model_tag}-lam${STEERF_LAM}-k${STEERF_KAPPA}-g${STEERF_GAMMA_H}-${STEERF_APPLY}-${STEERF_MAPPING}_${current_datetime}"

train_files="['$train_path']"
test_files="['$test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='left'  \
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
    actor_rollout_ref.actor.policy_loss.loss_mode=entropy_control \
    +actor_rollout_ref.actor.policy_loss.token_weight_min=${TOKEN_WEIGHT_MIN:-0.7} \
    +actor_rollout_ref.actor.policy_loss.token_weight_max=1.0 \
    +actor_rollout_ref.actor.policy_loss.linear=False \
    +actor_rollout_ref.actor.policy_loss.steerf_lam=${STEERF_LAM} \
    +actor_rollout_ref.actor.policy_loss.steerf_kappa=${STEERF_KAPPA} \
    +actor_rollout_ref.actor.policy_loss.steerf_gamma_h=${STEERF_GAMMA_H} \
    +actor_rollout_ref.actor.policy_loss.steerf_clip_c=${STEERF_CLIP_C} \
    +actor_rollout_ref.actor.policy_loss.steerf_norm=${STEERF_NORM} \
    +actor_rollout_ref.actor.policy_loss.steerf_baseline=${STEERF_BASELINE} \
    +actor_rollout_ref.actor.policy_loss.steerf_apply=${STEERF_APPLY} \
    +actor_rollout_ref.actor.policy_loss.steerf_mapping=${STEERF_MAPPING} \
    +actor_rollout_ref.actor.policy_loss.steerf_winsor_q=${STEERF_WINSOR_Q} \
    +actor_rollout_ref.actor.policy_loss.steerf_forecast=${STEERF_FORECAST} \
    +actor_rollout_ref.actor.policy_loss.steerf_heads_path=${STEERF_HEADS} \
    +actor_rollout_ref.actor.policy_loss.steerf_calib_path=${STEERF_CALIB} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE:-4} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.actor.checkpoint.save_contents=${save_contents} \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.rollout_data_dir=${STEER_ROOT}/rollout_data/$project_name/$run_name \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name=$project_name \
    trainer.experiment_name=$run_name \
    trainer.n_gpus_per_node=${N_GPUS:-8} \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=200 \
    trainer.resume_mode=disable \
    +trainer.save_best_only=False \
    +trainer.delete_old_best_checkpoint=True \
    +trainer.save_after=80 \
    +trainer.best_metric_key=val-core/codecontests/acc/mean@4 \
    ${SEED:+"+data.seed=${SEED}"} \
    "$@"
