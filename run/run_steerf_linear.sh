#!/usr/bin/env bash
# STEER-F 학습 스크립트 (STEER 레포의 run/run_linear.sh 기반).
#
# LAMBDA=0 으로 두면 STEER 재현 실행과 **비트 단위로 동일**하다
# (tests/test_lambda_zero_equiv.py 가 이를 강제한다). 따라서 이 스크립트 하나로
# λ 스윕 전체(E2-1)를 돌린다.
#
#   STEER 재현:  LAMBDA=0    bash run/run_steerf_linear.sh
#   STEER-F:     LAMBDA=0.5  MTP_HEADS=artifacts/mtp_heads_qwen7b.pt bash run/run_steerf_linear.sh
#
# 선행 조건:
#   1) STEER 레포에 patches/core_algos_steerf.patch 적용
#        cd $STEER_REPO && git apply /path/to/STEER-F/patches/core_algos_steerf.patch
#   2) STEER-F 저장소가 PYTHONPATH에 있을 것 (아래에서 처리)
#   3) λ>0이면 MTP_HEADS 체크포인트 필요 (scripts/phase1_warmup_heads.py 산출물)

set -x

export PYTHONHASHSEED=42
export PYTORCH_SEED=42
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEERF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STEER_ROOT="${STEER_ROOT:?STEER_ROOT (patched STEER repo 경로)를 설정하세요}"

# STEER-F 모듈이 core_algos.py 에서 import 되므로 반드시 경로에 있어야 한다.
export PYTHONPATH="${STEERF_ROOT}:${STEER_ROOT}:$PYTHONPATH"
echo "Current VERL path:"
python3 -c "import verl; print(verl.__file__)"
python3 -c "import steer_f; print('steer_f', steer_f.__version__, steer_f.__file__)"

# ---------------------------- STEER-F 하이퍼파라미터 ----------------------------
LAMBDA=${LAMBDA:-0.5}              # λ. 0 = 순수 STEER
KAPPA=${KAPPA:-4}                  # 예보 호라이즌 (Phase 1에서 결정)
GAMMA_H=${GAMMA_H:-0.85}           # 예보 할인 (Phase 1에서 결정)
CLIP_C=${CLIP_C:-1.0}              # A_H 클리핑
ETA=${ETA:-1.0}
STEERF_NORM=${STEERF_NORM:-z}      # z | robust
BASELINE=${BASELINE:-sibling}      # sibling | group  (Ablation A5)
BETA_MTP=${BETA_MTP:-0.05}         # MTP CE 보조 손실 (0 = 헤드 freeze, Ablation A7)
LOCAL_SCALE=${LOCAL_SCALE:-1.0}    # z(Ω) 계수 (0 = 미래 항만, Ablation A4)
DRIFT_KL=${DRIFT_KL:-0.5}          # 예보 표류 문턱
MTP_HEADS=${MTP_HEADS:-""}
ROLLOUT_N=${ROLLOUT_N:-8}          # = 그룹 크기 G (sibling baseline이 이 값에 의존)
# -------------------------------------------------------------------------------

if [ "${LAMBDA}" != "0" ] && [ "${LAMBDA}" != "0.0" ] && [ -z "${MTP_HEADS}" ]; then
  echo "ERROR: LAMBDA=${LAMBDA} 인데 MTP_HEADS 가 비어 있습니다." >&2
  echo "       scripts/phase1_warmup_heads.py 로 헤드를 먼저 학습하세요." >&2
  exit 1
fi

project_name='STEER-F'
train_path=${TRAIN_PATH:-${STEER_ROOT}/datasets/DAPO-Math-17k.parquet}
test_path=${TEST_PATH:-${STEER_ROOT}/datasets/aime24.parquet}
model_path=${MODEL_PATH:-"Qwen/Qwen2.5-Math-7B"}
run_name=STEERF-linear-lam${LAMBDA}-k${KAPPA}-g${GAMMA_H}

export WANDB_INIT_TIMEOUT=300
export WANDB_TIMEOUT=300
export WANDB_RETRY_DELAY=60
export WANDB_MAX_RETRIES=10

save_contents="['hf_model']"
current_datetime=$(date +"%Y%m%d_%H%M%S")
run_name="${run_name}_${current_datetime}"

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
    actor_rollout_ref.actor.policy_loss.loss_mode=entropy_control \
    +actor_rollout_ref.actor.policy_loss.token_weight_min=0.8 \
    +actor_rollout_ref.actor.policy_loss.token_weight_max=1.2 \
    +actor_rollout_ref.actor.policy_loss.linear=True \
    +actor_rollout_ref.actor.policy_loss.steerf_lam=${LAMBDA} \
    +actor_rollout_ref.actor.policy_loss.steerf_eta=${ETA} \
    +actor_rollout_ref.actor.policy_loss.steerf_clip_c=${CLIP_C} \
    +actor_rollout_ref.actor.policy_loss.steerf_norm=${STEERF_NORM} \
    +actor_rollout_ref.actor.policy_loss.steerf_local_scale=${LOCAL_SCALE} \
    +actor_rollout_ref.actor.policy_loss.steerf_kappa=${KAPPA} \
    +actor_rollout_ref.actor.policy_loss.steerf_gamma_h=${GAMMA_H} \
    +actor_rollout_ref.actor.policy_loss.steerf_baseline=${BASELINE} \
    +actor_rollout_ref.actor.policy_loss.steerf_beta_mtp=${BETA_MTP} \
    +actor_rollout_ref.actor.policy_loss.steerf_drift_kl_threshold=${DRIFT_KL} \
    +actor_rollout_ref.actor.policy_loss.steerf_group_size=${ROLLOUT_N} \
    +actor_rollout_ref.actor.policy_loss.steerf_heads_path=\"${MTP_HEADS}\" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.actor.checkpoint.save_contents=${save_contents} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.rollout_data_dir=${STEER_ROOT}/rollout_data/$project_name/$run_name \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name=$project_name \
    trainer.experiment_name=$run_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=1 \
    trainer.total_epochs=10 \
    trainer.resume_mode=disable \
    +trainer.save_best_only=False \
    +trainer.delete_old_best_checkpoint=True \
    +trainer.save_after=80 \
    +trainer.best_metric_key=val-core/math_dapo/acc/mean@32
