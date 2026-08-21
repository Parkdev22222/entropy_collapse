#!/usr/bin/env bash
# STEER-F on the paper's CODE-GENERATION track (Table 5, LCB-v5 rows).
#   models    : Qwen2.5-Coder-3B / 7B / 14B  (one run per model = one Table 5 cell)
#   training  : ArcherCodeR (6,753 tasks) — scripts/prepare_code_data.py
#   evaluation: LiveCodeBench v5 (279 problems), avg@4
#   reward    : prime_code (APPS-style tests) via data_source="codecontests"
# The paper's code-EDITING track (internal 51k corpus + Zeta) is not
# reproducible from public data and is out of scope.
set -x

export PYTHONHASHSEED=42
export PYTORCH_SEED=42
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# The rollout/train cycle allocates and frees tens of GiB every step, so the
# caching allocator wants help against fragmentation. It must NOT be
# expandable_segments:True: vLLM's sleep mode (rollout.free_cache_engine, on by
# default) allocates its pool through the cuMem APIs and asserts against
# expandable segments at engine construction -- see pytorch#147851. Sleep mode
# hands the whole vLLM pool back every step, which is the bigger win anyway.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-garbage_collection_threshold:0.8}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=run/_gpu_defaults.sh
. "${SCRIPT_DIR}/_gpu_defaults.sh"

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
STEERF_HEADS=${STEERF_HEADS:-${STEER_ROOT}/checkpoints/mtp_heads_${model_tag}${SCALE:+-${SCALE}}.pt}
STEERF_CALIB=${STEERF_CALIB:-${STEER_ROOT}/checkpoints/mtp_calibration_${model_tag}${SCALE:+-${SCALE}}.json}

if [ "${STEERF_LAM}" != "0" ] && [ "${STEERF_FORECAST}" = "mtp" ] && [ ! -f "${STEERF_HEADS}" ]; then
    echo "FATAL: steerf_lam=${STEERF_LAM} needs MTP heads at ${STEERF_HEADS}."
    echo "       Run run/collect_warmup_rollouts.sh then run/warmup_and_validate.sh first,"
    echo "       or set STEERF_FORECAST=oracle for the head-free control arm."
    exit 1
fi

# ---- scale profile ---------------------------------------------------------
# SCALE=paper (default) reproduces the paper exactly. SCALE=smoke shrinks every
# knob so the WHOLE pipeline (warm-up -> heads -> A_H -> reweighting -> eval ->
# collect) can be proven end-to-end on a single GPU in about an hour; its
# numbers are NOT comparable to the paper's tables and the run name says so.
SCALE=${SCALE:-paper}
case "${SCALE}" in
  paper) TRAIN_BS=512; MINI_BS=32; RESP_LEN=3072; STEPS=200; VAL_N=4;  TEST_FREQ=10; SAVE_FREQ=10 ;;
  smoke) TRAIN_BS=16;  MINI_BS=8;  RESP_LEN=512;  STEPS=3;   VAL_N=2;  TEST_FREQ=1;  SAVE_FREQ=1  ;;
  *) echo "FATAL: SCALE must be 'paper' or 'smoke', got '${SCALE}'"; exit 1 ;;
esac

# ---- memory profile --------------------------------------------------------
# On one 80GB card, FSDP cannot shard, so full fine-tuning needs
#   weights + grads + AdamW(fp32 master,m,v) = ~15x params in bytes
#   1.5B -> 23 GiB (fits)   7B -> 114 GiB (does NOT fit)   14B -> 219 GiB (no)
# OFFLOAD=1 pushes the optimizer state (and params) to CPU, which is what makes
# 7B possible at all on a single card — at a large speed cost.
OFFLOAD=${OFFLOAD:-0}
if [ "${OFFLOAD}" = "1" ]; then PARAM_OFF=True; OPT_OFF=True; else PARAM_OFF=False; OPT_OFF=False; fi

export WANDB_INIT_TIMEOUT=300
export WANDB_TIMEOUT=300
export WANDB_RETRY_DELAY=60
export WANDB_MAX_RETRIES=10

save_contents="['hf_model']"
current_datetime=$(date +"%Y%m%d_%H%M%S")
run_name=${RUN_NAME:-"STEERF-code-${SCALE}-${model_tag}-lam${STEERF_LAM}-k${STEERF_KAPPA}-g${STEERF_GAMMA_H}-${STEERF_APPLY}-${STEERF_MAPPING}_${current_datetime}"}

train_files="['$train_path']"
test_files="['$test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=${TRAIN_BS} \
    data.max_prompt_length=1024 \
    data.max_response_length=${RESP_LEN} \
    data.filter_overlong_prompts=True \
    data.truncation='left'  \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BS} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=entropy_control \
    ++actor_rollout_ref.actor.policy_loss.token_weight_min=${TOKEN_WEIGHT_MIN:-0.7} \
    ++actor_rollout_ref.actor.policy_loss.token_weight_max=1.0 \
    ++actor_rollout_ref.actor.policy_loss.linear=False \
    ++actor_rollout_ref.actor.policy_loss.steerf_lam=${STEERF_LAM} \
    ++actor_rollout_ref.actor.policy_loss.steerf_kappa=${STEERF_KAPPA} \
    ++actor_rollout_ref.actor.policy_loss.steerf_gamma_h=${STEERF_GAMMA_H} \
    ++actor_rollout_ref.actor.policy_loss.steerf_clip_c=${STEERF_CLIP_C} \
    ++actor_rollout_ref.actor.policy_loss.steerf_norm=${STEERF_NORM} \
    ++actor_rollout_ref.actor.policy_loss.steerf_baseline=${STEERF_BASELINE} \
    ++actor_rollout_ref.actor.policy_loss.steerf_apply=${STEERF_APPLY} \
    ++actor_rollout_ref.actor.policy_loss.steerf_mapping=${STEERF_MAPPING} \
    ++actor_rollout_ref.actor.policy_loss.steerf_winsor_q=${STEERF_WINSOR_Q} \
    ++actor_rollout_ref.actor.policy_loss.steerf_forecast=${STEERF_FORECAST} \
    ++actor_rollout_ref.actor.policy_loss.steerf_heads_path=${STEERF_HEADS} \
    ++actor_rollout_ref.actor.policy_loss.steerf_calib_path=${STEERF_CALIB} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    ++actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFF} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPT_OFF} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE:-4} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL:-0.6} \
    actor_rollout_ref.actor.checkpoint.save_contents=${save_contents} \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${PARAM_OFF} \
    algorithm.use_kl_in_reward=False \
    trainer.rollout_data_dir=${STEER_ROOT}/rollout_data/$project_name/$run_name \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name=$project_name \
    trainer.experiment_name=$run_name \
    trainer.n_gpus_per_node=${N_GPUS:-8} \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=10 \
    trainer.total_training_steps=${STEPS} \
    trainer.resume_mode=disable \
    ++trainer.save_best_only=False \
    ++trainer.delete_old_best_checkpoint=True \
    ++trainer.save_after=80 \
    ++trainer.best_metric_key=val-core/codecontests/acc/mean@${VAL_N} \
    ${SEED:+"++data.seed=${SEED}"} \
    "$@"
