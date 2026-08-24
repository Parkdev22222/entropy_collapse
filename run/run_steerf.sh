#!/usr/bin/env bash
# STEER-F training, paper-parity.
#
# Every hyperparameter below that is not prefixed steerf_ matches the paper
# (arXiv:2510.10150v4 §6.1 + Appendix E.2): symmetric mode, exponential
# mapping, lambda_min = 0.7 (paper v4; the released run_exp.sh still says 0.8
# — the paper is authoritative for table parity, override with
# TOKEN_WEIGHT_MIN to reproduce the release), GRPO on DAPO-Math-17k,
# 512/32/8, n=8, lr 1e-6, rollout temp 1.0 / top-p 1.0, AT MOST 200 rollout
# steps, checkpoints every 10 steps with the highest-AIME24 one selected for
# the final test, 8 GPUs. Main-table numbers are the average of TWO
# independent runs (SEED=1 / SEED=2). steerf_lam=0 is bit-identical to stock
# STEER (tests/test_lambda_zero_equiv.py).
#
# Paper table slots this run feeds (model chosen via MODEL_PATH):
#   Qwen/Qwen2.5-Math-7B          -> Table 3  (default)
#   Qwen/Qwen2.5-Math-1.5B        -> Table 12 (appendix F.3)
#   Qwen/Qwen2.5-14B              -> Table 4
#   meta-llama/Llama-3.2-3B-Instruct -> Figure 20a (appendix F.3)
# RL-algorithm generalization (appendix F.3): append
#   algorithm.adv_estimator=rloo   (or opo)   as a trailing hydra override.
# Evaluate the saved checkpoint with run/eval_steerf.sh to get the
# avg@32 / avg@1 numbers that drop into the tables.
#
# Prerequisites (once per base model):
#   bash run/collect_warmup_rollouts.sh     # base-policy rollouts
#   bash run/warmup_and_validate.sh         # -> checkpoints/mtp_heads_<tag>.pt
#                                           #    + mtp_calibration_<tag>.json
#                                           #    + selected kappa / gamma_H
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

project_name='STEER-F'
train_path=${STEER_ROOT}/datasets/DAPO-Math-17k.parquet
test_path=${STEER_ROOT}/datasets/aime24.parquet
model_path=${MODEL_PATH:-"Qwen/Qwen2.5-Math-7B"}
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
# The avg@32 benchmarks are ALREADY replicated 32x inside their parquets
# (aime24/aime25: 30 problems -> 960 rows; amc23: 40 -> 1280), which is how
# upstream STEER's run/eval.sh gets avg@32 with val_kwargs.n=1. verl names the
# metric mean@<samples-per-prompt>, so n>1 both multiplies the cost by n and
# renames the metric out from under select_best_checkpoint.py /
# collect_results.py, which look for mean@32.
case "${SCALE}" in
  paper) TRAIN_BS=512; MINI_BS=32; RESP_LEN=3072; STEPS=200; VAL_N=1;  TEST_FREQ=10; SAVE_FREQ=10 ;;
  smoke) TRAIN_BS=16;  MINI_BS=8;  RESP_LEN=512;  STEPS=3;   VAL_N=1;  TEST_FREQ=1;  SAVE_FREQ=1  ;;
  *) echo "FATAL: SCALE must be 'paper' or 'smoke'"; exit 1 ;;
esac
# verl names the val metric mean@<samples-per-prompt>, and the avg@32 parquets
# already hold 32 replicas of every problem -- so the suffix is replicas x VAL_N,
# NOT VAL_N. select_best_checkpoint.py and collect_results.py both look for
# mean@32, which is what VAL_N=1 yields.
VAL_REPLICAS=32
ACC_AT=$(( VAL_REPLICAS * VAL_N ))
# save_after: paper keeps the disk cost down by only checkpointing the back half
# of the run; smoke has to checkpoint from step 0 or there is nothing for
# select_best_checkpoint.py to pick and the eval stage can never be exercised.
case "${SCALE}" in
  paper) SAVE_AFTER=80 ;;
  smoke) SAVE_AFTER=0  ;;
  *) echo "FATAL: SCALE must be 'paper' or 'smoke', got '${SCALE}'"; exit 1 ;;
esac
# Set SAVE_AFTER_OVERRIDE=0 to checkpoint from the start. That is what makes a
# crashed run resumable rather than restartable -- but only together with
# SAVE_CONTENTS and RESUME_MODE below; a checkpoint without the optimizer is
# an eval artefact, not a resume point.
SAVE_AFTER=${SAVE_AFTER_OVERRIDE:-${SAVE_AFTER}}

# ---- memory: chunked entropy (numerically identical, ~27 GiB cheaper) -------
# The log-prob pass materialises fp32 logits AND fp32 softmax probs over
# [micro_bs*seq, V] — 29.5 GiB for 8x3250 at V=152k, on top of the 7.4 GiB bf16
# logits. verl's entropy_from_logits_with_chunking computes the same
# lse - <p, l> in 2048-row chunks; verified bit-identical, peak 36.8 -> 9.7 GiB.
# micro_batch_size_per_gpu stays at 8 because STEER's min-max is computed
# per micro-batch — shrinking it would change the method, not just the memory.

# ---- memory profile --------------------------------------------------------
# On one 80GB card, FSDP cannot shard, so full fine-tuning needs
#   weights + grads + AdamW(fp32 master,m,v) = ~15x params in bytes
#   1.5B -> 23 GiB (fits)   7B -> 114 GiB (does NOT fit)   14B -> 219 GiB (no)
# OFFLOAD=1 pushes the optimizer state (and params) to CPU, which is what makes
# 7B possible at all on a single card — at a large speed cost.
# log-prob micro batch. STEER mode makes compute_log_prob the peak: it holds
# entropy AND max_prob_log_probs AND log_probs over [micro_bs*seq, V] at once,
# and at 8 it OOMs on one 80GB card (measured: 8.03 GiB short, seed 1, step ~4).
# Safe to shrink -- compute_log_prob concatenates per-sequence results and
# reverts the order, and forecast_h_togo has no batch-level normalisation, so
# the outputs are identical. This is NOT true of ppo_micro_batch_size_per_gpu,
# which is the group STEER's min-max runs over; that one stays at 8.
# Measured on one A100-80GB: 8 OOMs, 2 costs 1357 s of a 2325 s step
# (old_log_prob alone, 59%) because it starves the GPU, 4 is the
# compromise -- peak was 67.2 GiB at 2, leaving ~12 GiB of headroom.
LOGP_MBS=${LOGP_MBS:-4}
OFFLOAD=${OFFLOAD:-0}
if [ "${OFFLOAD}" = "1" ]; then PARAM_OFF=True; OPT_OFF=True; else PARAM_OFF=False; OPT_OFF=False; fi

export WANDB_INIT_TIMEOUT=300
export WANDB_TIMEOUT=300
export WANDB_RETRY_DELAY=60
export WANDB_MAX_RETRIES=10

# verl's checkpoint_contents defaults to ['model','optimizer','extra']; naming
# save_contents REPLACES that list rather than adding to it. ['hf_model'] alone
# is therefore enough for select_best_checkpoint.py + eval, but it writes no
# optimizer or rng state, so trainer.resume_mode has nothing to resume FROM and
# a crash costs the whole run. To make a long run survivable:
#   SAVE_CONTENTS="['hf_model','model','optimizer','extra']" RESUME_MODE=auto \
#   SAVE_AFTER_OVERRIDE=0
# resume_mode=auto is safe with no checkpoints present -- it prints "Training
# from scratch" and starts at step 0.
# MAX_CKPT_KEEP bounds disk: verl deletes the oldest checkpoint BEFORE
# writing a new one, so peak usage stays flat. A full checkpoint here is
# 25 GiB (6.7 model + 12 optim + 6.7 hf), and /workspace is quota'd at
# ~80 GiB -- without this, saving every 10 steps fills the disk by step 40
# and the run dies mid-write.
save_contents=${SAVE_CONTENTS:-"['hf_model']"}
RESUME_MODE=${RESUME_MODE:-disable}
current_datetime=$(date +"%Y%m%d_%H%M%S")
run_name=${RUN_NAME:-"STEERF-${SCALE}-${model_tag}-lam${STEERF_LAM}-k${STEERF_KAPPA}-g${STEERF_GAMMA_H}-${STEERF_APPLY}-${STEERF_MAPPING}_${current_datetime}"}

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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGP_MBS} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE:-4} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL:-0.6} \
    actor_rollout_ref.actor.checkpoint.save_contents=${save_contents} \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOGP_MBS} \
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
    trainer.resume_mode=${RESUME_MODE} \
    ++trainer.save_best_only=False \
    ++trainer.delete_old_best_checkpoint=True \
    ++trainer.save_after=${SAVE_AFTER} \
    ${MAX_CKPT_KEEP:+++trainer.max_actor_ckpt_to_keep=${MAX_CKPT_KEEP}} \
    ++trainer.best_metric_key=val-core/aime_2024_dapo_boxed/acc/mean@${ACC_AT} \
    ${SEED:+"++data.seed=${SEED}"} \
    "$@"
