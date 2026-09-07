# Shared arm -> run-name table. Sourced, not executed.
#
# One definition, used by run_campaign.sh (what to train), run_eval_all.sh
# (what to evaluate) and run_followups.sh (what to add). When these lists live
# in three files they drift, and a drifted run name means an eval silently
# scores the wrong checkpoint -- which is exactly the failure that invalidated
# the first six-benchmark attempt.

MODEL_TAG=${MODEL_TAG:-Qwen2.5-Math-1.5B}

# --- the five campaign arms -------------------------------------------------
CAMPAIGN_ARMS=${CAMPAIGN_ARMS:-"grpo steer permuted signed uniform"}

run_name_for () {   # <arm> <seed>  -> the trainer's experiment_name
    case "$1" in
        grpo)     echo "grpo-${MODEL_TAG}-s$2" ;;
        steer)    echo "steer-${MODEL_TAG}-s$2" ;;
        signed)   echo "steer-f-${MODEL_TAG}-s$2-tree-rollout" ;;
        uniform)  echo "steer-f-${MODEL_TAG}-s$2-tree-rollout-uniform" ;;
        permuted) echo "steer-f-${MODEL_TAG}-s$2-tree-rollout-permuted" ;;
        # --- follow-up ablations, seed 1 only (see run_followups.sh) --------
        lam0.1)       echo "steer-f-${MODEL_TAG}-s$2-tree-rollout-lam0.1" ;;
        lam0.5)       echo "steer-f-${MODEL_TAG}-s$2-tree-rollout-lam0.5" ;;
        lam0-tree)    echo "steer-f-${MODEL_TAG}-s$2-tree-rollout-lam0" ;;
        xclip-signed) echo "steer-f-${MODEL_TAG}-s$2-tree-rollout-xclip" ;;
        xclip-steer)  echo "steer-${MODEL_TAG}-s$2-xclip" ;;
        rloo-signed)  echo "steer-f-${MODEL_TAG}-s$2-tree-rollout-rloo" ;;
        rloo-steer)   echo "steer-${MODEL_TAG}-s$2-rloo" ;;
        opo-signed)   echo "steer-f-${MODEL_TAG}-s$2-tree-rollout-opo" ;;
        opo-steer)    echo "steer-${MODEL_TAG}-s$2-opo" ;;
        *) return 1 ;;
    esac
}

# Seed 1 predates the naming convention above: those runs were launched by hand
# and their checkpoint directories carry the older experiment_name. The logs
# were renamed to the convention afterwards, the checkpoints were not, so an
# eval that only looks under run_name_for() finds nothing for them.
ckpt_alias_for () {   # <arm> <seed>  -> extra checkpoint directory names
    case "$1:$2" in
        steer:1) echo "math-${MODEL_TAG}-steer-s1" ;;
        *) : ;;
    esac
}

# A run is finished when its log carries the final optimisation step. The tqdm
# counter is not usable: run_steerf.sh declares total_training_steps=200 while
# the campaign stops at 110, so a complete run's bar reads "110/200".
# The glob picks up recovery runs with a tag suffix (the _0905 chain) too.
train_log_done () {   # <log-dir> <run-name> <steps>
    local f
    for f in "$1/train-$2"*.log; do
        [ -f "${f}" ] || continue
        grep -q "step:$3 - global_seqlen" "${f}" && return 0
    done
    return 1
}

# The trailing hydra overrides every campaign-parity run needs on top of
# run_steerf.sh's own defaults. run_steerf.sh hardcodes STEPS=200 inside its
# SCALE case, so an exported STEPS does NOT reach it -- the override below is
# the only thing that stops a "110 step" run at 110. logger picks tensorboard
# (select_best_checkpoint.py parses the event files, and wandb is not
# configured on the pod); rollout_data_dir=null because writing rollouts leaked
# ~300 GB of host RAM and killed the first tree run at step 10.
# run_uniform_ablation.sh already passes exactly these three for the tree arms.
steer_plain_args () {   # <steps>
    printf '%s\n' \
        "++trainer.total_training_steps=$1" \
        "trainer.logger=['console','tensorboard']" \
        "++trainer.rollout_data_dir=null"
}
