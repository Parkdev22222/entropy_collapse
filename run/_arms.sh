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

# --- the environment gate ---------------------------------------------------
# On 2026-09-08 an unpinned huggingface_hub upgrade (1.30.0, outside the
# >=0.34,<1.0 that transformers enforces at import) made every `import verl`
# raise. Two arms of a queued chain died in their first seconds and the queue
# kept going, so the failure only surfaced days later, in a traceback.
# Five seconds of checking in front of a 40-hour run is free.
env_preflight () {   # [root]  -> 0 when the training stack imports
    local root err
    root="${1:-${ROOT:-$(pwd)}}"
    # A symbol, not a bare import: this repo ships a verl/ directory, every
    # launcher puts the repo root on PYTHONPATH, and a directory without
    # __init__.py imports fine as a namespace package. `import verl` therefore
    # passes on a checkout that has no trainer in it at all. DataProto is what
    # verl/__init__.py:22 pulls in -- the exact line the 09-08 traceback died on.
    if err="$(cd "${root}" && PYTHONPATH="${root}:${PYTHONPATH:-}" \
                python3 -c 'from verl import DataProto; import steer_f.tree_rollout' 2>&1)"; then
        return 0
    fi
    echo "  the training stack does not import:"
    printf '%s\n' "${err}" | tail -6 | sed 's/^/    /'
    if [ -f "${root}/scripts/check_env_pins.py" ]; then
        echo
        (cd "${root}" && python3 scripts/check_env_pins.py) || true
    fi
    return 1
}

# A run that never logged step 1 did not crash during training -- it failed to
# start, and that is almost always the box rather than the arm. Saying so in
# the queue's own log is the difference between "one arm failed" and "the
# environment broke under us", which is the thing nobody could see last time.
diagnose_startup_failure () {   # <log-file> <label>
    grep -q "step:1 - global_seqlen" "$1" 2>/dev/null && return 0
    echo "[!] $2: STARTUP FAILURE -- never reached step 1."
    printf '    last lines of %s:\n' "$1"
    tail -4 "$1" 2>/dev/null | sed 's/^/      /'
    if env_preflight; then
        echo "    The environment imports fine now, so this was specific to the run."
    else
        echo "    ^^ THE ENVIRONMENT IS BROKEN. Every remaining run will die the same way."
        echo "       Fix it, then re-run this queue -- finished runs are skipped."
    fi
    return 1
}
