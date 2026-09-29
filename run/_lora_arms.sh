# The LoRA campaign: which runs exist, what each one is, how it is launched.
# Sourced after run/_arms.sh, not executed. run/run_lora_paper.sh trains and
# evaluates from this table and nothing else, so an arm cannot be trained under
# one definition and evaluated under another.
#
# THE METHOD. The treatment reads H_togo from the realised entropy of the
# rollout verl has already computed (STEERF_FORECAST=oracle): no MTP heads, no
# Phase 1, no extra forward pass. uniform and permuted -- the controls the
# registered contrasts pair it with -- read the same quantity, so a contrast
# changes one thing. mtp is the same treatment with the MTP forecaster,
# at every core seed, which is how the paper answers "does the forecaster
# matter" under its own configuration.
#
# TOKEN WEIGHTS. Every weighted arm runs in [0.7, 1.0]: STEER's paper value
# (lambda_min = 0.7, exponential map, which caps at 1.0); run_steerf.sh's
# defaults. wmin-steer is the released script's 0.8.
#
# Run names carry a "lora-" prefix: no LoRA run can ever be mistaken for, or
# resumed from, a full fine-tuning run of the same arm and seed.

LORA_PREFIX=${LORA_PREFIX:-lora-}
# Every LoRA run trains 150 steps (about 4.4 epochs of the 17k set at batch
# 512). _arms.sh's steps_for_arm reads STEPS, so the queue, the evaluation
# (final = step 150) and the checkpoint policy (50, 100, 150) all follow it.
STEPS=${STEPS:-150}
LORA_SEEDS=${LORA_SEEDS:-"1 2 3"}
LORA_CORE_ARMS=${LORA_CORE_ARMS:-"grpo steer uniform permuted signed mtp"}
# No grpo-long: the method has no MTP forward, so the compute-matched point is
# read off the regular GRPO run (analyze_seeds.py --lora), not a longer one.
LORA_FOLLOWUP_ARMS=${LORA_FOLLOWUP_ARMS:-"lam0.1 lam0.5 lam0-tree wmin-steer xclip-signed xclip-steer rloo-signed rloo-steer opo-signed opo-steer"}
LORA_FOLLOWUP_SEED=${LORA_FOLLOWUP_SEED:-1}
LORA_XCLIP="actor_rollout_ref.actor.clip_ratio_high=5 actor_rollout_ref.actor.clip_ratio_low=0.99"

# Backbones: does the sign of the headline contrast survive a change of scale
# (the 7B of the same family) and of pre-training family (Llama)? Three arms,
# seed 1, the same LoRA and step count. No MTP arm: STEER-V needs no heads,
# which is also why no backbone needs a Phase 1. Each backbone's model path,
# validation set and best-checkpoint key come from _arms.sh:backbone_profile.
LORA_BACKBONES=${LORA_BACKBONES:-"Qwen2.5-Math-7B Llama-3.2-3B-Instruct"}
LORA_BACKBONE_ARMS=${LORA_BACKBONE_ARMS:-"grpo steer signed"}

# The macro prefix a backbone's numbers carry (B<stem>), the names the
# manuscript's backbone table reads and tests/test_paper.py knows.
lora_backbone_stem () {   # <tag>
    case "$1" in
        Qwen2.5-Math-7B)        echo qwenbig ;;
        Llama-3.2-3B-Instruct)  echo llama ;;
        Mistral-7B-v0.3)        echo mistral ;;
        *) return 1 ;;
    esac
}

# Set up the environment of one backbone: its tag, and the model path,
# validation parquet and best key backbone_profile assigns. The three are
# unset first -- backbone_profile keeps a value that is already set, and
# sourcing _arms.sh has already set them for the 1.5B, so without this a 7B
# run would quietly train the 1.5B. Call it in a subshell.
lora_enter_backbone () {   # <tag>
    export MODEL_TAG="$1"
    unset MODEL_PATH VAL_PARQUET BEST_METRIC_KEY
    backbone_profile "$1"
}

lora_run_name () {   # <arm> <seed>
    local rn
    rn="$(run_name_for "$1" "$2")" || return 1
    echo "${LORA_PREFIX}${rn}"
}

# <kind> [K=V ...] [-- hydra overrides]
#   kind: grpo  -> run/run_grpo.sh
#         plain -> run/run_steerf.sh at lambda 0 (stock STEER)
#         tree  -> run/run_uniform_ablation.sh (the tree arms)
lora_arm_spec () {   # <arm>
    case "$1" in
        grpo|grpo-long) echo "grpo" ;;
        steer)          echo "plain STEERF_LAM=0" ;;
        signed)         echo "tree ARM=signed STEERF_LAM=0.25 STEERF_FORECAST=oracle" ;;
        uniform)        echo "tree ARM=uniform STEERF_LAM=0.25 STEERF_FORECAST=oracle" ;;
        permuted)       echo "tree ARM=permuted STEERF_LAM=0.25 STEERF_FORECAST=oracle" ;;
        mtp)     echo "tree ARM=signed STEERF_LAM=0.25 STEERF_FORECAST=mtp" ;;
        lam0.1)         echo "tree ARM=signed STEERF_LAM=0.1 STEERF_FORECAST=oracle" ;;
        lam0.5)         echo "tree ARM=signed STEERF_LAM=0.5 STEERF_FORECAST=oracle" ;;
        lam0-tree)      echo "tree ARM=signed STEERF_LAM=0 STEERF_FORECAST=oracle" ;;
        wmin-steer)     echo "plain STEERF_LAM=0 TOKEN_WEIGHT_MIN=0.8" ;;
        xclip-signed)   echo "tree ARM=signed STEERF_LAM=0.25 STEERF_FORECAST=oracle -- ${LORA_XCLIP}" ;;
        xclip-steer)    echo "plain STEERF_LAM=0 -- ${LORA_XCLIP}" ;;
        rloo-signed)    echo "tree ARM=signed STEERF_LAM=0.25 STEERF_FORECAST=oracle -- algorithm.adv_estimator=rloo" ;;
        rloo-steer)     echo "plain STEERF_LAM=0 -- algorithm.adv_estimator=rloo" ;;
        opo-signed)     echo "tree ARM=signed STEERF_LAM=0.25 STEERF_FORECAST=oracle -- algorithm.adv_estimator=opo" ;;
        opo-steer)      echo "plain STEERF_LAM=0 -- algorithm.adv_estimator=opo" ;;
        *) return 1 ;;
    esac
}

lora_needs_heads () {   # <arm>  -> 0 when the arm reads the MTP heads
    case "$(lora_arm_spec "$1")" in *STEERF_FORECAST=mtp*) return 0 ;; esac
    return 1
}

# Every run of the campaign, in the order it trains: the core seed by seed
# (a queue stopped part-way leaves whole seeds, never four arms at three seeds
# and two at one), then the follow-ups. One "stage:arm:seed" per line.
lora_plan () {   # [core|followups|backbones|all]
    local which=${1:-all} s a b
    if [ "${which}" = backbones ]; then
        # Backbone-major: all three arms of one backbone before the next, so a
        # stopped queue leaves whole backbones. Fourth field: the model tag.
        for b in ${LORA_BACKBONES}; do
            for a in ${LORA_BACKBONE_ARMS}; do echo "backbones:${a}:1:${b}"; done
        done
        return 0
    fi
    if [ "${which}" != followups ]; then
        for s in ${LORA_SEEDS}; do
            for a in ${LORA_CORE_ARMS}; do echo "core:${a}:${s}"; done
        done
    fi
    if [ "${which}" != core ]; then
        for a in ${LORA_FOLLOWUP_ARMS}; do echo "followups:${a}:${LORA_FOLLOWUP_SEED}"; done
    fi
}

# Deterministic split of a plan over N boxes: line k goes to shard (k mod N).
# Every box computes the same plan, so no coordination is needed -- only the
# same LORA_* settings on each.
lora_shard () {   # <i> <n>   (stdin: plan lines)
    awk -v i="$1" -v n="$2" '((NR - 1) % n) == i'
}
