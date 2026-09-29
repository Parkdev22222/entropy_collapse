# LoRA settings for every training launcher. Sourced, not executed.
#
# The first version of the paper trains with LoRA only, so a launcher started
# with no LoRA settings at all trains LoRA -- the paper's configuration -- and
# full fine-tuning has to be asked for by name (FULL_FT=1). Every arm reads
# the same values from here: a baseline and a treatment that differ in rank or
# learning rate are not a comparison of the two methods.
#
#   rank 64, alpha 32, every linear layer (lm_head is not one of them, so the
#   unembedding the forecast heads read stays the base model's).
#   lr 1e-5: the full fine-tuning runs used 1e-6, and LoRA at a full
#   fine-tuning learning rate barely moves; ten times is the usual ratio.
#   load_format=safetensors: with verl's default dummy_dtensor the first weight
#   sync loads only the base model and never registers the adapter, so the
#   first rollout after a resume comes from the base model without a warning.
#
# The launchers append LORA_ARGS after their own overrides and before "$@", so
# a later value on the command line still wins, and lr here replaces their
# hardcoded 1e-6 (hydra keeps the last value of a repeated key).

LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_TARGET=${LORA_TARGET:-all-linear}
LORA_LR=${LORA_LR:-1e-5}

if [ "${FULL_FT:-0}" = "1" ]; then
    LORA_ARGS=()
    echo "[lora] FULL_FT=1: full fine-tuning. The LoRA paper's runs never set this."
else
    case "${LORA_RANK}" in
        ''|0|*[!0-9]*)
            echo "REFUSE: LORA_RANK='${LORA_RANK}'. This tree trains LoRA; set FULL_FT=1" >&2
            echo "        to run full fine-tuning on purpose." >&2
            exit 2 ;;
    esac
    LORA_ARGS=(
        actor_rollout_ref.model.lora_rank="${LORA_RANK}"
        actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}"
        actor_rollout_ref.model.target_modules="${LORA_TARGET}"
        actor_rollout_ref.actor.optim.lr="${LORA_LR}"
        actor_rollout_ref.rollout.load_format=safetensors
    )
    # huggingface/ under LoRA holds PEFT-shaped keys that no loader maps back
    # onto the model, so it is not an evaluable checkpoint -- the adapter in
    # actor/lora_adapter/ is, and verl writes it on every save. Keep the
    # sharded state for resuming instead of the hf_model copy.
    SAVE_CONTENTS=${SAVE_CONTENTS:-"['model','optimizer','extra']"}
    echo "[lora] rank=${LORA_RANK} alpha=${LORA_ALPHA} target=${LORA_TARGET} lr=${LORA_LR}"
fi
export SAVE_CONTENTS
