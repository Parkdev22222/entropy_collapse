# Shared arm -> run-name table. Sourced, not executed.
#
# One definition, used by run_campaign.sh (what to train), run_eval_all.sh
# (what to evaluate) and run_followups.sh (what to add). When these lists live
# in three files they drift, and a drifted run name means an eval silently
# scores the wrong checkpoint -- which is exactly the failure that invalidated
# the first six-benchmark attempt.

MODEL_TAG=${MODEL_TAG:-Qwen2.5-Math-1.5B}

# The model lives here, beside the tag, because the two must not drift.
#
# 2026-09-13: they had. run_grpo.sh:83 and run_uniform_ablation.sh:81 default to
# 1.5B, but run_steerf.sh:58 defaults to Qwen2.5-Math-7B, and the campaign's
# steer arm is the one arm that calls run_steerf.sh directly. On the pod where
# the campaign started, run_steerf.sh had been edited to 1.5B by hand and the
# edit was never committed, so the default never bit; a second pod cloned from
# the committed tree and its steer arm resolved
#     model.path = Qwen/Qwen2.5-Math-7B
#     experiment_name = steer-Qwen2.5-Math-1.5B-s5
# A 7B model training under a 1.5B run name produces plausible numbers under the
# wrong label and raises nothing. Exporting MODEL_PATH from here means no
# launcher's own default is ever reached.
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-Math-${MODEL_TAG#Qwen2.5-Math-}}

# Refuse a run whose name claims a model the run is not using. Cheap, and the
# only thing standing between a mismatched default and a table full of numbers
# from the wrong network.
model_guard () {   # 0 = MODEL_PATH's basename matches MODEL_TAG
    local got="${MODEL_PATH##*/}"
    [ "${got}" = "${MODEL_TAG}" ] && return 0
    echo "REFUSE: MODEL_PATH resolves to '${got}' but the run names say '${MODEL_TAG}'." >&2
    echo "        MODEL_PATH=${MODEL_PATH}" >&2
    echo "        A run trained on one model and logged under another is worse than" >&2
    echo "        a crash: nothing about it looks wrong afterwards." >&2
    return 1
}

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
        # The compute-matched control. STEER-F costs ~1.75x GRPO per step, so
        # "is the gain worth the wall clock" is only answered by giving GRPO the
        # same wall clock -- 200 steps covers a ratio up to ~1.8. The analysis
        # reads the step whose cumulative seconds match STEER-F at 110 rather
        # than assuming a ratio, so the log has to run past the crossing.
        grpo-long)    echo "grpo-${MODEL_TAG}-s$2-long" ;;
        *) return 1 ;;
    esac
}

# Almost every arm stops at the campaign's 110 steps. grpo-long is the one
# exception: it is the compute-matched control and has to run past the point
# where its cumulative wall clock crosses STEER-F's at 110, which 200 covers.
# Both the training queue and the eval queue need the same answer, so it lives
# here rather than in either of them.
steps_for_arm () {   # <arm>
    case "$1" in
        grpo-long) echo "${LONG_STEPS:-200}" ;;
        *)         echo "${STEPS:-110}" ;;
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
#
# The only suffix allowed on top of the run name is _<tag>, which is how the
# recovery chain names its logs (train-<run>_0905.log). A bare "train-<run>"*
# glob would ALSO swallow the arm suffixes, and the run names are prefixes of
# each other by construction:
#
#     signed    steer-f-<tag>-s2-tree-rollout
#     permuted  steer-f-<tag>-s2-tree-rollout-permuted     <- signed + "-..."
#
# so one finished permuted would mark signed done, and the campaign runs
# permuted BEFORE signed. The queue is built once per invocation, so a running
# campaign is unaffected -- but every restart after the first tree arm of a
# seed completes would drop the treatment arm silently, with no error anywhere.
# Every arm suffix starts with "-", every recovery tag with "_", which is
# exactly the line these two patterns draw. tests/test_run_names.py pins it.
train_log_done () {   # <log-dir> <run-name> <steps>
    local f
    for f in "$1/train-$2.log" "$1/train-$2"_*.log; do
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
        # `import steer_f.tree_rollout` was the wrong thing to check. Both
        # branches of this repo carry a steer_f/, their histories are unrelated,
        # and tree_rollout.py is the ONE file that happens to be byte-identical
        # between them -- so this gate passed on a box whose steer_f could not
        # serve verl at all (7 symbols missing, among them the forecast_h_togo
        # that dp_actor.py:407 imports). The arms died inside worker init with
        # the gate still saying OK. Check verl's actual import statements.
        if ! (cd "${root}" && PYTHONPATH="${root}:${PYTHONPATH:-}" \
                python3 run/_check_steer_f.py "${root}"); then
            return 1
        fi
        # The reward scorer is imported lazily, the first time verl scores a
        # generation -- step-0 validation, several minutes in. A missing
        # word2number there killed three H100 arms in a row on 2026-09-14, each
        # after a full model load and a full generation pass. Importing it here
        # costs a second and moves that failure ahead of the queue.
        if ! err="$(cd "${root}" && PYTHONPATH="${root}:${PYTHONPATH:-}" python3 -c '
import importlib, importlib.util, sys
if importlib.util.find_spec("verl.utils.reward_score") is None:
    sys.exit(0)                      # no trainer in this checkout
importlib.import_module("verl.utils.reward_score.multi_datasets_eval")' 2>&1)"; then
            echo "  verl and steer_f import, but the reward scorer does not:"
            printf '%s\n' "${err}" | tail -6 | sed 's/^/    /'
            echo
            echo "  verl reaches this only at step-0 validation, minutes into a run,"
            echo "  so it does not look like an environment problem in the log."
            echo "  Install what the traceback names. Most likely:"
            echo "      pip install word2number sympy"
            return 1
        fi
        # verl importing is necessary and not sufficient. On 2026-09-13 every
        # arm on a fresh box died with "The current node timed out during
        # startup" while this check passed, because the broken package was
        # opentelemetry: ray's dashboard failed to import, ray.init() timed out,
        # and main_ppo never reached a training step. The queue then reported
        # "the environment imports fine, so this was specific to the run" --
        # the opposite of the truth -- and burned the rest of the queue.
        # Starting a one-CPU ray costs a few seconds in front of a 40-hour run.
        if err="$(cd "${root}" && python3 -c '
import ray, sys
ray.init(num_cpus=1, ignore_reinit_error=True, log_to_driver=False)
ray.shutdown()' 2>&1)"; then
            # verl imports flash_attn unconditionally on a CUDA box --
            # dp_actor.py:43 is `if is_cuda_available: from flash_attn...`, and
            # the elif beside it is for Ascend NPUs, not a fallback. So a stale
            # flash_attn_2_cuda.so does not degrade anything: it kills the run at
            # worker init, one to two minutes in, long after this gate said OK.
            # Only required where CUDA is actually present, so CPU checkouts and
            # CI still pass.
            if err="$(cd "${root}" && python3 -c '
import sys
try:
    import torch
except Exception:
    sys.exit(0)                      # no torch here: not a training box
if not torch.cuda.is_available():
    sys.exit(0)
import flash_attn.bert_padding' 2>&1)"; then
                # Everything imports. Two things that are not import problems but
                # do stop a multi-GPU run at NCCL init, reported rather than
                # enforced because neither is ours to decide.
                local shm_kb
                shm_kb="$(df -k /dev/shm 2>/dev/null | awk 'NR==2{print $2}')"
                if [ -n "${shm_kb}" ] && [ "${shm_kb}" -lt 1048576 ]; then
                    echo "  WARNING: /dev/shm is $((shm_kb / 1024)) MB."
                    echo "    NCCL uses shared memory for intra-node transport and fails"
                    echo "    obscurely when it is small -- 'Cuda failure 401' at init is one"
                    echo "    way it shows up. A pod with a larger --shm-size is the fix;"
                    echo "    NCCL_SHM_DISABLE=1 is the workaround, at a cost in throughput."
                fi
                if ! gpus_free; then
                    echo "  WARNING: something already holds VRAM on this box:"
                    gpu_holders | head -5 | sed 's/^/    /'
                    echo "    A run that died leaves CUDA contexts behind, and the next"
                    echo "    NCCL init can fail on them. ray stop --force, then check"
                    echo "    nvidia-smi returns to 0 MiB before starting a queue."
                fi
                return 0
            fi
            echo "  verl and ray are fine, but flash_attn does not load:"
            printf '%s\n' "${err}" | tail -6 | sed 's/^/    /'
            echo
            echo "  verl imports it unconditionally (dp_actor.py:43). It is not optional,"
            echo "  and there is no sdpa fallback on that path. Rebuild it against the"
            echo "  torch that is installed -- BOTH flags matter:"
            echo "      pip uninstall -y flash-attn flash_attn"
            echo "      pip install flash-attn --no-cache-dir --no-build-isolation"
            echo "  --no-cache-dir skips the wheel built against the old torch;"
            echo "  --no-build-isolation builds against the installed one rather than"
            echo "  a torch pip downloads into a sandbox."
            return 1
        fi
        echo "  verl imports, but ray cannot start a node:"
        printf '%s\n' "${err}" | tail -8 | sed 's/^/    /'
        # "The current node timed out during startup" names nothing. What
        # actually failed is in ray's own session logs -- on 2026-09-13 it was
        # first an opentelemetry ImportError and then, after that was rolled
        # back, a TypeError from ray calling a newer opentelemetry API than
        # vllm's pins allow. Neither string appears in the exception above.
        local rl
        for rl in /tmp/ray/session_latest/logs/dashboard.log \
                  /tmp/ray/session_latest/logs/gcs_server.err \
                  /tmp/ray/session_latest/logs/raylet.err; do
            [ -s "${rl}" ] || continue
            echo
            echo "  ${rl} (tail):"
            tail -12 "${rl}" | sed 's/^/    /'
        done
        if [ -f "${root}/scripts/check_env_pins.py" ]; then
            echo
            (cd "${root}" && python3 scripts/check_env_pins.py) || true
        fi
        return 1
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
        echo "    verl imports and ray starts now, so this looks specific to the run."
    else
        echo "    ^^ THE ENVIRONMENT IS BROKEN. Every remaining run will die the same way."
        echo "       Fix it, then re-run this queue -- finished runs are skipped."
    fi
    return 1
}

# --- is a trainer running? ---------------------------------------------------
# `pgrep -f` matches whole command lines, so the shell that launched this script
# matches too whenever the launch command mentions main_ppo. Excluding our own
# ancestors removes exactly those without hiding a real trainer. Three queues
# had their own copy of this; a second trainer on the same two cards OOMs both,
# so the check is worth having in exactly one place.
BUSY_RE=${BUSY_RE:-"[m]ain_ppo|[r]un_0905_chain"}
busy_pids () {
    local anc p
    anc=" "; p=$$
    while [ "${p}" != "1" ] && [ -r "/proc/${p}/status" ]; do
        anc="${anc}${p} "
        p="$(awk '/^PPid:/{print $2}' "/proc/${p}/status" 2>/dev/null)"
        [ -n "${p}" ] || break
    done
    # Our own cmdline, to drop forks of ourselves. Excluding ancestors is not
    # enough when the checking shell's OWN command line mentions main_ppo (a
    # tmux launch string, say): every subshell it forks inherits that cmdline,
    # matches pgrep, and is a DESCENDANT rather than an ancestor. A real trainer
    # is "python3 -m verl.trainer.main_ppo ..." and never byte-identical to the
    # shell asking the question.
    local self_cmd
    self_cmd="$(tr '\0' ' ' < "/proc/$$/cmdline" 2>/dev/null || true)"
    pgrep -f "${BUSY_RE}" 2>/dev/null | while read -r pid; do
        case "${anc}" in *" ${pid} "*) continue ;; esac
        [ -n "${self_cmd}" ] && \
            [ "$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)" = "${self_cmd}" ] \
            && continue
        echo "${pid}"
    done
}
is_busy () { [ -n "$(busy_pids)" ]; }

# --- are the GPUs actually usable? -------------------------------------------
# is_busy only matches main_ppo on the command line. A run that died leaves
# ray::WorkerDict and vLLM engine processes holding CUDA contexts, none of which
# match, so the queue reports an idle box and starts -- and NCCL then fails at
# init with
#     Cuda failure 401 'the operation cannot be performed in the present state'
# which names neither the leftover process nor the GPU. Checking what the run
# actually needs (a GPU it can make a context on) rather than only what the
# queue happens to grep for.
#
# Never kills anything: the holder may be someone else's job, and this is the
# box the campaign shares.
gpu_holders () {   # prints "pid used_memory" per process holding VRAM, ours excluded
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    local anc p line pid
    anc=" "; p=$$
    while [ "${p}" != "1" ] && [ -r "/proc/${p}/status" ]; do
        anc="${anc}${p} "
        p="$(awk '/^PPid:/{print $2}' "/proc/${p}/status" 2>/dev/null)"
        [ -n "${p}" ] || break
    done
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
    | while IFS=, read -r pid rest; do
        pid="$(printf '%s' "${pid}" | tr -dc '0-9')"
        [ -n "${pid}" ] || continue
        case "${anc}" in *" ${pid} "*) continue ;; esac
        printf '%s %s\n' "${pid}" "$(printf '%s' "${rest}" | sed 's/^ *//')"
    done
}

gpus_free () { [ -z "$(gpu_holders)" ]; }

# Wait for VRAM to come back after a run exits, then say so either way. Warning
# rather than refusing: a holder we did not start is not ours to wait out
# forever, and a queue that stops on it stops for hours unattended.
await_gpus () {   # [seconds]
    local left="${1:-${GPU_WAIT:-60}}"
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    while [ "${left}" -gt 0 ] && ! gpus_free; do sleep 5; left=$((left - 5)); done
    gpus_free && return 0
    echo "[gpu] VRAM is still held after waiting; starting anyway:"
    gpu_holders | head -5 | while read -r pid mem; do
        printf '[gpu]   pid %-8s %-10s %s\n' "${pid}" "${mem}" \
            "$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null | cut -c1-60)"
    done
    echo "[gpu] If NCCL then fails with 'Cuda failure 401', that is why."
    return 1
}

# Is one of the other queues holding its lock with a live process?
lock_holder () {   # <lock-dir>  -> prints the live pid, or nothing
    local h
    [ -d "$1" ] || return 1
    h="$(cat "$1/pid" 2>/dev/null || true)"
    [ -n "${h}" ] && kill -0 "${h}" 2>/dev/null && { echo "${h}"; return 0; }
    return 1
}
