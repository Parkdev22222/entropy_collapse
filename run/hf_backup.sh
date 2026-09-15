#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Back a training run's checkpoints up to the Hugging Face Hub, verify the
# upload byte for byte, and (only when asked) delete the local copy.
#
#   REPO=<user>/<repo> bash run/hf_backup.sh <run-name> [step ...]
#
# Examples
#   REPO=DSDSh/steer-f_2 bash run/hf_backup.sh steer-f-Qwen2.5-Math-1.5B-s1-tree-rollout-permuted
#   REPO=DSDSh/steer-f_2 DELETE=1 bash run/hf_backup.sh <run> 110
#   PRUNE=1 bash run/hf_backup.sh <run>        # drop optimizer state, no upload
#
# With no step given it processes every global_step_* under the run.
#
# Only actor/huggingface is uploaded: that is the whole HF model directory
# (weights, config, tokenizer) and the only thing run/eval_subset.sh needs.
# The optimizer and extra state next to it exist for resume, are four times
# larger, and are worthless once a run has finished.
#
# Nothing is deleted unless DELETE=1, and then only after the verification
# below reports every file matching in size on the Hub.
#
# A run that is still being written is refused. That decision is made from
# STATE, not from the run's name:
#   - a trainer process whose command line carries this run name,
#   - a training log that has not reached STEPS,
#   - a checkpoint directory touched within FRESH_MIN minutes.
# The name test (LIVE_TAG) is kept for compatibility but is no longer what
# protects you: LIVE_TAG defaulted to _0905, which only run_0905_chain.sh ever
# appends, so every campaign run -- steer-f-<tag>-s3-tree-rollout and friends --
# sailed straight past it. Uploading a half-written checkpoint also passes
# verification, because verification compares local size against Hub size and a
# truncated file matches the truncated copy of itself.
#
# PRUNE=1 deletes the optimizer and extra state of a finished run without
# uploading anything. That is most of the bytes on disk and none of what an
# evaluation needs, so it is the fastest way to get a full volume back. It runs
# the same safety checks first.
#
# Restore:
#   hf download "$REPO" --repo-type model \
#       --include "<run>/global_step_<N>/*" --local-dir /tmp/restore
#   mkdir -p checkpoints/STEER-F/<run>/global_step_<N>/actor
#   mv /tmp/restore/<run>/global_step_<N> \
#      checkpoints/STEER-F/<run>/global_step_<N>/actor/huggingface
# ---------------------------------------------------------------------------
set -uo pipefail

STEER_ROOT=${STEER_ROOT:-/workspace/entropy_collapse}
cd "${STEER_ROOT}" || { echo "FATAL: no ${STEER_ROOT}"; exit 1; }

CKPT_ROOT="${STEER_ROOT}/checkpoints/STEER-F"
LIVE_TAG=${LIVE_TAG:-_0905}       # legacy name test; see the header
DELETE=${DELETE:-0}
PRUNE=${PRUNE:-0}
FRESH_MIN=${FRESH_MIN:-30}        # a checkpoint touched this recently is live
LOG_DIR=${LOG_DIR:-${STEER_ROOT}/logs/experiments}
STEPS=${STEPS:-110}

# is_busy/busy_pids/train_log_done live in _arms.sh, which is the same judgement
# migrate_pod.sh --export was taught to use (task 14). Sourcing it is optional:
# on a box that only has this one script the mtime check below still stands.
# shellcheck source=/dev/null
[ -f "${STEER_ROOT}/run/_arms.sh" ] && . "${STEER_ROOT}/run/_arms.sh" 2>/dev/null

RUN=${1:-}
if [ -z "${RUN}" ]; then
    echo "usage: REPO=<user>/<repo> bash run/hf_backup.sh <run-name> [step ...]"
    echo
    echo "runs available:"
    ls -1 "${CKPT_ROOT}" 2>/dev/null | sed 's/^/  /'
    exit 2
fi
shift

[ -d "${CKPT_ROOT}/${RUN}" ] || { echo "FATAL: no ${CKPT_ROOT}/${RUN}"; exit 1; }

# --- is this run still being written? ---------------------------------------
# Three independent tests, because each one has a hole the others cover:
# a trainer in another container is invisible to pgrep, a log written elsewhere
# never reaches LOG_DIR, and a run that died mid-save leaves a stale directory
# with no process and no final step. Any one of them firing is enough to stop.
live_reason () {   # <run> -> prints why the run is unsafe to read, or nothing
    local run="$1" pid cmd fresh

    # 1. a trainer whose command line carries this run name. Our own process
    #    tree is already excluded by busy_pids (see _arms.sh), so a match here
    #    is someone else's live job -- or ours, which is worse.
    if command -v busy_pids >/dev/null 2>&1; then
        for pid in $(busy_pids); do
            cmd="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
            case "${cmd}" in
                *"${run}"*) echo "pid ${pid} is training it right now"; return 0 ;;
            esac
        done
    fi

    # 2. a checkpoint directory touched in the last FRESH_MIN minutes. This is
    #    the test that does not depend on seeing the process or the log, and it
    #    is the one that would have caught the campaign runs.
    fresh="$(find "${CKPT_ROOT}/${run}" -maxdepth 3 -mmin "-${FRESH_MIN}" -print -quit 2>/dev/null)"
    if [ -n "${fresh}" ]; then
        echo "${fresh#"${CKPT_ROOT}/"} was written in the last ${FRESH_MIN} min"
        return 0
    fi

    # 3. the legacy name test. Kept so an explicit LIVE_TAG still works, but it
    #    is last now: it protects only runs that happen to carry the tag.
    case "${run}" in
        *"${LIVE_TAG}"*) echo "the name contains LIVE_TAG=${LIVE_TAG}"; return 0 ;;
    esac
    return 0
}

live_guard () {   # <run> -> 1 when the run must not be read
    local run="$1" why other
    why="$(live_reason "${run}")"
    if [ -z "${why}" ]; then
        # Not live, but possibly incomplete. train_log_done is the same test
        # the queues use to decide a run is finished; a run that never reached
        # STEPS is a crashed run, whose checkpoints are still worth keeping --
        # so this warns rather than refuses.
        if command -v train_log_done >/dev/null 2>&1 \
           && ! train_log_done "${LOG_DIR}" "${run}" "${STEPS}"; then
            echo "WARN: no train log for '${run}' reaching step ${STEPS}."
            echo "      It either crashed or its log lives elsewhere. The"
            echo "      checkpoints are readable; just know they are not a"
            echo "      completed run. (LOG_DIR=${LOG_DIR} STEPS=${STEPS})"
            echo
        fi
        return 0
    fi
    if [ "${FORCE:-0}" = "1" ]; then
        echo "WARN: ${why} -- FORCE=1, continuing anyway."
        echo
        return 0
    fi
    echo "REFUSE: '${run}' is still being written -- ${why}."
    echo "        Reading a checkpoint mid-write gives a corrupt copy, and"
    echo "        verification will not catch it: it compares local size to Hub"
    echo "        size, and a truncated file matches its own truncated upload."
    echo
    other=""
    for d in "${CKPT_ROOT}"/*/; do
        d="$(basename "${d}")"
        [ "${d}" = "${run}" ] && continue
        [ -z "$(live_reason "${d}")" ] && other="${other}          ${d}"$'\n'
    done
    if [ -n "${other}" ]; then
        echo "        These are safe to back up instead:"
        printf '%s' "${other}"
    else
        echo "        No other run here is safe to back up either."
    fi
    echo
    echo "        FORCE=1 overrides. FRESH_MIN=${FRESH_MIN} sets the quiet window."
    return 1
}

live_guard "${RUN}" || exit 1

# --- PRUNE: reclaim the bytes that never needed to be uploaded --------------
# actor/ holds the FSDP shards, optimizer and extra state next to huggingface/.
# Only huggingface/ is needed to evaluate or to publish; the rest exists to
# resume, which a finished run will not do. This is the fastest way to get a
# full volume back, and it needs no network.
if [ "${PRUNE}" = "1" ]; then
    echo "=========================================================="
    echo " prune   ${RUN}  (keeping actor/huggingface, dropping the rest)"
    echo "=========================================================="
    df -h "${STEER_ROOT}" | tail -1
    freed=0
    for d in "${CKPT_ROOT}/${RUN}"/global_step_*/actor; do
        [ -d "${d}" ] || continue
        # No huggingface/ means the eval copy is not there yet; removing the
        # shards would leave nothing at all.
        if [ ! -d "${d}/huggingface" ]; then
            echo "  skip $(basename "$(dirname "${d}")"): no actor/huggingface"
            continue
        fi
        while IFS= read -r item; do
            [ -n "${item}" ] || continue
            echo "  rm $(du -sh "${item}" 2>/dev/null | cut -f1)  ${item#"${CKPT_ROOT}/"}"
            rm -rf "${item}"
            freed=$((freed + 1))
        done <<< "$(find "${d}" -maxdepth 1 -mindepth 1 ! -name huggingface 2>/dev/null)"
    done
    echo
    df -h "${STEER_ROOT}" | tail -1
    echo "pruned ${freed} item(s). actor/huggingface kept everywhere."
    exit 0
fi

# Steps: the ones named on the command line, else every one on disk.
steps=("$@")
if [ ${#steps[@]} -eq 0 ]; then
    for d in "${CKPT_ROOT}/${RUN}"/global_step_*; do
        [ -d "${d}/actor/huggingface" ] || continue
        steps+=("$(basename "${d}" | sed 's/^global_step_//')")
    done
fi
if [ ${#steps[@]} -eq 0 ]; then
    # "nothing to upload" has three very different causes and the old message
    # named none of them, so show what is actually on disk.
    echo "FATAL: no global_step_*/actor/huggingface under ${RUN}"
    echo
    echo "  what is there:"
    find "${CKPT_ROOT}/${RUN}" -maxdepth 3 2>/dev/null | head -20 | sed 's/^/    /'
    echo
    echo "  the usual causes:"
    echo "    - the run has not saved yet. verl writes actor/huggingface at its"
    echo "      first save, so an empty run directory means no checkpoint exists."
    echo "      Check how far it got:"
    echo "        grep -o 'step:[0-9]* - global_seqlen' \\"
    echo "          ${LOG_DIR}/train-${RUN}.log | tail -1"
    echo "    - it was already backed up and deleted (DELETE=1 leaves the run"
    echo "      directory behind, empty). Check the Hub before re-uploading."
    echo "    - a step was named on the command line that is not on disk."
    exit 1
fi

: "${REPO:?set REPO to the Hub repo id, e.g. REPO=DSDSh/steer-f_2}"

# huggingface_hub renamed its CLI from `huggingface-cli` to `hf` in 0.34. Accept
# either. This matters more than it looks: on 2026-09-08 a missing `hf` is the
# most likely reason someone ran an unpinned `pip install -U huggingface_hub`,
# which pulled 1.30.0, broke the pin transformers enforces at import, and killed
# two queued training arms. Taking the old name removes the reason to upgrade.
HF_CLI=${HF_CLI:-$(command -v hf || command -v huggingface-cli || true)}
if [ -z "${HF_CLI}" ]; then
    echo "FATAL: neither 'hf' nor 'huggingface-cli' is on PATH."
    echo "       Install it INSIDE the pin transformers declares -- an unpinned"
    echo "       upgrade breaks every training run:"
    echo "         pip install \"huggingface_hub>=0.34,<1.0\""
    exit 1
fi

echo "=========================================================="
echo " repo    ${REPO}"
echo " run     ${RUN}"
echo " steps   ${steps[*]}"
echo " delete  $([ "${DELETE}" = "1" ] && echo 'yes, after verification' || echo 'no (DELETE=1 to enable)')"
echo "=========================================================="
df -h "${STEER_ROOT}" | tail -1
echo

verify () {   # <run> <step> -> 0 when every local file matches the Hub
    RUN="$1" STEP="$2" REPO="${REPO}" python3 - <<'PY'
import os, sys
from huggingface_hub import HfApi

repo, run, step = os.environ["REPO"], os.environ["RUN"], os.environ["STEP"]
src    = f"checkpoints/STEER-F/{run}/global_step_{step}/actor/huggingface"
prefix = f"{run}/global_step_{step}"

if not os.path.isdir(src):
    print(f"  FAILED: no local {src}"); sys.exit(1)
local = {f: os.path.getsize(os.path.join(src, f))
         for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))}
if not local:
    print("  FAILED: local directory is empty"); sys.exit(1)
try:
    remote = {os.path.basename(e.path): e.size
              for e in HfApi().list_repo_tree(repo, path_in_repo=prefix,
                                              recursive=True, repo_type="model")
              if getattr(e, "size", None) is not None}
except Exception as exc:
    print(f"  FAILED: cannot read the Hub tree: {exc}"); sys.exit(1)

bad = [f for f, s in local.items() if remote.get(f) != s]
for f, s in sorted(local.items()):
    print(f"  {'ok      ' if remote.get(f) == s else 'MISMATCH'}  {f}  local={s} remote={remote.get(f)}")
if bad:
    print(f"  FAILED: {len(bad)} file(s) differ"); sys.exit(1)
print(f"  VERIFIED: {len(local)} files match")
PY
}

rc=0
for step in "${steps[@]}"; do
    src="${CKPT_ROOT}/${RUN}/global_step_${step}/actor/huggingface"
    echo "---- global_step_${step}  ($(du -sh "${src}" 2>/dev/null | cut -f1))"
    if [ ! -d "${src}" ]; then echo "  skip: ${src} missing"; rc=1; continue; fi

    "${HF_CLI}" upload "${REPO}" "${src}" "${RUN}/global_step_${step}" \
        --repo-type model --commit-message "backup ${RUN} step ${step}"
    if [ $? -ne 0 ]; then echo "  upload FAILED -- not verifying, not deleting"; rc=1; continue; fi

    if ! verify "${RUN}" "${step}"; then
        echo "  verification FAILED -- local copy kept"; rc=1; continue
    fi

    if [ "${DELETE}" = "1" ]; then
        rm -rf "${CKPT_ROOT}/${RUN}/global_step_${step}"
        echo "  deleted ${CKPT_ROOT}/${RUN}/global_step_${step}"
    fi
    echo
done

echo "=========================================================="
df -h "${STEER_ROOT}" | tail -1
[ "${rc}" = "0" ] && echo "all steps done" || echo "finished with errors -- see above"
exit "${rc}"
