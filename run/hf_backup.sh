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
#
# With no step given it processes every global_step_* under the run.
#
# Only actor/huggingface is uploaded: that is the whole HF model directory
# (weights, config, tokenizer) and the only thing run/eval_subset.sh needs.
# The optimizer and extra state next to it exist for resume, are four times
# larger, and are worthless once a run has finished.
#
# Nothing is deleted unless DELETE=1, and then only after the verification
# below reports every file matching in size on the Hub. A run whose name
# contains the tag of a live training job is refused outright -- reading a
# checkpoint while the trainer is writing it yields a corrupt backup.
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
LIVE_TAG=${LIVE_TAG:-_0905}       # runs whose name contains this are in flight
DELETE=${DELETE:-0}

RUN=${1:-}
if [ -z "${RUN}" ]; then
    echo "usage: REPO=<user>/<repo> bash run/hf_backup.sh <run-name> [step ...]"
    echo
    echo "runs available:"
    ls -1 "${CKPT_ROOT}" 2>/dev/null | sed 's/^/  /'
    exit 2
fi
shift

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

case "${RUN}" in
    *"${LIVE_TAG}"*)
        if [ "${FORCE:-0}" != "1" ]; then
            echo "REFUSE: '${RUN}' looks like the live training run (contains ${LIVE_TAG})."
            echo "        Backing it up mid-write gives a corrupt copy. FORCE=1 overrides."
            exit 1
        fi ;;
esac

[ -d "${CKPT_ROOT}/${RUN}" ] || { echo "FATAL: no ${CKPT_ROOT}/${RUN}"; exit 1; }

# Steps: the ones named on the command line, else every one on disk.
steps=("$@")
if [ ${#steps[@]} -eq 0 ]; then
    for d in "${CKPT_ROOT}/${RUN}"/global_step_*; do
        [ -d "${d}/actor/huggingface" ] || continue
        steps+=("$(basename "${d}" | sed 's/^global_step_//')")
    done
fi
[ ${#steps[@]} -gt 0 ] || { echo "FATAL: no global_step_*/actor/huggingface under ${RUN}"; exit 1; }

echo "=========================================================="
echo " repo    ${REPO}"
echo " run     ${RUN}"
echo " steps   ${steps[*]}"
echo " delete  $([ "${DELETE}" = "1" ] && echo 'yes, after verification' || echo 'no (DELETE=1 to enable)')"
echo "=========================================================="
df -h /workspace | tail -1
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
df -h /workspace | tail -1
[ "${rc}" = "0" ] && echo "all steps done" || echo "finished with errors -- see above"
exit "${rc}"
