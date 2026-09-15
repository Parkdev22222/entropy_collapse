#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Commit this box's finished training logs to the branch that holds them.
#
#   bash run/publish_logs.sh                 # show what would go, change nothing
#   bash run/publish_logs.sh --push          # commit and push
#   bash run/publish_logs.sh --done-only     # only runs that reached their last step
#   bash run/publish_logs.sh --queue-logs    # also the campaign/followups drivers
#
# The logs are the experiment record: every number in the manuscript is
# recomputed from them (scripts/analyze_seeds.py, scripts/seed1_table.py
# --git-ref origin/paper). A box that is reclaimed with an uncommitted log
# takes that run's evidence with it -- which has already happened once, to the
# GRPO seed-1 arm, whose numbers now survive only as a transcript in
# docs/seed1_grpo_transcript.json.
#
# Three things make this more than `git add logs`:
#
#   1. NEVER CHECK OUT ANOTHER BRANCH IN A TRAINING TREE. The running trainer
#      reads run/ and steer_f/ off the working tree; switching branches under
#      it swaps the code mid-run. This uses `git worktree`, so the training
#      tree is never touched.
#
#   2. A LIVE RUN'S LOG IS STILL BEING WRITTEN. Committing it freezes a partial
#      record, and the commit says "results" while holding half of one. Runs
#      with a trainer on them are skipped unless --force.
#
#   3. validation_data/ IS NOT GITIGNORED and is ~150 MB per run. `git add -A`
#      in the wrong directory puts it in history permanently. Only the files
#      listed below are ever staged.
#
# Two boxes publishing to one branch is expected (A100 and H100): the commit
# message names the box, the push rebases onto whatever the other one pushed,
# and the file sets are disjoint so there is nothing to resolve.
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}" || { echo "FATAL: cannot cd to ${ROOT}" >&2; exit 1; }

# shellcheck source=run/_arms.sh
. "${ROOT}/run/_arms.sh"

BRANCH=${BRANCH:-paper}
LOG_DIR=${LOG_DIR:-${ROOT}/logs/experiments}
WORKTREE=${WORKTREE:-${ROOT}/../entropy_logs_${BRANCH}}
BIG_MB=${BIG_MB:-20}

# LOG_DIR must belong to THIS checkout. The script resolves ROOT from its own
# location, so calling it by an absolute path while pointing LOG_DIR somewhere
# else publishes those files into THIS repository's branch -- which is how
# three synthetic test fixtures reached origin/paper on 2026-09-15, named
# exactly like real arms (train-grpo-<tag>-s2.log), where analyze_seeds.py
# would have read them as results. Overridable for tests that drive a fixture
# checkout, never as a convenience.
log_dir_guard () {
    local real_root real_logs
    real_root="$(cd "${ROOT}" 2>/dev/null && pwd -P)" || return 1
    real_logs="$(cd "${LOG_DIR}" 2>/dev/null && pwd -P)" || {
        echo "FATAL: no LOG_DIR at ${LOG_DIR}" >&2; return 1; }
    case "${real_logs}/" in
        "${real_root}"/*) return 0 ;;
    esac
    echo "REFUSE: LOG_DIR is outside this checkout." >&2
    echo "        LOG_DIR ${real_logs}" >&2
    echo "        repo    ${real_root}" >&2
    echo "        Publishing would commit those files to THIS repository's" >&2
    echo "        '${BRANCH}' branch. Run the script from the checkout the logs" >&2
    echo "        belong to, or set ALLOW_FOREIGN_LOGS=1 if you really mean it." >&2
    return 1
}

PUSH=0; DONE_ONLY=0; QUEUE_LOGS=0; FORCE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --push)       PUSH=1 ;;
        --done-only)  DONE_ONLY=1 ;;
        --queue-logs) QUEUE_LOGS=1 ;;
        --force)      FORCE=1 ;;
        --branch)     shift; BRANCH="$1"; WORKTREE="${ROOT}/../entropy_logs_${BRANCH}" ;;
        -h|--help)    sed -n '2,40p' "$0"; exit 0 ;;
        *)            echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

[ "${ALLOW_FOREIGN_LOGS:-0}" = "1" ] || log_dir_guard || exit 1

# --- what box is this? ------------------------------------------------------
# The commit message has to say, because the same arm name means different
# things on different hardware: timings are not comparable across boxes, and
# scripts/analyze_seeds.py's compute-matched control integrates seconds.
box_tag () {
    local gpu n
    gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    n="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
    if [ -n "${gpu}" ]; then
        # "NVIDIA A100-SXM4-80GB" -> "A100x2"
        printf '%sx%s' "$(echo "${gpu}" | grep -oE '(A100|H100|H200|L40S|A6000|V100)' | head -1)" "${n}"
    else
        printf '%s' "$(hostname 2>/dev/null || echo unknown-box)"
    fi
}

# --- pick the files ---------------------------------------------------------
# Only names, and only from LOG_DIR. Nothing globs outside it.
candidates=()
for f in "${LOG_DIR}"/train-*.log "${LOG_DIR}"/warmup-*.log; do
    [ -f "${f}" ] && candidates+=("${f}")
done
if [ "${QUEUE_LOGS}" = "1" ]; then
    for f in "${LOG_DIR}"/campaign.log "${LOG_DIR}"/followups.log \
             "${LOG_DIR}"/backbones.log "${LOG_DIR}"/paper_*.log \
             "${LOG_DIR}"/chain_*.log; do
        [ -f "${f}" ] && candidates+=("${f}")
    done
fi
if [ ${#candidates[@]} -eq 0 ]; then
    echo "nothing to publish: no train-*.log or warmup-*.log under ${LOG_DIR}"
    exit 0
fi

# run name from the file name, so train_log_done and trainer_pid_for can be
# asked about it. train-<run>.log and the recovery chain's train-<run>_<tag>.log.
run_of () {   # <path> -> run name
    local b; b="$(basename "$1" .log)"
    b="${b#train-}"
    printf '%s' "${b%%_*}"
}

# A run's expected final step. steps_for_arm knows the one exception
# (grpo-long, the compute-matched control, runs to 200), so ask it by arm
# rather than assuming 110 everywhere.
steps_of () {   # <run name> -> final step
    local a s
    for a in ${CAMPAIGN_ARMS} ${FOLLOWUP_ARMS:-} grpo-long; do
        for s in 1 2 3 4 5; do
            [ "$(run_name_for "${a}" "${s}" 2>/dev/null)" = "$1" ] && {
                steps_for_arm "${a}"; return 0; }
        done
    done
    echo "${STEPS:-110}"
}

take=(); skipped=(); unfinished=()
for f in "${candidates[@]}"; do
    case "$(basename "${f}")" in
        train-*) ;;
        *) take+=("${f}"); continue ;;          # warmup / queue logs: no run name
    esac
    rn="$(run_of "${f}")"
    if run_is_live "${rn}" && [ "${FORCE}" != "1" ]; then
        skipped+=("${f}")
        continue
    fi
    if ! train_log_done "${LOG_DIR}" "${rn}" "$(steps_of "${rn}")"; then
        unfinished+=("${f}")
        [ "${DONE_ONLY}" = "1" ] && continue
    fi
    take+=("${f}")
done

# --done-only drops the unfinished ones from `take`, so count what is actually
# going rather than what was considered -- a header that says "1 unfinished"
# about a set that has none is the kind of small lie that costs trust later.
n_unfinished=0
for u in ${unfinished[@]+"${unfinished[@]}"}; do
    for t in ${take[@]+"${take[@]}"}; do
        [ "${u}" = "${t}" ] && n_unfinished=$(( n_unfinished + 1 ))
    done
done

echo "=========================================================="
echo " box       $(box_tag)"
echo " branch    ${BRANCH}"
if [ "$(git rev-parse --abbrev-ref HEAD 2>/dev/null)" = "${BRANCH}" ]; then
    echo " worktree  ${ROOT}  (already on ${BRANCH})"
else
    echo " worktree  ${WORKTREE}"
fi
echo " files     ${#take[@]} to publish, ${n_unfinished} of them unfinished,"
echo "           ${#skipped[@]} skipped as live"
echo "=========================================================="
for f in "${take[@]}"; do
    rn="$(run_of "${f}")"
    mark="  "
    for u in ${unfinished[@]+"${unfinished[@]}"}; do
        [ "${u}" = "${f}" ] && mark="~ "
    done
    printf ' %s%-58s %s\n' "${mark}" "$(basename "${f}")" \
        "$(du -h "${f}" 2>/dev/null | cut -f1)"
done
if [ ${#skipped[@]} -gt 0 ]; then
    echo
    echo " skipped (a trainer is writing these right now; --force overrides):"
    for f in "${skipped[@]}"; do printf '   %s\n' "$(basename "${f}")"; done
fi
if [ "${n_unfinished}" -gt 0 ]; then
    echo
    echo " '~' never reached its final step. They are published anyway: a crashed"
    echo " run's log is the evidence for why it crashed, and diagnose_run_failure"
    echo " reads it. --done-only leaves them out."
fi

# A log that has grown past a sane size is usually a queue driver that captured
# a preflight loop, and git keeps it forever.
for f in "${take[@]}"; do
    mb=$(( $(stat -c %s "${f}" 2>/dev/null || echo 0) / 1048576 ))
    [ "${mb}" -ge "${BIG_MB}" ] && echo " WARN: $(basename "${f}") is ${mb} MB"
done

if [ ${#take[@]} -eq 0 ]; then
    echo
    echo "nothing to publish."
    exit 0
fi

if [ "${PUSH}" != "1" ]; then
    echo
    echo "dry run -- nothing was committed. Add --push to commit and push."
    exit 0
fi

# --- the worktree -----------------------------------------------------------
echo
for i in 1 2 3 4; do
    git fetch origin "${BRANCH}" && break
    echo "fetch failed, retry ${i}"; sleep $(( 2 ** i ))
done
# If this checkout is ALREADY on the target branch there is nothing to switch,
# so a worktree would only fail ("'paper' is already used by worktree at ...").
# Use the tree itself -- but not while a trainer is reading it, because the
# rebase below pulls whatever the other box pushed, which may include code.
cur_branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
if [ "${cur_branch}" = "${BRANCH}" ]; then
    if is_busy && [ "${FORCE}" != "1" ]; then
        echo "REFUSE: this tree is already on '${BRANCH}' and a trainer is running." >&2
        echo "        Pulling would change files under it. Publish from another" >&2
        echo "        checkout, or wait, or --force if you know the pull is logs only." >&2
        exit 1
    fi
    WORKTREE="${ROOT}"
elif [ ! -d "${WORKTREE}/.git" ] && [ ! -f "${WORKTREE}/.git" ]; then
    if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
        git worktree add "${WORKTREE}" "${BRANCH}" || exit 1
    else
        git worktree add --track -b "${BRANCH}" "${WORKTREE}" "origin/${BRANCH}" || exit 1
    fi
fi
git -C "${WORKTREE}" pull --rebase origin "${BRANCH}" || {
    echo "REFUSE: could not rebase ${WORKTREE} onto origin/${BRANCH}." >&2
    echo "        Resolve it there, then re-run. The training tree is untouched." >&2
    exit 1
}

mkdir -p "${WORKTREE}/logs/experiments"
staged=()
for f in "${take[@]}"; do
    cp "${f}" "${WORKTREE}/logs/experiments/$(basename "${f}")"
    staged+=("logs/experiments/$(basename "${f}")")
done
git -C "${WORKTREE}" add -- "${staged[@]}" || exit 1

if git -C "${WORKTREE}" diff --cached --quiet; then
    echo "already up to date on ${BRANCH} -- nothing changed."
    exit 0
fi
git -C "${WORKTREE}" diff --cached --stat | tail -5

msg="logs: $(box_tag) training logs, $(date -u +%Y-%m-%d)

Published by run/publish_logs.sh from $(hostname 2>/dev/null || echo a pod).
${#take[@]} log(s), ${n_unfinished} of which never reached their final step.
These are the record every number is recomputed from -- analyze_seeds.py and
seed1_table.py read this branch."

git -C "${WORKTREE}" commit -q -m "${msg}" || exit 1
for i in 1 2 3 4; do
    git -C "${WORKTREE}" push origin "${BRANCH}" && break
    echo "push failed, retry ${i}"; sleep $(( 2 ** i ))
    git -C "${WORKTREE}" pull --rebase origin "${BRANCH}" || true
done
echo
git -C "${WORKTREE}" log --oneline -1
echo "done. The training tree was never checked out to ${BRANCH}."
