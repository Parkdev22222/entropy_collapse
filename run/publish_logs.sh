#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Commit this box's finished training logs to the branch that holds them.
#
#   bash run/publish_logs.sh                 # show what would go, change nothing
#   bash run/publish_logs.sh --push          # commit and push
#   bash run/publish_logs.sh --done-only     # only runs that reached their last step
#   bash run/publish_logs.sh --queue-logs    # also the campaign/followups drivers
#   bash run/publish_logs.sh --pull          # bring the OTHER box's logs down
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

PUSH=0; DONE_ONLY=0; QUEUE_LOGS=0; FORCE=0; PULL=0
while [ $# -gt 0 ]; do
    case "$1" in
        --push)       PUSH=1 ;;
        --pull)       PULL=1 ;;
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

# How far a log got, so two boxes holding the same run name cannot destroy
# each other's work. On 2026-09-15 the H100 publish replaced four A100 logs
# with its own dead stubs of the same name -- including a finished 1 MB GRPO
# seed-2 run, overwritten by an 8 KB carcass from a start that failed in
# seconds. Nothing was lost (git keeps the old blob) but the branch tip, which
# is what every analysis reads, was wrong and said nothing about it.
last_step_in () {   # <file> -> the highest optimisation step it recorded
    grep -o 'step:[0-9]* - global_seqlen' "$1" 2>/dev/null \
        | grep -o '[0-9]\+' | sort -n | tail -1
}
# A log only ever grows, and it grows in whole lines. Two numbers say whether a
# copy is the same record or a damaged one, and the step count says neither.
n_lf ()    { tr -dc '\n' < "$1" 2>/dev/null | wc -c | tr -d ' '; }
n_bytes () { wc -c < "$1" 2>/dev/null | tr -d ' '; }

# Why this exists: on 2026-09-21 a push replaced train-grpo-<tag>-s4.log with a
# copy of itself whose every newline had become a carriage return -- 4057 LF and
# 133 CR became 0 LF and 4194 CR. Both copies reached step 110, so the clobber
# test below saw nothing to object to and overwrote the good one. The data
# survived (Python's splitlines() cuts on \r too, so analyze_seeds.py still read
# the run) but these logs are the ledger every number in the manuscript is
# recomputed from, and a path that silently degrades them is the defect.
degraded () {   # <incoming> <committed> -> prints why the incoming copy is worse
    local mine_lf theirs_lf mine_b theirs_b
    mine_lf="$(n_lf "$1")";     theirs_lf="$(n_lf "$2")"
    mine_b="$(n_bytes "$1")";   theirs_b="$(n_bytes "$2")"
    # Line endings mangled: the committed copy has real lines and ours lost them.
    if [ "${theirs_lf:-0}" -gt 100 ] \
       && [ "$(( mine_lf * 2 ))" -lt "${theirs_lf:-0}" ]; then
        echo "line endings: ours has ${mine_lf} newline(s), the branch has ${theirs_lf}"
        return 0
    fi
    # Truncated: append-only means a later copy of the same run cannot shrink.
    if [ "${mine_b:-0}" -lt "${theirs_b:-0}" ]; then
        echo "truncated: ours is ${mine_b} bytes, the branch has ${theirs_b}"
        return 0
    fi
    return 1
}
# run name from the file name, so train_log_done and trainer_pid_for can be
# asked about it. train-<run>.log and the recovery chain's train-<run>_<tag>.log.
run_of () {   # <path> -> run name
    local b; b="$(basename "$1" .log)"
    b="${b#train-}"
    printf '%s' "${b%%_*}"
}
# --- pull: teach this box what the other one has already finished -----------
# A queue decides what is left by reading logs/experiments on the box it runs
# on, so two boxes sharing a campaign disagree about what is done. On
# 2026-09-15 the A100 listed all ten follow-up arms as remaining while the
# H100 had already finished two of them: starting that queue would have
# re-run 29 GPU-hours of completed work.
#
# Only files this box does not have, or has a SHORTER version of, are taken --
# the same comparison the push side makes, for the same reason.
if [ "${PULL}" = "1" ]; then
    echo "pulling logs from origin/${BRANCH} into ${LOG_DIR}"
    for i in 1 2 3 4; do
        git fetch origin "${BRANCH}" && break
        echo "fetch failed, retry ${i}"; sleep $(( 2 ** i ))
    done
    peer="$(mktemp -d)"
    trap 'rm -rf "${peer}"' EXIT
    if ! git archive "origin/${BRANCH}" logs/experiments 2>/dev/null \
         | tar -x -C "${peer}" 2>/dev/null; then
        echo "REFUSE: could not read logs/experiments from origin/${BRANCH}." >&2
        exit 1
    fi
    took=0; kept=0
    mkdir -p "${LOG_DIR}"
    for f in "${peer}"/logs/experiments/train-*.log; do
        [ -f "${f}" ] || continue
        dest="${LOG_DIR}/$(basename "${f}")"
        if [ -f "${dest}" ]; then
            mine="$(last_step_in "${dest}")";  mine=${mine:-0}
            theirs="$(last_step_in "${f}")";   theirs=${theirs:-0}
            # Never overwrite a run this box got further on, and never touch a
            # log a trainer here is still appending to.
            rn="$(run_of "${dest}")"
            if [ "${mine}" -ge "${theirs}" ] || run_is_live "${rn}"; then
                kept=$(( kept + 1 )); continue
            fi
        fi
        cp "${f}" "${dest}"
        printf '   + %-58s step %s\n' "$(basename "${f}")" "$(last_step_in "${f}")"
        took=$(( took + 1 ))
    done
    echo "took ${took} log(s), left ${kept} alone (this box is at or ahead)."
    echo "Re-run the queues with DRY=1 to see what is actually left now."
    exit 0
fi

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
fetched=0
for i in 1 2 3 4; do
    if git fetch origin "${BRANCH}"; then fetched=1; break; fi
    echo "fetch failed, retry ${i}"; sleep $(( 2 ** i ))
done
# Four failed fetches is not a flaky network to shrug at. Carrying on rebases
# onto a stale ref and reports whatever goes wrong next, which on 2026-09-21
# meant a full disk ("Disk quota exceeded" on every fetch) surfacing three
# steps later as "You have unstaged changes" -- and the advice printed for
# THAT is to run the command that had just silently failed for the same
# reason. Stop where the real error is.
if [ "${fetched}" != "1" ]; then
    echo >&2
    echo "REFUSE: could not fetch origin/${BRANCH} after 4 tries." >&2
    echo "        The training tree is untouched and nothing was committed." >&2
    if ! df_out="$(df -h "${ROOT}" 2>/dev/null | tail -1)"; then df_out=""; fi
    [ -n "${df_out}" ] && echo "        disk: ${df_out}" >&2
    echo "        If the errors above say 'Disk quota exceeded' or 'No space" >&2
    echo "        left on device', free space before retrying -- git needs room" >&2
    echo "        to unpack objects. The fastest reclaim that uploads nothing:" >&2
    echo "          for r in \$(ls -1 checkpoints/STEER-F); do" >&2
    echo "            PRUNE=1 bash run/hf_backup.sh \"\$r\"; done" >&2
    exit 1
fi
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
# A run that died between `git add` and `git commit` leaves the index dirty,
# and every later run then failed at the rebase with "Your index contains
# uncommitted changes" -- permanently, because nothing here cleaned up after
# itself. The worktree is scratch space this script creates and only this
# script writes to, so reset it. Commits survive: an earlier run that committed
# but could not push keeps its work and pushes it below.
#
# NEVER when WORKTREE is ROOT. That branch is taken when the checkout is
# already on ${BRANCH}, and it is the user's real tree -- resetting it would
# throw away whatever they have in progress.
if [ "${WORKTREE}" != "${ROOT}" ]; then
    # Not `|| true`: reset --hard WRITES files, so it is the first thing a full
    # disk kills, and swallowing that turns "no space" into "unstaged changes"
    # at the rebase below -- whose advice is to run this very command.
    if ! reset_err="$(git -C "${WORKTREE}" reset -q --hard HEAD 2>&1)"; then
        echo "REFUSE: could not reset the scratch worktree ${WORKTREE}." >&2
        printf '        %s\n' "${reset_err}" >&2
        case "${reset_err}" in
            *"Disk quota exceeded"*|*"No space left"*)
                echo "        That is a full disk, not a git problem. Free space first:" >&2
                echo "          for r in \$(ls -1 checkpoints/STEER-F); do" >&2
                echo "            PRUNE=1 bash run/hf_backup.sh \"\$r\"; done" >&2 ;;
        esac
        exit 1
    fi
    git -C "${WORKTREE}" clean -qfd logs/experiments 2>/dev/null || true
elif ! git -C "${WORKTREE}" diff --quiet || ! git -C "${WORKTREE}" diff --cached --quiet; then
    echo "REFUSE: ${WORKTREE} has uncommitted changes and it is your working" >&2
    echo "        tree, not a scratch worktree -- this script will not reset it." >&2
    echo "        Commit or stash them, then re-run." >&2
    exit 1
fi

# rebase, not `pull --rebase`: the fetch above already ran.
git -C "${WORKTREE}" rebase -q "origin/${BRANCH}" || {
    git -C "${WORKTREE}" rebase --abort 2>/dev/null || true
    echo "REFUSE: could not rebase ${WORKTREE} onto origin/${BRANCH}." >&2
    echo "        The training tree is untouched. To see which case this is:" >&2
    echo "          git -C ${WORKTREE} log --oneline -2" >&2
    echo "          git -C ${WORKTREE} status --short" >&2
    echo "        A commit of your own that has not been pushed:" >&2
    echo "          git -C ${WORKTREE} pull --rebase origin ${BRANCH} && \\" >&2
    echo "            git -C ${WORKTREE} push origin ${BRANCH}" >&2
    echo "        Nothing committed (the usual case -- a half-finished run):" >&2
    echo "          git -C ${WORKTREE} reset --hard HEAD, then re-run this." >&2
    exit 1
}


mkdir -p "${WORKTREE}/logs/experiments"
staged=(); clobber=(); damaged=()
for f in "${take[@]}"; do
    dest="${WORKTREE}/logs/experiments/$(basename "${f}")"
    if [ -f "${dest}" ]; then
        mine="$(last_step_in "${f}")";  mine=${mine:-0}
        theirs="$(last_step_in "${dest}")"; theirs=${theirs:-0}
        if [ "${theirs}" -gt "${mine}" ] && [ "${FORCE}" != "1" ]; then
            clobber+=("$(basename "${f}") ours=${mine} theirs=${theirs}")
            continue
        fi
        # Same step, and still not the same record. Checked separately because
        # the step test above passes exactly when both copies finished.
        if [ "${FORCE}" != "1" ] && why="$(degraded "${f}" "${dest}")"; then
            clobber+=("$(basename "${f}") -- ${why}")
            damaged+=("$(basename "${f}")")
            continue
        fi
    fi
    cp "${f}" "${dest}"
    staged+=("logs/experiments/$(basename "${f}")")
done
if [ ${#clobber[@]} -gt 0 ]; then
    echo
    echo " NOT overwritten -- the branch already has a run that got further:"
    for c in "${clobber[@]}"; do printf '   %s\n' "${c}"; done
    echo " Yours are almost certainly carcasses of a start that failed here."
    echo " --force overwrites anyway."
fi
if [ ${#damaged[@]} -gt 0 ]; then
    echo
    echo " Those are not shorter runs -- they are damaged copies of the same"
    echo " run. The branch has the intact one, which is what this branch is"
    echo " for. Restore each from it and the next publish is a no-op:"
    for d in "${damaged[@]}"; do
        printf '   git show origin/%s:logs/experiments/%s > %s/%s\n' \
            "${BRANCH}" "${d}" "${LOG_DIR}" "${d}"
    done
fi
if [ ${#staged[@]} -eq 0 ]; then
    echo
    echo "nothing to stage -- every file is already there, or newer there."
    exit 0
fi
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
    # The other box pushed while we were committing; rebase onto it and retry.
    git -C "${WORKTREE}" fetch origin "${BRANCH}" || true
    git -C "${WORKTREE}" rebase -q "origin/${BRANCH}" \
        || git -C "${WORKTREE}" rebase --abort 2>/dev/null || true
done
echo
git -C "${WORKTREE}" log --oneline -1
echo "done. The training tree was never checked out to ${BRANCH}."
