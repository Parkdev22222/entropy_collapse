#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Everything that happens AFTER the 20-run campaign, as one unattended queue.
#
#   tmux new -d -s phase2 \
#     "cd /workspace/entropy_collapse && \
#      REPO=DSDSh/steer-f_2 bash run/run_phase2.sh > logs/experiments/phase2.log 2>&1"
#
#   DRY=1 bash run/run_phase2.sh     # print the whole plan, run nothing
#
# Order, and why:
#   1. wait   for .campaign.lock to clear AND every main_ppo to exit. Starting
#             a second trainer on the same two cards OOMs both, so this waits
#             rather than refusing -- the point of arming it early is that
#             nobody has to be awake when the campaign finishes.
#   2. patch  run/instrument_phase2.sh --apply. Safe here and nowhere else:
#             it edits eval_steerf.sh and run_uniform_ablation.sh, and this is
#             the first moment no run is reading them.
#   3. eval   the 20 campaign runs on six benchmarks (~13 GPU-h, no training).
#             First, because it turns work already paid for into table rows.
#   4. train  the nine follow-up ablations (~352 GPU-h, seed 1 each).
#   5. eval   those nine (~6 GPU-h).
#
# Steps 3-5 are individually resumable: each queue skips what is already done,
# so killing this and re-arming it costs nothing.
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}" || { echo "FATAL: cannot cd to ${ROOT}" >&2; exit 1; }

DRY=${DRY:-0}
REPO=${REPO:-}
WAIT_POLL=${WAIT_POLL:-300}
SKIP_EVAL=${SKIP_EVAL:-0}
SKIP_FOLLOWUPS=${SKIP_FOLLOWUPS:-0}
EVAL_FOLLOWUPS=${EVAL_FOLLOWUPS:-1}
BUSY_RE="[m]ain_ppo|[r]un_0905_chain"

banner () { printf '\n############################################\n# %s\n############################################\n' "$*"; }

busy_pids () {
    local anc p
    anc=" "; p=$$
    while [ "${p}" != "1" ] && [ -r "/proc/${p}/status" ]; do
        anc="${anc}${p} "
        p="$(awk '/^PPid:/{print $2}' "/proc/${p}/status" 2>/dev/null)"
        [ -n "${p}" ] || break
    done
    pgrep -f "${BUSY_RE}" 2>/dev/null | while read -r pid; do
        case "${anc}" in *" ${pid} "*) continue ;; esac
        echo "${pid}"
    done
}
campaign_running () {
    local holder
    if [ -d "${ROOT}/.campaign.lock" ]; then
        holder="$(cat "${ROOT}/.campaign.lock/pid" 2>/dev/null || echo '')"
        if [ -n "${holder}" ] && kill -0 "${holder}" 2>/dev/null; then return 0; fi
    fi
    [ -n "$(busy_pids)" ]
}

if [ "${DRY}" = "1" ]; then
    banner "DRY RUN -- the plan"
    echo "--- 3. evaluate the campaign runs"
    DRY=1 bash run/run_eval_all.sh
    echo
    echo "--- 4. the follow-up ablations"
    DRY=1 bash run/run_followups.sh
    echo
    echo "--- 5. evaluate the follow-ups"
    DRY=1 EVAL_SET=followups bash run/run_eval_all.sh
    echo
    echo "(DRY=1: nothing was started, nothing was patched)"
    exit 0
fi

# --------------------------------------------------------------------- lock
LOCK="${ROOT}/.phase2.lock"
if ! mkdir "${LOCK}" 2>/dev/null; then
    holder="$(cat "${LOCK}/pid" 2>/dev/null || echo '?')"
    if [ "${holder}" != "?" ] && kill -0 "${holder}" 2>/dev/null; then
        echo "REFUSE: phase 2 is already queued (pid ${holder})." >&2
        exit 2
    fi
    echo "[phase2] reclaiming a stale lock from pid ${holder}"
    rm -rf "${LOCK}"; mkdir "${LOCK}" || { echo "FATAL: cannot take ${LOCK}" >&2; exit 2; }
fi
echo $$ > "${LOCK}/pid"
trap 'rm -rf "${LOCK}"' EXIT INT TERM

# ------------------------------------------------------------------ 1. wait
banner "$(date -Is)  phase 2 armed -- waiting for the campaign"
if campaign_running; then
    echo "[phase2] campaign or trainer still up; polling every ${WAIT_POLL}s"
    busy_pids | head -3 | while read -r h; do
        printf '[phase2]   holder: %s %s\n' "${h}" \
            "$(tr '\0' ' ' < "/proc/${h}/cmdline" 2>/dev/null | cut -c1-100)"
    done
    while campaign_running; do sleep "${WAIT_POLL}"; done
fi
echo "[phase2] $(date -Is)  the box is free"
sleep 60      # let the GPUs actually release before vLLM grabs them

# ----------------------------------------------------------------- 2. patch
banner "$(date -Is)  applying the phase-2 instrumentation"
bash run/instrument_phase2.sh --apply || {
    echo "FATAL: instrument_phase2.sh --apply failed; not continuing" >&2
    exit 2
}

# ------------------------------------------------------------------ 3. eval
rc=0
if [ "${SKIP_EVAL}" != "1" ]; then
    banner "$(date -Is)  six-benchmark evaluation of the campaign runs"
    REPO="${REPO}" bash run/run_eval_all.sh || { rc=1; echo "[phase2] campaign eval finished with errors"; }
    python3 scripts/collect_results.py --logs logs/experiments \
        --out results/summary_campaign.tsv || true
fi

# ------------------------------------------------------------- 4. followups
if [ "${SKIP_FOLLOWUPS}" != "1" ]; then
    banner "$(date -Is)  follow-up ablations"
    REPO="${REPO}" WAIT=1 bash run/run_followups.sh || { rc=1; echo "[phase2] follow-ups finished with errors"; }
fi

# ------------------------------------------------------------------ 5. eval
if [ "${EVAL_FOLLOWUPS}" = "1" ] && [ "${SKIP_FOLLOWUPS}" != "1" ]; then
    banner "$(date -Is)  six-benchmark evaluation of the follow-ups"
    REPO="${REPO}" EVAL_SET=followups bash run/run_eval_all.sh || { rc=1; echo "[phase2] follow-up eval finished with errors"; }
fi

banner "$(date -Is)  phase 2 finished (rc=${rc})"
python3 scripts/collect_results.py --logs logs/experiments --out results/summary.tsv || true
exit "${rc}"
