#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Is a run actually progressing, or did it die at startup?
#
#   bash run/run_status.sh              # answer once
#   WATCH=1 bash run/run_status.sh      # refresh every 60s
#   ARMS="lam0.1 lam0.5" bash ...       # only these
#   SEEDS="1" bash ...                  # default: campaign 1-5, followups 1
#
# Read-only. It starts nothing, kills nothing, deletes nothing.
#
# WHY THIS EXISTS
#   Between 2026-09-12 and 09-14 a fresh box failed to start six times running
#   -- hub pins, ray, the wrong model default, flash-attn, NCCL/NVLS, the wrong
#   steer_f lineage, a missing word2number -- and every one of them was
#   diagnosed by pasting log tails back and forth. Each answer came out of the
#   same four or five greps. They belong in a file.
#
#   The distinction that matters is the one diagnose_startup_failure already
#   draws: a run that never reached `step:1 - global_seqlen` did not crash
#   during training, it failed to start, and those have completely different
#   causes. Everything here is built on the helpers in _arms.sh so that the
#   status view and the queues can never disagree about what "done" means.
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}" || { echo "FATAL: cannot cd to ${ROOT}" >&2; exit 1; }

# shellcheck source=run/_arms.sh
. "${ROOT}/run/_arms.sh"

LOG_DIR="${LOG_DIR:-${ROOT}/logs/experiments}"
STEPS=${STEPS:-110}
WATCH=${WATCH:-0}
SEEDS=${SEEDS:-}
ARMS=${ARMS:-}

hr ()  { printf '\033[2m%s\033[0m\n' "------------------------------------------------------------"; }
say () { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# The last step the trainer logged, across the main log and any _<tag> recovery
# log -- the same pair train_log_done looks at, so the two never disagree.
last_step () {   # <run-name>
    local f best n
    best=0
    for f in "${LOG_DIR}/train-$1.log" "${LOG_DIR}/train-$1"_*.log; do
        [ -f "${f}" ] || continue
        n="$(grep -o 'step:[0-9]* - global_seqlen' "${f}" 2>/dev/null | tail -1 \
             | tr -dc '0-9')"
        [ -n "${n}" ] && [ "${n}" -gt "${best}" ] && best="${n}"
    done
    printf '%s' "${best}"
}

newest_log () {  # <run-name> -> the log file to quote, newest wins
    local f newest
    newest=
    for f in "${LOG_DIR}/train-$1.log" "${LOG_DIR}/train-$1"_*.log; do
        [ -f "${f}" ] || continue
        if [ -z "${newest}" ] || [ "${f}" -nt "${newest}" ]; then newest="${f}"; fi
    done
    printf '%s' "${newest}"
}

report () {
    say "1. is anything training?"
    if is_busy; then
        printf '  \033[32mYES\033[0m   trainer running\n'
        busy_pids | head -3 | while read -r h; do
            printf '        pid %-8s %s\n' "${h}" \
                "$(tr '\0' ' ' < "/proc/${h}/cmdline" 2>/dev/null | cut -c1-70)"
        done
    else
        printf '  \033[33mNO\033[0m    no trainer process\n'
    fi
    command -v tmux >/dev/null 2>&1 && tmux ls 2>/dev/null | sed 's/^/        tmux: /'

    say "2. GPUs"
    if command -v nvidia-smi >/dev/null 2>&1; then
        local total held
        total="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
        held="$(gpu_holders | wc -l)"
        printf '  %s card(s) on this box, %s process(es) holding VRAM\n' "${total}" "${held}"
        gpu_holders | head -8 | sed 's/^/        /'
    else
        printf '  no nvidia-smi here\n'
    fi
    # The banner run_uniform_ablation.sh:162 and run_grpo.sh:152 print. This is
    # how you see that six arms took two cards and three took four -- it comes
    # from the logs, so it works on a box that is only holding the logs.
    local seen n
    seen="$(grep -ho 'gpus *[0-9]* / [0-9]* (tp=[0-9]*)' "${LOG_DIR}"/train-*.log 2>/dev/null \
            | sed 's/.*\/ *//' | sort -u | tr '\n' ' ')"
    if [ -n "${seen}" ]; then
        printf '  topology the logs report: %s\n' "${seen}"
        n="$(printf '%s\n' ${seen} | grep -c '(tp=')"
        if [ "${n}" -gt 1 ]; then
            printf '  \033[33mNOTE\033[0m  more than one topology above -- arms in the same table\n'
            printf '        are not all on the same number of cards.\n'
        fi
    fi

    say "3. runs"
    local seeds arms a s rn step log
    seeds="${SEEDS:-1 2 3 4 5}"
    arms="${ARMS:-${CAMPAIGN_ARMS} ${FOLLOWUP_ARMS:-lam0-tree lam0.1 lam0.5 xclip-signed xclip-steer rloo-signed rloo-steer opo-signed opo-steer}}"
    printf '  %-52s %-10s %s\n' "run" "step" "state"
    hr
    for s in ${seeds}; do
        for a in ${arms}; do
            rn="$(run_name_for "${a}" "${s}" 2>/dev/null)" || continue
            log="$(newest_log "${rn}")"
            [ -n "${log}" ] || continue          # never started: not interesting
            step="$(last_step "${rn}")"
            if train_log_done "${LOG_DIR}" "${rn}" "${STEPS}"; then
                printf '  %-52s %-10s \033[32mdone\033[0m\n' "${rn}" "${step}/${STEPS}"
            elif [ "${step}" -eq 0 ]; then
                printf '  %-52s %-10s \033[31mNEVER STARTED\033[0m\n' "${rn}" "-"
            elif [ -n "$(find "${log}" -newermt '-20 minutes' 2>/dev/null)" ]; then
                printf '  %-52s %-10s \033[32mrunning\033[0m\n' "${rn}" "${step}/${STEPS}"
            else
                printf '  %-52s %-10s \033[33mstalled/stopped\033[0m\n' "${rn}" "${step}/${STEPS}"
            fi
        done
    done

    say "4. anything that never reached step 1"
    local any
    any=0
    for s in ${seeds}; do
        for a in ${arms}; do
            rn="$(run_name_for "${a}" "${s}" 2>/dev/null)" || continue
            log="$(newest_log "${rn}")"
            [ -n "${log}" ] || continue
            [ "$(last_step "${rn}")" -eq 0 ] || continue
            any=1
            diagnose_startup_failure "${log}" "${a} seed ${s}" || true
        done
    done
    [ "${any}" = "0" ] && printf '  none -- every run that has a log got past step 1\n'

    # A run that trained for hours and then hit a CUDA OOM is invisible above:
    # last_step is not 0, so it is not a startup failure, and nothing else here
    # looks at why an unfinished run stopped. One line each -- the full remedy
    # is long and belongs in the queue log, not in a status sweep.
    say "4b. anything that died on a CUDA OOM"
    any=0
    for s in ${seeds}; do
        for a in ${arms}; do
            rn="$(run_name_for "${a}" "${s}" 2>/dev/null)" || continue
            log="$(newest_log "${rn}")"
            [ -n "${log}" ] || continue
            grep -qE 'OutOfMemoryError|CUDA out of memory' "${log}" 2>/dev/null || continue
            train_log_done "${LOG_DIR}" "${rn}" "${STEPS}" && continue
            any=1
            printf '  %-46s OOM at step %s\n' "${rn}" "$(last_step "${rn}")"
        done
    done
    if [ "${any}" = "0" ]; then
        printf '  none\n'
    else
        printf '  full diagnosis:  . run/_arms.sh && diagnose_run_failure <log> <label> %s\n' "${STEPS}"
    fi

    say "5. latest validation"
    local v
    v="$(grep -h 'val-core/aime_2024_dapo_boxed/acc/mean@32' "${LOG_DIR}"/train-*.log 2>/dev/null \
         | tail -1 | grep -o 'val-core/aime_2024_dapo_boxed/acc/mean@32:[0-9.]*')"
    if [ -n "${v}" ]; then
        printf '  %s\n' "${v}"
    else
        printf '  no validation line in any log yet\n'
    fi
    v="$(grep -h 'perf/time_per_step' "${LOG_DIR}"/train-*.log 2>/dev/null \
         | tail -1 | grep -o 'perf/time_per_step:[0-9.]*')"
    if [ -n "${v}" ]; then
        printf '  %s  (x %s steps = %s h)\n' "${v}" "${STEPS}" \
            "$(awk -v t="${v##*:}" -v n="${STEPS}" 'BEGIN{printf "%.1f", t*n/3600}')"
    fi
    # An empty box is a normal answer, not a failure. Without this the exit
    # status is whatever the last test happened to be -- 1 on a fresh pod,
    # which is the one box where someone is most likely to be scripting around
    # this command.
    return 0
}

if [ "${WATCH}" = "1" ]; then
    while true; do
        clear 2>/dev/null
        printf '\033[1m%s\033[0m\n' "$(date -Is)"
        report
        sleep 60
    done
else
    report
fi
