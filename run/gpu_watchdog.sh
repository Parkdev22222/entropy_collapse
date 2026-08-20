#!/usr/bin/env bash
# Keep the GPU busy for as long as Phase 2 has work left.
#
#   nohup ./run/gpu_watchdog.sh > logs/gpu_watchdog.log 2>&1 &
#
# The sweep queue already chains its runs, so the failure this guards against is
# not a slow queue but a *dead* one: the queue process exits (killed, crashed,
# session teardown) and the GPU then sits at 0 % while the remaining runs never
# start. That state is indistinguishable from "still training" unless something
# is watching, and it has cost idle hours twice already.
#
# Restarting the queue is safe to do automatically: phase2_queue.sh skips any
# (lam, seed) already recorded `ok` in docs/phase2_runs.tsv, so a restart
# resumes rather than repeats.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEERF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${STEERF_ROOT}"

IDLE_LIMIT=${IDLE_LIMIT:-6}      # consecutive idle polls before acting
POLL=${POLL:-60}
SUMMARY=docs/phase2_runs.tsv
# A trainer holding <2 GB is not training; real runs hold tens of GB.
STUCK_LIMIT=${STUCK_LIMIT:-5}
TOTAL=${TOTAL:-48}   # 21 main grid + 9 apply=weight + 6 apply=branch
                     # + 12 paper/theorem arms. A low value would retire the
                     # watchdog while runs remain unguarded.

log(){ echo "[watchdog $(date +%m-%d\ %H:%M:%S)] $*"; }

idle=0
stuck=0
while true; do
  # `grep -c` prints 0 and *exits 1* when nothing matches, so a trailing
  # `|| echo 0` appends a second line and the count becomes "0\n0", which then
  # fails every integer test below. Take the first line and default it instead.
  done_n=0
  if [ -f "$SUMMARY" ]; then
    done_n=$(grep -cP "\tok\t" "$SUMMARY" 2>/dev/null | head -1)
    done_n=${done_n:-0}
  fi
  if [ "$done_n" -ge "$TOTAL" ]; then
    log "all ${TOTAL} runs recorded ok — nothing left to keep busy"
    exit 0
  fi

  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  training=$(pgrep -f "verl.trainer.main_ppo" >/dev/null && echo yes || echo no)

  # A live process is not evidence of progress. A verl run that dies inside a
  # Ray worker can leave the parent python alive and sleeping forever, holding
  # no GPU memory. Every queue gates on `pgrep main_ppo`, so one such corpse
  # blocks all of them -- that is how 5.5 hours were lost on 08-18: main_ppo
  # alive, GPU at 0%%/4MiB, watchdog logging nothing because it read
  # training=yes. Progress is GPU memory, not process existence.
  if [ "$training" = "yes" ] && [ "${mem:-0}" -lt 2000 ]; then
    stuck=$((stuck+1))
    log "main_ppo alive but GPU empty (${stuck}/${STUCK_LIMIT}) — util=${util}% mem=${mem}MiB"
    if [ "$stuck" -ge "$STUCK_LIMIT" ]; then
      log "HUNG TRAINER — killing it so the queues' wait loops release"
      pkill -9 -f "verl.trainer.main_ppo"
      pkill -9 -f "ray::"
      stuck=0
    fi
    sleep "$POLL"
    continue
  fi
  stuck=0

  if [ "$training" = "no" ] && [ "${mem:-0}" -lt 2000 ]; then
    idle=$((idle+1))
    log "GPU idle (${idle}/${IDLE_LIMIT}) — util=${util}% mem=${mem}MiB, ${done_n}/${TOTAL} runs done"
    if [ "$idle" -ge "$IDLE_LIMIT" ]; then
      # The owner is phase2_tonight.sh now, not phase2_queue.sh. Reviving the
      # old queue here would recreate exactly the four-way race that made
      # phase2_tonight.sh necessary: two queues launching into one GPU, the
      # loser dying in 0m, and the last queue in the chain never starting.
      if pgrep -f phase3_queue.sh >/dev/null; then
        log "queue alive but GPU idle — a run is probably starting; not intervening"
      else
        log "RUNNER IS DEAD with ${done_n}/${TOTAL} runs done — restarting it"
        nohup "${SCRIPT_DIR}/phase3_queue.sh" >> logs/phase3_queue.log 2>&1 &
        log "restarted; it resumes at the first unrecorded (lam, seed)"
      fi
      idle=0
    fi
  else
    idle=0
  fi
  sleep "$POLL"
done
