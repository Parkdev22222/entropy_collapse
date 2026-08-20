#!/usr/bin/env bash
# Follow-on queue: the apply="weight" arms, run after the main grid finishes.
#
#   nohup ./run/phase2_queue_e.sh > logs/phase2_queue_e.log 2>&1 &
#
# Waits for the main queue to exit rather than racing it, so the GPU is never
# shared and the main grid's step times stay comparable.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
SUMMARY=docs/phase2_runs.tsv
LAMS=${LAMS:-"0.25 0.5 1.0"}
SEEDS=${SEEDS:-"1 2 3"}
log(){ echo "[queue-e $(date +%m-%d\ %H:%M:%S)] $*"; }

log "waiting for the main grid to finish"
while pgrep -f "phase2_queue.sh" >/dev/null; do sleep 120; done
while pgrep -f "verl.trainer.main_ppo" >/dev/null; do sleep 60; done
log "main grid done — starting apply=weight arms"

for seed in $SEEDS; do
  for lam in $LAMS; do
    tag="phase2-applyw-lam${lam}-s${seed}"
    key="applyw${lam}"
    if grep -qP "^${key}\t${seed}\tok\t" "$SUMMARY" 2>/dev/null; then
      log "$tag already recorded — skipping"; continue
    fi
    log "starting $tag"
    started=$(date +%s)
    LAM="$lam" SEED="$seed" "${SCRIPT_DIR}/phase2_applyw.sh"
    rc=$?
    mins=$(( ($(date +%s) - started) / 60 ))
    log "$tag exited rc=$rc after ${mins}m"
    python scripts/summarize_run.py "logs/${tag}.log" \
        --lam "$key" --seed "$seed" --minutes "$mins" >> "$SUMMARY" || true
    while pgrep -f "verl.trainer.main_ppo" >/dev/null; do sleep 20; done
  done
  log "seed ${seed} pass complete"
done
log "all apply=weight arms done"
cat "$SUMMARY"
