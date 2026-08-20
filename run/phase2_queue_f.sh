#!/usr/bin/env bash
# Third queue: apply="branch" — uniform protection at branch points.
#
#   nohup ./run/phase2_queue_f.sh > logs/phase2_queue_f.log 2>&1 &
#
# The weakest of the three hypotheses, and the one that would be the most
# useful if it held.
#
#   metric / weight (queues 1-2)  grade the adjustment by A_H, so siblings at a
#       branch point are pushed apart. A_H is a deviation from the sibling mean
#       and therefore zero-centred by construction, so this spreads the branch
#       tokens without moving their mean: measured offline, mean branch-vs-other
#       +0.002 on a 0.1 band while the weight std rose 0.0034 -> 0.0043.
#
#   branch (this queue)  attenuates every position that still has siblings by
#       the same amount, ignoring the score's sign and magnitude. It moves the
#       mean, which the signed forms cannot.
#
# What makes it worth a queue of its own: "does this rollout still have siblings
# here" is answered by the rollout group alone. No forecast, no MTP heads, no
# Phase 1 warm-up, no extra forward pass. If it matches the signed arms, none of
# that machinery is carrying the effect, and the method collapses to something
# far simpler than the paper describes. If it loses, the discrimination between
# siblings is the load-bearing part -- which is the claim, now with a control
# behind it.
#
# Waits for the apply="weight" queue so the GPU is never shared and step times
# stay comparable across every arm.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"

SUMMARY=docs/phase2_runs.tsv
LAMS=${LAMS:-"0.5 1.0"}
SEEDS=${SEEDS:-"1 2 3"}

log() { echo "[queue-f $(date +%m-%d\ %H:%M:%S)] $*"; }

log "waiting for the earlier queues to finish"
while pgrep -f "phase2_queue.sh" >/dev/null \
   || pgrep -f "phase2_queue_e.sh" >/dev/null \
   || pgrep -f "phase2_handover" >/dev/null; do
  sleep 120
done
while pgrep -f "verl.trainer.main_ppo" >/dev/null; do sleep 60; done
log "earlier queues done — starting apply=branch arms"

for seed in $SEEDS; do
  for lam in $LAMS; do
    key="branch${lam}"
    tag="phase2-branch-lam${lam}-s${seed}"
    if grep -qP "^${key}\t${seed}\tok\t" "$SUMMARY" 2>/dev/null; then
      log "$tag already recorded — skipping"
      continue
    fi
    log "starting $tag"
    started=$(date +%s)
    LAM="$lam" SEED="$seed" PROJECT_NAME=STEER-F-phase2 \
      "${SCRIPT_DIR}/phase2_1gpu.sh" \
        +actor_rollout_ref.actor.policy_loss.steerf_apply=branch
    rc=$?
    mins=$(( ($(date +%s) - started) / 60 ))
    # phase2_1gpu.sh names its log by (lam, seed) only, so move it aside before
    # the next formulation of the same cell overwrites it.
    mv -f "logs/phase2-lam${lam}-s${seed}.log" "logs/${tag}.log" 2>/dev/null || true
    log "$tag exited rc=$rc after ${mins}m"
    python scripts/summarize_run.py "logs/${tag}.log" \
        --lam "$key" --seed "$seed" --minutes "$mins" >> "$SUMMARY" || true
    while pgrep -f "verl.trainer.main_ppo" >/dev/null; do sleep 20; done
  done
  log "seed ${seed} pass complete"
done
log "all apply=branch arms done"
cat "$SUMMARY"
