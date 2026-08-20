#!/usr/bin/env bash
# Phase 3. Two questions the Phase 2 grid raised and cannot answer.
#
#   nohup ./run/phase3_queue.sh > logs/phase3_queue.log 2>&1 &
#
# ---------------------------------------------------------------------------
# Q1. IS THE FORECAST LOAD-BEARING?   (branch control, ~8 h)
#
# apply="weight" with mode="signed" grades the correction by A_H, so siblings
# at one branch point are pushed apart. apply="branch" is mode="uniform": it
# attenuates every position that still has surviving siblings by the same
# amount, ignoring the score's sign and magnitude entirely.
#
# The point is what uniform does NOT need. "Does this rollout still have
# siblings here" is answered by the rollout group alone -- no MTP heads, no
# Phase 1 warm-up, no forecast pass, and none of the 3x step-time cost. So:
#
#   uniform ~= signed  ->  the forecast is not carrying the effect, and the
#                          method collapses to something far simpler than the
#                          methodology document describes.
#   uniform  < signed  ->  discrimination between siblings is the load-bearing
#                          part, now with a control behind it rather than an
#                          assertion.
#
# Run at lam=0.5 seed 1 to sit directly against weight0.5 seed 1 (pass@16
# 0.7901, entropy 0.549), same everything else.
#
# ---------------------------------------------------------------------------
# Q2. DOES PRESERVED ENTROPY EVENTUALLY PAY?   (300-step runs, ~9 h + ~23 h)
#
# At 100 steps the two differences are of completely different sizes. Using
# GRPO's own seed-to-seed spread as the noise scale (pass@16 0.0068, entropy
# 0.0820 across seeds 1 and 2):
#
#     weight0.5 vs GRPO mean:  pass@16 -0.0044   0.65x the spread  -- invisible
#                              entropy +0.2740   3.34x the spread  -- real
#
# So entropy is preserved at no measurable cost, which is weaker than the claim
# the method wants: that preserving it *helps*. The reason it cannot yet help
# is visible in the slopes -- nothing has bottomed out at step 100. GRPO seed 1
# is still falling (0.251 -> 0.234 over steps 70-100) and weight0.5 is still
# rising (0.463 -> 0.549). The run ends before collapse has had a chance to
# cost anything. The paper trains 200 steps; this runs 300.
#
# PREREGISTERED, so that whatever comes out is not read after the fact:
#
#   H1  Past the step where GRPO's entropy first drops below 0.15, weight0.5's
#       pass@16 exceeds GRPO's by more than one seed spread (0.0068).
#   H0  The gap stays inside 0.0068 for the whole 300 steps -- preserved
#       entropy is free but useless at this scale, and the honest result is a
#       negative one.
#
# test_freq=50 gives validation at 50/100/150/200/250/300: enough resolution to
# say *where* any divergence starts, and step 100 doubles as a consistency
# check against the runs already recorded.
# ---------------------------------------------------------------------------
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
SUMMARY=docs/phase2_runs.tsv
log() { echo "[phase3 $(date +%m-%d\ %H:%M:%S)] $*"; }

record() {   # tag key seed minutes
  python scripts/summarize_run.py "logs/${1}.log" \
      --lam "$2" --seed "$3" --minutes "$4" >> "$SUMMARY" || true
}

wait_gpu() { while pgrep -f "verl.trainer.main_ppo" >/dev/null; do sleep 60; done; }

log "waiting for the in-flight run"
wait_gpu

# --- Q1 -------------------------------------------------------------------
if ! grep -qP "^branch0.5\t1\tok\t" "$SUMMARY" 2>/dev/null; then
  log "Q1: branch control, apply=branch lam=0.5 seed 1"
  t0=$(date +%s)
  RUN_TAG="phase3-branch-lam0.5-s1" LAM=0.5 SEED=1 PROJECT_NAME=STEER-F-phase3 \
    "${SCRIPT_DIR}/phase2_1gpu.sh" \
      +actor_rollout_ref.actor.policy_loss.steerf_apply=branch
  record phase3-branch-lam0.5-s1 branch0.5 1 $(( ($(date +%s)-t0)/60 ))
  wait_gpu
fi

# --- Q2 -------------------------------------------------------------------
if ! grep -qP "^grpo300\t1\tok\t" "$SUMMARY" 2>/dev/null; then
  log "Q2a: GRPO, 300 steps"
  t0=$(date +%s)
  RUN_TAG="phase3-grpo300-s1" SEED=1 STEPS=300 PROJECT_NAME=STEER-F-phase3 \
    "${SCRIPT_DIR}/phase2_grpo.sh" trainer.test_freq=50
  record phase3-grpo300-s1 grpo300 1 $(( ($(date +%s)-t0)/60 ))
  wait_gpu
fi

if ! grep -qP "^weight300\t1\tok\t" "$SUMMARY" 2>/dev/null; then
  log "Q2b: apply=weight lam=0.5, 300 steps"
  t0=$(date +%s)
  RUN_TAG="phase3-weight300-lam0.5-s1" LAM=0.5 SEED=1 STEPS=300 \
    PROJECT_NAME=STEER-F-phase3 "${SCRIPT_DIR}/phase2_1gpu.sh" \
      +actor_rollout_ref.actor.policy_loss.steerf_apply=weight \
      trainer.test_freq=50
  record phase3-weight300-lam0.5-s1 weight300 1 $(( ($(date +%s)-t0)/60 ))
  wait_gpu
fi

log "phase 3 complete"
cat "$SUMMARY"
