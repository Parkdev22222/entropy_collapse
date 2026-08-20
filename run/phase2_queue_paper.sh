#!/usr/bin/env bash
# Paper-faithful STEER, and STEER-F on top of it.
#
#   nohup ./run/phase2_queue_paper.sh > logs/phase2_queue_paper.log 2>&1 &
#
# WHY THIS EXISTS. None of the three configurations shipped in STEER's own
# run/ directory is the configuration its paper reports. The paper's method is
# Eq. 9,
#
#     lambda(s) = exp( -alpha * |Omega(s)| / max_B |Omega| ),  lambda_min = e^-alpha
#
# with lambda_min = 0.7 in every main experiment, and Appendix F.1 Table 11
# ablates the mapping explicitly: exponential 48.6 > linear 48.0 > binary 46.5.
# Eq. 9 is reproduced exactly by the code path
#
#     linear=False, mode="symmetric", token_weight_min=L, token_weight_max=1.0
#
# because that branch computes k = -log(w_min)/max|Omega| and then
# w = exp(-k|Omega|) = w_min^(|Omega|/max|Omega|) = exp(-alpha |Omega|/max|Omega|).
#
# What has been run so far instead:
#
#   run_linear.sh       linear,      [0.8, 1.2]  -- G0 FAILED here. w_max = 1.2
#                       amplifies above 1, which appears nowhere in the paper,
#                       and the weights degenerated to a near-constant 1.07,
#                       collapsing entropy faster than plain GRPO.
#   run_asymmetric.sh   linear,      [0.9, 1.0]  -- G0 passes, and this is what
#                       the main grid is running. Internally valid, but it is
#                       not the published method, so "STEER-F beats STEER" would
#                       only mean "beats a variant we picked".
#
# This queue runs the published configuration as its own sub-grid: the lambda=0
# arm is then STEER as published, and the lambda>0 arms are STEER-F built on it.
# Only that pairing supports the claim the paper-facing writeup wants to make.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"

SUMMARY=docs/phase2_runs.tsv
LAMS=${LAMS:-"0.0 0.5"}
SEEDS=${SEEDS:-"1 2 3"}

# Two Omega definitions, same Eq. 9 mapping, so the mapping is held fixed and
# the only thing that varies is the estimator itself.
#
#   paper  released STEER: Omega = -(A/pi_old) * pi(1-pi)(log pi + H)
#   thm    Theorem 1:      Omega = -(I_clip * r * A) * pi(1-pi)(log pi + H)
#
# Appendix G Step 3 says only the sampled action's logit moves, so the realised
# entropy change is a single inner-product term carrying r = pi_theta/pi_old --
# and Eq. 7's expectation is over a ~ pi_theta while rollouts come from pi_old,
# so importance correction supplies the same r. The released code has A/pi_old,
# i.e. the theorem's value divided by pi_theta.
#
# Measured offline on real rollouts, pooled across queries so the batch max
# spans several groups (docs/omega_forms.json): under the released form the
# batch maximum -- Eq. 9's denominator for *every* token -- is set by a token of
# probability 2.0e-6. Under Theorem 1 it is set by one of probability 0.30.
# max/p99 6.4 -> 3.7; tokens pinned within 1% of w_max 83.4% -> 66.7%; weight
# std 0.0124 -> 0.0180. Rank correlation 0.945, so the two also disagree about
# which tokens to attenuate, not merely by how much.
#
# 1/L was tried and rejected: the paper's L is per group while Eq. 9's max is
# over the batch, so the shortest group captures the maximum and compresses
# everything else (max/p99 72.9, 97.9% pinned, std 0.0026). verl's token-mean
# normaliser is batch-wide and constant, so it cancels. Dropping it is correct.
ARMS=${ARMS:-"paper thm"}

log() { echo "[queue-paper $(date +%m-%d\ %H:%M:%S)] $*"; }

log "waiting for the earlier queues"
while pgrep -f "phase2_queue.sh" >/dev/null \
   || pgrep -f "phase2_queue_e.sh" >/dev/null \
   || pgrep -f "phase2_queue_f.sh" >/dev/null \
   || pgrep -f "phase2_handover" >/dev/null; do
  sleep 120
done
while pgrep -f "verl.trainer.main_ppo" >/dev/null; do sleep 60; done
log "starting paper-faithful arms"

for seed in $SEEDS; do
 for arm in $ARMS; do
  case "$arm" in
    paper) ICLIP=False; RATIO=False ;;
    thm)   ICLIP=True;  RATIO=True  ;;
    *) log "unknown arm $arm — skipping"; continue ;;
  esac
  for lam in $LAMS; do
    key="${arm}${lam}"
    tag="phase2-${arm}-lam${lam}-s${seed}"
    if grep -qP "^${key}\t${seed}\tok\t" "$SUMMARY" 2>/dev/null; then
      log "$tag already recorded — skipping"
      continue
    fi
    log "starting $tag"
    started=$(date +%s)
    # Eq. 9 exactly: exponential mapping, |Omega|, weights in [0.7, 1.0].
    #
    # Passed as trailing hydra overrides rather than environment variables:
    # phase2_1gpu.sh exports the asymmetric configuration unconditionally, so
    # env vars would be ignored. Both phase2_1gpu.sh and run_steerf.sh forward
    # "$@" to the end of the command line, and hydra takes the last assignment
    # for a key, so these win. No `+` prefix -- the keys already exist by then,
    # having been added by run_steerf.sh's own `+`-prefixed arguments.
    # The estimator flags come from $arm; everything else is held fixed, so
    # paper-vs-thm at the same (lam, seed) isolates the Omega definition.
    LAM="$lam" SEED="$seed" PROJECT_NAME=STEER-F-phase2 \
      "${SCRIPT_DIR}/phase2_1gpu.sh" \
        actor_rollout_ref.actor.policy_loss.entropy_control_mode=symmetric \
        actor_rollout_ref.actor.policy_loss.token_weight_min=0.7 \
        actor_rollout_ref.actor.policy_loss.token_weight_max=1.0 \
        actor_rollout_ref.actor.policy_loss.linear=False \
        actor_rollout_ref.actor.policy_loss.steerf_iclip=${ICLIP} \
        actor_rollout_ref.actor.policy_loss.steerf_ratio=${RATIO}
    rc=$?
    mins=$(( ($(date +%s) - started) / 60 ))
    mv -f "logs/phase2-lam${lam}-s${seed}.log" "logs/${tag}.log" 2>/dev/null || true
    log "$tag exited rc=$rc after ${mins}m"
    python scripts/summarize_run.py "logs/${tag}.log" \
        --lam "$key" --seed "$seed" --minutes "$mins" >> "$SUMMARY" || true
    while pgrep -f "verl.trainer.main_ppo" >/dev/null; do sleep 20; done
  done
  log "arm ${arm} seed ${seed} complete"
 done
  log "seed ${seed} pass complete"
done
log "all paper-faithful arms done"
cat "$SUMMARY"
