#!/usr/bin/env bash
# One runner, in priority order, replacing four queues that were racing.
#
#   nohup ./run/phase2_tonight.sh > logs/phase2_tonight.log 2>&1 &
#
# WHY. At 15:49 on 08-18 four queues (main, e, f, paper) were alive at once.
# They gate on each other with `pgrep`, and that gate failed: the main queue and
# queue_e both launched at 14:16, the main queue's run died in 0m against the
# GPU queue_e had taken, and queue_paper -- last in the chain -- would never
# have started at all. Ordering by "who waits for whom" does not survive a
# queue restarting; ordering by one sequential script does.
#
# PRIORITY. A STEER-F arm costs ~8 h (the MTP forecast pass roughly triples step
# time); a lambda=0 arm costs ~3 h. Roughly 15 h remain before the results are
# next looked at, so this buys 3 slots, and they go to the newest finding
# rather than to more seeds of one already answered at seed 1.
#
#   1. paper lam=0.0   published STEER, Eq. 9 exactly, released Omega
#   2. thm   lam=0.0   same mapping, Theorem 1's Omega (I_clip and r*A)
#
# That pair is the whole estimator question with everything else held fixed.
# It is worth the slots because the change is already carried by two
# independent measurements: pooled over real rollouts the released form hands
# Eq. 9's denominator to a token of probability 2.0e-6 while Theorem 1 hands it
# to one of probability 0.30 (docs/omega_forms.json), and against ground-truth
# per-token entropy change over 32 groups the released form scores PCC 0.263
# against Theorem 1's 0.387 -- the latter being the value the paper itself
# reports (docs/omega_ground_truth.json).
#
# Then lam=0.5 for both arms, then everything previously queued.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
log() { echo "[tonight $(date +%m-%d\ %H:%M:%S)] $*"; }

log "waiting for the in-flight run to finish"
while pgrep -f "verl.trainer.main_ppo" >/dev/null; do sleep 60; done
log "GPU free — starting the priority passes"

LAMS="0.0" SEEDS="1" "${SCRIPT_DIR}/phase2_queue_paper.sh"
log "lam=0.0 pass done"
LAMS="0.5" SEEDS="1" "${SCRIPT_DIR}/phase2_queue_paper.sh"
log "lam=0.5 pass done"
LAMS="0.0 0.5" SEEDS="2 3" "${SCRIPT_DIR}/phase2_queue_paper.sh"
log "paper/thm arms complete — resuming the rest"

nohup "${SCRIPT_DIR}/phase2_queue_e.sh" >> logs/phase2_queue_e.log 2>&1
nohup "${SCRIPT_DIR}/phase2_queue_f.sh" >> logs/phase2_queue_f.log 2>&1
nohup "${SCRIPT_DIR}/phase2_queue.sh"   >> logs/phase2_queue.log   2>&1
log "all queues drained"
