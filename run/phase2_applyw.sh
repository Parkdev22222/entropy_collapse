#!/usr/bin/env bash
# Phase 2 arm with the branch correction applied AFTER STEER's mapping.
#
#   LAM=0.5 SEED=1 ./run/phase2_applyw.sh
#
# A separate launcher rather than a flag on phase2_1gpu.sh, because that script
# is executing while the main grid runs and bash reads a script by byte offset:
# editing a running one makes it resume at a stale position. Nothing here edits
# a file the live queue is inside.
#
# WHY THIS ARM EXISTS. The shipped form adds the branch term to Omega and lets
# STEER's per-micro-batch min-max map the sum. Omega is heavy-tailed (it divides
# by pi_old, floored at 1e-8), so its extremes set the scale and the bulk of
# tokens land within ~5% of the band of each other. Measured offline on real
# rollouts (docs/weight_forms.json), a branch signal covering ~1% of positions
# survives as a mean weight difference of 0.0007 on a 0.1-wide band -- nothing --
# while still costing the attenuating third of the range 8-34% of its occupancy,
# because it shifted the distribution the min-max is computed over. Four
# metric-level variants were tried; all four failed the same way. Applying the
# correction after the mapping scored:
#
#     frac_low 0.716 vs 0.717 at lambda=0   (budget untouched)
#     drift on non-branch tokens 0.00000    (STEER's other 99% bit-identical)
#     branch-vs-other grows 12x from lambda 0.25 to 1.0  (lambda is a real knob)
#
# The steerf_apply override is passed as a trailing hydra argument, which
# phase2_1gpu.sh and run_steerf.sh both forward, so neither has to be edited.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAM=${LAM:-0.5}
SEED=${SEED:-1}
export RUN_TAG_OVERRIDE="phase2-applyw-lam${LAM}-s${SEED}"
LOG="$(cd "${SCRIPT_DIR}/.." && pwd)/logs/${RUN_TAG_OVERRIDE}.log"
echo "[phase2] apply=weight lambda=${LAM} seed=${SEED} -> ${LOG}"
LAM="${LAM}" SEED="${SEED}" PROJECT_NAME=STEER-F-phase2 \
  "${SCRIPT_DIR}/phase2_1gpu.sh" \
    +actor_rollout_ref.actor.policy_loss.steerf_apply=weight \
    "$@"
# phase2_1gpu.sh writes logs/phase2-lam<LAM>-s<SEED>.log; rename so the two
# formulations of the same (lambda, seed) do not overwrite each other.
mv -f "$(cd "${SCRIPT_DIR}/.." && pwd)/logs/phase2-lam${LAM}-s${SEED}.log" "$LOG" 2>/dev/null || true
