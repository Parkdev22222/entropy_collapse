#!/usr/bin/env bash
# Re-run the lambda>0 plumbing test once the GPU frees, to confirm the FSDP
# tied-unembedding fix works inside the real training loop.
#
# The fix was verified against the real 1.5B model under verl's FSDP
# configuration in isolation; this checks it in situ, where the micro-batching,
# remove-padding and A_H assembly also run. Still a plumbing test: the heads are
# random, so nothing here is a STEER-F result.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEERF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${STEERF_ROOT}"

log() { echo "[verify $(date +%H:%M:%S)] $*"; }

while pgrep -f "verl.trainer.main_ppo" >/dev/null; do sleep 30; done
log "GPU free — starting steerf plumbing re-test"

PROJECT_NAME=STEERF-smoke VARIANT=fixcheck RUN_TAG=smoke-steerf-fix \
STEERF_HEADS="${STEERF_ROOT}/checkpoints/mtp_heads_smoke.pt" \
STEERF_CALIB="" \
ARM=steerf STEERF_LAM=0.5 STEPS=2 \
    "${SCRIPT_DIR}/phase0_1gpu.sh" \
    trainer.val_before_train=False \
    trainer.test_freq=-1
rc=$?
log "steerf plumbing re-test exited rc=${rc} -> logs/phase0_steerf_fixcheck.log"

if grep -qE "steerf/visit_rel_mag|steerf/branch_entropy" logs/phase0_steerf_fixcheck.log 2>/dev/null; then
  log "PASS: lambda>0 diagnostics were emitted"
else
  log "CHECK: no lambda>0 diagnostics in the log — inspect before Phase 2"
fi
