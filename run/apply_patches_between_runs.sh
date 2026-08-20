#!/usr/bin/env bash
# Sync third_party/STEER with the committed patches, at an inter-run boundary.
#
#   nohup ./run/apply_patches_between_runs.sh > logs/patch_sync.log 2>&1 &
#
# The applied tree is missing `steerf_apply` entirely: the patches were edited
# and verified on a scratch copy, but never re-applied here. Consequences, both
# silent:
#
#   * queue_e (9 runs, apply="weight") and queue_f (6 runs, apply="branch") pass
#     the override via hydra's `+` prefix, which adds the key without error --
#     and nothing reads it. All 15 would have run apply="metric" and been
#     recorded under labels claiming otherwise, i.e. ~50 GPU-hours of duplicates
#     of the one formulation already measured to destroy the branch signal.
#   * the committed patches were themselves inconsistent: dp_actor passed
#     steerf_apply into compute_policy_loss_with_entropy, whose signature in the
#     core_algos patch never accepted it. Re-applying as-was would have raised
#     TypeError on the first entropy_control step. Fixed, and verified by
#     calling the real function for all three modes on a pristine-plus-patches
#     copy before this script was written.
#
# Applying resets the tree to HEAD for about a second. A training process that
# has already imported these modules is unaffected, but a Ray worker respawning
# inside that window would import stock verl. So this waits for a gap: no
# main_ppo alive. It does not wait for the queue, which starts the next run
# immediately -- hence the retry loop and the verification afterwards.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"

log() { echo "[patch-sync $(date +%m-%d\ %H:%M:%S)] $*"; }

if grep -rq "steerf_apply" third_party/STEER/verl/ 2>/dev/null; then
  log "tree already carries steerf_apply — nothing to do"
  exit 0
fi

log "waiting for a gap between runs"
while true; do
  if ! pgrep -f "verl.trainer.main_ppo" >/dev/null; then
    log "gap detected — syncing"
    git -C third_party/STEER checkout -- . && bash scripts/setup_steer.sh
    rc=$?
    if [ $rc -eq 0 ] && grep -rq "steerf_apply" third_party/STEER/verl/; then
      log "sync OK — steerf_apply now reaches the loss"
      OMP_NUM_THREADS=4 STEER_REPO="$PWD/third_party/STEER" \
        python -m pytest tests/ -q 2>&1 | tail -2
      exit 0
    fi
    log "sync FAILED (rc=$rc) — leaving the tree as found, retrying"
  fi
  sleep 20
done
