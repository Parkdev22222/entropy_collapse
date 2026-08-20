#!/usr/bin/env bash
# Second handover: retire the queue instance that predates the control-arm key
# fix, without discarding the run it is currently supervising.
#
# The old instance would have skipped every oracle control arm. It keyed its
# "already done?" check on the lambda value, so `oracle:0.25` produced the same
# key as the main grid's own lambda=0.25 row and matched it — dropping, from all
# three seeds, the arm that answers whether the forecaster earns its parameters.
#
# The in-flight training is a grandchild of the retired queue and survives it;
# killing it would throw away hours. Its TSV row is written here, because the
# queue that would have written it is gone and the corrected queue skips only
# what is already recorded `ok`.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"

SUMMARY=docs/phase2_runs.tsv
TAG=${TAG:-phase2-lam0.5-s1}
LAM=${LAM:-0.5}
SEED=${SEED:-1}
STARTED=${STARTED:-0}

log() { echo "[handover2 $(date +%m-%d\ %H:%M:%S)] $*"; }

log "waiting for the in-flight ${TAG}"
while pgrep -f "verl.trainer.main_ppo" >/dev/null; do sleep 60; done
log "${TAG} finished"

if ! grep -qP "^${LAM}\t${SEED}\tok\t" "$SUMMARY" 2>/dev/null; then
  mins=0
  if [ "$STARTED" -gt 0 ]; then
    mins=$(( ($(date +%s) - STARTED) / 60 ))
  fi
  python scripts/summarize_run.py "logs/${TAG}.log" \
      --lam "$LAM" --seed "$SEED" --minutes "$mins" >> "$SUMMARY" || true
  log "recorded: $(tail -1 "$SUMMARY")"
fi

log "starting the corrected queue"
exec "${SCRIPT_DIR}/phase2_queue.sh"
