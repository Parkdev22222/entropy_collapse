#!/usr/bin/env bash
# One-shot handover: finish the in-flight run under the old queue, then start
# the extended queue.
#
# The sweep queue was edited while running. bash reads a script by byte offset,
# so a running instance that returns from a child resumes at a now-stale
# position; rather than find out what it executes, the old instance is retired
# and this takes over. The in-flight training is left alone -- it is a
# grandchild, it survives its parent, and killing it would throw away hours.
#
# The TSV row for the in-flight run is written here, because the queue that
# would have written it is gone and the extended queue skips only what is
# already recorded `ok`.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
SUMMARY=docs/phase2_runs.tsv
TAG=${TAG:-phase2-lam0.25-s1}
LAM=${LAM:-0.25}
SEED=${SEED:-1}
STARTED=${STARTED:-0}

log(){ echo "[handover $(date +%m-%d\ %H:%M:%S)] $*"; }

log "waiting for the in-flight ${TAG} to finish"
while pgrep -f "verl.trainer.main_ppo" >/dev/null; do sleep 60; done
log "${TAG} finished"

if ! grep -qP "^${LAM}\t${SEED}\tok\t" "$SUMMARY" 2>/dev/null; then
  mins=0
  [ "$STARTED" -gt 0 ] && mins=$(( ($(date +%s) - STARTED) / 60 ))
  python scripts/summarize_run.py "logs/${TAG}.log" \
      --lam "$LAM" --seed "$SEED" --minutes "$mins" >> "$SUMMARY" || true
  log "recorded: $(tail -1 "$SUMMARY")"
fi

log "starting the extended queue (main grid + oracle/grpo controls)"
exec "${SCRIPT_DIR}/phase2_queue.sh"
