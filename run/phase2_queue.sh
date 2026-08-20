#!/usr/bin/env bash
# Phase 2, all arms, serialised on the single GPU.
#
#   nohup ./run/phase2_queue.sh > logs/phase2_queue.log 2>&1 &
#
# Grid: lambda in {0, 0.25, 0.5, 1.0} x seed in {1, 2, 3} = 12 runs.
# lambda=0 is the STEER baseline (bit-identical to upstream), so it is part of
# the grid rather than a separate experiment.
#
# Order is seed-major on purpose: after the first pass the grid is already
# complete at one seed, so an interruption leaves a usable — if
# underpowered — result set rather than three finished arms and one missing.
#
# Each run appends a one-line summary to docs/phase2_runs.tsv as it finishes,
# so a partial grid is still readable without re-parsing every log.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEERF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${STEERF_ROOT}"

LAMS=${LAMS:-"0.0 0.25 0.5 1.0"}
SEEDS=${SEEDS:-"1 2 3"}
# Control arms, run after the main grid at every seed.
#
#   oracle:<lam>  STEER-F with H_togo read from the rollout's own entropy
#                 instead of forecast by the MTP heads. Isolates what the
#                 forecaster contributes: if this matches the MTP arm, the
#                 25M-parameter heads and the whole of Phase 1 are unnecessary
#                 and the method reduces to reweighting by realised downstream
#                 entropy. Costs the lambda=0 step time -- no heads, no extra
#                 forward -- so it is nearly free to include.
#   grpo          plain GRPO under Phase 2's exact evaluation protocol. Phase 0
#                 measured it without pass@16, so without this arm the headline
#                 comparison would be STEER-F against STEER only, with no
#                 no-reweighting reference on the same metric.
CONTROLS=${CONTROLS:-"oracle:0.25 oracle:0.5 grpo"}
SUMMARY="${STEERF_ROOT}/docs/phase2_runs.tsv"

# Needed by the one-off alignment analysis below; the per-arm launchers source
# this themselves, but the queue reads MODEL_PATH before delegating to them.
source "${SCRIPT_DIR}/env_1gpu.sh"
ALIGN_ROLLOUTS=${ALIGN_ROLLOUTS:-rollout_data/STEER-F-small/steer_20260815_105124/rollouts.jsonl}

log() { echo "[phase2-queue $(date +%m-%d\ %H:%M:%S)] $*"; }

[ -f "$SUMMARY" ] || printf 'lam\tseed\tstatus\tfinal_entropy\tbranch_gap\tmean@16\tbest@16\tsteps\tminutes\n' > "$SUMMARY"

while pgrep -f "verl.trainer.main_ppo" >/dev/null; do sleep 30; done

# --- one-off analysis, run before the grid so it cannot contend with training --
#
# Measures whether the branch STEER-F protects is the branch that goes on to be
# correct. It needs no training, only the Phase-0 rollouts (which carry the
# verifier's verdict), and it answers the objection that A_H optimises a proxy:
# the reward supplies correctness, so how badly can an entropy proxy mislead?
# The honest answer is "measure it" -- and the number belongs in §Limitations
# whichever way it comes out. ~15 min, once.
if [ ! -f docs/ah_vs_correctness.json ] && [ -f "${ALIGN_ROLLOUTS:-}" ]; then
  log "running the A_H-vs-correctness alignment analysis (one-off, ~15 min)"
  python scripts/analyze_ah_vs_correctness.py \
      --model "${MODEL_PATH}" --heads checkpoints/mtp_heads.pt \
      --calib checkpoints/mtp_calibration.json \
      --rollouts "${ALIGN_ROLLOUTS}" \
      --kappa 2 --gamma-h 0.7 --max-groups 120 \
      --out docs/ah_vs_correctness.json >> logs/ah_alignment.log 2>&1 \
      && log "alignment analysis done -> docs/ah_vs_correctness.json" \
      || log "alignment analysis failed (see logs/ah_alignment.log) — continuing"
fi

for seed in $SEEDS; do
  for lam in $LAMS; do
    tag="phase2-lam${lam}-s${seed}"
    if grep -qP "^${lam}\t${seed}\tok\t" "$SUMMARY" 2>/dev/null; then
      log "$tag already recorded ok — skipping"
      continue
    fi
    log "starting $tag"
    started=$(date +%s)
    LAM="$lam" SEED="$seed" "${SCRIPT_DIR}/phase2_1gpu.sh"
    rc=$?
    mins=$(( ($(date +%s) - started) / 60 ))
    log "$tag exited rc=$rc after ${mins}m"
    python scripts/summarize_run.py "logs/${tag}.log" \
        --lam "$lam" --seed "$seed" --minutes "$mins" >> "$SUMMARY" || true
    while pgrep -f "verl.trainer.main_ppo" >/dev/null; do sleep 20; done
  done
  for c in $CONTROLS; do
    case "$c" in
      oracle:*) lam="${c#oracle:}"; fc=oracle; key="oracle${lam}"
                tag="phase2-oracle-lam${lam}-s${seed}" ;;
      grpo)     lam=0.0;            fc=mtp;    key="grpo"
                tag="phase2-grpo-s${seed}" ;;
      *) log "unknown control '$c' — skipping"; continue ;;
    esac
    # The TSV key must differ from the main grid's, or the oracle control at
    # lambda=0.25 reads the main grid's own 0.25 row as "already done" and is
    # silently skipped -- losing the arm that tells us whether the forecaster
    # earns its parameters at all.
    if grep -qP "^${key}\t${seed}\tok\t" "$SUMMARY" 2>/dev/null; then
      log "$tag already recorded ok — skipping"; continue
    fi
    log "starting $tag"
    started=$(date +%s)
    if [ "$key" = "grpo" ]; then
      RUN_TAG="$tag" SEED="$seed" "${SCRIPT_DIR}/phase2_grpo.sh"
    else
      FORECAST="$fc" LAM="$lam" SEED="$seed" "${SCRIPT_DIR}/phase2_1gpu.sh"
    fi
    rc=$?
    mins=$(( ($(date +%s) - started) / 60 ))
    log "$tag exited rc=$rc after ${mins}m"
    python scripts/summarize_run.py "logs/${tag}.log" \
        --lam "$key" --seed "$seed" --minutes "$mins" >> "$SUMMARY" || true
    while pgrep -f "verl.trainer.main_ppo" >/dev/null; do sleep 20; done
  done
  log "seed ${seed} pass complete — grid is usable at $(echo $SEEDS | tr ' ' '\n' | grep -n "^${seed}$" | cut -d: -f1) seed(s)"
done

log "all done"
cat "$SUMMARY"
