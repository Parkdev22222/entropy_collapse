#!/usr/bin/env bash
# Phase 1 (gate G1), end to end, serialised on the single GPU.
#
#   nohup ./run/phase1_queue.sh > logs/phase1_queue.log 2>&1 &
#
# Waits for the head warm-up to finish, then:
#   1. Monte-Carlo validation + (kappa, gamma_H) grid + calibration
#   2. extract the calibration where the training script looks for it
#   3. branch recall@10% on real rollout groups  -> the other half of G1
#   4. the same recall statistic with UNTRAINED heads, as a control
#
# Step 4 is not in the plan and is the reason the verdict can be trusted.
# A_H is supported only where a rollout still has siblings, which is also where
# branch points live, so the *support* of A_H is partially informative on its
# own: measured with random heads the recall already came out at 0.60. Without
# the control, a pass on criterion 2 cannot be attributed to the forecast's
# ranking rather than to that support.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEERF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${STEERF_ROOT}"
source "${SCRIPT_DIR}/env_1gpu.sh"

ROLLOUTS=${ROLLOUTS:-rollout_data/STEER-F-small/steer_20260815_105124/rollouts.jsonl}
PROBLEMS=${PROBLEMS:-${STEER_ROOT}/datasets/math500.parquet}

log() { echo "[phase1 $(date +%H:%M:%S)] $*"; }

while pgrep -f "phase1_warmup_heads" >/dev/null; do sleep 30; done
[ -f checkpoints/mtp_heads.pt ] || { log "no checkpoints/mtp_heads.pt — warm-up failed"; exit 1; }
log "heads ready"

# --- 1. MC validation, grid, calibration ------------------------------------
# --cont-temperature/--cont-top-p are set to the *training* rollout settings
# (1.0 / 1.0) rather than the script defaults (0.7 / 0.9). H_togo is defined as
# expected future entropy under the policy, so the continuations that define
# the measured counterpart have to come from that same policy; sampling them
# from a sharpened one would score the forecast against a distribution the
# policy never visits during training.
log "MC validation"
python scripts/phase1_validate.py \
    --model "${MODEL_PATH}" \
    --heads checkpoints/mtp_heads.pt \
    --problems "${PROBLEMS}" \
    --n-problems 25 --n-trajectories 32 --n-continuations 16 \
    --max-prefixes 600 \
    --traj-temperature 1.0 --cont-temperature 1.0 --cont-top-p 1.0 \
    --calibrate \
    --out docs/phase1_results.json
log "validation exited $?"

# --- 2. calibration where run_steerf.sh expects it --------------------------
if [ -f docs/phase1_results.json ]; then
  python -c "
import json
d = json.load(open('docs/phase1_results.json'))
c = d.get('calibration')
if c:
    json.dump(c, open('checkpoints/mtp_calibration.json', 'w'), indent=2)
    print('[phase1] wrote checkpoints/mtp_calibration.json')
else:
    print('[phase1] no calibration in results — Phase 2 would run uncalibrated (ablation A6)')
"
fi

# --- 3. branch recall on real rollout groups --------------------------------
KAPPA=$(python -c "
import json
try:
    print(json.load(open('docs/phase1_results.json'))['chosen']['kappa'])
except Exception:
    print(4)
" 2>/dev/null || echo 4)
GAMMA=$(python -c "
import json
try:
    print(json.load(open('docs/phase1_results.json'))['chosen']['gamma_h'])
except Exception:
    print(0.85)
" 2>/dev/null || echo 0.85)
log "selected kappa=${KAPPA} gamma_h=${GAMMA}"

log "branch recall (trained heads)"
python scripts/phase1_branch_recall.py \
    --model "${MODEL_PATH}" --heads checkpoints/mtp_heads.pt \
    --rollouts "${ROLLOUTS}" --calib checkpoints/mtp_calibration.json \
    --kappa "${KAPPA}" --gamma-h "${GAMMA}" \
    --phase1-results docs/phase1_results.json \
    --max-groups 96 \
    --out docs/phase1_recall.json
log "recall exited $?"

# --- 4. control: the same statistic with untrained heads --------------------
log "branch recall (UNTRAINED control)"
python scripts/make_dummy_heads.py --model "${MODEL_PATH}" \
    --out checkpoints/mtp_heads_control.pt --num-heads 8
python - <<'EOF'
import torch
d = torch.load('checkpoints/mtp_heads_control.pt', map_location='cpu', weights_only=False)
d.pop('untrained', None)          # the guard exists to keep dummies out of Phase 2;
torch.save(d, 'checkpoints/mtp_heads_control_nogaurd.pt')  # this path is a measurement, not a run
EOF
python scripts/phase1_branch_recall.py \
    --model "${MODEL_PATH}" --heads checkpoints/mtp_heads_control_nogaurd.pt \
    --rollouts "${ROLLOUTS}" \
    --kappa "${KAPPA}" --gamma-h "${GAMMA}" \
    --max-groups 96 \
    --out docs/phase1_recall_control.json
log "control exited $?"

log "done — compare docs/phase1_recall.json against docs/phase1_recall_control.json"
