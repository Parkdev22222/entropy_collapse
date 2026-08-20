#!/usr/bin/env bash
# STEER-F Phase 1: warm up the MTP heads on base-policy rollouts, then
# Monte-Carlo-validate the forecast, fit the per-head calibration and select
# (kappa, gamma_H). The policy is frozen throughout, so nothing here touches
# the paper-parity training setup.
#
# Gate G1 (steer_f.validation.evaluate_gate_g1) must PASS before any
# steerf_lam > 0 run is worth GPU time:
#   1. within-problem Spearman rho(H_togo, measured future entropy) >= 0.2
#   2. branch recall@10% above chance (p < 0.05) on the sibling support —
#      ALWAYS compare against the untrained-heads control this script also
#      produces; the support alone already localises branch points.
set -xe

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=run/_gpu_defaults.sh
. "${SCRIPT_DIR}/_gpu_defaults.sh"
export PYTHONPATH="${STEER_ROOT}:$PYTHONPATH"

model_path=${MODEL_PATH:-"Qwen/Qwen2.5-Math-7B"}
model_tag=$(basename "${model_path}")
SCALE=${SCALE:-paper}
rollouts=${ROLLOUTS:-${STEER_ROOT}/rollout_data/warmup/${model_tag}${SCALE:+-${SCALE}}/rollouts.jsonl}
case "${SCALE}" in
  paper) N_PROB=25; N_TRAJ=32; N_CONT=16; MAX_PRE=600 ;;
  smoke) N_PROB=3;  N_TRAJ=4;  N_CONT=4;  MAX_PRE=24  ;;
esac
ckpt_dir=${STEER_ROOT}/checkpoints
mkdir -p "${ckpt_dir}" "${STEER_ROOT}/docs"

heads_out=${ckpt_dir}/mtp_heads_${model_tag}${SCALE:+-${SCALE}}.pt
calib_out=${ckpt_dir}/mtp_calibration_${model_tag}${SCALE:+-${SCALE}}.json
results_out=${STEER_ROOT}/docs/phase1_results_${model_tag}${SCALE:+-${SCALE}}.json

# 1) head warm-up (K=8; kappa is selected afterwards, no retraining needed)
python3 ${STEER_ROOT}/scripts/phase1_warmup_heads.py \
    --model "${model_path}" \
    --rollouts "${rollouts}" \
    --out "${heads_out}" \
    --num-heads 8 --head-hidden 1024 --epochs 1 \
    --max-length 1024 ${WARMUP_EXTRA_ARGS:-}

# 2) MC validation + (kappa, gamma_H) grid + calibration fit
python3 ${STEER_ROOT}/scripts/phase1_validate.py \
    --model "${model_path}" \
    --heads "${heads_out}" \
    --problems ${PROBLEMS:-${STEER_ROOT}/datasets/math500.parquet} \
    --n-problems ${N_PROB} --n-trajectories ${N_TRAJ} --n-continuations ${N_CONT} \
    --max-prefixes ${MAX_PRE} --calibrate \
    --out "${results_out}"

python3 - "$results_out" "$calib_out" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
json.dump(d["calibration"], open(sys.argv[2], "w"), indent=2)
print(f"[phase1] calibration -> {sys.argv[2]}")
print(f"[phase1] selected kappa={d['chosen']['kappa']} gamma_H={d['chosen']['gamma_h']} "
      f"(rho={d['chosen']['rho_mean']:.4f})")
print("[phase1] export these for run_steerf.sh:")
print(f"         STEERF_KAPPA={d['chosen']['kappa']} STEERF_GAMMA_H={d['chosen']['gamma_h']}")
PY

# 3) branch recall on real rollout groups: trained heads vs UNTRAINED control
python3 ${STEER_ROOT}/scripts/phase1_branch_recall.py \
    --model "${model_path}" --heads "${heads_out}" --calib "${calib_out}" \
    --rollouts "${rollouts}" --phase1-results "${results_out}" \
    --out ${STEER_ROOT}/docs/phase1_recall_${model_tag}.json

python3 ${STEER_ROOT}/scripts/make_dummy_heads.py \
    --model "${model_path}" --out ${ckpt_dir}/mtp_heads_control_${model_tag}.pt
python3 ${STEER_ROOT}/scripts/phase1_branch_recall.py \
    --model "${model_path}" --heads ${ckpt_dir}/mtp_heads_control_${model_tag}.pt \
    --rollouts "${rollouts}" \
    --out ${STEER_ROOT}/docs/phase1_recall_control_${model_tag}.json

echo "[phase1] compare docs/phase1_recall_${model_tag}.json vs the _control file"
echo "         before trusting gate G1's recall criterion."
