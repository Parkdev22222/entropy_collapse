#!/usr/bin/env bash
# Two-channel decomposition of measured entropy change (docs/channel_decomposition.md).
#
#   DRY=1 bash run/channel_decomp.sh            # where every checkpoint is
#         bash run/channel_decomp.sh            # primary: base -> {grpo,perm,signed}100
#   SET=all bash run/channel_decomp.sh          # + the secondary intervals
#
# One GPU. Every stage is skipped when its output exists AND was made from the
# same checkpoint, so a rerun resumes and a changed path is redone.
# Hub checkpoints missing on disk are fetched (EVAL_REPOS) into checkpoints/_manip
# with the layout run_eval_all.sh reads, and kept for the next stage.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
# shellcheck source=run/_arms.sh
source run/_arms.sh

BASE=${BASE_MODEL:-Qwen/Qwen2.5-Math-1.5B}
DATA=${DATA:-datasets/DAPO-Math-17k.parquet}
OUT=${OUT_DIR:-results/decomp}
RESULT=${RESULT:-results/channel_decomp.json}
PILOT=${PILOT:-/workspace/entropy_lora/checkpoints/STEER-F/pilot-steer-f-1p5b-s1-150}
EVAL_REPOS=${EVAL_REPOS:-"DSDSh/steer-f_2 DSDSh/sssss"}
SET=${SET:-primary}
DRY=${DRY:-0}
PY="python3 scripts/channel_decomp.py"
GRPO_RUN=grpo-Qwen2.5-Math-1.5B-s1
PERM_RUN=steer-f-Qwen2.5-Math-1.5B-s1-tree-rollout-permuted

# label -> "path|hub-run|step"; hub-run empty for checkpoints that are local.
declare -A SPEC=(
  [base]="${BASE}||"
  [grpo80]="checkpoints/_manip/${GRPO_RUN}/global_step_80|${GRPO_RUN}|80"
  [grpo100]="checkpoints/_manip/${GRPO_RUN}/global_step_100|${GRPO_RUN}|100"
  [grpo110]="checkpoints/_manip/${GRPO_RUN}/global_step_110|${GRPO_RUN}|110"
  [perm80]="checkpoints/_manip/${PERM_RUN}/global_step_80|${PERM_RUN}|80"
  [perm100]="checkpoints/_manip/${PERM_RUN}/global_step_100|${PERM_RUN}|100"
  [perm110]="checkpoints/_manip/${PERM_RUN}/global_step_110|${PERM_RUN}|110"
  [signed50]="${PILOT}/global_step_50/actor/huggingface||"
  [signed100]="${PILOT}/global_step_100/actor/huggingface||"
  [signed150]="${PILOT}/global_step_150/actor/huggingface||"
  [signed200]="${PILOT}/global_step_200/actor/huggingface||"
)
PRIMARY="base:grpo100 base:perm100 base:signed100"
SECONDARY="grpo80:grpo110 perm80:perm110 base:signed50 signed50:signed100 signed100:signed150 signed150:signed200"
case "${SET}" in
  primary) INTERVALS="${PRIMARY}" ;;
  all)     INTERVALS="${PRIMARY} ${SECONDARY}" ;;
  *) echo "SET must be primary or all" >&2; exit 2 ;;
esac
LABELS=$(for iv in ${INTERVALS}; do echo "${iv%%:*}"; echo "${iv##*:}"; done | awk '!s[$0]++')

path_of () { local s="${SPEC[$1]}"; echo "${s%%|*}"; }
present () { [ "$1" = "${BASE}" ] || hf_weights_present "$1"; }
# Output counts as done only if it was made from the same checkpoint(s).
done_for () {   # <json> <key=value>...
    [ -s "$1" ] || return 1
    python3 - "$@" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
sys.exit(any(r.get(k) != v for k, v in (a.split("=", 1) for a in sys.argv[2:])))
PY
}

echo "set=${SET}  out=${OUT}"
echo "intervals: ${INTERVALS}"
echo "checkpoints:"
missing=0
for l in ${LABELS}; do
    p="$(path_of "$l")"; rest="${SPEC[$l]#*|}"; run="${rest%%|*}"; step="${rest##*|}"
    if present "$p"; then printf '  %-10s %s\n' "$l" "$p"
    elif [ -n "${run}" ]; then printf '  %-10s fetch %s step %s\n' "$l" "${run}" "${step}"
    else printf '  %-10s MISSING %s\n' "$l" "$p"; missing=$((missing + 1)); fi
done
[ "${missing}" -eq 0 ] || { echo "STOP: ${missing} local checkpoint(s) missing."; exit 1; }
if [ "${DRY}" = "1" ]; then exit 0; fi

mkdir -p "${OUT}"
# 1) fetch and sample every checkpoint once
for l in ${LABELS}; do
    p="$(path_of "$l")"; rest="${SPEC[$l]#*|}"; run="${rest%%|*}"; step="${rest##*|}"
    if ! present "$p"; then
        python3 scripts/manip_check.py fetch --repos "${EVAL_REPOS}" --run "${run}" \
            --step "${step}" --dest checkpoints/_manip
        present "$p" || { echo "STOP: could not fetch ${l}"; exit 1; }
    fi
    done_for "${OUT}/samples/${l}.json" "model=${p}" && continue
    ${PY} sample --model "$p" --label "$l" --base "${BASE}" --data "${DATA}" --out "${OUT}"
done
# 2) every (samples, model) pair an interval needs
for iv in ${INTERVALS}; do
    a="${iv%%:*}"; b="${iv##*:}"
    for pair in "$a:$a" "$b:$b" "$a:$b" "$b:$a"; do
        s="${pair%%:*}"; m="${pair##*:}"; mp="$(path_of "$m")"
        done_for "${OUT}/ent/${s}__${m}.json" "model=${mp}" "samples_model=$(path_of "$s")" && continue
        ${PY} entropy --model "${mp}" --label "$m" --samples "$s" --out "${OUT}"
    done
done
# 3) decomposition
${PY} analyze --out "${OUT}" --result "${RESULT}"
