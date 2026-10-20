#!/usr/bin/env bash
# Two-channel decomposition of measured entropy change (docs/channel_decomposition.md).
#
#   DRY=1 bash run/channel_decomp.sh            # where every checkpoint is
#         bash run/channel_decomp.sh            # primary: base -> {grpo,perm,signed}100
#   SET=all bash run/channel_decomp.sh          # + the secondary intervals
#   SEED=3 bash run/channel_decomp.sh           # replication on another seed
#
# SEED other than 1 writes to results/decomp_s<SEED>, reuses the base model's
# samples and entropies from results/decomp (same prompts, same sampler seed),
# and runs base -> {grpo,perm}100 of that seed; SIGNED_RUN=<hub run> adds
# base -> signed100 from that run. INTERVALS overrides the list.
#
# Any Hub checkpoint can be given a label with HUB_CKPTS="label=run@step ...",
# then used in INTERVALS (Amendment 2, docs/channel_decomposition.md):
#   OUT_DIR=results/decomp_rep RESULT=results/channel_decomp_rep.json \
#   HUB_CKPTS="grpos2at80=grpo-Qwen2.5-Math-1.5B-s2@80" \
#   INTERVALS="base:grpos2at80" bash run/channel_decomp.sh
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
SEED=${SEED:-1}
if [ "${SEED}" = "1" ]; then SFX=""; else SFX="_s${SEED}"; fi
OUT=${OUT_DIR:-results/decomp${SFX}}
RESULT=${RESULT:-results/channel_decomp${SFX}.json}
PILOT=${PILOT:-/workspace/entropy_lora/checkpoints/STEER-F/pilot-steer-f-1p5b-s1-150}
EVAL_REPOS=${EVAL_REPOS:-"DSDSh/steer-f_2 DSDSh/sssss"}
SET=${SET:-primary}
DRY=${DRY:-0}
PY="python3 scripts/channel_decomp.py"
GRPO_RUN=grpo-Qwen2.5-Math-1.5B-s${SEED}
PERM_RUN=steer-f-Qwen2.5-Math-1.5B-s${SEED}-tree-rollout-permuted
SIGNED_RUN=${SIGNED_RUN:-}

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
for e in ${HUB_CKPTS:-}; do
    l="${e%%=*}"; rs="${e#*=}"; r="${rs%@*}"; st="${rs##*@}"
    SPEC[$l]="checkpoints/_manip/${r}/global_step_${st}|${r}|${st}"
done
if [ "${SEED}" != "1" ]; then
    # The pilot is seed 1; another seed's signed arm comes only from SIGNED_RUN.
    if [ -n "${SIGNED_RUN}" ]; then
        SPEC[signed100]="checkpoints/_manip/${SIGNED_RUN}/global_step_100|${SIGNED_RUN}|100"
    else
        unset 'SPEC[signed100]'
    fi
fi
PRIMARY="base:grpo100 base:perm100"
[ -n "${SPEC[signed100]+x}" ] && PRIMARY="${PRIMARY} base:signed100"
SECONDARY="grpo80:grpo110 perm80:perm110 base:signed50 signed50:signed100 signed100:signed150 signed150:signed200"
NAMED="${INTERVALS:-}"
if [ -n "${NAMED}" ]; then :
else case "${SET}" in
  primary) INTERVALS="${PRIMARY}" ;;
  all)     [ "${SEED}" = "1" ] || { echo "SET=all is seed 1 only" >&2; exit 2; }
           INTERVALS="${PRIMARY} ${SECONDARY}" ;;
  *) echo "SET must be primary or all" >&2; exit 2 ;;
esac; fi
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

echo "seed=${SEED}  set=${SET}  out=${OUT}"
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
# The base model's samples and self-entropies do not depend on the seed under
# study: same prompts, same sampler seed. Reuse them rather than redraw.
if [ "${OUT}" != "results/decomp" ] && [ -s results/decomp/samples/base.json ]; then
    mkdir -p "${OUT}/samples" "${OUT}/ent"
    cp -n results/decomp/prompts.json "${OUT}/"
    cp -n results/decomp/samples/base.json "${OUT}/samples/"
    [ -s results/decomp/ent/base__base.json ] && cp -n results/decomp/ent/base__base.json "${OUT}/ent/"
fi
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
# Named intervals are analysed alone; otherwise every registered interval on disk.
${PY} analyze --out "${OUT}" --result "${RESULT}" ${NAMED:+--intervals "${NAMED}"}
