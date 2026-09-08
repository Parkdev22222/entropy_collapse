#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Evaluate every finished training run on the six-benchmark suite.
#
#   bash run/run_eval_all.sh                       # the 20-run campaign
#   DRY=1 bash run/run_eval_all.sh                 # print the queue, run nothing
#   EVAL_SET=followups bash run/run_eval_all.sh    # the nine ablations
#   EVAL_SET=all bash run/run_eval_all.sh          # both
#   REPO=user/repo bash run/run_eval_all.sh        # fetch checkpoints from the Hub
#
# No training. Each run is one call to run/eval_steerf.sh, which does verl's
# val_only pass twice (avg@32 on AIME24/AIME25/AMC23, avg@1 on MATH500 /
# Minerva / OlympiadBench / GSM8K) -- about 40 minutes per checkpoint on 2xA100.
#
# THE FAILURE THIS SCRIPT EXISTS TO PREVENT
#   The first six-benchmark attempt produced five eval logs whose MODEL_PATH
#   was the SAME checkpoint (grpo .../global_step_110). Every number in them
#   was the GRPO row printed five times, and nothing in the pipeline noticed:
#   collect_results.py reads whatever the log contains. So this script
#     - resolves the checkpoint per run and prints it before starting,
#     - reads actor_rollout_ref.model.path back OUT of the finished log and
#       aborts the queue if it is not the path it asked for,
#     - re-checks at the end that no two runs shared a path.
#
# PER-PROBLEM SCORES
#   VAL_DATA_DIR is set per run, so each eval leaves {input, output, score}
#   JSONL behind -- the paired across-problem error bars the manuscript's
#   Limitations section currently says are not computable. Needs
#   `bash run/instrument_phase2.sh --apply` first; without it the variable is
#   simply ignored and the queue still runs (it warns).
#
# CHECKPOINTS
#   The campaign uploads each finished run to the Hub and deletes the local
#   copy, so most checkpoints are not on disk by the time this runs. With REPO
#   set, each one is downloaded just before its eval and deleted straight
#   after, which keeps the working set at a single ~3.1 GB directory.
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}" || { echo "FATAL: cannot cd to ${ROOT}" >&2; exit 1; }
# shellcheck source=run/_arms.sh
. "${ROOT}/run/_arms.sh"

SEEDS=${SEEDS:-"1 2 3 4 5"}
EVAL_SET=${EVAL_SET:-campaign}          # campaign | followups | all
FOLLOWUP_ARMS=${FOLLOWUP_ARMS:-"lam0.1 lam0.5 lam0-tree xclip-signed xclip-steer rloo-signed rloo-steer opo-signed opo-steer"}
STEPS=${STEPS:-110}
LOG_DIR="${ROOT}/logs/experiments"
CKPT_ROOT="${ROOT}/checkpoints/STEER-F"
STAGE="${ROOT}/checkpoints/_eval_stage"  # where Hub downloads land
MIN_FREE_GB=${MIN_FREE_GB:-15}
DRY=${DRY:-0}
REPO=${REPO:-}
FORCE=${FORCE:-0}                        # 1 = re-evaluate runs that already have a log

banner () { printf '\n========================================\n%s\n========================================\n' "$*"; }
free_gb () { df -BG --output=avail "${ROOT}" 2>/dev/null | tail -1 | tr -dc '0-9'; }

# ------------------------------------------------------------- the queue
declare -a QUEUE=()
add_run () {   # <arm> <seed>
    local rn
    rn="$(run_name_for "$1" "$2")" || { echo "FATAL: unknown arm '$1'" >&2; exit 2; }
    if ! train_log_done "${LOG_DIR}" "${rn}" "${STEPS}"; then
        printf '  skip  %-24s s%-2s %-52s (training log never reached step %s)\n' "$1" "$2" "${rn}" "${STEPS}"
        return
    fi
    if [ "${FORCE}" != "1" ] && [ -s "${LOG_DIR}/eval-$1-s$2.log" ]; then
        printf '  skip  %-24s s%-2s %-52s (already evaluated)\n' "$1" "$2" "${rn}"
        return
    fi
    printf '  QUEUE %-24s s%-2s %-52s\n' "$1" "$2" "${rn}"
    QUEUE+=("$1|$2|${rn}")
}

case "${EVAL_SET}" in
    campaign|followups|all) ;;
    *) echo "FATAL: EVAL_SET must be campaign|followups|all, got '${EVAL_SET}'" >&2; exit 2 ;;
esac
if [ "${EVAL_SET}" = "campaign" ] || [ "${EVAL_SET}" = "all" ]; then
    for s in ${SEEDS}; do for a in ${CAMPAIGN_ARMS}; do add_run "${a}" "${s}"; done; done
fi
if [ "${EVAL_SET}" = "followups" ] || [ "${EVAL_SET}" = "all" ]; then
    for a in ${FOLLOWUP_ARMS}; do add_run "${a}" 1; done
fi

printf '\n%s eval(s) queued\n' "${#QUEUE[@]}"
[ "${#QUEUE[@]}" -eq 0 ] && { echo "Nothing to do."; exit 0; }
[ "${DRY}" = "1" ] && { echo "(DRY=1, stopping here)"; exit 0; }

# ---------------------------------------------------------------- guards
if ! bash run/instrument_phase2.sh --check >/dev/null 2>&1; then
    echo "WARNING: run/instrument_phase2.sh --check does not pass."
    echo "         The evals will run, but validation_data_dir will be ignored and"
    echo "         the six benchmarks will have no per-problem scores. Fix with:"
    echo "           bash run/instrument_phase2.sh --apply"
    if [ "${ALLOW_UNINSTRUMENTED:-0}" != "1" ]; then
        echo "REFUSE: set ALLOW_UNINSTRUMENTED=1 to evaluate without them anyway." >&2
        exit 2
    fi
fi
# The eval passes are verl val_only runs, so they die on a broken stack exactly
# as training does -- and 20 of them failing in sequence looks like 20 missing
# checkpoints rather than one broken box.
if ! env_preflight "${ROOT}"; then
    echo "REFUSE: the training environment is broken -- nothing would evaluate." >&2
    exit 2
fi
# `hf` in huggingface_hub >= 0.34, `huggingface-cli` before that. Accept either
# so nobody upgrades the package to get the new name -- see run/hf_backup.sh.
HF_CLI=${HF_CLI:-$(command -v hf || command -v huggingface-cli || true)}
if [ -n "${REPO}" ] && [ -z "${HF_CLI}" ]; then
    echo "REFUSE: REPO is set but neither 'hf' nor 'huggingface-cli' is on PATH." >&2
    echo "        pip install \"huggingface_hub>=0.34,<1.0\"   (an unpinned upgrade breaks training)" >&2
    exit 2
fi

mkdir -p "${LOG_DIR}" "${STAGE}"

# -------------------------------------------------- checkpoint resolution
# Prints the HF model directory to use, or nothing. Sets FETCHED=<dir to rm>
# when the checkpoint had to come off the Hub.
FETCHED=""
resolve_ckpt () {   # <arm> <seed> <run-name>
    FETCHED=""
    local arm="$1" seed="$2" rn="$3" cand d best=""

    for cand in "${rn}" $(ckpt_alias_for "${arm}" "${seed}"); do
        # With save_best_only there is exactly one global_step_* per run, and
        # it IS the AIME24 argmax. Older runs kept several; take the highest
        # step and say so, so the selection rule is never silent.
        for d in $(ls -d "${CKPT_ROOT}/${cand}"/global_step_* 2>/dev/null |
                   sed 's/.*global_step_//' | sort -n); do
            [ -d "${CKPT_ROOT}/${cand}/global_step_${d}/actor/huggingface" ] || continue
            best="${CKPT_ROOT}/${cand}/global_step_${d}/actor/huggingface"
        done
        [ -n "${best}" ] && { echo "${best}"; return 0; }
    done

    [ -n "${REPO}" ] || return 1

    # Hub layout, written by run/hf_backup.sh: <run>/global_step_<N>/<files>
    local step
    step="$(REPO="${REPO}" RUN="${rn}" python3 - <<'PY' 2>/dev/null
import os, re
from huggingface_hub import HfApi
repo, run = os.environ["REPO"], os.environ["RUN"]
steps = set()
try:
    for e in HfApi().list_repo_tree(repo, path_in_repo=run, recursive=True, repo_type="model"):
        m = re.search(r"/global_step_(\d+)/", "/" + e.path + "/")
        if m:
            steps.add(int(m.group(1)))
except Exception:
    pass
print(max(steps) if steps else "")
PY
)"
    [ -n "${step}" ] || return 1
    local dest="${STAGE}/${rn}/global_step_${step}"
    rm -rf "${dest}"; mkdir -p "${dest}"
    "${HF_CLI}" download "${REPO}" --repo-type model \
        --include "${rn}/global_step_${step}/*" --local-dir "${STAGE}/_dl" >/dev/null 2>&1 || return 1
    mv "${STAGE}/_dl/${rn}/global_step_${step}"/* "${dest}/" 2>/dev/null || return 1
    rm -rf "${STAGE}/_dl"
    FETCHED="${STAGE}/${rn}"
    echo "${dest}"
}

# ------------------------------------------------------------------- run
banner "eval: ${#QUEUE[@]} run(s), set=${EVAL_SET}"
declare -a FAILED=()
for item in "${QUEUE[@]}"; do
    arm="${item%%|*}"; rest="${item#*|}"; seed="${rest%%|*}"; rn="${rest##*|}"
    log="${LOG_DIR}/eval-${arm}-s${seed}.log"

    avail="$(free_gb)"
    if [ -n "${avail}" ] && [ "${avail}" -lt "${MIN_FREE_GB}" ]; then
        echo "STOP: only ${avail} GB free, need ${MIN_FREE_GB}." >&2
        exit 1
    fi

    mp="$(resolve_ckpt "${arm}" "${seed}" "${rn}")"
    if [ -z "${mp}" ] || [ ! -d "${mp}" ]; then
        echo "[eval] MISSING checkpoint for ${arm} s${seed} (${rn}) -- skipping"
        FAILED+=("${arm}-s${seed}:no-checkpoint")
        continue
    fi

    banner "${arm}  seed ${seed}"
    echo "  run        ${rn}"
    echo "  model      ${mp}"
    echo "  log        ${log}"
    echo "  fetched    ${FETCHED:-no (local)}"
    start=$(date +%s)

    MODEL_PATH="${mp}" \
    VAL_DATA_DIR="${ROOT}/validation_data/eval/${arm}-s${seed}" \
    EVAL_RESULTS_DIR="${ROOT}/results/eval_json/${arm}-s${seed}" \
        bash run/eval_steerf.sh > "${log}" 2>&1
    st=$?
    printf '[eval] %s s%s exit %s after %s min\n' \
        "${arm}" "${seed}" "${st}" "$(( ($(date +%s) - start) / 60 ))"

    # Read the path back out of the log. This is the check that would have
    # caught the five-identical-logs failure at run 2 instead of at analysis.
    seen="$(grep -oE 'actor_rollout_ref\.model\.path=[^ ]+' "${log}" | sed 's/.*path=//' | sort -u)"
    if [ "$(printf '%s\n' "${seen}" | grep -c .)" != "1" ] || [ "${seen}" != "${mp}" ]; then
        echo "ABORT: ${log} evaluated the wrong checkpoint." >&2
        echo "       asked for : ${mp}" >&2
        echo "       log says  : ${seen:-<nothing>}" >&2
        exit 1
    fi
    echo "  verified   model.path matches"

    if [ "${st}" -ne 0 ]; then
        FAILED+=("${arm}-s${seed}:exit${st}")
    else
        for k in aime_2024_dapo_boxed:mean@32 math500:mean@1; do
            printf '  %-28s %s\n' "${k}" \
                "$(grep -oE "val-core/${k%%:*}/acc/${k##*:}:[0-9.]+" "${log}" | tail -1)"
        done
    fi

    [ -n "${FETCHED}" ] && { rm -rf "${FETCHED}"; echo "  staged copy removed"; }
done

# --------------------------------------------------------------- summary
banner "eval finished"
python3 - "${LOG_DIR}" <<'PY'
import re, sys, collections
from pathlib import Path
paths = collections.defaultdict(list)
for f in sorted(Path(sys.argv[1]).glob("eval-*.log")):
    m = re.match(r"eval-(.+)-s(\d+)\.log$", f.name)
    if not m:
        print(f"  {f.name}: name does not match eval-<arm>-s<seed>.log "
              "-- collect_results.py will skip it silently")
        continue
    found = set(re.findall(r"actor_rollout_ref\.model\.path=(\S+)", f.read_text(errors="replace")))
    for p in found:
        paths[p].append(f.name)
    print(f"  {f.name:44s} {' '.join(sorted(found)) or '<no model.path>'}")
dupes = {p: v for p, v in paths.items() if len(v) > 1}
if dupes:
    print("\n  COLLISION -- these logs share a checkpoint, their numbers are not independent:")
    for p, v in dupes.items():
        print(f"    {p}\n      {', '.join(v)}")
    sys.exit(1)
print("\n  no two eval logs share a checkpoint")
PY
collision=$?

echo
echo "Next:  python3 scripts/collect_results.py --logs ${LOG_DIR} --out results/summary.tsv"
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo
    echo "${#FAILED[@]} run(s) did not produce a clean eval:"
    printf '  %s\n' "${FAILED[@]}"
fi
exit $(( collision != 0 ? 1 : (${#FAILED[@]} > 0 ? 1 : 0) ))
