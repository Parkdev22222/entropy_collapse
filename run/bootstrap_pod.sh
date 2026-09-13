#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Assemble a runnable tree on a fresh pod, out of the two branches this repo is
# split across.
#
#   bash run/bootstrap_pod.sh            # take what is missing
#   DRY=1 bash run/bootstrap_pod.sh      # list it, change nothing
#   PAPER_REF=origin/paper bash ...      # a different donor ref
#
# WHY THIS EXISTS
#   The work lives on two branches and neither is runnable alone:
#
#     this branch   run/ (the queues), scripts/, paper/, results/, tests/
#     origin/paper  verl/, datasets/*.parquet, logs/, requirements.txt,
#                   setup.py, and the launchers the queues invoke
#                   (run_steerf.sh, run_grpo.sh, run_uniform_ablation.sh,
#                   eval_steerf.sh, _gpu_defaults.sh, warmup_and_validate.sh)
#
#   Clone this branch alone and `import verl` raises, datasets/ is empty, and
#   run_campaign.sh calls launchers that are not there. That is exactly the
#   report a fresh H100 box produced on 2026-09-13.
#
# WHY NOT `git checkout origin/paper -- run`
#   Both branches carry run/setup_env.sh, run/_check_deps.py,
#   run/run_steerf_linear.sh and run/run_tree_2x2.sh. Checking out the donor's
#   whole run/ overwrites this branch's setup_env.sh with an older copy that has
#   no huggingface_hub pin check -- the guard against the failure that has now
#   killed runs twice. So the rule here is strictly additive:
#
#       take a path from the donor only if HEAD does not track it.
#
#   Nothing this branch owns is ever touched, which makes the script safe to
#   re-run and impossible to get wrong by hand.
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}" || { echo "FATAL: cannot cd to ${ROOT}" >&2; exit 1; }

PAPER_REF=${PAPER_REF:-origin/paper}
DRY=${DRY:-0}

# Donor paths that are history rather than inputs. Everything else the donor has
# and HEAD lacks is taken.
SKIP_RE='^(archive/|experiments_state|results/\.ipynb_checkpoints/|logs/experiments_smoke/)'

say () { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok ()  { printf '  \033[32mOK\033[0m    %s\n' "$*"; }
warn (){ printf '  \033[33mWARN\033[0m  %s\n' "$*"; }
bad () { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; }

say "1. refs"
git rev-parse --verify HEAD >/dev/null 2>&1 || { bad "not a git checkout"; exit 1; }
ok "HEAD      $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
if ! git rev-parse --verify "${PAPER_REF}" >/dev/null 2>&1; then
    warn "${PAPER_REF} not present, fetching"
    git fetch origin "${PAPER_REF#origin/}" || { bad "fetch failed"; exit 1; }
fi
git rev-parse --verify "${PAPER_REF}" >/dev/null 2>&1 \
    || { bad "no such ref: ${PAPER_REF}"; exit 1; }
ok "donor     ${PAPER_REF} @ $(git rev-parse --short "${PAPER_REF}")"

say "2. what the donor has and HEAD does not"
# Set difference over tracked paths. Doing it on full paths rather than on
# directories is what keeps run/setup_env.sh out of the take-list: the donor has
# it, but so does HEAD, so it never appears here.
mapfile -t TAKE < <(
    comm -23 \
        <(git ls-tree -r --name-only "${PAPER_REF}" | sort) \
        <(git ls-tree -r --name-only HEAD           | sort) \
    | grep -Ev "${SKIP_RE}" || true
)

if [ "${#TAKE[@]}" -eq 0 ]; then
    ok "nothing to take -- this tree already has everything ${PAPER_REF} carries"
    exit 0
fi

# Report by top-level directory; the file list runs to thousands under verl/.
printf '%s\n' "${TAKE[@]}" | awk -F/ '{print (NF>1 ? $1"/" : $1)}' \
    | sort | uniq -c | sort -rn \
    | while read -r n d; do printf '  %6s file(s)  %s\n' "${n}" "${d}"; done

say "3. paths HEAD owns, left untouched"
for p in run/setup_env.sh run/_check_deps.py run/run_steerf_linear.sh run/run_tree_2x2.sh; do
    git ls-tree -r --name-only HEAD -- "${p}" | grep -q . && ok "${p}"
done

if [ "${DRY}" = "1" ]; then
    say "DRY=1, nothing taken"
    exit 0
fi

say "4. taking them"
# One checkout call per batch rather than per file: verl/ alone is ~1500 paths.
printf '%s\0' "${TAKE[@]}" | xargs -0 -n 200 git checkout "${PAPER_REF}" -- \
    || { bad "checkout failed"; exit 1; }
ok "${#TAKE[@]} path(s) checked out from ${PAPER_REF}"

say "5. verify"
rc=0
if PYTHONPATH="${ROOT}:${PYTHONPATH:-}" python3 -c 'from verl import DataProto' 2>/dev/null; then
    ok "from verl import DataProto"
else
    warn "verl still does not import -- run bash run/setup_env.sh (vllm/ray/torch)"
fi
for f in datasets/DAPO-Math-17k.parquet datasets/aime24.parquet \
         run/run_steerf.sh run/run_uniform_ablation.sh run/run_grpo.sh run/eval_steerf.sh; do
    [ -e "${f}" ] && ok "${f}" || { bad "${f} still missing"; rc=1; }
done

# The tree-rollout patch ships already applied in the donor's verl/. setup_env.sh
# cannot tell "applied" from "patch file present" and will suggest re-applying
# it; doing so conflicts. Say so here, where the answer is known.
spmd=verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py
if grep -q steerf_tree_depths "${spmd}" 2>/dev/null; then
    ok "tree-rollout patch already in ${PAPER_REF}'s verl -- do NOT git apply it again"
else
    warn "tree rollout not patched: git apply patches/steerf_tree_rollout.patch"
fi

say "next"
cat <<'EOT'
  1. bash run/setup_env.sh                      # vllm, ray, flash-attn, pins
  2. REPO=<hub repo> bash run/migrate_pod.sh --import   # the MTP heads
  3. docs/RUNBOOK.md                            # what to launch, and where
EOT
exit "${rc}"
