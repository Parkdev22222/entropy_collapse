#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Move this workspace to another pod, with the move verified rather than hoped.
#
#   OLD pod:   REPO=DSDSh/steer-f_2 bash run/migrate_pod.sh --export
#   NEW pod:   REPO=DSDSh/steer-f_2 bash run/migrate_pod.sh --import
#   either:    bash run/migrate_pod.sh --check      # inventory only, no writes
#
#   Splitting work across two pods that BOTH keep running is not a migration:
#              REPO=DSDSh/steer-f_2 bash run/migrate_pod.sh --share
#   uploads the artefacts and the manifest and nothing else. It leaves the
#   trained checkpoints alone, and tolerates a dirty tree, because it does not
#   end with anyone deleting the source pod.
#
# WHAT ACTUALLY HAS TO MOVE
#   Three classes, and only one of them is a real problem.
#
#   In git already -- code, run/, scripts/, datasets/*.parquet, logs/,
#     tensorboard_log/. `--export` refuses if any of it is uncommitted, because
#     "I'll commit it later" is how a pod gets deleted with the only copy on it.
#
#   Rebuildable -- the pip environment and the HuggingFace model cache. Do NOT
#     copy these. run/setup_env.sh reinstalls the first against the new pod's
#     CUDA build, and the second re-downloads. Copying a pip tree between
#     images is how flash-attn ends up compiled against the wrong torch.
#
#   Neither, and this is the class that bites -- gitignored artefacts that
#     cost GPU-hours to recreate:
#       checkpoints/mtp_heads_<tag>-paper.pt        Phase 0+1. Without it no
#       checkpoints/mtp_calibration_<tag>-paper.json  tree arm starts at all --
#                                               run_uniform_ablation.sh refuses.
#       rollout_data/warmup/<tag>/rollouts.jsonl  the locality and recall
#                                               measurements need it.
#       checkpoints/STEER-F/<run>/...           the trained models.
#     These go to the Hub, with sizes and sha256 recorded in a manifest that
#     `--import` checks. A silently truncated .pt would otherwise surface as a
#     strange training curve three days later.
#
# The manifest is the point. Anyone can copy files; the question a migration
# has to answer is "did everything arrive intact", and that needs a list made
# before the move.
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}" || { echo "FATAL: cannot cd to ${ROOT}" >&2; exit 1; }

MODE="${1:-}"
case "${MODE}" in
    --export|--share|--import|--check) ;;
    *) echo "usage: bash run/migrate_pod.sh --export|--share|--import|--check" >&2; exit 2 ;;
esac

# shellcheck source=run/_arms.sh
. "${ROOT}/run/_arms.sh"      # is_busy, train_log_done, run_name_for, MODEL_TAG

REPO=${REPO:-}
MODEL_TAG=${MODEL_TAG:-Qwen2.5-Math-1.5B}
export MODEL_TAG          # the manifest writer reads it from the environment
PREFIX=${PREFIX:-migration}          # path inside the Hub repo
MANIFEST="${ROOT}/.migration_manifest.json"
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"

HF_CLI=${HF_CLI:-$(command -v hf || command -v huggingface-cli || true)}

say () { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok ()  { printf '  \033[32mOK\033[0m    %s\n' "$*"; }
warn () { printf '  \033[33mWARN\033[0m  %s\n' "$*"; }
bad () { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; }

# The artefacts that are not in git and cost GPU time to rebuild, split by who
# needs them.
#
# Which pair is required is decided by the launcher, not by run_steerf.sh's
# generic default. run_uniform_ablation.sh:95-96 HARDCODES the -paper pair and
# refuses at :149 when the heads file is absent -- and that check reads only
# STEERF_FORECAST, so it fires at lambda=0 too (lam0-tree included). That
# launcher runs signed, uniform, permuted and six of the ten follow-ups: three
# of the campaign's five arms and most of the ablations.
#
# run_steerf.sh's own default has no -paper suffix, but it only opens the file
# when lambda != 0 (:76), and every arm it launches here runs at lambda=0. The
# _0905 STEER log printing the un-suffixed path is therefore not evidence about
# which pair matters: that run never opened one.
required_paths () {     # no tree arm starts without these
    printf '%s\n' \
        "checkpoints/mtp_heads_${MODEL_TAG}-paper.pt" \
        "checkpoints/mtp_calibration_${MODEL_TAG}-paper.json"
}
optional_paths () {     # nothing in the current queues opens these; absence is not fatal
    printf '%s\n' \
        "checkpoints/mtp_heads_${MODEL_TAG}.pt" \
        "checkpoints/mtp_calibration_${MODEL_TAG}.json" \
        "checkpoints/mtp_heads_control_${MODEL_TAG}.pt" \
        "rollout_data/warmup/${MODEL_TAG}/rollouts.jsonl" \
        "rollout_data/warmup/${MODEL_TAG}-paper/rollouts.jsonl"
}
critical_paths () { required_paths; optional_paths; }

human () { du -sh "$1" 2>/dev/null | cut -f1; }

# ------------------------------------------------------------------ check
inventory () {
    local missing=0
    say "1. git"
    if [ "${BRANCH}" = "?" ]; then
        bad "not a git checkout"
        missing=1
    else
        ok "branch ${BRANCH}"
        # git state is blocking for --export and --check, informational for
        # --share: sharing does not delete the source pod, so uncommitted work
        # cannot be lost by it, and the pod that shares is usually mid-campaign
        # with local edits.
        local dirty; dirty="$(git status --porcelain | grep -v '^?? ' || true)"
        if [ -n "${dirty}" ]; then
            if [ "${MODE}" = "--share" ]; then
                warn "uncommitted tracked changes (not blocking a share):"
            else
                bad "uncommitted tracked changes:"; missing=1
            fi
            printf '%s\n' "${dirty}" | head -10 | sed 's/^/        /'
        else
            ok "no uncommitted tracked changes"
        fi
        local ahead; ahead="$(git log --oneline "@{u}..HEAD" 2>/dev/null | wc -l || echo '?')"
        if [ "${ahead}" = "0" ]; then ok "pushed to origin"
        elif [ "${MODE}" = "--share" ]; then
            warn "${ahead} commit(s) not pushed (not blocking a share)"
        else
            bad "${ahead} commit(s) not pushed -- push before you delete the pod"; missing=1
        fi
        local untracked; untracked="$(git status --porcelain | grep -c '^?? ' || true)"
        [ "${untracked}" != "0" ] && warn "${untracked} untracked path(s); check none of them is a result"
    fi

    say "2. artefacts that are NOT in git -- required by every tree arm"
    local p
    while IFS= read -r p; do
        if [ -e "${p}" ]; then ok "$(printf '%-58s %s' "${p}" "$(human "${p}")")"
        else bad "MISSING  ${p}"; missing=1; fi
    done < <(required_paths)

    say "2b. artefacts nothing in the current queues opens -- absence is not fatal"
    local n_opt=0
    while IFS= read -r p; do
        if [ -e "${p}" ]; then ok "$(printf '%-58s %s' "${p}" "$(human "${p}")")"; n_opt=$((n_opt + 1))
        else warn "absent   ${p}"; fi
    done < <(optional_paths)
    [ "${n_opt}" = "0" ] && warn "none present: every training arm can still run; STAGES=measure has nothing to read"

    say "3. trained checkpoints"
    local n=0
    for d in checkpoints/STEER-F/*/global_step_*/actor/huggingface; do
        [ -d "${d}" ] || continue
        n=$((n + 1))
        printf '  %-72s %s\n' "${d%/actor/huggingface}" "$(human "${d}")"
    done
    [ "${n}" = "0" ] && warn "none on disk (already uploaded and deleted?)" \
                     || ok "${n} checkpoint(s) on disk"

    say "4. rebuilt on the new pod, do NOT copy"
    printf '  pip env       -> bash run/setup_env.sh\n'
    printf '  HF model cache-> re-downloads on first run (~3 GB)\n'
    return "${missing}"
}

if [ "${MODE}" = "--check" ]; then
    inventory; rc=$?
    say "verdict"
    [ "${rc}" = "0" ] && ok "ready to export" || bad "fix the FAIL lines first"
    exit "${rc}"
fi

: "${REPO:?set REPO to the Hub repo id, e.g. REPO=DSDSh/steer-f_2}"
[ -n "${HF_CLI}" ] || { echo "FATAL: neither 'hf' nor 'huggingface-cli' on PATH." >&2
                        echo "       pip install \"huggingface_hub>=0.34,<1.0\"" >&2; exit 1; }

# --------------------------------------------------------- export / share
# Both put the gitignored artefacts and a manifest on the Hub; only --export
# also uploads the trained checkpoints. --share exists because splitting work
# across two LIVE pods is not a migration: the second box needs the MTP heads
# and nothing else, and uploading checkpoints there is at best slow and at
# worst dangerous (see the live guard below).
if [ "${MODE}" = "--export" ] || [ "${MODE}" = "--share" ]; then
    inventory || { say "refusing"; bad "commit and push first, or recreate the missing artefacts"; exit 1; }

    say "5. manifest"
    # The paths come in as argv rather than being rebuilt here, so the manifest
    # can never disagree with what critical_paths() actually uploads.
    python3 - "${MANIFEST}" "${BRANCH}" $(critical_paths) > /dev/null <<'PY'
import hashlib, json, os, subprocess, sys, time
manifest, branch = sys.argv[1], sys.argv[2]
tag = os.environ.get("MODEL_TAG", "Qwen2.5-Math-1.5B")
paths = sys.argv[3:]
entries = {}
for p in paths:
    if not os.path.isfile(p):
        continue
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    entries[p] = {"bytes": os.path.getsize(p), "sha256": h.hexdigest()}
runs = sorted({d.split("/")[2] for d in
               subprocess.run(["bash", "-c",
                               "ls -d checkpoints/STEER-F/*/global_step_*/actor/huggingface 2>/dev/null"],
                              capture_output=True, text=True).stdout.split()})
json.dump({"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "git_branch": branch,
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"],
                                        capture_output=True, text=True).stdout.strip(),
           "model_tag": tag,
           "artifacts": entries,
           "checkpoint_runs_on_disk": runs},
          open(manifest, "w"), indent=2)
print(len(entries))
PY
    ok "wrote $(basename "${MANIFEST}") ($(python3 -c "import json;print(len(json.load(open('${MANIFEST}'))['artifacts']))") artefact(s))"

    say "6. upload the artefacts to ${REPO}/${PREFIX}"
    while IFS= read -r p; do
        [ -f "${p}" ] || continue
        echo "  -> ${p}"
        "${HF_CLI}" upload "${REPO}" "${p}" "${PREFIX}/${p}" --repo-type model \
            --commit-message "migration: ${p}" >/dev/null || { bad "upload failed: ${p}"; exit 1; }
    done < <(critical_paths)
    "${HF_CLI}" upload "${REPO}" "${MANIFEST}" "${PREFIX}/manifest.json" --repo-type model \
        --commit-message "migration manifest" >/dev/null || { bad "manifest upload failed"; exit 1; }
    ok "artefacts and manifest uploaded"

    say "7. trained checkpoints"
    if [ "${MODE}" = "--share" ]; then
        ok "--share leaves them alone (the other pod needs the heads, not the weights)"
        ok "to move the weights too, use --export on a pod that is not training"
    else
    # hf_backup.sh refuses a run whose NAME contains LIVE_TAG (_0905 by default),
    # which was the naming the recovery chain used. The campaign's runs are
    # steer-f-<tag>-s<N>-tree-rollout with no suffix at all, so that guard does
    # not fire for them: a --export on a training box would upload the directory
    # verl is writing and then "verify" it byte for byte. Judge by state, not by
    # name -- a run is safe to upload only once its log has reached its final
    # step and nothing is training right now.
    training_now=0
    if is_busy; then
        training_now=1
        warn "training is running -- only runs whose log reached the final step will be uploaded"
    fi
    n_runs=0; n_skip=0
    for d in checkpoints/STEER-F/*/; do
        [ -d "${d}" ] || continue
        run="$(basename "${d}")"
        ls -d "${d}"global_step_*/actor/huggingface >/dev/null 2>&1 || continue
        if [ "${training_now}" = "1" ] \
           && ! train_log_done "${ROOT}/logs/experiments" "${run}" "${STEPS:-110}"; then
            warn "skip ${run} -- still training (log has not reached step ${STEPS:-110})"
            n_skip=$((n_skip + 1))
            continue
        fi
        n_runs=$((n_runs + 1))
        echo "  -> ${run}"
        REPO="${REPO}" bash run/hf_backup.sh "${run}" || warn "backup failed for ${run}"
    done
    [ "${n_runs}" = "0" ] && ok "nothing on disk to upload" \
                             || ok "${n_runs} run(s) uploaded (local copies kept -- delete the pod, not the files)"
    [ "${n_skip}" != "0" ] && warn "${n_skip} run(s) skipped as in-flight; re-run --export once they finish"
    fi

    say "done"
    cat <<EOT
  On the other pod:

    git clone <this repo> /workspace/entropy_collapse
    cd /workspace/entropy_collapse
    git checkout ${BRANCH}
    bash run/setup_env.sh
    REPO=${REPO} bash run/migrate_pod.sh --import
    REPO=${REPO} bash run/run_paper.sh
EOT
    exit 0
fi

# ----------------------------------------------------------------- import
say "1. fetch the manifest"
tmp="$(mktemp -d)"
"${HF_CLI}" download "${REPO}" --repo-type model --include "${PREFIX}/*" \
    --local-dir "${tmp}" >/dev/null || { bad "download failed"; exit 1; }
src="${tmp}/${PREFIX}"
if [ ! -f "${src}/manifest.json" ]; then
    bad "no manifest at ${PREFIX}/manifest.json in ${REPO}"
    echo "  The manifest is written by the pod that HAS the artefacts. Run there:"
    echo "      REPO=${REPO} bash run/migrate_pod.sh --share    # artefacts only, safe while training"
    echo "      REPO=${REPO} bash run/migrate_pod.sh --export   # artefacts + trained checkpoints"
    echo "  then re-run this --import."
    exit 1
fi
ok "manifest fetched"

say "2. place and verify"
python3 - "${src}" "${ROOT}" <<'PY'
import hashlib, json, os, shutil, sys
src, root = sys.argv[1], sys.argv[2]
man = json.load(open(os.path.join(src, "manifest.json")))
print(f"  exported {man['created']} from {man['git_branch']} @ {man['git_commit'][:8]}")
bad = 0
for rel, meta in man["artifacts"].items():
    s, d = os.path.join(src, rel), os.path.join(root, rel)
    if not os.path.isfile(s):
        print(f"  \033[31mFAIL\033[0m  not in the download: {rel}"); bad += 1; continue
    os.makedirs(os.path.dirname(d), exist_ok=True)
    shutil.copy2(s, d)
    h = hashlib.sha256()
    with open(d, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    size_ok = os.path.getsize(d) == meta["bytes"]
    hash_ok = h.hexdigest() == meta["sha256"]
    if size_ok and hash_ok:
        print(f"  \033[32mOK\033[0m    {rel}  ({meta['bytes']} bytes, sha256 matches)")
    else:
        print(f"  \033[31mFAIL\033[0m  {rel}  size_ok={size_ok} sha256_ok={hash_ok}"); bad += 1
if man.get("checkpoint_runs_on_disk"):
    print("\n  trained checkpoints to pull when you need them "
          "(run/hf_backup.sh documents the restore):")
    for r in man["checkpoint_runs_on_disk"]:
        print(f"    {r}")
sys.exit(1 if bad else 0)
PY
rc=$?
rm -rf "${tmp}"

say "3. environment"
bash run/setup_env.sh CHECK_ONLY=1 >/dev/null 2>&1 && ok "setup_env passes" \
    || warn "run: bash run/setup_env.sh"
if bash -c '. run/_arms.sh; env_preflight "'"${ROOT}"'"' >/dev/null 2>&1; then
    ok "verl + steer_f import"
else
    bad "the training stack does not import -- run bash run/setup_env.sh"
    rc=1
fi

say "verdict"
if [ "${rc}" = "0" ]; then
    ok "migration verified. Next: REPO=${REPO} bash run/run_paper.sh"
else
    bad "something did not arrive intact -- do not delete the old pod yet"
fi
exit "${rc}"
