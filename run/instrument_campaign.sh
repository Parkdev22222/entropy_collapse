#!/usr/bin/env bash
# Turn on the instrumentation the 5-seed campaign needs, before any GPU time is spent.
#
#   bash run/instrument_campaign.sh --check     # report only, change nothing
#   bash run/instrument_campaign.sh --apply     # patch, with .bak backups
#
# Three changes, all additive and none of which touch a gradient:
#
#   1. run_steerf.sh gains  ++trainer.validation_data_dir=${VAL_DATA_DIR}
#      Without it verl throws away per-problem validation scores (ray_trainer.py
#      only dumps them when that key is set). With it every validation step writes
#      {input, output, score} JSONL, which is what the paired across-problem error
#      bars need. AIME24 is 30 problems x 32 replicas = 960 rows per step.
#
#   2. run_steerf.sh stops hardcoding  ++trainer.save_best_only=False
#      It becomes ${SAVE_BEST_ONLY:-False}, so the campaign can ask for the
#      best-by-AIME24 checkpoint without editing the file on every pod.
#
#   3. verl's ray_trainer.py uncomments seq_entropy_agg, which logs
#      actor/seq_entropy -- the real trajectory entropy rather than
#      entropy-divided-by-length.
#
#   4. steer_f/monitors.py gains four keys on the a_h != 0 support:
#      support_entropy / nonsupport_entropy / support_entropy_gap /
#      support_frac. The existing branch_entropy keys are NOT touched.
#      Why the new keys are needed: branch_token_entropy splits on the top
#      DECILE of A_H (top_frac=0.1, never overridden at the call site), while
#      the positions the correction reaches are branch_corr_frac ~= .012. A_H
#      is exactly zero wherever a rollout is its own only sibling, so the rest
#      of that decile is the tie-break among zeros -- measured, 94% of the
#      bucket, taken as a contiguous index slab rather than a random sample.
#      `a_h != 0` is the predicate omega_tilde.branch_weight_correction itself
#      uses, so the new split covers what the method touches and nothing else.
#      Why branch_entropy is left alone: seeds already finished logged it, and
#      redefining a key mid-campaign makes those runs incomparable.
#
# Every edit is idempotent: running twice changes nothing the second time.

set -uo pipefail

MODE="${1:-}"
case "${MODE}" in
    --check|--apply) ;;
    *) echo "usage: bash run/instrument_campaign.sh --check|--apply" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# run_uniform_ablation.sh delegates to run_steerf.sh, so patching these two
# covers all five arms.
LAUNCHERS=("${ROOT}/run/run_steerf.sh" "${ROOT}/run/run_grpo.sh")
rc=0
declare -a TODO=()

say()  { printf '%s\n' "$*"; }
ok()   { printf '  [ok]     %s\n' "$*"; }
need() { printf "  [needed] %s\n" "$2"; TODO+=("$1"); rc=1; }
bad()  { printf '  [ERROR]  %s\n' "$*"; rc=2; }

# ---------------------------------------------------------------- locate verl
RT="$(python3 - <<'PY' 2>/dev/null
try:
    import verl, os
    p = os.path.join(os.path.dirname(verl.__file__), "trainer", "ppo", "ray_trainer.py")
    print(p if os.path.exists(p) else "")
except Exception:
    print("")
PY
)"

# steer_f is taken from the donor branch on a pod (run/bootstrap_pod.sh:35-51):
# this branch carries a steer_f of a different lineage, and patching that one
# would change nothing the trainer runs. Match on a donor-only marker rather
# than on the path, so a wrong-lineage checkout is skipped and said out loud.
MON="$(python3 - <<'MONPY' 2>/dev/null
# find_spec locates the package without executing monitors.py, which on a
# machine missing torch or a sibling module would otherwise raise and hide a
# file that is sitting right there.
import importlib.util, os
try:
    sp = importlib.util.find_spec("steer_f")
    d = list(sp.submodule_search_locations)[0] if sp else ""
    f = os.path.join(d, "monitors.py") if d else ""
    print(f if f and os.path.exists(f) else "")
except Exception:
    print("")
MONPY
)"

say "root : ${ROOT}"
say "verl : ${RT:-<not importable>}"
say "steer_f monitors : ${MON:-<not importable>}"
say ""

# ------------------------------------------------------------------ 1 + 2
for RS in "${LAUNCHERS[@]}"; do
    say "$(basename "${RS}")"
    if [ ! -f "${RS}" ]; then
        bad "not found -- run this on the pod checkout, not a bare clone"
        continue
    fi
    if grep -q 'validation_data_dir' "${RS}"; then
        ok "validation_data_dir already wired"
    else
        need "VAL_DATA_DIR:${RS}"  "add ++trainer.validation_data_dir"
    fi

    if grep -qE '\+\+trainer\.save_best_only=\$\{SAVE_BEST_ONLY' "${RS}"; then
        ok "save_best_only is env-driven"
    elif grep -qE '\+\+trainer\.save_best_only=(True|False)' "${RS}"; then
        need "SAVE_BEST_ONLY:${RS}"  "save_best_only is hardcoded; make it \${SAVE_BEST_ONLY:-False}"
    else
        bad "no save_best_only line found -- inspect the file by hand"
    fi
    say ""
done

# ---------------------------------------------------------------------- 3
say "verl ray_trainer.py"
if [ -z "${RT}" ]; then
    printf '  [skip]   verl is not importable here; seq_entropy left unchecked\n'
    printf '           (expected on a machine without the training env -- re-run on the pod)\n'
    [ "${rc}" -eq 0 ] && rc=1
elif grep -qE '^\s*"actor/seq_entropy"' "${RT}"; then
    ok "seq_entropy already logged"
elif grep -qE '^\s*#\s*seq_entropy_agg\s*=' "${RT}"; then
    need "SEQ_ENTROPY"  "seq_entropy_agg is commented out"
else
    bad "neither an active nor a commented seq_entropy_agg line found"
fi
say ""

# ---------------------------------------------------------------------- 4
say "steer_f monitors.py"
if [ -z "${MON}" ]; then
    printf '  [skip]   steer_f is not importable here; support_entropy left unchecked\n'
    [ "${rc}" -eq 0 ] && rc=1
elif ! grep -q '_top_k_selection' "${MON}"; then
    printf '  [skip]   this steer_f is the other lineage (no _top_k_selection).\n'
    printf '           Run this on a pod, after run/bootstrap_pod.sh has put the\n'
    printf "           donor's steer_f in place.\n"
    [ "${rc}" -eq 0 ] && rc=1
elif grep -q 'steerf/support_entropy' "${MON}"; then
    ok "support-restricted entropy already logged"
elif grep -q 'steerf/a_h_std' "${MON}"; then
    need "SUPPORT_ENTROPY"  "branch_token_entropy splits on the top decile only"
else
    bad "branch_token_entropy does not look as expected -- inspect by hand"
fi
say ""

if [ "${rc}" -eq 2 ]; then
    say "Refusing to continue: something is not where this script expects it."
    exit 2
fi

if [ "${#TODO[@]}" -eq 0 ]; then
    say "Nothing to do -- instrumentation is already in place."
    exit 0
fi

if [ "${MODE}" = "--check" ]; then
    say "${#TODO[@]} change(s) pending. Re-run with --apply."
    exit 1
fi

# ------------------------------------------------------------------- apply
stamp="$(date +%Y%m%d_%H%M%S)"
for what in "${TODO[@]}"; do
    RS="${what#*:}"
    case "${what%%:*}" in
    VAL_DATA_DIR)
        [ -f "${RS}.bak.${stamp}" ] || cp -p "${RS}" "${RS}.bak.${stamp}"
        # insert immediately after the rollout_data_dir line, keeping its indent
        python3 - "${RS}" <<'PYEOF'
import re, sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
# Anchor on the real save_best_only argument -- never a comment -- so the flag
# lands inside the command (run_steerf.sh) or inside ARGS=( ) (run_grpo.sh).
m = None
for cand in re.finditer(r'^([ \t]*)((?:\$\{[A-Z_]+:\+)?\+\+trainer\.save_best_only=.*)$', s, re.M):
    if not cand.group(2).lstrip().startswith('#'):
        m = cand
        break
assert m, "no save_best_only argument line found"
ind, line = m.group(1), m.group(2)
cont = ' \\' if line.rstrip().endswith('\\') else ''
add = ind + '${VAL_DATA_DIR:+++trainer.validation_data_dir=${VAL_DATA_DIR}}' + cont
s = s[:m.start()] + add + "\n" + s[m.start():]
open(p, "w", encoding="utf-8").write(s)
PYEOF
        ok "added ++trainer.validation_data_dir (backup ${RS}.bak.${stamp})"
        ;;
    SAVE_BEST_ONLY)
        [ -f "${RS}.bak.${stamp}" ] || cp -p "${RS}" "${RS}.bak.${stamp}"
        sed -i -E 's/(\+\+trainer\.save_best_only=)(True|False)/\1${SAVE_BEST_ONLY:-False}/' "${RS}"
        ok "save_best_only is now \${SAVE_BEST_ONLY:-False}"
        ;;
    SUPPORT_ENTROPY)
        [ -f "${MON}.bak.${stamp}" ] || cp -p "${MON}" "${MON}.bak.${stamp}"
        python3 - "${MON}" <<'MONEOF'
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
assert "steerf/support_entropy" not in s, "already applied"
anchor = '    other = float(ent_v[~is_branch].mean()) if int((~is_branch).sum()) > 0 else float("nan")'
assert anchor in s, "the branch/non-branch split is not where this patch expects it"
block = anchor + """

    # The split above is a fixed top DECILE of A_H. The positions the
    # correction actually reaches are `a_h != 0` -- the same predicate
    # omega_tilde.branch_weight_correction applies -- and that is about an
    # eightieth of the tokens, an eighth of the decile. A_H is exactly zero
    # wherever a rollout has become its own only sibling, so the remainder of
    # the decile is torch.topk's tie-break among those zeros, which takes a
    # contiguous index slab rather than a random sample. Anything read off the
    # decile keys is therefore mostly positional. These four are not.
    sup = a_v != 0
    n_sup = int(sup.sum())
    sup_e = float(ent_v[sup].mean()) if n_sup > 0 else float("nan")
    nsup_e = float(ent_v[~sup].mean()) if int((~sup).sum()) > 0 else float("nan")"""
s = s.replace(anchor, block, 1)
key = '        "steerf/a_h_std": float(a_v.std(unbiased=False)),'
assert key in s, "the a_h_std key is not where this patch expects it"
s = s.replace(key, key + """
        "steerf/support_entropy": sup_e,
        "steerf/nonsupport_entropy": nsup_e,
        "steerf/support_entropy_gap": sup_e - nsup_e,
        "steerf/support_frac": n_sup / n,""", 1)
open(p, "w", encoding="utf-8").write(s)
MONEOF
        ok "added the support-restricted entropy keys (backup ${MON}.bak.${stamp})"
        ;;
    SEQ_ENTROPY)
        [ -f "${RT}.bak.${stamp}" ] || cp -p "${RT}" "${RT}.bak.${stamp}"
        sed -i -E 's/^(\s*)#\s*(seq_entropy_agg\s*=)/\1\2/; s/^(\s*)#\s*("actor\/seq_entropy")/\1\2/' "${RT}"
        ok "uncommented seq_entropy_agg (backup ${RT}.bak.${stamp})"
        ;;
    esac
done

say ""
say "verifying syntax"
for RS in "${LAUNCHERS[@]}"; do
    [ -f "${RS}" ] || continue
    if bash -n "${RS}"; then ok "$(basename "${RS}") parses"
    else bad "$(basename "${RS}") no longer parses -- restore ${RS}.bak.${stamp}"; fi
done
if [ -n "${RT}" ]; then
    if python3 -m py_compile "${RT}" 2>/dev/null; then ok "ray_trainer.py compiles"
    else bad "ray_trainer.py no longer compiles -- restore ${RT}.bak.${stamp}"; fi
fi
if [ -n "${MON}" ]; then
    if python3 -m py_compile "${MON}" 2>/dev/null; then ok "monitors.py compiles"
    else bad "monitors.py no longer compiles -- restore ${MON}.bak.${stamp}"; fi
fi

say ""
say "re-checking"
exec bash "${BASH_SOURCE[0]}" --check
