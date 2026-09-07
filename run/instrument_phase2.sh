#!/usr/bin/env bash
# Two more instrumentation patches, needed by phase 2 (evaluation + follow-up
# ablations) and by nothing else.
#
#   bash run/instrument_phase2.sh --check     # report only, change nothing
#   bash run/instrument_phase2.sh --apply     # patch, with .bak backups
#
# WHY THIS IS NOT PART OF instrument_campaign.sh
#   run_campaign.sh refuses to start unless `instrument_campaign.sh --check`
#   passes. The 20-run campaign is already armed; adding items to that check
#   would make a restart of the running campaign refuse for reasons that have
#   nothing to do with training. So the phase-2 items live here, and
#   run_phase2.sh applies them after the campaign has released the GPUs.
#
# Two changes, both additive, neither touching a gradient:
#
#   1. run/eval_steerf.sh gains
#        ${VAL_DATA_DIR:+++trainer.validation_data_dir=${VAL_DATA_DIR}/${tag}}
#      Without it verl throws away the per-problem validation scores, and the
#      six-benchmark table has means with no paired error bars. The ${tag}
#      suffix is load-bearing: in val_only mode global_steps is 0
#      (ray_trainer.py:1078 then the early return at :1091), so _dump_generations
#      writes "0.jsonl" -- the avg@1 pass would silently overwrite the avg@32
#      pass if both passes shared a directory.
#
#   2. run/run_uniform_ablation.sh forwards "$@" to run_steerf.sh.
#      The follow-up ablations are tree runs that differ from the signed arm by
#      one hydra override each (clip_ratio_high=5, adv_estimator=rloo, ...).
#      Without the pass-through there is no way to reach run_steerf.sh's own
#      trailing-override slot through the wrapper, and every follow-up would
#      have to re-declare the tree arm by hand -- which is how two arms drift
#      apart.
#
# Every edit is idempotent: running twice changes nothing the second time.

set -uo pipefail

MODE="${1:-}"
case "${MODE}" in
    --check|--apply) ;;
    *) echo "usage: bash run/instrument_phase2.sh --check|--apply" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_SH="${ROOT}/run/eval_steerf.sh"
ABL_SH="${ROOT}/run/run_uniform_ablation.sh"
rc=0
declare -a TODO=()

say()  { printf '%s\n' "$*"; }
ok()   { printf '  [ok]     %s\n' "$*"; }
need() { printf '  [needed] %s\n' "$2"; TODO+=("$1"); rc=1; }
bad()  { printf '  [ERROR]  %s\n' "$*"; rc=2; }

say "root : ${ROOT}"
say ""

# -------------------------------------------------------------------- 1
say "$(basename "${EVAL_SH}")"
if [ ! -f "${EVAL_SH}" ]; then
    bad "not found -- run this on the pod checkout, not a bare clone"
elif grep -q 'validation_data_dir' "${EVAL_SH}"; then
    ok "validation_data_dir already wired"
elif grep -qE '^\s*trainer\.val_only=True \\$' "${EVAL_SH}"; then
    need "EVAL_VAL_DATA_DIR" "add ++trainer.validation_data_dir to run_eval()"
else
    bad "no 'trainer.val_only=True \\' anchor line -- inspect the file by hand"
fi
say ""

# -------------------------------------------------------------------- 2
say "$(basename "${ABL_SH}")"
if [ ! -f "${ABL_SH}" ]; then
    bad "not found -- run this on the pod checkout, not a bare clone"
elif grep -qF 'run_steerf.sh" "${ARGS[@]}" "$@"' "${ABL_SH}"; then
    ok 'trailing overrides ("$@") already forwarded'
elif grep -qF 'run_steerf.sh" "${ARGS[@]}" >' "${ABL_SH}"; then
    need "ABL_PASSTHROUGH" 'forward "$@" to run_steerf.sh'
else
    bad "no recognisable run_steerf.sh invocation -- inspect the file by hand"
fi
say ""

if [ "${rc}" -eq 2 ]; then
    say "Refusing to continue: something is not where this script expects it."
    exit 2
fi
if [ "${#TODO[@]}" -eq 0 ]; then
    say "Nothing to do -- phase-2 instrumentation is already in place."
    exit 0
fi
if [ "${MODE}" = "--check" ]; then
    say "${#TODO[@]} change(s) pending. Re-run with --apply."
    exit 1
fi

# --------------------------------------------------------------- apply
stamp="$(date +%Y%m%d_%H%M%S)"
for what in "${TODO[@]}"; do
    case "${what}" in
    EVAL_VAL_DATA_DIR)
        [ -f "${EVAL_SH}.bak.${stamp}" ] || cp -p "${EVAL_SH}" "${EVAL_SH}.bak.${stamp}"
        python3 - "${EVAL_SH}" <<'PYEOF'
import re, sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
m = re.search(r'^([ \t]*)trainer\.val_only=True \\$', s, re.M)
assert m, "no 'trainer.val_only=True \\' anchor"
ind = m.group(1)
add = ind + '${VAL_DATA_DIR:+++trainer.validation_data_dir=${VAL_DATA_DIR}/${tag}} \\\n'
s = s[:m.start()] + add + s[m.start():]
open(p, "w", encoding="utf-8").write(s)
PYEOF
        ok "added ++trainer.validation_data_dir (backup ${EVAL_SH}.bak.${stamp})"
        ;;
    ABL_PASSTHROUGH)
        [ -f "${ABL_SH}.bak.${stamp}" ] || cp -p "${ABL_SH}" "${ABL_SH}.bak.${stamp}"
        python3 - "${ABL_SH}" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
old = 'run_steerf.sh" "${ARGS[@]}" >'
new = 'run_steerf.sh" "${ARGS[@]}" "$@" >'
assert s.count(old) == 1, f"expected exactly one invocation, found {s.count(old)}"
s = s.replace(old, new)
# the DRY_RUN preview should show them too, when there are any
old_dry = """    printf '      %s \\\\\\n' "${ARGS[@]}"\n"""
new_dry = """    printf '      %s \\\\\\n' "${ARGS[@]}" "$@"\n"""
if old_dry in s:
    s = s.replace(old_dry, new_dry, 1)
open(p, "w", encoding="utf-8").write(s)
PYEOF
        ok "run_uniform_ablation.sh now forwards \"\$@\" (backup ${ABL_SH}.bak.${stamp})"
        ;;
    esac
done

say ""
say "verifying syntax"
for f in "${EVAL_SH}" "${ABL_SH}"; do
    [ -f "${f}" ] || continue
    if bash -n "${f}"; then ok "$(basename "${f}") parses"
    else bad "$(basename "${f}") no longer parses -- restore ${f}.bak.${stamp}"; fi
done

say ""
say "re-checking"
exec bash "${BASH_SOURCE[0]}" --check
