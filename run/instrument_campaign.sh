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
# Every edit is idempotent: running twice changes nothing the second time.

set -uo pipefail

MODE="${1:-}"
case "${MODE}" in
    --check|--apply) ;;
    *) echo "usage: bash run/instrument_campaign.sh --check|--apply" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RS="${ROOT}/run/run_steerf.sh"
rc=0
declare -a TODO=()

say()  { printf '%s\n' "$*"; }
ok()   { printf '  [ok]     %s\n' "$*"; }
need() { printf '  [needed] %s\n' "$*"; TODO+=("$1"); rc=1; }
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

say "root : ${ROOT}"
say "verl : ${RT:-<not importable>}"
say ""

# ------------------------------------------------------------------ 1 + 2
say "run/run_steerf.sh"
if [ ! -f "${RS}" ]; then
    bad "not found -- run this on the pod checkout, not a bare clone"
else
    if grep -q 'validation_data_dir' "${RS}"; then
        ok "validation_data_dir already wired"
    else
        need "VAL_DATA_DIR"  "add ++trainer.validation_data_dir"
    fi

    if grep -qE '\+\+trainer\.save_best_only=\$\{SAVE_BEST_ONLY' "${RS}"; then
        ok "save_best_only is env-driven"
    elif grep -qE '\+\+trainer\.save_best_only=(True|False)' "${RS}"; then
        need "SAVE_BEST_ONLY"  "save_best_only is hardcoded; make it \${SAVE_BEST_ONLY:-False}"
    else
        bad "no save_best_only line found -- inspect the file by hand"
    fi
fi
say ""

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
    case "${what}" in
    VAL_DATA_DIR)
        cp -p "${RS}" "${RS}.bak.${stamp}"
        # insert immediately after the rollout_data_dir line, keeping its indent
        python3 - "${RS}" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
m = re.search(r'^([ \t]*)(.*rollout_data_dir=.*)$', s, re.M)
assert m, "rollout_data_dir line not found"
ind = m.group(1)
add = (ind + '${VAL_DATA_DIR:+++trainer.validation_data_dir=${VAL_DATA_DIR}} \\')
s = s[:m.end()] + "\n" + add + s[m.end():]
open(p, "w", encoding="utf-8").write(s)
PY
        ok "added ++trainer.validation_data_dir (backup ${RS}.bak.${stamp})"
        ;;
    SAVE_BEST_ONLY)
        [ -f "${RS}.bak.${stamp}" ] || cp -p "${RS}" "${RS}.bak.${stamp}"
        sed -i -E 's/(\+\+trainer\.save_best_only=)(True|False)/\1${SAVE_BEST_ONLY:-False}/' "${RS}"
        ok "save_best_only is now \${SAVE_BEST_ONLY:-False}"
        ;;
    SEQ_ENTROPY)
        cp -p "${RT}" "${RT}.bak.${stamp}"
        sed -i -E 's/^(\s*)#\s*(seq_entropy_agg\s*=)/\1\2/; s/^(\s*)#\s*("actor\/seq_entropy")/\1\2/' "${RT}"
        ok "uncommented seq_entropy_agg (backup ${RT}.bak.${stamp})"
        ;;
    esac
done

say ""
say "verifying syntax"
if bash -n "${RS}"; then ok "run_steerf.sh parses"; else bad "run_steerf.sh no longer parses -- restore ${RS}.bak.${stamp}"; fi
if [ -n "${RT}" ]; then
    if python3 -m py_compile "${RT}" 2>/dev/null; then ok "ray_trainer.py compiles"
    else bad "ray_trainer.py no longer compiles -- restore ${RT}.bak.${stamp}"; fi
fi

say ""
say "re-checking"
exec bash "${BASH_SOURCE[0]}" --check
