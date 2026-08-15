#!/usr/bin/env bash
# Fetch STEER, pin it, and apply the STEER-F patches.
#
#   bash scripts/setup_steer.sh              # clone + patch
#   bash scripts/setup_steer.sh --check      # verify patches apply, change nothing
#   bash scripts/setup_steer.sh --revert     # undo the patches
#
# The STEER tree is never edited by hand — patches/ is the only channel, which
# is what keeps `git -C third_party/STEER diff` a readable statement of every
# change STEER-F makes to verl (plan §7).
set -euo pipefail

STEERF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STEER_ROOT="${STEER_ROOT:-${STEERF_ROOT}/third_party/STEER}"
STEER_REPO="https://github.com/zz-haooo/STEER.git"
# Pinned: the commit docs/steer_code_map.md was written against. verl 0.4.1.dev.
STEER_COMMIT="${STEER_COMMIT:-08add1cc27f4d32a78a6c0d6cb857aa52b8f2a55}"

MODE="apply"
case "${1:-}" in
  --check)  MODE="check" ;;
  --revert) MODE="revert" ;;
  "")       ;;
  *) echo "usage: $0 [--check|--revert]"; exit 1 ;;
esac

if [ ! -d "${STEER_ROOT}/.git" ]; then
  if [ "$MODE" != "apply" ]; then
    echo "no STEER checkout at ${STEER_ROOT}; run without flags first"
    exit 1
  fi
  echo "[setup] cloning STEER into ${STEER_ROOT}"
  mkdir -p "$(dirname "${STEER_ROOT}")"
  git clone "${STEER_REPO}" "${STEER_ROOT}"
fi

cd "${STEER_ROOT}"
CURRENT="$(git rev-parse HEAD)"
if [ "${CURRENT}" != "${STEER_COMMIT}" ] && [ "$MODE" = "apply" ]; then
  echo "[setup] pinning to ${STEER_COMMIT} (was ${CURRENT})"
  git fetch --depth 50 origin "${STEER_COMMIT}" 2>/dev/null || git fetch origin
  git checkout --quiet "${STEER_COMMIT}"
fi

PATCH_ARGS="-p1"
[ "$MODE" = "check" ]  && PATCH_ARGS="$PATCH_ARGS --dry-run"
[ "$MODE" = "revert" ] && PATCH_ARGS="$PATCH_ARGS -R"

failed=0
for p in "${STEERF_ROOT}"/patches/*.patch; do
  name="$(basename "$p")"
  if [ "$MODE" = "apply" ] && patch -p1 -R --dry-run --force -s -i "$p" >/dev/null 2>&1; then
    echo "[setup] $name: already applied, skipping"
    continue
  fi
  if patch $PATCH_ARGS -s -i "$p"; then
    echo "[setup] $name: ${MODE} ok"
  else
    echo "[setup] $name: ${MODE} FAILED"
    failed=1
  fi
done

find "${STEER_ROOT}" -name '*.orig' -o -name '*.rej' | xargs -r rm -f

if [ "$failed" -ne 0 ]; then
  echo
  echo "[setup] A patch did not apply. The pinned commit is ${STEER_COMMIT};"
  echo "        if you moved off it, regenerate the patches rather than forcing them —"
  echo "        docs/steer_code_map.md records exact line numbers that will have shifted."
  exit 1
fi

if [ "$MODE" = "apply" ]; then
  echo
  echo "[setup] done. Verify with:"
  echo "        STEER_REPO=${STEER_ROOT} python3 -m pytest tests/ -q"
  echo "        git -C ${STEER_ROOT} diff --stat"
fi
