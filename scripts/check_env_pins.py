#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
# Licensed under the Apache License, Version 2.0
"""Check that the installed packages satisfy the pins the training stack declares.

    python3 scripts/check_env_pins.py            # 0 = clean, 1 = violations, 2 = cannot tell,
                                                 # 3 = only the accepted exceptions below
    python3 scripts/check_env_pins.py --quiet    # print only the fix command

WHY THIS EXISTS
    On 2026-09-08 an unpinned `pip install -U huggingface_hub` put 1.30.0 into
    the container. transformers 4.x does not merely declare
    ``huggingface-hub>=0.34.0,<1.0`` -- it enforces it at import time, from
    ``dependency_versions_check`` at the top of ``transformers/__init__.py``.
    So every ``import verl`` raised ImportError, every ``main_ppo`` died in the
    first seconds, and a queued training chain burned two arms before anyone
    looked. The traceback named the cause exactly; nothing was watching for it.

    Any check that hardcodes "huggingface_hub must be <1.0" goes stale the day
    transformers is upgraded. So this reads the requirement out of the INSTALLED
    metadata instead: whatever transformers currently declares is what we test
    against. New pins are picked up for free.

WHAT IT DOES NOT DO
    It does not check that a package imports -- a package can satisfy every
    version pin and still be broken (flash-attn compiled against a different
    torch, say). ``run/setup_env.sh`` section 6 covers that, and the queues call
    ``env_preflight`` in ``run/_arms.sh``, which imports verl for real.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import sys

try:
    from packaging.requirements import Requirement
except ImportError:  # packaging ships with pip; if it is gone, say so rather than crash
    print("[pins] packaging is not importable -- cannot check", file=sys.stderr)
    raise SystemExit(2)

# Distributions whose declared pins are load-bearing for training. Each one is
# checked only if it is installed, so this list can name optional packages.
# ray is here because of 2026-09-13: an unpinned `pip install wandb` moved
# opentelemetry 1.26 -> 1.44, ray's dashboard failed to import
# PrometheusMetricReader, ray.init() timed out, and every arm of a 20-run queue
# died with "The current node timed out during startup" -- a message that names
# neither opentelemetry nor pip.
WATCH = ["transformers", "tokenizers", "vllm", "ray", "datasets", "accelerate", "peft"]

GREEN, YELLOW, RED, OFF = "\033[32m", "\033[33m", "\033[31m", "\033[0m"

# Declared-only conflicts that are resolved on purpose, not by accident. They are
# reported, never "fixed": the fix line used to pin them back, and following it
# broke the stack. vllm 0.8.4 declares opentelemetry-*<1.27 and uses it only for
# optional tracing; ray 2.58's dashboard needs a much newer opentelemetry, and
# with 1.26 installed ray.init() times out ("The current node timed out during
# startup"). run/setup_env.sh records the same decision next to RAY_PIN.
ACCEPTED = {("vllm", "opentelemetry-"):
            "ray 2.58 needs a newer opentelemetry; vllm uses it only for optional tracing"}


def _accepted(holder: str, name: str) -> str | None:
    for (h, prefix), why in ACCEPTED.items():
        if holder.lower() == h and name.lower().startswith(prefix):
            return why
    return None


def _requirements(dist: str) -> list[Requirement]:
    """Declared requirements of `dist`, minus the ones behind an extra."""
    out = []
    for raw in md.requires(dist) or []:
        try:
            req = Requirement(raw)
        except Exception:
            continue
        # "; extra == 'dev'" requirements are not installed unless the extra was
        # asked for, so a version mismatch there means nothing.
        if req.marker is not None:
            if "extra" in str(req.marker):
                continue
            try:
                if not req.marker.evaluate():
                    continue
            except Exception:
                continue
        if req.specifier:
            out.append(req)
    return out


def _enforced_at_import(holder: str) -> set[str] | None:
    """Requirements `holder` re-checks when imported; empty set if it does not.

    transformers runs dependency_versions_check at the top of its __init__, so a
    violation of one of ITS pins really does stop `import verl` in its tracks --
    that is the 2026-09-08 huggingface_hub failure. Nothing else in WATCH does
    that: vllm declares its opentelemetry range and never looks again, so a
    violation there is an untested combination rather than a dead run. Saying
    both in the same words made this script cry wolf on 2026-09-13, when the
    pins were violated and the stack imported and ray started anyway.

    None means "could not determine", which is reported as such rather than
    guessed either way.
    """
    if holder != "transformers":
        return set()
    try:
        from transformers import dependency_versions_check as dvc
        return {n.lower() for n in getattr(dvc, "pkgs_to_check_at_runtime", [])}
    except Exception:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true",
                    help="print only the pip command that fixes things")
    args = ap.parse_args(argv)

    checked = 0
    violations = []          # (holder, requirement, installed version)
    for dist in WATCH:
        try:
            held = md.version(dist)
        except md.PackageNotFoundError:
            continue
        checked += 1
        if not args.quiet:
            print(f"  {GREEN}--{OFF}    {dist} {held}")
        for req in _requirements(dist):
            try:
                got = md.version(req.name)
            except md.PackageNotFoundError:
                continue          # not installed: a missing dep, not a wrong pin
            # prereleases=True so a 1.0.0rc1 is judged by the specifier, not skipped
            if req.specifier.contains(got, prereleases=True):
                continue
            violations.append((dist, req, got))

    if checked == 0:
        print("[pins] none of the watched distributions are installed -- cannot check",
              file=sys.stderr)
        return 2

    # An accepted conflict stays accepted only while its holder does not enforce
    # it at import; if a future vllm re-checks the range, it is a real failure.
    noted = []
    for v in list(violations):
        holder, req, got = v
        why = _accepted(holder, req.name)
        enforced = _enforced_at_import(holder) if why else None
        if why and enforced is not None and req.name.lower() not in enforced:
            noted.append((holder, req, got, why))
            violations.remove(v)

    if not violations:
        if not args.quiet:
            for holder, req, got, why in noted:
                print(f"  {YELLOW}NOTE{OFF}  {holder} declares {req.name}{req.specifier}, "
                      f"{req.name}=={got} installed -- accepted: {why}")
            print(f"  {GREEN}OK{OFF}    {checked} distribution(s), every declared pin satisfied"
                  + (" apart from the accepted exceptions above" if noted else ""))
        return 3 if noted else 0

    if not args.quiet:
        print()
        for holder, req, got in violations:
            print(f"  {RED}FAIL{OFF}  {holder} requires {req.name}{req.specifier} "
                  f"but {req.name}=={got} is installed")
        print()
        hard, soft, unknown = [], [], []
        for holder, req, _ in violations:
            enforced = _enforced_at_import(holder)
            if enforced is None:
                unknown.append((holder, req))
            elif req.name.lower() in enforced:
                hard.append((holder, req))
            else:
                soft.append((holder, req))
        if hard:
            print("  ENFORCED AT IMPORT -- these stop the stack dead:")
            for holder, req in hard:
                print(f"    {holder} re-checks {req.name} when it is imported, so"
                      " `import verl` raises")
            print("    and every training run dies in its first seconds.")
            print()
        if soft:
            print("  DECLARED ONLY -- not re-checked at import:")
            for holder, req in soft:
                print(f"    {holder} declares {req.name}{req.specifier} and does not"
                      " enforce it at runtime.")
            print("    The stack may well import and train. It is an untested combination,")
            print("    and two boxes that differ here are not running the same experiment.")
            print()
        if unknown:
            print("  ENFORCEMENT UNKNOWN (the holder could not be imported here):")
            for holder, req in unknown:
                print(f"    {holder} -> {req.name}{req.specifier}")
            print()

    # One pip line, pinning each offender to the range its holder asks for.
    # Deduplicated by package, keeping the first holder's specifier.
    seen, parts = set(), []
    for _, req, _ in violations:
        if req.name in seen:
            continue
        seen.add(req.name)
        parts.append(f'"{req.name}{req.specifier}"')
    print("  fix: pip install " + " ".join(parts))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
