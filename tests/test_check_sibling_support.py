# Copyright 2026 STEER-F authors
# Licensed under the Apache License, Version 2.0
"""The cheap pre-check for the locality measurement.

It exists so that an hour of CPU is not spent discovering that there were no
sibling positions to measure.  These tests pin the two readings that decide
that: prefix-sharing rollouts have support, independent ones do not.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_sibling_support.py"


def _jsonl(tmp_path, records, name="r.jsonl"):
    p = tmp_path / name
    p.write_text("".join(json.dumps(r) + "\n" for r in records))
    return p


def _run(path, *extra):
    # --model is a path that cannot resolve, so the whitespace fallback runs
    # and the test needs neither transformers nor the network.
    out = subprocess.run(
        [sys.executable, str(SCRIPT), str(path), "--model", "/nonexistent", *extra],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    return out


def test_prefix_sharing_rollouts_have_support(tmp_path):
    recs = [{"input": f"p{p}", "output": "a b c d " + " ".join(
        f"{p}_{j}_{k}" for k in range(20))}
        for p in range(3) for j in range(8)]
    out = _run(_jsonl(tmp_path, recs))
    assert out.returncode == 0, out.stderr
    assert "groups             3" in out.stdout
    frac = float(out.stdout.split("support_frac")[1].split()[0])
    assert frac > 0, out.stdout
    # The shared prefix is at the start, so the script must say so.
    assert "decile 0" in out.stdout


def test_independent_rollouts_have_support_only_at_position_zero(tmp_path):
    """The plain-rollout case, and the reason the fraction is ~1/T.

    Every rollout is trivially prefix-matched before its own first token, so
    t=0 is always in the support and nothing after it is.  With responses of
    T tokens that is 1/T -- which is where training's .003 on plain rollouts
    comes from, against .40 on tree rollouts.  A_H is zero everywhere else.
    """
    T = 20
    recs = [{"input": f"p{p}", "output": " ".join(f"{p}_{j}_{k}"
                                                  for k in range(T))}
            for p in range(3) for j in range(8)]
    out = _run(_jsonl(tmp_path, recs))
    assert out.returncode == 0, out.stderr
    frac = float(out.stdout.split("support_frac")[1].split()[0])
    assert abs(frac - 1.0 / T) < 1e-6, out.stdout

    deciles = out.stdout.split("by decile")[1].split("\n")[0].split()
    assert float(deciles[0]) > 0 and all(float(d) == 0 for d in deciles[1:]), \
        out.stdout
    assert "too few positions" in out.stdout
    assert "not the branch points" in out.stdout


def test_accepts_both_key_spellings(tmp_path):
    """verl dumps validation as input/output and warm-up as prompt/response."""
    a = _jsonl(tmp_path, [{"input": "p", "output": "a b c d e"} for _ in range(4)],
               "in_out.jsonl")
    b = _jsonl(tmp_path, [{"prompt": "p", "response": "a b c d e"} for _ in range(4)],
               "prompt_resp.jsonl")
    ra, rb = _run(a), _run(b)
    assert ra.returncode == 0 and rb.returncode == 0, (ra.stderr, rb.stderr)
    take = lambda s: s.split("support_frac")[1].split()[0]
    assert take(ra.stdout) == take(rb.stdout)


def test_no_usable_group_exits_with_a_message(tmp_path):
    out = _run(_jsonl(tmp_path, [{"input": f"p{i}", "output": "x"}
                                 for i in range(4)]))
    assert out.returncode != 0
    assert "no prompt group of size >= 2" in (out.stderr + out.stdout)


def test_writes_json_when_asked(tmp_path):
    recs = [{"input": "p", "output": "a b c " + str(j)} for j in range(4)]
    dest = tmp_path / "out" / "s.json"
    out = _run(_jsonl(tmp_path, recs), "--json", str(dest))
    assert out.returncode == 0, out.stderr
    got = json.loads(dest.read_text())
    assert got["n_groups"] == 1 and "support_frac" in got
