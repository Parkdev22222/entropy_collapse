# Copyright 2026 STEER-F authors
# Licensed under the Apache License, Version 2.0
"""The locality script calls into the steer_f the pods actually run.

WHY THIS EXISTS
    `scripts/measure_forecast_locality.py` had never been executed -- the
    measure stage skipped it whenever the warm-up rollouts were missing, which
    was always.  Its one call into the trainer's own code read

        sibling_support(resp, B, mask)

    against a signature of ``(responses, mask, group_index)``: the batch size
    was passed where the mask belongs.  The first real execution would have
    died on ``torch.zeros_like(int)`` in the first group, after loading a
    1.5B model.

    The reason a plain import test does not catch it is the branch split.
    `steer_f/entropy_forecast.py` on THIS branch is the early Phase 0-2 file
    and has no `sibling_support` at all; the file the pods run lives on
    `origin/paper`.  So the check has to reach for that copy explicitly --
    which is exactly what makes it catch drift between the two.
"""
from __future__ import annotations

import ast
import importlib.util
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "measure_forecast_locality.py"
PAPER_REF = "origin/paper:steer_f/entropy_forecast.py"


def _paper_entropy_forecast(tmp_path):
    """The trainer's entropy_forecast, loaded standalone (torch only)."""
    try:
        src = subprocess.run(["git", "show", PAPER_REF], cwd=ROOT,
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:      # no git
        pytest.skip(f"cannot reach {PAPER_REF}: {exc}")
    if src.returncode != 0:
        pytest.skip(f"cannot reach {PAPER_REF}: {src.stderr.strip()[:80]}")

    path = tmp_path / "ef_paper.py"
    path.write_text(src.stdout)
    spec = importlib.util.spec_from_file_location("ef_paper", path)
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules, so register first.
    sys.modules["ef_paper"] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("ef_paper", None)
    return mod


def _calls_to(name):
    """Every call to `name` in the locality script, as (posargs, kwnames)."""
    tree = ast.parse(SCRIPT.read_text())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == name:
            out.append((len(node.args), [k.arg for k in node.keywords]))
    return out


def test_script_imports_exist_in_the_trainers_module(tmp_path):
    """Names imported from steer_f.entropy_forecast are really there."""
    mod = _paper_entropy_forecast(tmp_path)
    tree = ast.parse(SCRIPT.read_text())
    wanted = [a.name for n in ast.walk(tree)
              if isinstance(n, ast.ImportFrom)
              and n.module == "steer_f.entropy_forecast"
              for a in n.names]
    assert wanted, "the script no longer imports from steer_f.entropy_forecast"
    missing = [w for w in wanted if not hasattr(mod, w)]
    assert not missing, f"not in {PAPER_REF}: {missing}"


def test_sibling_support_is_called_by_keyword(tmp_path):
    """Order-proof call, and the keywords are the real parameter names."""
    import inspect

    calls = _calls_to("sibling_support")
    assert len(calls) == 1, f"expected one sibling_support call, got {calls}"
    n_pos, kwnames = calls[0]
    assert n_pos == 0, (
        "sibling_support takes (responses, mask, group_index); a positional "
        "call is how the batch size ended up where the mask belongs")

    mod = _paper_entropy_forecast(tmp_path)
    params = list(inspect.signature(mod.sibling_support).parameters)
    assert sorted(kwnames) == sorted(params[:len(kwnames)]) or \
        all(k in params for k in kwnames), f"{kwnames} vs {params}"
    assert set(params[:3]) <= set(kwnames), (
        f"call passes {kwnames}, signature needs {params[:3]}")


def test_sibling_support_runs_on_the_shapes_the_script_builds(tmp_path):
    """The regression itself: these arguments used to raise TypeError."""
    torch = pytest.importorskip("torch")
    mod = _paper_entropy_forecast(tmp_path)

    # One prompt group, padded exactly as the script pads it.
    resp = torch.tensor([[5, 6, 7, 1, 2, 3],
                         [5, 6, 7, 4, 4, 0],     # shares the 5,6,7 prefix
                         [9, 9, 9, 9, 0, 0]])    # diverges at t=0
    mask = torch.tensor([[1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 0, 0]])
    uid = [0, 0, 0]

    sup = mod.sibling_support(responses=resp, mask=mask, group_index=uid)
    assert sup.shape == resp.shape and sup.dtype == torch.bool
    # The two prefix-sharing rows have support through their divergence at t=3.
    assert sup[0, :4].all() and sup[1, :4].all()
    # The outlier only shares position 0, where every rollout is still together.
    assert bool(sup[2, 0]) and not bool(sup[2, 1:].any())

    with pytest.raises(TypeError):
        mod.sibling_support(resp, len(uid), mask)      # the shipped bug


# --- the local twin -----------------------------------------------------


def test_local_twin_is_read_at_the_same_offset_as_the_forecast():
    """The defect that made the first real run a null test.

    slice_response_hidden takes hidden[:, -T:] -- "the hidden produced after
    consuming responses[:, i]" -- so H_togo[t] is conditioned on s_t + y_t.
    The twin was read from logits[:, P-1 : P-1+T], i.e. H(pi(.|s_t)),
    conditioned on the prefix alone. entropy_advantage applies no shift of its
    own, so prefix-matched siblings all carry the same value there and the
    sibling baseline cancels it exactly: the run reported
    a_local_absmean = 1.0e-07 against a_h_absmean = 0.283, with a correlation
    of -6e-09 that could not have come out any other way.
    """
    src = SCRIPT.read_text()
    assert re.search(r"hidden\[:, P\s*:\s*P \+ T, :\]", src), \
        "the local twin must be read at P..P+T-1, matching slice_response_hidden"
    assert not re.search(r"\[:, P - 1\s*:\s*P - 1 \+ T, :\]", src), \
        "P-1 is the pre-branch state; the sibling baseline annihilates it"


def test_a_void_run_is_announced_not_reported_as_a_result():
    """A collapsed twin must be called out, not read as 'uncorrelated'."""
    src = SCRIPT.read_text()
    assert "VOID" in src, "no guard against a degenerate local twin"
    assert "a_local_absmean" in src.split("def main")[1], \
        "the guard has to look at the twin's own magnitude"


# --- memory: the group forward used to be 33 GB ------------------------


def _load_local_entropy():
    """Compile just local_entropy out of the script.

    Importing the module pulls steer_f, whose copy on this branch is the
    early Phase 0-2 file (see the header). Lifting the one function keeps the
    test on the real source instead of a paraphrase of it.
    """
    torch = pytest.importorskip("torch")
    tree = ast.parse(SCRIPT.read_text())
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "local_entropy"), None)
    assert fn is not None, "local_entropy is gone; the chunking went with it"
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<local_entropy>", "exec"), ns)
    return torch, ns["local_entropy"]


def test_chunking_the_local_entropy_is_exact():
    """Softmax is per position, so the chunk size must not move the answer.

    This is what licenses the memory fix: one group's logits were
    [32, 768, 151936] float32 = 13.9 GB, plus 4.6 GB each for logp and its
    exponential, on a box that is also training.
    """
    torch, local_entropy = _load_local_entropy()
    torch.manual_seed(0)
    B, T, H, V = 3, 7, 16, 41
    h = torch.randn(B, T, H)
    head = torch.nn.Linear(H, V, bias=False)

    whole = local_entropy(h, head, B * T)
    for chunk in (11, 5, 1):
        got = local_entropy(h, head, chunk)
        assert torch.allclose(whole, got, atol=1e-6), f"chunk={chunk}"
    assert whole.shape == (B, T)
    assert (whole > 0).all() and (whole <= math.log(V) + 1e-5).all()


def test_the_group_forward_does_not_build_every_layer_or_the_full_logits():
    """The regression: one discarded 13.9 GB tensor and 28 discarded layers."""
    src = SCRIPT.read_text()
    body = src.split("for gi, (prompt, responses) in enumerate(groups):")[1]
    assert "get_decoder()" in body, "the CausalLM head rebuilds [B, S, V] logits"
    assert "out.logits" not in body, "the local twin must come from hidden states"
    # Comments here discuss the old path, so count code lines only. The one
    # survivor is the fallback for a model without get_decoder().
    code = [ln for ln in body.splitlines() if not ln.lstrip().startswith("#")]
    assert sum("output_hidden_states=True" in ln for ln in code) <= 1
