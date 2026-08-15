"""End-to-end check of `patches/` against a real STEER checkout.

The unit tests prove that `steer_f.omega_tilde` reproduces STEER.  They do not
prove that the *patch* wires it up correctly — a patch that applies cleanly can
still pass the wrong argument, drop a keyword, or reverse a default.  This test
closes that gap: it applies the patches to a fresh copy of the repo, extracts
the patched `compute_token_weights` by AST, and runs it against the untouched
reference.

Skipped unless a STEER checkout is available:

    git clone https://github.com/zz-haooo/STEER third_party/STEER
    # or
    STEER_REPO=/path/to/STEER pytest tests/test_patch_integration.py
"""

import ast
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from reference_steer import compute_token_weights as steer_reference  # noqa: E402
from test_lambda_zero_equiv import make_batch  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent
PATCH_DIR = REPO_ROOT / "patches"

TARGETS = {
    "core_algos_steerf.patch": "verl/trainer/ppo/core_algos.py",
    "dp_actor_steerf.patch": "verl/workers/actor/dp_actor.py",
}

# Files touched by any patch, so the fixture stages a complete tree.
PATCHED_FILES = [
    "verl/trainer/ppo/core_algos.py",
    "verl/trainer/ppo/ray_trainer.py",
    "verl/workers/actor/dp_actor.py",
    "verl/workers/fsdp_workers.py",
]


def find_steer_repo():
    for candidate in (os.environ.get("STEER_REPO"), REPO_ROOT / "third_party" / "STEER"):
        if candidate and (Path(candidate) / "verl" / "trainer" / "ppo" / "core_algos.py").exists():
            return Path(candidate)
    return None


steer_repo = find_steer_repo()
requires_steer = pytest.mark.skipif(
    steer_repo is None,
    reason="no STEER checkout; set STEER_REPO or clone into third_party/STEER",
)


def pristine_source(rel: str) -> str:
    """The upstream file, regardless of the checkout's current patch state.

    `scripts/setup_steer.sh` patches the working tree in place, so reading the
    files off disk would feed already-patched input to the patcher and fail.
    Reading them out of git guarantees this test always starts from upstream.
    """
    proc = subprocess.run(
        ["git", "-C", str(steer_repo), "show", f"HEAD:{rel}"],
        capture_output=True, text=True,
    )
    if proc.returncode == 0:
        return proc.stdout
    return (steer_repo / rel).read_text()  # not a git checkout — take it as-is


@pytest.fixture(scope="module")
def patched_tree():
    """A throwaway copy of the pristine STEER files with every patch applied."""
    tmp = Path(tempfile.mkdtemp(prefix="steerf-patch-"))
    try:
        for rel in PATCHED_FILES:
            dst = tmp / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(pristine_source(rel))
        for patch_file in sorted(PATCH_DIR.glob("*.patch")):
            proc = subprocess.run(
                ["patch", "-p1", "-s", "-i", str(patch_file)],
                cwd=tmp, capture_output=True, text=True,
            )
            assert proc.returncode == 0, f"{patch_file.name} failed:\n{proc.stdout}{proc.stderr}"
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def extract_function(path: Path, name: str):
    """Compile one top-level function out of a module we cannot import.

    verl's `core_algos.py` pulls in ray, vllm and friends at import time, none
    of which exist on a CPU box.  `compute_token_weights` only needs `torch`,
    so lifting it out by AST lets the patch be tested anywhere.
    """
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            module = ast.Module(body=[node], type_ignores=[])
            ns = {"torch": torch, "__name__": "patched_core_algos"}
            exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
            return ns[name]
    raise AssertionError(f"{name} not found in {path}")


@requires_steer
def test_patches_apply_cleanly(patched_tree):
    for rel in PATCHED_FILES:
        assert (patched_tree / rel).exists()


@requires_steer
def test_every_patch_targets_a_real_file():
    for rel in PATCHED_FILES:
        assert (steer_repo / rel).exists(), f"patch target missing upstream: {rel}"
    for patch_file in TARGETS:
        assert (PATCH_DIR / patch_file).exists(), f"missing patch {patch_file}"


@requires_steer
def test_patched_tree_is_valid_python(patched_tree):
    for rel in PATCHED_FILES:
        ast.parse((patched_tree / rel).read_text())


@requires_steer
def test_forecast_patch_keeps_the_return_arity_consistent(patched_tree):
    """compute_log_prob grew a 4th value; every call site must agree.

    Upstream's ref-policy call site already unpacked 4 values from a 3-tuple,
    so this also pins the incidental fix — a regression here would resurrect a
    crash that only fires when a standalone reference policy is enabled.
    """
    actor = (patched_tree / "verl/workers/actor/dp_actor.py").read_text()
    workers = (patched_tree / "verl/workers/fsdp_workers.py").read_text()
    assert "return log_probs, entropys, max_prob_log_probs, h_togo_vals" in actor
    assert "output, entropys, max_prob_log_probs, h_togo = self.actor.compute_log_prob" in workers
    assert "output, _, _, _ = self.ref_policy.compute_log_prob" in workers


@requires_steer
def test_a_h_is_built_where_the_whole_group_is_visible(patched_tree):
    """The sibling baseline needs every rollout of a prompt, so A_H must be
    assembled in ray_trainer (which has `uid`), never inside the actor."""
    trainer = (patched_tree / "verl/trainer/ppo/ray_trainer.py").read_text()
    actor = (patched_tree / "verl/workers/actor/dp_actor.py").read_text()
    assert 'batch.batch["a_h"] = a_h' in trainer
    assert 'batch.non_tensor_batch["uid"]' in trainer
    assert "compute_a_h" not in actor, "A_H must not be computed per micro-batch"


@requires_steer
def test_forecast_is_skipped_when_steerf_is_off(patched_tree):
    """No heads loaded, no extra forward, when lambda is 0."""
    actor = (patched_tree / "verl/workers/actor/dp_actor.py").read_text()
    assert 'float(self._steerf_config().get("steerf_lam", 0.0)) != 0.0' in actor
    assert "if not self._steerf_enabled():\n            return None" in actor


@requires_steer
@pytest.mark.parametrize("mode", ["symmetric", "asymmetric"])
@pytest.mark.parametrize("linear", [True, False])
def test_patched_default_call_is_stock_steer(patched_tree, mode, linear):
    """The default keyword path must be untouched STEER, bit for bit.

    This is what protects every non-STEER-F run in the experiment matrix: the
    GRPO and STEER arms go through the patched file too.
    """
    patched = extract_function(patched_tree / TARGETS["core_algos_steerf.patch"],
                               "compute_token_weights")
    batch = make_batch(seed=21)
    expected = steer_reference(**batch, linear=linear, mode=mode)
    got = patched(**batch, linear=linear, mode=mode)
    assert torch.equal(got, expected)


@requires_steer
def test_patched_return_stats_shape(patched_tree):
    """return_stats must not change the stock return type by accident."""
    patched = extract_function(patched_tree / TARGETS["core_algos_steerf.patch"],
                               "compute_token_weights")
    batch = make_batch(seed=22)
    plain = patched(**batch)
    assert isinstance(plain, torch.Tensor)
    weights, stats = patched(**batch, return_stats=True)
    assert torch.equal(weights, plain)
    assert isinstance(stats, dict) and stats["steerf/lam"] == 0.0


@requires_steer
def test_patched_delegates_when_lambda_nonzero(patched_tree):
    """With lambda>0 the patched function must produce the steer_f result."""
    from steer_f.omega_tilde import compute_token_weights_steerf

    patched = extract_function(patched_tree / TARGETS["core_algos_steerf.patch"],
                               "compute_token_weights")
    batch = make_batch(seed=23)
    a_h = torch.randn(batch["advantages"].shape, generator=torch.Generator().manual_seed(23))

    got = patched(**batch, a_h=a_h, steerf_lam=0.5)
    expected = compute_token_weights_steerf(**batch, a_h=a_h, lam=0.5)
    assert torch.equal(got, expected)
    assert not torch.equal(got, steer_reference(**batch)), "delegation was a no-op"


@requires_steer
def test_patched_ignores_a_h_when_lambda_zero(patched_tree):
    patched = extract_function(patched_tree / TARGETS["core_algos_steerf.patch"],
                               "compute_token_weights")
    batch = make_batch(seed=24)
    got = patched(**batch, a_h=torch.randn(batch["advantages"].shape), steerf_lam=0.0)
    assert torch.equal(got, steer_reference(**batch))


@requires_steer
def test_dp_actor_still_unpacks_nine_values(patched_tree):
    """Guard the return-arity contract documented in steer_code_map §7."""
    src = (patched_tree / TARGETS["dp_actor_steerf.patch"]).read_text()
    assert "pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, pg_clipfrac_high, " \
           "pg_clipfrac_low, clipped_by_high, clipped_by_low, clip_stats = result" in src


@requires_steer
def test_dp_actor_passes_every_steerf_knob(patched_tree):
    """A dropped keyword here would silently pin lambda to its default."""
    src = (patched_tree / TARGETS["dp_actor_steerf.patch"]).read_text()
    for knob in ("a_h=a_h", "steerf_lam=steerf_lam", "steerf_eta=steerf_eta",
                 "steerf_clip_c=steerf_clip_c", "steerf_norm=steerf_norm"):
        assert knob in src, f"dp_actor patch does not forward {knob}"
