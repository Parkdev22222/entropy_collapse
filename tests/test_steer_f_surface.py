"""The gate must check the steer_f surface verl actually imports.

Two branches with unrelated histories both carry a ``steer_f/``. Only the
``origin/paper`` one satisfies the ``verl/`` that ships beside it, and the one
file the two lineages share byte for byte is ``tree_rollout.py`` -- which is
exactly what every gate in the repo used to import. So the gate passed on a box
where no arm could start. These tests pin the replacement.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "run" / "_check_steer_f.py"


def run(root: Path):
    p = subprocess.run([sys.executable, str(CHECKER), str(root)],
                       capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def make_tree(root: Path, *, defines: str) -> None:
    """A minimal box: a verl that imports two names, and a steer_f."""
    actor = root / "verl" / "workers" / "actor"
    actor.mkdir(parents=True)
    (root / "verl" / "__init__.py").write_text("")
    actor.joinpath("dp_actor.py").write_text(textwrap.dedent("""
        def build():
            from steer_f.verl_integration import forecast_h_togo
            from steer_f.entropy_forecast import sibling_support
            return forecast_h_togo, sibling_support
    """))
    sf = root / "steer_f"
    sf.mkdir()
    sf.joinpath("__init__.py").write_text("")
    sf.joinpath("verl_integration.py").write_text(defines)
    sf.joinpath("entropy_forecast.py").write_text("def sibling_support():\n    pass\n")


def test_missing_symbol_is_caught(tmp_path):
    """The exact H100 failure: the module imports, the symbol is not there."""
    make_tree(tmp_path, defines="def something_else():\n    pass\n")
    rc, out = run(tmp_path)
    assert rc == 1
    assert "steer_f.verl_integration.forecast_h_togo" in out
    assert "git checkout origin/paper -- steer_f" in out


def test_complete_surface_passes(tmp_path):
    make_tree(tmp_path, defines="def forecast_h_togo():\n    pass\n")
    rc, out = run(tmp_path)
    assert rc == 0, out


def test_reexport_counts(tmp_path):
    """verl imports the name, not the definition site: a re-export is fine."""
    make_tree(tmp_path, defines="def forecast_h_togo():\n    pass\n")
    impl = tmp_path / "steer_f" / "_impl.py"
    impl.write_text("def forecast_h_togo():\n    pass\n")
    (tmp_path / "steer_f" / "verl_integration.py").write_text(
        "from steer_f._impl import forecast_h_togo\n")
    rc, out = run(tmp_path)
    assert rc == 0, out


def test_stub_verl_is_not_a_pass(tmp_path):
    """A checkout of the orchestration branch alone has an empty verl/ skeleton.
    Finding no import sites there must read as 'nothing to check', not 'OK'."""
    (tmp_path / "verl" / "trainer").mkdir(parents=True)
    (tmp_path / "steer_f").mkdir()
    (tmp_path / "steer_f" / "__init__.py").write_text("")
    rc, out = run(tmp_path)
    assert rc == 0
    assert "not a trainer checkout" in out


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True)


@pytest.mark.skipif(_git("rev-parse", "--verify", "origin/paper").returncode != 0,
                    reason="origin/paper not fetched here")
def test_this_branch_steer_f_does_not_satisfy_the_donor_verl():
    """The finding itself, pinned. If this ever starts failing, the two
    lineages have converged and bootstrap_pod.sh's override can be dropped --
    check that deliberately rather than deleting this test."""
    need = {
        "steer_f/verl_integration.py": ["forecast_h_togo", "compute_a_h"],
        "steer_f/entropy_forecast.py": ["sibling_support", "oracle_h_togo",
                                        "first_divergence"],
        "steer_f/monitors.py": ["token_weight_distribution"],
    }
    for path, names in need.items():
        donor = _git("show", f"origin/paper:{path}").stdout
        here = _git("show", f"HEAD:{path}").stdout
        for name in names:
            assert f"def {name}" in donor, f"{path}:{name} missing from origin/paper"
            assert f"def {name}" not in here, (
                f"{path}:{name} now exists on this branch too -- revisit "
                "bootstrap_pod.sh's RUNTIME_RE override")
