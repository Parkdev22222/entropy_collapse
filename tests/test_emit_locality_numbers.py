# Copyright 2026 STEER-F authors
# Licensed under the Apache License, Version 2.0
"""The locality measurement reaches the manuscript as macros, not by hand.

Section 12's cells are `\\num{<name>}` resolved against results/*.tex. Typing
a measured number straight into the prose is the defect fixed on 2026-09-15,
where `\\num{0.145}` passed a literal to a macro lookup and three sentences
rendered as red placeholders.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "emit_locality_numbers.py"

FULL = {
    "config": {"n_groups": 8, "kappa": 2, "gamma_h": 0.7},
    "sibling_support": {
        "n_positions": 6576,
        "levels_pearson_h_togo_vs_h_local": 0.3720568276830094,
        "levels_spearman_h_togo_vs_h_local": 0.40475992381125514,
        "advantage_pearson_a_h_vs_a_local": -6.341370426280739e-10,
        "advantage_spearman_a_h_vs_a_local": 0.02507130460600682,
        "advantage_r_squared": 4.021297888330796e-19,
        "sign_agreement": 0.5272727272727272,
        "a_h_absmean": 0.049557995051145554,
        "a_local_absmean": 1.85734059243714e-08,
        "h_togo_mean": 0.456428200006485,
        "h_local_mean": 0.4367567002773285,
    },
    "branch_points_only": {
        "n_positions": 1151,
        "levels_pearson_h_togo_vs_h_local": 0.23499780164171555,
        "levels_spearman_h_togo_vs_h_local": 0.521363010530553,
        "advantage_pearson_a_h_vs_a_local": -6.195422255026088e-09,
        "advantage_spearman_a_h_vs_a_local": 0.031140043460902114,
        "advantage_r_squared": 3.8383256918072545e-17,
        "sign_agreement": 0.5272727272727272,
        "a_h_absmean": 0.28313931822776794,
        "a_local_absmean": 1.0191900656764119e-07,
        "h_togo_mean": 0.9435131549835205,
        "h_local_mean": 1.6212263107299805,
    },
}


def _run(tmp_path, doc):
    src = tmp_path / "loc.json"
    src.write_text(json.dumps(doc))
    dest = tmp_path / "out" / "numbers-locality.tex"
    out = subprocess.run([sys.executable, str(SCRIPT), "--json", str(src),
                          "--out", str(dest)],
                         cwd=ROOT, capture_output=True, text=True, timeout=60)
    return out, dest


def _macros(text):
    return dict(re.findall(r"\\providecommand\{\\([A-Za-z]+)\}\{([^}]*)\}", text))


def test_emits_the_branch_point_scope_as_the_headline(tmp_path):
    out, dest = _run(tmp_path, FULL)
    assert out.returncode == 0, out.stderr
    m = _macros(dest.read_text())
    assert m["Locn"] == "1151" and m["Locnsupport"] == "6576"
    assert m["Locgroups"] == "8"
    assert m["Loclevels"] == "+0.235"
    assert m["Locadv"] == "-0.000"
    assert m["Locadvrho"] == "+0.031"
    assert m["Locsign"] == "0.527"


def test_emits_both_magnitudes_so_a_void_run_is_visible(tmp_path):
    """|A_H| vs |A_local| is what separates 'unrelated' from 'nothing there'."""
    out, dest = _run(tmp_path, FULL)
    assert out.returncode == 0, out.stderr
    m = _macros(dest.read_text())
    assert m["Locahmag"] == "0.2831"
    # The first real run's twin was float32 dust; it must still render.
    assert float(m["Locloctwinmag"]) == 0.0 or m["Locloctwinmag"] == "0.0000"


def test_every_name_is_a_valid_macro(tmp_path):
    """test_paper.py requires \\num{} names to be letters only."""
    out, dest = _run(tmp_path, FULL)
    assert out.returncode == 0, out.stderr
    for name in _macros(dest.read_text()):
        assert re.fullmatch(r"[A-Za-z]+", name), name


def test_a_truncated_run_leaves_its_slots_red(tmp_path):
    """Too few branch points: emit what exists, never invent the rest."""
    doc = {"config": {"n_groups": 2},
           "sibling_support": FULL["sibling_support"],
           "branch_points_only": {"n_positions": 1}}
    out, dest = _run(tmp_path, doc)
    assert out.returncode == 0, out.stderr
    m = _macros(dest.read_text())
    assert m["Locn"] == "1"
    assert "Locadv" not in m and "Locsign" not in m
    assert "left red" in out.stdout


def test_missing_json_is_an_error_with_the_command_to_fix_it(tmp_path):
    out = subprocess.run([sys.executable, str(SCRIPT),
                          "--json", str(tmp_path / "nope.json"),
                          "--out", str(tmp_path / "o.tex")],
                         cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert out.returncode == 1
    assert "measure_forecast_locality.py" in out.stdout
