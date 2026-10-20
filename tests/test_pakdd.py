"""Checks on the PAKDD submission: paper/pakdd.tex (main) and
paper/pakdd_supp.tex (supplementary material)."""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAIN, SUPP = ROOT / "paper" / "pakdd.tex", ROOT / "paper" / "pakdd_supp.tex"
PAGE_LIMIT = 12        # PAKDD 2025 main/special track: everything included
ABSTRACT_WORDS = 200
IDENTITY = re.compile(r"parkdev|jeonghobag|dsdsh|runpod|anthropic|claude|"
                      r"\bACL\b|\bARR\b|NAACL|COLING|openreview", re.I)


def _pdftext(tex: Path) -> str:
    pdf = tex.with_suffix(".pdf")
    if not pdf.is_file():
        pytest.skip(f"no {pdf.name}; build paper/ first")
    return subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True,
                          text=True, check=True).stdout


def test_main_paper_fits_the_page_limit():
    pdf = MAIN.with_suffix(".pdf")
    if not pdf.is_file():
        pytest.skip("no pakdd.pdf")
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
    assert pages <= PAGE_LIMIT, f"{pages} pages"


def test_abstract_word_limit():
    tex = MAIN.read_text()
    body = re.search(r"\\begin\{abstract\}(.*?)\\keywords", tex, re.S).group(1)
    words = re.findall(r"[A-Za-z0-9$\\{}.]+", re.sub(r"%.*", "", body))
    assert len(words) <= ABSTRACT_WORDS, f"{len(words)} words"


@pytest.mark.parametrize("tex", [MAIN, SUPP], ids=["main", "supp"])
def test_no_pending_cell_and_no_identity(tex):
    text = _pdftext(tex)
    assert "PENDING" not in text
    hits = sorted(set(m.group(0) for m in IDENTITY.finditer(text)))
    assert not hits, hits


@pytest.mark.parametrize("tex", [MAIN, SUPP], ids=["main", "supp"])
def test_every_slot_can_be_produced(tex):
    import importlib.util
    spec = importlib.util.spec_from_file_location("tp", ROOT / "tests" / "test_paper.py")
    tp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tp)
    src = re.sub(r"(?<!\\)%.*", "", tex.read_text())      # comments are not slots
    want = set(re.findall(r"\\num\{([A-Za-z]+)\}", src))
    orphans = sorted(want - tp.emittable_names())
    assert not orphans, orphans


def test_supplement_is_current():
    """pakdd_supp.tex is generated; a stale copy would drift from steerf.tex."""
    before = SUPP.read_text()
    subprocess.run(["python3", str(ROOT / "scripts" / "make_pakdd_supp.py")],
                   check=True, capture_output=True)
    assert SUPP.read_text() == before, "rerun scripts/make_pakdd_supp.py"
