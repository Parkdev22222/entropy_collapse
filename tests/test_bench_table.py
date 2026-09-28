"""The six-benchmark section's words and table come from the data.

The manuscript prints the registered verdict (\\num{Dirverdict}) and the arms
with no evaluated row (\\num{Benchmissing}) rather than asserting them, because
one evaluated seed per arm is what exists now and more are coming: a sentence
that names a verdict or lists absent arms by hand goes stale the day a row
arrives.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_seeds as A  # noqa: E402

HEAD = "arm\tseed\tAIME24\tAIME25\tAMC23\tMATH500\tMinerva\tOlympiad\tGSM8K*\tLCB-v5\tAvg(math6)\n"


def tsv(tmp_path, rows):
    f = tmp_path / "summary.tsv"
    f.write_text(HEAD + "".join("\t".join(map(str, r)) + "\n" for r in rows))
    return f


@pytest.mark.parametrize("hits,word", [(6, "support"), (4, "support"),
                                       (3, "equivocal"),
                                       (2, "evidence against"), (0, "evidence against")])
def test_the_registered_thresholds(hits, word):
    assert A.direction_verdict(hits, 6) == word


def test_the_thresholds_are_only_for_the_registered_six():
    """The held-out five has no rule; a verdict word for it would be invented."""
    with pytest.raises(ValueError):
        A.direction_verdict(3, 5)


def test_each_row_says_which_seed_it_is(tmp_path):
    f = tsv(tmp_path, [("signed", 4, 17.1, 7.3, 52.5, 69.4, 23.9, 34.5, 81.0, "-", 34.1),
                       ("permuted", 3, 15.2, 6.4, 55.8, 69.0, 23.9, 31.2, 76.8, "-", 33.6)])
    tbl = A.benchmarks_table(f)
    assert "& seed \\\\" in tbl
    assert [l.strip().rsplit("&", 1)[1].strip(" \\") for l in tbl.splitlines()
            if l.strip().startswith(("\\textbf", "\\textsc"))] == ["3", "4"]


def test_the_oracle_row_is_printed_after_a_rule(tmp_path):
    """It used to be dropped: the table looped MAIN_ARMS only, so a finished
    oracle evaluation would never have reached the paper."""
    f = tsv(tmp_path, [("signed", 4, 17.1, 7.3, 52.5, 69.4, 23.9, 34.5, 81.0, "-", 34.1),
                       ("oracle", 4, 14.2, 8.2, 58.2, 70.8, 24.3, 33.7, 81.5, "-", 34.9)])
    lines = [l.strip() for l in A.benchmarks_table(f).splitlines()]
    i = next(k for k, l in enumerate(lines) if l.startswith("\\textsc{oracle}"))
    assert lines[i - 1] == "\\midrule" and i > lines.index("\\midrule") + 1


def test_missing_arms_are_named(tmp_path):
    f = tsv(tmp_path, [("signed", 4, 17.1, 7.3, 52.5, 69.4, 23.9, 34.5, 81.0, "-", 34.1),
                       ("uniform", 4, 15.7, 6.7, 56.0, 72.2, 25.7, 33.6, 80.6, "-", 35.0)])
    assert A.bench_missing(f) == ["grpo", "steer", "permuted", "oracle"]
