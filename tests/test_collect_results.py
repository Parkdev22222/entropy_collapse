"""collect_results.py reads a log at the sample count it carries, and will not mix.

The second eval pass used to be one sample per problem and the lookup asked for
mean@1 only, so a log evaluated at 32 samples would have left MATH500, Minerva,
OlympiadBench and GSM8K empty without a word. And one table must not hold an
avg@1 row beside an avg@32 row: the arm means and direction counts built on it
would subtract two different measurements.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PASS_A = ("val-core/aime_2024_dapo_boxed/acc/mean@32:0.171 - "
          "val-core/aime_2025_dapo_boxed/acc/mean@32:0.073 - "
          "val-core/amc2023_dapo_boxed/acc/mean@32:0.525\n")


def pass_b(k, m=0.694):
    return (f"val-core/math500/acc/mean@{k}:{m} - val-core/minerva_math/acc/mean@{k}:0.239 - "
            f"val-core/olympiadbench/acc/mean@{k}:0.345 - val-core/gsm8k_test/acc/mean@{k}:0.810\n")


def collect(tmp_path, logs):
    d = tmp_path / "logs"
    d.mkdir()
    for name, text in logs.items():
        (d / name).write_text(text)
    out = tmp_path / "summary.tsv"
    p = subprocess.run([sys.executable, str(ROOT / "scripts/collect_results.py"),
                        "--logs", str(d), "--out", str(out)], capture_output=True, text=True)
    return p, out


def test_an_avg1_log_reads_as_before(tmp_path):
    p, out = collect(tmp_path, {"eval-signed-s4.log": PASS_A + pass_b(1)})
    assert p.returncode == 0, p.stderr
    assert out.read_text().splitlines()[1].split("\t")[:9] == \
        ["signed", "4", "17.1", "7.3", "52.5", "69.4", "23.9", "34.5", "81.0"]


def test_a_k32_log_fills_every_column(tmp_path):
    p, out = collect(tmp_path, {"eval-signed-s4.log": PASS_A + pass_b(32)})
    assert p.returncode == 0, p.stderr
    row = out.read_text().splitlines()[1].split("\t")
    assert row[5:9] == ["69.4", "23.9", "34.5", "81.0"], row
    assert "MATH500@32" in p.stdout


def test_mixed_sample_counts_are_refused(tmp_path):
    p, _ = collect(tmp_path, {"eval-signed-s4.log": PASS_A + pass_b(32),
                              "eval-uniform-s4.log": PASS_A + pass_b(1, 0.722)})
    assert p.returncode != 0
    assert "REFUSE" in p.stderr and "eval-uniform-s4.log @1" in p.stderr, p.stderr
