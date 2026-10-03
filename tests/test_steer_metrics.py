"""The STEER-metrics path: one-copy Pass@k inputs, both pass@k numbers, curves.

run/eval_all_metrics.sh strings these together; each piece is tested here on
CPU with fakes, because the evaluation itself needs a GPU.
"""
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from passk_from_dump import ks_for, pass_at_k  # noqa: E402


def run(*args):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)


def write_parquet(path, prompts):
    rows = [{"data_source": "aime_2024_dapo_boxed",
             "prompt": [{"role": "user", "content": p}],
             "reward_model": {"ground_truth": "1"}} for p in prompts]
    pq.write_table(pa.Table.from_pylist(rows), path)


# ---------------------------------------------------------------- dedupe
def test_dedupe_keeps_one_copy_in_file_order(tmp_path):
    src, dst = tmp_path / "a.parquet", tmp_path / "out" / "a.parquet"
    problems = [f"problem {i}" for i in range(30)]
    write_parquet(src, [p for _ in range(32) for p in problems])     # 960 rows, interleaved
    p = run(ROOT / "scripts/dedupe_replicas.py", "--expect", 30, src, dst)
    assert p.returncode == 0, p.stderr
    got = [r["prompt"][0]["content"] for r in pq.read_table(dst).to_pylist()]
    assert got == problems


def test_dedupe_refuses_uneven_replication(tmp_path):
    src = tmp_path / "a.parquet"
    write_parquet(src, ["x", "x", "y"])
    p = run(ROOT / "scripts/dedupe_replicas.py", src, tmp_path / "b.parquet")
    assert p.returncode != 0 and "unevenly" in (p.stdout + p.stderr)


def test_dedupe_refuses_wrong_problem_count(tmp_path):
    src = tmp_path / "a.parquet"
    write_parquet(src, ["x", "y", "x", "y"])
    p = run(ROOT / "scripts/dedupe_replicas.py", "--expect", 30, src, tmp_path / "b.parquet")
    assert p.returncode != 0 and "expected 30" in (p.stdout + p.stderr)


# ---------------------------------------------------------------- pass@k
def test_unbiased_estimator_small_cases():
    assert pass_at_k(4, 1, 1) == pytest.approx(0.25)
    assert pass_at_k(4, 1, 4) == 1.0
    assert pass_at_k(4, 0, 4) == 0.0
    assert pass_at_k(4, 2, 2) == pytest.approx(1 - 1 / 6)        # 1 - C(2,2)/C(4,2)
    # one solve in 1024 is pass@1024 = 1, where verl's bootstrap reads ~.63
    assert pass_at_k(1024, 1, 1024) == 1.0


def test_ks_are_powers_of_two_ending_at_n():
    assert ks_for(1024) == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    assert ks_for(6) == [1, 2, 4, 6]


def write_dump(path, problems):
    """problems: {prompt: [correct, ...]}, written in verl's interleaved order."""
    path.mkdir(parents=True, exist_ok=True)
    with (path / "0.jsonl").open("w") as fh:
        for prompt, results in problems.items():
            for ok in results:
                fh.write(json.dumps({"input": prompt, "output": "", "acc": bool(ok),
                                     "score": 1.0 if ok else -1.0,
                                     "data_source": "aime_2024_dapo_boxed"}) + "\n")


def test_passk_from_dump_table_and_rerun(tmp_path):
    dump, out = tmp_path / "passk", tmp_path / "passk_unbiased.tsv"
    write_dump(dump, {"p1": [1, 0, 0, 0], "p2": [0, 0, 0, 0]})
    for _ in range(2):                                   # a re-run replaces, not duplicates
        p = run(ROOT / "scripts/passk_from_dump.py", dump, "--label", "r2-s1", "--out", out)
        assert p.returncode == 0, p.stderr
    rows = [ln.split("\t") for ln in out.read_text().splitlines()[1:]]
    assert len(rows) == len(ks_for(4))
    by_k = {int(r[4]): float(r[5]) for r in rows}
    assert by_k[1] == pytest.approx(12.5)                # (.25 + 0) / 2
    assert by_k[4] == pytest.approx(50.0)                # (1 + 0) / 2
    p = run(ROOT / "scripts/passk_from_dump.py", dump, "--label", "base", "--out", out)
    assert {ln.split("\t")[0] for ln in out.read_text().splitlines()[1:]} == {"r2-s1", "base"}


def test_passk_from_dump_refuses_unequal_n(tmp_path):
    dump = tmp_path / "passk"
    write_dump(dump, {"p1": [1, 0, 0, 0], "p2": [0, 0]})
    p = run(ROOT / "scripts/passk_from_dump.py", dump, "--label", "x", "--out", tmp_path / "o.tsv")
    assert p.returncode != 0 and "different sample counts" in (p.stdout + p.stderr)


# ---------------------------------------------------------------- verl best@k
def test_collect_passk_reads_core_and_aux(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    line = ("step:0 - val-aux/aime_2024_dapo_boxed/acc/best@256/mean:0.40 - "
            "val-aux/aime_2024_dapo_boxed/acc/best@512/mean:0.45 - "
            "val-core/aime_2024_dapo_boxed/acc/best@1024/mean:0.50 - "
            "val-aux/aime_2025_dapo_boxed/acc/best@256/mean:0.30 - "
            "val-aux/aime_2025_dapo_boxed/acc/best@512/mean:0.33 - "
            "val-core/aime_2025_dapo_boxed/acc/best@1024/mean:0.36\n")
    (logs / "eval-r2-s1.log").write_text(line)
    (logs / "eval-r2-s2.log").write_text(line.replace("0.50", "0.60"))
    out = tmp_path / "passk_verl.tsv"
    p = run(ROOT / "scripts/collect_results.py", "--passk", "--logs", logs, "--out", out)
    assert p.returncode == 0, p.stderr
    lines = out.read_text().splitlines()
    assert lines[0].split("\t")[2:] == ["AIME24 best@256", "AIME24 best@512", "AIME24 best@1024",
                                        "AIME25 best@256", "AIME25 best@512", "AIME25 best@1024"]
    assert lines[1].split("\t") == ["r2", "1", "40.0", "45.0", "50.0", "30.0", "33.0", "36.0"]
    assert lines[3].split("\t")[:5] == ["r2", "avg", "40.0", "45.0", "55.0"]


def test_collect_six_benchmarks_unchanged_by_passk_mode(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "eval-r2-s1.log").write_text("val-core/aime_2024_dapo_boxed/acc/mean@32:0.171\n")
    out = tmp_path / "bench.tsv"
    p = run(ROOT / "scripts/collect_results.py", "--logs", logs, "--out", out)
    assert p.returncode == 0, p.stderr
    assert out.read_text().splitlines()[1].split("\t")[:3] == ["r2", "1", "17.1"]


# ---------------------------------------------------------------- curves
def test_export_curves_steps_and_resume_override(tmp_path):
    a, b = tmp_path / "a.log", tmp_path / "b.log"
    a.write_text("step:0 - val-core/aime_2024_dapo_boxed/acc/mean@32:0.033\n"
                 "step:1 - actor/entropy:0.599 - critic/score/mean:-0.868 - response_length/mean:1264.0\n"
                 "step:10 - actor/entropy:0.357 - val-core/aime_2024_dapo_boxed/acc/mean@32:0.072 - "
                 "val-core/aime_2024_dapo_boxed/acc/maj@32/mean:0.126\n")
    b.write_text("step:10 - val-core/aime_2024_dapo_boxed/acc/mean@32:0.080\n")   # resumed re-validation
    out = tmp_path / "c.tsv"
    p = run(ROOT / "scripts/export_curves.py", a, b, "--out", out)
    assert p.returncode == 0, p.stderr
    lines = out.read_text().splitlines()
    head = lines[0].split("\t")
    rows = {int(ln.split("\t")[0]): dict(zip(head, ln.split("\t"))) for ln in lines[1:]}
    assert sorted(rows) == [0, 1, 10]
    assert rows[1]["entropy"] == "0.5990" and rows[1]["train_acc"] == "0.0660"
    assert rows[1]["aime24_mean@32"] == ""
    assert rows[10]["aime24_mean@32"] == "0.0800"       # the later log wins
    assert rows[10]["aime24_maj@32"] == "0.1260"


def test_wrapper_parses():
    assert subprocess.run(["bash", "-n", str(ROOT / "run/eval_all_metrics.sh")]).returncode == 0
    assert subprocess.run(["bash", "-n", str(ROOT / "run/eval_steerf.sh")]).returncode == 0


def test_wrapper_refuses_missing_weights(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "POINT": "final", "ARM": "x", "SEED": "1",
           "RUN_NAME": "nope", "CKPT_ROOT": str(tmp_path)}
    p = subprocess.run(["bash", str(ROOT / "run/eval_all_metrics.sh")], env=env,
                       capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 2 and "no model weights" in p.stderr
