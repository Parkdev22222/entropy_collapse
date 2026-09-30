"""run/bench_lora.sh: the cases that decide the campaign's topology and vLLM
setting launch what they say they do (DRY, no GPU)."""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def bench(tmp_path, **env):
    e = dict(os.environ, DRY="1", BENCH_DIR=str(tmp_path / "bench"), **env)
    return subprocess.run(["bash", str(ROOT / "run/bench_lora.sh")], cwd=ROOT,
                          capture_output=True, text=True, env=e, timeout=120)


def test_signed_times_the_tree_arm_and_graph_cases_turn_graphs_on(tmp_path):
    r = bench(tmp_path, BENCH_ARM="signed",
              BENCH_CASES="lora_tp4 lora_tp4_graph lora_1gpu_x4_graph")
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    # the x4 trainers start in the background, so their DRY prints interleave:
    # count lines rather than parse blocks
    assert out.count(" arm            signed") == 6
    assert out.count("trainer.test_freq=-1") == 6
    assert out.count("lora_rank=64") == 6
    assert out.count("enforce_eager=False") == 1 + 4
    serial_eager = out[:out.index("[bench] lora_tp4_graph:")]      # lora_tp4 runs first, alone
    assert "arm            signed" in serial_eager and "enforce_eager" not in serial_eager
    assert out.count("alone, tp=1") == 4
    assert out.count("(tp=1)") == 4


def test_grpo_stays_the_default(tmp_path):
    r = bench(tmp_path, BENCH_CASES="lora_tp4")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "arm            signed" not in r.stdout
    assert "enforce_eager=False" not in r.stdout


def test_bad_arm_and_missing_baseline_log_are_refused(tmp_path):
    assert bench(tmp_path, BENCH_ARM="uniform").returncode == 2
    assert bench(tmp_path, BENCH_BASELINE_LOG=str(tmp_path / "nope.log")).returncode == 2
