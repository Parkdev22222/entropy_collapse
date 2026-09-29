"""scripts/bench_lora_summary.py: per-case step timing out of verl logs."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bls", ROOT / "scripts" / "bench_lora_summary.py")
bls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bls)


def _log(tmp_path, name, steps, step_s, gen_s=100.0, mem=40.0, cr=False):
    lines = ["### run_grpo.sh  arm=grpo", "some noise"]
    for k in range(1, steps + 1):
        s = step_s * (3 if k == 1 else 1)  # warm-up step is slow and must not count
        lines.append(f"step:{k} - actor/entropy:0.5 - timing_s/gen:{gen_s} - "
                     f"timing_s/old_log_prob:20.0 - timing_s/update_actor:30.0 - "
                     f"timing_s/step:{s} - perf/max_memory_allocated_gb:{mem}")
    p = tmp_path / name
    p.write_text(("\r" if cr else "\n").join(lines) + "\n")
    return p


def test_steps_parsed_and_warmup_and_last_excluded(tmp_path):
    p = _log(tmp_path, "a.log", 6, 100.0)
    rows = bls.parse_log(p)
    assert [r["step"] for r in rows] == [1, 2, 3, 4, 5, 6]
    s = bls.summarise_case("full_tp4", [p], expected_steps=6)
    assert s["status"] == "ok"
    assert s["step_s"] == 100.0          # step 1 (300 s) is excluded
    assert s["gen_s"] == 100.0 and s["update_actor_s"] == 30.0
    assert s["mem_gb"] == 40.0
    assert s["box_steps_per_hour"] == 36.0


def test_concurrent_case_multiplies_throughput(tmp_path):
    logs = [_log(tmp_path, f"g{i}.log", 6, 200.0) for i in range(4)]
    s = bls.summarise_case("lora_1gpu_x4", logs, expected_steps=6)
    assert s["runs"] == 4
    assert s["box_steps_per_hour"] == 4 * 3600 / 200.0


def test_short_run_is_failed_not_averaged(tmp_path):
    good = _log(tmp_path, "g0.log", 6, 200.0)
    died = _log(tmp_path, "g1.log", 2, 200.0)
    s = bls.summarise_case("lora_1gpu_x4", [good, died], expected_steps=6)
    assert s["status"].startswith("FAILED")
    assert "g1.log" in s["status"]


def test_carriage_returns_are_lines(tmp_path):
    p = _log(tmp_path, "cr.log", 5, 50.0, cr=True)
    assert len(bls.parse_log(p)) == 5


def test_relative_to_baseline(tmp_path):
    a = bls.summarise_case("full_tp4", [_log(tmp_path, "a.log", 6, 100.0)], 6)
    b = bls.summarise_case("lora_1gpu_x4",
                           [_log(tmp_path, f"b{i}.log", 6, 200.0) for i in range(4)], 6)
    rows = bls.with_speedup([a, b], baseline="full_tp4")
    assert rows[0]["speedup"] == 1.0
    assert rows[1]["speedup"] == 2.0
