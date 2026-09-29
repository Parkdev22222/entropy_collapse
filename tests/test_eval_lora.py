"""run/eval_lora.sh: which checkpoint each evaluation point resolves to."""
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(tmp_path, *args, dry=True):
    env = dict(os.environ, CKPT_ROOT=str(tmp_path / "ckpt"), LOG_ROOT=str(tmp_path / "logs"),
               DRY="1" if dry else "0")
    return subprocess.run(["bash", str(ROOT / "run/eval_lora.sh"), *args],
                          capture_output=True, text=True, env=env)


def make_run(tmp_path, arm="signed", seed=1, steps=(50, 70, 100, 110), best=70):
    name = subprocess.run(["bash", "-c", f"cd {ROOT}; . run/_arms.sh; . run/_lora_arms.sh; "
                                         f"lora_run_name {arm} {seed}"],
                          capture_output=True, text=True).stdout.strip()
    d = tmp_path / "ckpt" / name
    for s in steps:
        a = d / f"global_step_{s}" / "actor" / "lora_adapter"
        a.mkdir(parents=True)
        (a / "adapter_model.safetensors").write_bytes(b"x")
        (a / "adapter_config.json").write_text("{}")
    if best is not None:
        (d / "best_checkpoint_info.json").write_text(json.dumps({"best_checkpoint_step": best}))
    return d


def test_final_is_the_arms_last_step_not_the_best(tmp_path):
    make_run(tmp_path)
    r = run(tmp_path, "signed", "1", "final")
    assert r.returncode == 0, r.stderr
    assert "step 110" in r.stdout and "global_step_110/actor/lora_adapter" in r.stdout
    assert "eval-final-k32/eval-signed-s1.log" in r.stdout


def test_best_reads_the_json(tmp_path):
    make_run(tmp_path)
    r = run(tmp_path, "signed", "1", "best")
    assert r.returncode == 0, r.stderr
    assert "step 70" in r.stdout and "eval-best-k32" in r.stdout


def test_best_without_json_fails(tmp_path):
    make_run(tmp_path, best=None)
    r = run(tmp_path, "signed", "1", "best")
    assert r.returncode == 1 and "best_checkpoint_info.json missing" in r.stderr


def test_missing_adapter_fails_even_dry(tmp_path):
    make_run(tmp_path, steps=(50, 100))
    r = run(tmp_path, "signed", "1", "final")
    assert r.returncode == 1 and "no adapter" in r.stderr


def test_grpo_long_final_is_step_200(tmp_path):
    make_run(tmp_path, arm="grpo-long", steps=(200,), best=None)
    r = run(tmp_path, "grpo-long", "1", "final")
    assert r.returncode == 0, r.stderr
    assert "step 200" in r.stdout


def test_bad_point_and_arm(tmp_path):
    assert run(tmp_path, "signed", "1", "middle").returncode == 2
    assert run(tmp_path, "nosuch", "1", "final").returncode == 2
