"""A backbone changes several things at once, or it changes nothing safely.

run_uniform_ablation.sh:95 hardcodes the 1.5B MTP head path without consulting
the model tag, and its own guard at :149 asks only whether that FILE EXISTS --
which it does on every box that ever trained the 1.5B. So a 7B STEER-F run
would sail past the check and load a forecaster built for a 1536-wide model.
These tests pin the replacement: the tag, the weights path, the validation set,
the selection key and the heads travel together, and a mismatch refuses.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE_ENV = {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": "/tmp"}


def arms(snippet: str, **env):
    """Source _arms.sh with a clean environment and run a snippet."""
    e = dict(BASE_ENV)
    e.update({k: v for k, v in env.items() if v is not None})
    return subprocess.run(["bash", "-c", f'. "{ROOT}/run/_arms.sh"\n{snippet}'],
                          capture_output=True, text=True, cwd=str(ROOT), env=e)


PROFILES = [
    ("Qwen2.5-Math-1.5B", "Qwen/Qwen2.5-Math-1.5B", "aime24", "aime_2024_dapo_boxed/acc/mean@32"),
    ("Qwen2.5-Math-7B",   "Qwen/Qwen2.5-Math-7B",   "aime24", "aime_2024_dapo_boxed/acc/mean@32"),
    ("Llama-3.1-8B",      "meta-llama/Llama-3.1-8B", "math500", "math500/acc/mean@1"),
    ("Mistral-7B-v0.3",   "mistralai/Mistral-7B-v0.3", "math500", "math500/acc/mean@1"),
]


@pytest.mark.parametrize("tag,path,val,key", PROFILES)
def test_profile_moves_as_one(tag, path, val, key):
    """Whatever the backbone is, its weights, validation set and selection
    metric come from the same place, so they cannot drift apart."""
    p = arms('echo "$MODEL_PATH|$VAL_PARQUET|$BEST_METRIC_KEY"', MODEL_TAG=tag)
    got = p.stdout.strip().splitlines()[-1]
    assert got == f"{path}|datasets/{val}.parquet|val-core/{key}", p.stderr


def test_non_math_backbones_do_not_select_on_aime24():
    """30 problems put every arm inside one SE for a model that scores near
    zero, and save_best_only would then pick the checkpoint by that noise."""
    for tag, _, val, _ in PROFILES:
        p = arms('echo "$VAL_PARQUET"', MODEL_TAG=tag)
        got = p.stdout.strip().splitlines()[-1]
        if tag.startswith("Qwen2.5-Math"):
            assert "aime24" in got
        else:
            assert "math500" in got, f"{tag} still validates on {got}"


def test_unknown_tag_refuses_rather_than_guessing():
    """The old derivation turned MODEL_TAG=Llama-3.1-8B into
    'Qwen/Qwen2.5-Math-Llama-3.1-8B'."""
    p = arms("backbone_profile Gemma-2-9B", MODEL_TAG="Gemma-2-9B")
    assert p.returncode != 0
    assert "not a known backbone" in p.stdout + p.stderr


def test_unknown_tag_is_allowed_when_fully_specified():
    p = arms("backbone_profile Gemma-2-9B && echo FINE", MODEL_TAG="Gemma-2-9B",
             MODEL_PATH="google/gemma-2-9b", VAL_PARQUET="datasets/math500.parquet",
             BEST_METRIC_KEY="val-core/math500/acc/mean@1")
    assert "FINE" in p.stdout, p.stderr


# ------------------------------------------------------------------- heads
def test_heads_guard_refuses_another_models_forecaster(tmp_path):
    """THE bug: the 1.5B head file exists, so an existence check passes."""
    f = tmp_path / "mtp_heads_Qwen2.5-Math-1.5B-paper.pt"
    f.write_bytes(b"not really a checkpoint")
    p = arms("heads_guard", MODEL_TAG="Qwen2.5-Math-7B", STEERF_HEADS=str(f))
    assert p.returncode != 0
    assert "not Qwen2.5-Math-7B's forecaster" in p.stdout + p.stderr


def test_heads_guard_refuses_a_missing_file(tmp_path):
    p = arms("heads_guard", MODEL_TAG="Qwen2.5-Math-7B",
             STEERF_HEADS=str(tmp_path / "nope.pt"))
    assert p.returncode != 0
    assert "no MTP heads" in p.stdout + p.stderr


def test_heads_guard_accepts_the_matching_name(tmp_path):
    """Name matches and the deep check cannot run here (no transformers), so
    it must say so and pass rather than block a correct run."""
    f = tmp_path / "mtp_heads_Qwen2.5-Math-7B.pt"
    f.write_bytes(b"x")
    p = arms("heads_guard", MODEL_TAG="Qwen2.5-Math-7B", STEERF_HEADS=str(f))
    assert p.returncode == 0, p.stdout + p.stderr


@pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None,
                    reason="torch not installed here")
def test_check_heads_detects_a_width_mismatch(tmp_path):
    """A right-looking name over the wrong tensors. Needs transformers to read
    the model config; without it the checker must SKIP, not pass silently."""
    import torch
    heads = tmp_path / "mtp_heads_Fake-7B.pt"
    torch.save({"h0.weight": torch.zeros(8, 1536)}, heads)
    model = tmp_path / "Fake-7B"
    model.mkdir()
    (model / "config.json").write_text(
        '{"model_type":"qwen2","hidden_size":3584,"architectures":["Qwen2ForCausalLM"]}')
    p = subprocess.run([sys.executable, str(ROOT / "run" / "_check_heads.py"),
                        str(heads), str(model)], capture_output=True, text=True)
    out = p.stdout + p.stderr
    if __import__("importlib").util.find_spec("transformers") is None:
        assert p.returncode == 0 and "SKIP" in out, out
    else:
        assert p.returncode == 1, out
        assert "1536" in out and "3584" in out


# ------------------------------------------------------------------ queue
def dry(**env):
    e = dict(BASE_ENV); e["DRY"] = "1"; e.update(env)
    return subprocess.run(["bash", str(ROOT / "run" / "run_backbones.sh")],
                          capture_output=True, text=True, cwd=str(ROOT), env=e)


def test_queue_is_three_arms_per_backbone():
    p = dry()
    assert p.returncode == 0, p.stderr
    assert p.stdout.count("QUEUE") == 9, p.stdout


def test_queue_honours_an_arm_subset():
    p = dry(ARMS="grpo signed")
    assert p.stdout.count("QUEUE") == 6, p.stdout


def test_run_names_stay_distinct_across_backbones():
    """A collision would make one backbone's log mark another's run done."""
    names = set()
    for tag, *_ in PROFILES:
        for arm in ("grpo", "steer", "signed"):
            p = arms(f'run_name_for {arm} 1', MODEL_TAG=tag)
            n = p.stdout.strip().splitlines()[-1]
            assert n not in names, f"{n} is claimed twice"
            names.add(n)
    assert len(names) == len(PROFILES) * 3


def test_an_unknown_backbone_stops_the_queue():
    p = dry(BACKBONES="Qwen2.5-Math-7B Gemma-2-9B")
    assert p.returncode != 0
    assert "unknown backbone" in p.stdout + p.stderr
