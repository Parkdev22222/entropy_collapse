"""scripts/make_supplement.py: what it removes, and what it refuses to ship."""
from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "make_supplement", ROOT / "scripts" / "make_supplement.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_accounts_hosts_and_tokens_are_redacted():
    text = ("uploading run to DSDSh/steer-f_2 at https://huggingface.co/DSDSh/x "
            "mfs#us-mo-1.runpod.net:9421 /workspace mail me@example.com "
            "token hf_abcdefghijklmnopqrstuvwxyz12 see github.com/Parkdev22222/repo")
    out, n = mod.redact(text)
    assert n >= 5
    assert mod.check("t", out) == []
    for s in ("DSDSh", "runpod", "example.com", "hf_abc", "Parkdev22222"):
        assert s not in out


def test_public_model_paths_are_kept():
    text = "load huggingface.co/Qwen/Qwen2.5-Math-1.5B and huggingface.co/meta-llama/x"
    out, _ = mod.redact(text)
    assert out == text


def test_a_surviving_identity_string_refuses_the_build(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "REDACTIONS", [])          # redaction disabled
    out = tmp_path / "s.zip"
    _, leaks = mod.build({"logs/a.log": "pushed by Parkdev22222"}, out)
    assert leaks and not out.exists()


def test_archive_layout_and_fixed_timestamps(tmp_path):
    out = tmp_path / "s.zip"
    redacted, leaks = mod.build({"logs/train-x.log": "step:1", "README.txt": "r"}, out)
    assert leaks == []
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert names == ["supplement/README.txt", "supplement/logs/train-x.log"]
        assert all(i.date_time == (1980, 1, 1, 0, 0, 0) for i in z.infolist())


def test_only_training_and_evaluation_logs_are_selected():
    assert mod.LOG_RE.match("train-grpo-Qwen2.5-Math-1.5B-s1.log")
    assert mod.LOG_RE.match("eval-r2-s2-step190.log")
    assert not mod.LOG_RE.match("paper_h100.log")
    assert not mod.LOG_RE.match("warmup-rollouts-Qwen2.5-Math-1.5B.log")
