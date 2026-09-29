"""scripts/merge_lora.py: the parts that decide whether to merge (no torch needed)."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("merge_lora", ROOT / "scripts" / "merge_lora.py")
ml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml)


def _adapter(tmp_path, payload=b"weights"):
    a = tmp_path / "lora_adapter"
    a.mkdir()
    (a / "adapter_config.json").write_text('{"r": 64}')
    (a / "adapter_model.safetensors").write_bytes(payload)
    return a


def test_digest_changes_with_the_adapter(tmp_path):
    a = _adapter(tmp_path)
    d1 = ml.adapter_digest(a)
    (a / "adapter_model.safetensors").write_bytes(b"other")
    assert ml.adapter_digest(a) != d1


def test_merged_dir_is_reused_only_for_the_same_adapter_and_base(tmp_path):
    out = tmp_path / "merged"
    out.mkdir()
    assert not ml.is_current(out, "abc", "base")                  # no stamp
    (out / ml.STAMP).write_text(json.dumps({"adapter_sha256": "abc", "base": "base"}))
    assert not ml.is_current(out, "abc", "base")                  # no weights
    (out / "model.safetensors").write_bytes(b"x")
    assert ml.is_current(out, "abc", "base")
    assert not ml.is_current(out, "abd", "base")
    assert not ml.is_current(out, "abc", "other")


def test_peft_key_drops_the_adapter_name():
    k = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    assert ml._peft_key(k) == "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"


def test_missing_adapter_file_is_an_error(tmp_path):
    a = tmp_path / "lora_adapter"
    a.mkdir()
    assert ml.main(["--base", "b", "--adapter", str(a), "--out", str(tmp_path / "m")]) == 1
