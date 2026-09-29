"""LoRA reaches every vLLM generate call, the tree's included, or the run stops."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lora_request", ROOT / "steer_f" / "lora_request.py")
lr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lr)
SPMD = (ROOT / "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py").read_text()


def test_off_means_no_adapter():
    assert lr.active_lora_id(False, []) is None
    assert lr.active_lora_id(False, [7, 8]) is None


def test_on_returns_the_one_adapter():
    assert lr.active_lora_id(True, iter([42])) == 42


@pytest.mark.parametrize("ids", [[], [1, 2]])
def test_on_without_exactly_one_adapter_raises(ids):
    with pytest.raises(RuntimeError, match="expected exactly"):
        lr.active_lora_id(True, ids)


def test_tree_rollout_no_longer_refuses_lora_and_passes_the_adapter():
    assert "does not support LoRA" not in SPMD
    tree = SPMD[SPMD.index("def _generate_tree"):SPMD.index("result = generate_tree(")]
    assert "lora_request=None if lora_id is None else" in tree
    assert "* len(prompt_token_ids)" in tree
    assert "self._generate_tree(vllm_inputs, lora_int_id)" in SPMD


def test_flat_path_uses_the_checked_id():
    assert "active_lora_id(bool(self.lora_kwargs)" in SPMD
    assert "if len(lora_int_ids) > 0" not in SPMD


def test_adapter_save_failure_is_not_swallowed():
    w = (ROOT / "verl/workers/fsdp_workers.py").read_text()
    block = w[w.index('Save LoRA Adapter Error'):w.index('Saved LoRA adapter to')]
    assert "raise" in block
