"""Which LoRA adapter a vLLM generate call must use.

verl registers a fresh adapter with vLLM at every weight sync
(``fsdp_vllm.py: update_params -> llm_engine.add_lora``) under an id taken from
``time.time_ns() % 0x7FFFFFFF``, and vLLM's CPU LoRA cache holds ``max_loras``
of them (verl builds the engine with ``max_loras=1``). So after a sync there is
exactly one adapter, and it is the current policy.

Upstream verl reads ``list_loras()`` and, when it is empty, generates with
``lora_request=None`` -- the base model -- without a word. That happens for real:
with ``load_format=dummy_*`` the first sync loads only the base weights and
registers no adapter, so the first rollout after a resume comes from the base
model. More than one adapter would mean the cache kept a stale policy and the
ids, being time modulo 2**31, cannot say which one is newest. Both are errors
here, not fallbacks.
"""
from __future__ import annotations

from typing import Iterable, Optional


def active_lora_id(lora_enabled: bool, registered: Iterable[int]) -> Optional[int]:
    """The id of the adapter to generate with, or None when LoRA is off."""
    if not lora_enabled:
        return None
    ids = list(registered)
    if len(ids) != 1:
        raise RuntimeError(
            f"LoRA is enabled but vLLM holds {len(ids)} adapter(s) {ids}; expected exactly "
            "one. None means generation would silently use the base model (use "
            "rollout.load_format=safetensors so the first sync registers the adapter); "
            "several mean a stale policy is still cached (max_loras must be 1).")
    return ids[0]
