#!/usr/bin/env python3
"""Fold a LoRA run's adapter into its base model, for evaluation.

    python3 scripts/merge_lora.py --base Qwen/Qwen2.5-Math-1.5B \
        --adapter checkpoints/STEER-F/<run>/global_step_110/actor/lora_adapter \
        --out     checkpoints/STEER-F/<run>/global_step_110/actor/merged

WHY. Under LoRA, verl's actor/huggingface/ holds the FSDP state dict of a
PeftModel saved through a plain save_pretrained: keys like
``base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight``, nothing
merged. It still has a config.json, so every loader accepts it, maps none of
its tensors, initialises the model at random and scores near zero without an
error. The evaluable copy is actor/lora_adapter/ (standard PEFT names and
adapter_config.json), which verl writes at every save; this merges it.

CHECKS, because a merge that loads nothing is the same silent failure:
  - every tensor in the adapter file must reach the model unchanged (compared
    against PEFT's own view of the loaded adapter, key for key);
  - an adapter whose B matrices are all zero is the untrained initial state --
    refused, since evaluating it evaluates the base model.
A stamp next to the merged weights records the adapter's hash, so a re-run with
the same adapter is skipped and a changed adapter is re-merged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

STAMP = "merge_stamp.json"


def adapter_digest(adapter: Path) -> str:
    h = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        with open(adapter / name, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def is_current(out: Path, digest: str, base: str) -> bool:
    """A merged directory is reusable only if it came from this adapter and base."""
    try:
        stamp = json.loads((out / STAMP).read_text())
    except (OSError, ValueError):
        return False
    has_weights = any(out.glob("*.safetensors"))
    return has_weights and stamp.get("adapter_sha256") == digest and stamp.get("base") == base


def _peft_key(k: str) -> str:
    """The adapter file's key for a PEFT state-dict key (drop the adapter name)."""
    return k.replace(".default", "")


def merge(base: str, adapter: Path, out: Path) -> dict:
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    saved = load_file(str(adapter / "adapter_model.safetensors"))
    if not saved:
        raise SystemExit(f"{adapter}: adapter file holds no tensors")
    b_max = max((float(t.abs().max()) for k, t in saved.items() if "lora_B" in k), default=0.0)
    if b_max == 0.0:
        raise SystemExit(f"{adapter}: every lora_B is zero -- the untrained initial adapter. "
                         "Evaluating it would evaluate the base model.")

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(adapter))
    loaded = {_peft_key(k): v for k, v in get_peft_model_state_dict(model).items()}
    missing = sorted(set(saved) - set(loaded))
    if missing:
        raise SystemExit(f"{adapter}: {len(missing)} adapter tensor(s) did not reach the model, "
                         f"e.g. {missing[:3]}")
    for k, t in saved.items():
        if not torch.equal(loaded[k].to(t.dtype).cpu(), t.cpu()):
            raise SystemExit(f"{adapter}: tensor {k} changed on load")

    merged = model.merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(str(out))
    return {"n_adapter_tensors": len(saved), "max_abs_lora_B": b_max}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="the base model the run trained from")
    p.add_argument("--adapter", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)

    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (args.adapter / name).is_file():
            print(f"[merge] {args.adapter}/{name} missing", file=sys.stderr)
            return 1
    digest = adapter_digest(args.adapter)
    if is_current(args.out, digest, args.base):
        print(f"[merge] up to date: {args.out}")
        return 0
    info = merge(args.base, args.adapter, args.out)
    (args.out / STAMP).write_text(json.dumps(
        {"adapter_sha256": digest, "base": args.base, "adapter": str(args.adapter), **info},
        indent=2) + "\n")
    print(f"[merge] {args.adapter} -> {args.out} ({info['n_adapter_tensors']} tensors, "
          f"max|B|={info['max_abs_lora_B']:.3g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
