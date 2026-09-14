#!/usr/bin/env python3
"""Do these MTP heads fit this model?

The forecaster is trained per backbone: its heads read the base model's hidden
states, so their width is the model's ``hidden_size``.  Handing a 1.5B model's
heads to a 7B one is not a degraded run, it is a different method.

``run_uniform_ablation.sh:95`` hardcodes the 1.5B head path without consulting
the model tag, and its guard at :149 asks only whether the file EXISTS -- which
it does, on every box that ever trained the 1.5B.  So the check passes and the
wrong forecaster is loaded.  This compares the tensors against the model.

Usage:  python3 run/_check_heads.py <heads.pt> <model path or name>
        exit 0 = fits, or cannot be determined here (says which)
        exit 1 = definitely does not fit
"""
import sys

GREEN, YELLOW, RED, OFF = "\033[32m", "\033[33m", "\033[31m", "\033[0m"


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    heads, model = sys.argv[1], sys.argv[2]
    try:
        import torch
        from transformers import AutoConfig
    except Exception as exc:
        print(f"  {YELLOW}SKIP{OFF}  cannot check head shapes here "
              f"({type(exc).__name__}); the name check still applied",
              file=sys.stderr)
        return 0
    try:
        sd = torch.load(heads, map_location="cpu", weights_only=True)
    except Exception as exc:
        print(f"  {RED}FAIL{OFF}  {heads} does not load: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    if not isinstance(sd, dict) or not sd:
        print(f"  {YELLOW}SKIP{OFF}  {heads} is not a plain state dict; "
              f"not checking shapes", file=sys.stderr)
        return 0
    try:
        want = AutoConfig.from_pretrained(model).hidden_size
    except Exception as exc:
        # Offline, or the weights are not cached yet. Say so rather than
        # pretending the check ran -- a silent pass is what put us here.
        print(f"  {YELLOW}SKIP{OFF}  cannot read {model}'s config "
              f"({type(exc).__name__}); the name check still applied",
              file=sys.stderr)
        return 0
    widths = {tuple(v.shape)[-1] for v in sd.values()
              if hasattr(v, "shape") and len(getattr(v, "shape", ())) >= 1}
    if widths and want not in widths:
        print(f"  {RED}FAIL{OFF}  these heads are for another model.",
              file=sys.stderr)
        print(f"          heads carry width(s) {sorted(widths)}", file=sys.stderr)
        print(f"          {model} has hidden_size={want}", file=sys.stderr)
        print(f"""
        MTP heads are trained against one backbone's hidden states. Build this
        model's own:

            MODEL_TAG={model.split('/')[-1]} bash run/collect_warmup_rollouts.sh
            MODEL_TAG={model.split('/')[-1]} bash run/warmup_and_validate.sh
""", file=sys.stderr)
        return 1
    print(f"  {GREEN}OK{OFF}    heads match {model} (hidden_size={want})",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
