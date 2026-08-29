#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Count the positions where `A_H` is actually nonzero, on an ideal tree batch.

Why this script exists
----------------------
`docs/STEERF_tree_rollout.md` originally asked `steerf/branch_corr_frac` to go
from 0.003 to "order 0.5" once rollouts were sampled as a tree.  The training
run reports 0.008 -> 0.010 instead, which reads as the tree having failed.  It
did not.  The target was unreachable.

`H_togo` is a deterministic function of the causal prefix, so two rollouts that
share `y_<=t` carry *identical* forecasts.  `A_H[i,t]` is `i`'s forecast minus
the mean over the rollouts sharing `y_<t`, so

    A_H[i, t] != 0  <=>  siblings share y_<t but diverge at y_t
                    <=>  t is a branch point of the group

and inside a tree, sharing a prefix means sharing every token up to the next
cut.  `A_H` is therefore identically zero across the whole support region
except at the cuts.  Since `n` sequences form a trie with at most `n - 1`
internal branching nodes, and `entropy_forecast._shift_forward` widens each
into two columns,

    branch_corr_frac <= 2 (n - 1) / T          for any sampler whatsoever.

This script builds the ideal tree -- tokens and forecasts made exact functions
of the sub-trunk, which is the best case the sampler can produce -- and counts
the live positions directly, so the bound can be read off rather than argued
about.

It also prints the threshold sweep behind the `1e-8 -> 1e-6` change in
`steer_f/verl_integration.py`: the float32 round-off in the sibling mean leaves
`|a_h| ~ 6e-8` at positions that are interpretively zero, and the old test
counted those as live.

Usage
-----
    python3 scripts/measure_ah_support.py
    python3 scripts/measure_ah_support.py --n 8 --seq-len 1000 --cuts 64,192,384
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steer_f.entropy_forecast import entropy_advantage  # noqa: E402

# Reported by the tree run this script explains (train-math-Qwen2.5-Math-1.5B-s1,
# n=8, cuts 64,192,384, response_length/mean ~= 1000).
MEASURED_FLAT = 0.003
MEASURED_TREE = 0.010


def build_ideal_tree(n: int, seq_len: int, cuts: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokens and forecasts as exact functions of the sub-trunk a rollout is in.

    This is the best case: every rollout below a cut shares its prefix with all
    the others under the same parent, and nothing else leaks in.  Real rollouts
    only add refills and early-EOS trunks, both of which *raise* the live count
    slightly -- which is why the training run reads 0.010 against the 0.006 this
    produces.
    """
    sizes = [n]                       # rollouts per sub-trunk in each segment
    for c in cuts:
        del c
        sizes.append(max(1, sizes[-1] // 2))

    def sub_trunk(i: int, t: int) -> tuple[int, int]:
        """(segment index, sub-trunk index) for rollout i at position t."""
        seg = sum(t >= c for c in cuts)
        return seg, i // sizes[seg]

    responses = torch.zeros(n, seq_len, dtype=torch.long)
    h_togo = torch.zeros(n, seq_len)
    for i in range(n):
        for t in range(seq_len):
            key = sub_trunk(i, t)
            responses[i, t] = (hash((key, t)) % 30000) + 1
            h_togo[i, t] = ((hash((key, t, "h")) % 10007) / 10007.0) * 0.7
    return responses, h_togo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=8, help="rollout.n / group size")
    ap.add_argument("--seq-len", type=int, default=1000, help="T; the run measured ~1000")
    ap.add_argument("--cuts", type=str, default="64,192,384", help="steerf_tree_depths")
    ap.add_argument("--a-h-std", type=float, default=0.009, help="steerf/a_h_std from the run")
    ap.add_argument("--h-togo-mean", type=float, default=0.351, help="steerf/h_togo_mean from the run")
    ap.add_argument("--lam", type=float, default=0.25, help="STEERF_LAM, for the amplification line")
    args = ap.parse_args()

    cuts = [int(x) for x in args.cuts.split(",") if x.strip()]
    n, seq_len = args.n, args.seq_len
    if cuts and cuts[-1] >= seq_len:
        print(f"error: deepest cut {cuts[-1]} is past T={seq_len}", file=sys.stderr)
        return 2

    responses, h_togo = build_ideal_tree(n, seq_len, cuts)
    mask = torch.ones(n, seq_len)

    a_h, _ = entropy_advantage(h_togo, response_ids=responses, group_size=n,
                               response_mask=mask, baseline="sibling")
    mag = a_h.abs()

    print("=" * 74)
    print(f"ideal tree: n={n}  T={seq_len}  cuts={cuts}")
    print("=" * 74)
    print("  threshold   nonzero_frac   positions/rollout")
    for thr in (1e-8, 1e-7, 1e-6, 1e-4, 1e-2):
        frac = (mag >= thr).float().mean().item()
        print(f"    {thr:<9.0e} {frac:8.4f}      {frac * seq_len:7.1f}")

    live = mag >= 1e-6
    cols = sorted(set(live.any(dim=0).nonzero().flatten().tolist()))
    frac = live.float().mean().item()
    print(f"\n  live columns (>=1e-6)          {cols}")
    print(f"  cut depths                     {cuts}")
    print("  -> A_H is nonzero at the cuts and nowhere else "
          "(two columns per cut: _shift_forward)")

    noise = mag[(mag > 0) & (mag < 1e-6)]
    if noise.numel():
        print(f"\n  sub-1e-6 entries               {noise.numel()} of {n * seq_len} "
              f"({noise.numel() / (n * seq_len):.4f}), max {noise.max().item():.2e}")
        print("  -> float32 round-off in the sibling mean; the old `< 1e-8` zero test")
        print("     in verl_integration.py counted these as live signal.")

    ceiling = 2 * (n - 1) / seq_len
    print("\n" + "=" * 74)
    print("the ceiling")
    print("=" * 74)
    print(f"  ideal tree, counted here       {frac:.4f}   (2L/T, L={len(cuts)})")
    print(f"  ceiling 2(n-1)/T               {ceiling:.4f}   (a trie on n leaves has "
          f"<= n-1 branch nodes)")
    print(f"  flat i.i.d. sampler, measured  {MEASURED_FLAT:.4f}   "
          f"({MEASURED_FLAT / ceiling:.0%} of ceiling)")
    print(f"  tree, measured                 {MEASURED_TREE:.4f}   "
          f"({MEASURED_TREE / ceiling:.0%} of ceiling)")
    print("  -> the tree worked. The ceiling is just small, and 0.5 is not on it.")

    p = max(frac, 1e-12)
    sigma_live = args.a_h_std / p ** 0.5
    print("\n" + "=" * 74)
    print("what the sparsity does NOT mean")
    print("=" * 74)
    print(f"  a_h_std (over ALL valid positions, {1 - frac:.1%} of them exactly zero)")
    print(f"                                 {args.a_h_std:.4f}")
    print(f"  conditional spread at branch points  a_h_std / sqrt(p) = {sigma_live:.4f}")
    print(f"  against h_togo_mean = {args.h_togo_mean:.3f}          "
          f"{sigma_live / args.h_togo_mean:.0%}")
    print("  -> sibling futures differ a great deal where the term is defined.")
    print("     C5 corollary 1's 'futures are near-equally diverse' branch is rejected:")
    print("     the visitation term is sparse, not small.")
    print(f"\n  omega_tilde.normalize rescales to unit variance, amplifying live")
    print(f"  entries by 1/sqrt(p) = {1 / p ** 0.5:.1f}; at lam={args.lam} the future term is")
    print(f"  worth ~{args.lam / p ** 0.5:.1f} z-units at branch points and a constant elsewhere.")

    ag, _ = entropy_advantage(h_togo, response_ids=responses, group_size=n,
                              response_mask=mask, baseline="group")
    print(f"\n  baseline='group' (Ablation A5) would give "
          f"{(ag.abs() >= 1e-6).float().mean().item():.4f} coverage,")
    print("  at the cost of comparing against rollouts that do not share the prefix.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
