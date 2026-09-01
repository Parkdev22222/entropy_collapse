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
from 0.003 to "order 0.5" once rollouts were sampled as a tree.  The 1.5B run
reports 0.007 -> 0.011 instead, which reads as the tree having failed.  It did
not; the target was unreachable, and 0.011 is itself above what a real count
can produce.

`H_togo` is a deterministic function of the causal prefix, so two rollouts that
share `y_<=t` carry *identical* forecasts.  `sibling_prefix_baseline` averages
over the rollouts sharing `y_<t`, so

    A_H[i, t] != 0  <=>  siblings share y_<t but diverge at y_t
                    <=>  t is a branch point of the group

and inside a tree, sharing a prefix means sharing every token up to the next
cut.  `A_H` is therefore identically zero across the support region except at
the cuts.  Counting nonzero `(rollout, position)` entries gives the sum of
subtree sizes over the internal nodes of the group's trie, which for `n` leaves
is at most `n + (n-1) + ... + 2`, so

    branch_corr_frac <= (n(n+1)/2 - 1) / (n * T)     for any sampler whatsoever.

This script builds the ideal tree -- tokens and forecasts made exact functions
of the sub-trunk, the best case the sampler can produce -- and counts the live
positions directly, so the bound can be read off rather than argued about.

It also prints the threshold sweep behind `steerf/branch_corr_frac_strict`
(`branch_weight_correction` tests `visit != 0`, and the float32 round-off in
the sibling mean leaves `|A_H| ~ 6e-8` at interpretively-zero positions), and a
demonstration that `steerf/branch_recall` is pinned at ~0.5 by the sign of a
zero-centred score and so cannot distinguish a real forecast from noise.

Usage
-----
    python3 scripts/measure_ah_support.py
    python3 scripts/measure_ah_support.py --n 8 --seq-len 1005 --cuts 64,192,384
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steer_f.entropy_forecast import entropy_advantage  # noqa: E402
from steer_f.monitors import _top_k_selection  # noqa: E402

# Reported by the tree run this script explains: train-steer-f-Qwen2.5-Math-1.5B
# -s1-tree-rollout, n=8, cuts 64,192,384, response_length/mean ~= 1005, lam=0.25.
MEASURED_FLAT = 0.003
MEASURED_TREE = 0.011


def build_ideal_tree(n: int, seq_len: int, cuts: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokens and forecasts as exact functions of the sub-trunk a rollout is in.

    The best case: every rollout below a cut shares its prefix with all the
    others under the same parent and nothing else leaks in.  Real rollouts add
    refills and early-EOS trunks, which raise the live count a little.
    """
    sizes = [n]
    for _ in cuts:
        sizes.append(max(1, sizes[-1] // 2))

    def sub_trunk(i: int, t: int) -> tuple[int, int]:
        seg = sum(t >= c for c in cuts)
        return seg, i // sizes[seg]

    responses = torch.zeros(n, seq_len, dtype=torch.long)
    h_togo_vals = torch.zeros(n, seq_len)
    for i in range(n):
        for t in range(seq_len):
            key = sub_trunk(i, t)
            responses[i, t] = (hash((key, t)) % 30000) + 1
            h_togo_vals[i, t] = ((hash((key, t, "h")) % 10007) / 10007.0) * 0.7
    return responses, h_togo_vals


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=8, help="rollout.n / group size")
    ap.add_argument("--seq-len", type=int, default=1005, help="T; the run measured ~1005")
    ap.add_argument("--cuts", type=str, default="64,192,384", help="steerf_tree_depths")
    ap.add_argument("--a-h-std", type=float, default=0.009, help="steerf/a_h_std from the run")
    ap.add_argument("--h-togo-mean", type=float, default=0.211, help="steerf/h_togo_mean from the run")
    args = ap.parse_args()

    cuts = [int(x) for x in args.cuts.split(",") if x.strip()]
    n, seq_len = args.n, args.seq_len
    if cuts and cuts[-1] >= seq_len:
        print(f"error: deepest cut {cuts[-1]} is past T={seq_len}", file=sys.stderr)
        return 2

    responses, h_togo_vals = build_ideal_tree(n, seq_len, cuts)
    mask = torch.ones(n, seq_len)
    group_index = [0] * n

    a_h = entropy_advantage(h_togo_vals, group_index, mask, responses=responses, baseline="sibling")
    mag = a_h.abs()
    peak = float(mag.max())

    print("=" * 74)
    print(f"ideal tree: n={n}  T={seq_len}  cuts={cuts}")
    print("=" * 74)
    print("  threshold (rel. to peak)   nonzero_frac   positions/rollout")
    for rel in (0.0, 1e-8, 1e-6, 1e-4, 1e-2):
        sel = mag > rel * peak if rel else mag != 0
        frac = sel.float().mean().item()
        label = "!= 0" if rel == 0 else f"> {rel:.0e} * peak"
        print(f"    {label:<22s} {frac:9.4f}      {frac * seq_len:7.1f}")

    live = mag > 1e-4 * peak
    cols = sorted(set(live.any(dim=0).nonzero().flatten().tolist()))
    frac = live.float().mean().item()
    print(f"\n  live columns              {cols}")
    print(f"  cut depths                {cuts}")
    print("  -> A_H is nonzero at the cuts and nowhere else")

    noise = mag[(mag > 0) & (mag <= 1e-4 * peak)]
    if noise.numel():
        print(f"\n  round-off entries         {noise.numel()} of {n * seq_len} "
              f"({noise.numel() / (n * seq_len):.4f}), max {noise.max().item():.2e}")
        print("  -> counted as live by `visit != 0` in branch_weight_correction")

    ceiling = (n * (n + 1) / 2 - 1) / (n * seq_len)
    print("\n" + "=" * 74)
    print("the ceiling")
    print("=" * 74)
    print(f"  balanced tree, counted    {frac:.4f}")
    print(f"  ceiling (n(n+1)/2-1)/nT   {ceiling:.4f}")
    print(f"  flat sampler, measured    {MEASURED_FLAT:.4f}   ({MEASURED_FLAT / ceiling:.0%} of ceiling)")
    print(f"  tree, measured            {MEASURED_TREE:.4f}   ({MEASURED_TREE / ceiling:.0%} of ceiling)")
    if MEASURED_TREE > ceiling:
        share = 1 - ceiling / MEASURED_TREE
        print(f"  -> measured EXCEEDS the ceiling: >= {share:.0%} of the reported")
        print("     support is float32 round-off. Read branch_corr_frac_strict.")
        print(f"  -> rms is deflated by sqrt({ceiling:.4f}/{MEASURED_TREE:.3f}) = "
              f"{(ceiling / MEASURED_TREE) ** 0.5:.2f}, inflating every tanh")
        print(f"     argument by {(MEASURED_TREE / ceiling) ** 0.5:.2f}x -- a training effect, not logging.")

    p = max(frac, 1e-12)
    sigma_live = args.a_h_std / p ** 0.5
    print("\n" + "=" * 74)
    print("what the sparsity does NOT mean")
    print("=" * 74)
    print(f"  a_h_std over ALL valid positions ({1 - frac:.1%} exactly zero)   {args.a_h_std:.4f}")
    print(f"  conditional spread at branch points  a_h_std/sqrt(p)  = {sigma_live:.4f}")
    print(f"  against h_togo_mean = {args.h_togo_mean:.3f}                     "
          f"{sigma_live / args.h_togo_mean:.0%}")
    print("  -> sibling futures differ a great deal where the term is defined.")
    print("     C5 corollary 1's 'near-equally diverse' branch is rejected:")
    print("     the visitation term is sparse, not small.")

    ag = entropy_advantage(h_togo_vals, group_index, mask, baseline="group")
    print(f"\n  baseline='group' would give {(ag.abs() > 1e-4 * peak).float().mean().item():.4f} coverage,")
    print("  at the cost of comparing against rollouts that do not share the prefix.")

    recall_degeneracy(p)
    print("=" * 74)
    return 0


def recall_degeneracy(p: float, n: int = 200_000, top_frac: float = 0.1, seed: int = 0) -> None:
    """`branch_recall` is pinned at ~0.5 by the sign of a zero-centred score.

    `_top_k_selection` ranks the *signed* `a_h`.  `A_H` is a deviation from a
    sibling mean, so at branch points it is zero-centred by construction: the
    positive half sorts above the mass of exact zeros and the negative half
    below it.  Recall is then the positive half, ~0.5, whatever the forecast
    says -- so the metric cannot tell an informative score from noise.

    Ranking `|a_h|` does not fix it either: inside the support `a_h != 0` iff
    the position is a branch point, so `|a_h|` identifies them perfectly by
    definition and recall becomes 1.0.
    """
    torch.manual_seed(seed)
    n_branch = max(1, int(n * p))
    k = max(1, int(round(n * top_frac)))
    print("\n" + "=" * 74)
    print(f"branch_recall is degenerate (p={p:.4f}, top_frac={top_frac})")
    print("=" * 74)
    print("  score at branch points              recall    lift")
    for label, scale in (("informative a_h", 1.0), ("PURE NOISE, zero-centred", 1.0),
                         ("a_h scaled 100x", 100.0)):
        a = torch.zeros(n)
        idx = torch.randperm(n)[:n_branch]
        a[idx] = torch.randn(n_branch) * scale
        is_branch = torch.zeros(n, dtype=torch.bool)
        is_branch[idx] = True
        rec = int((is_branch & _top_k_selection(a, k)).sum()) / n_branch
        print(f"    {label:<33s} {rec:.4f}   {rec / top_frac:.2f}")
        if label.startswith("informative"):
            abs_rec = int((is_branch & _top_k_selection(a.abs(), k)).sum()) / n_branch
            print(f"    {'(same score, ranked by |a_h|)':<33s} {abs_rec:.4f}   "
                  f"{abs_rec / top_frac:.2f}   <- trivially 1.0")
    print("  training log: branch_recall 0.497 -> 0.498, lift 4.98, flat over 79 steps")
    print("  -> the metric is uninformative in BOTH directions. Use Part B of")
    print("     scripts/phase1_sibling_spread.py to test the forecast instead.")


if __name__ == "__main__":
    raise SystemExit(main())
