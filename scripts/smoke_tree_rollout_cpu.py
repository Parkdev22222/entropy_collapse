#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""CPU smoke of `steer_f/tree_rollout.py` -- no GPU, no model, no dataset.

    python scripts/smoke_tree_rollout_cpu.py

The fake engine is a first-order Markov sampler over a small vocabulary with a
per-token stop probability, so continuations genuinely diverge and trunks
genuinely die early; the tree driver cannot tell it from vLLM.

What is actually being proven, in order of how much it matters:

1. `sibling_support_stats` agrees, exactly, with the pairwise
   `first_divergence` rule that `entropy_forecast.sibling_prefix_baseline`
   uses in training.  Every other number here is worthless if this one is a
   different definition wearing the same name.
2. Flat i.i.d. sampling reproduces the support collapse that motivates the
   module, and the tree fixes it -- measured, on the same engine, same budget.
3. The tree returns exactly `n` rollouts per prompt with no duplicated
   sequences, which is what keeps GRPO's group statistics honest.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from steer_f.tree_rollout import (  # noqa: E402
    TreeRolloutConfig,
    TreeSample,
    generate_tree,
    parse_int_list,
    sibling_support_stats,
)

fails: list[str] = []


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        fails.append(label)


# ----------------------------------------------------------------------
# a fake engine
# ----------------------------------------------------------------------
class MarkovEngine:
    """Sampling engine whose continuations diverge like a real policy's.

    The next token depends on the last one, so a shared prefix really does
    make two continuations correlated for a while and then not -- which is the
    only property of a policy the tree driver depends on.
    """

    def __init__(self, vocab=16, stop_prob=0.0, seed=0):
        self.vocab = vocab
        self.stop_prob = stop_prob
        self.rng = random.Random(seed)
        self.calls = 0
        self.generated_tokens = 0

    def __call__(self, prompts, n, max_tokens):
        self.calls += 1
        out = []
        for prompt in prompts:
            group = []
            for _ in range(n):
                toks, lps = [], []
                last = prompt[-1] if prompt else 0
                finished = False
                for _ in range(max_tokens):
                    # a mildly peaked conditional: 40% mass on `last + 1`
                    if self.rng.random() < 0.4:
                        nxt = (last + 1) % self.vocab
                        lp = -0.92
                    else:
                        nxt = self.rng.randrange(self.vocab)
                        lp = -2.77
                    toks.append(nxt)
                    lps.append(lp)
                    last = nxt
                    if self.stop_prob and self.rng.random() < self.stop_prob:
                        finished = True
                        break
                self.generated_tokens += len(toks)
                group.append(TreeSample(token_ids=toks, finished=finished, logprobs=lps))
            out.append(group)
        return out


# ----------------------------------------------------------------------
# the reference the diagnostics must match
# ----------------------------------------------------------------------
def reference_support_frac(groups):
    """`sibling_support` computed the slow, obvious, pairwise way.

    Mirrors `first_divergence`'s torch semantics on ragged lists: a position
    where exactly one of the two rollouts is still alive counts as a
    divergence; a position where neither is counts as nothing.
    """
    alive_total = supported = 0
    for group in groups:
        g = len(group)
        lens = [len(r) for r in group]
        t_max = max(lens) if lens else 0
        div = [[t_max] * g for _ in range(g)]
        for i in range(g):
            for j in range(g):
                if i == j:
                    continue
                for t in range(t_max):
                    ai, aj = lens[i] > t, lens[j] > t
                    if not ai and not aj:
                        continue
                    if ai != aj or group[i][t] != group[j][t]:
                        div[i][j] = t
                        break
        for i in range(g):
            for t in range(lens[i]):
                alive_total += 1
                n_sib = 1 + sum(1 for j in range(g) if j != i and div[i][j] >= t and lens[j] > t)
                if n_sib > 1:
                    supported += 1
    return supported / alive_total if alive_total else 0.0


# ----------------------------------------------------------------------
print("\n[1] config validation")
ok = TreeRolloutConfig(n=8, response_length=1024, depths="128,384,640", factors="2,2,2", roots=1)
check("valid config accepted", ok.enabled and ok.num_stages == 4, ok.describe())
check("stage budgets sum to response_length", sum(ok.stage_budgets()) == 1024, ok.stage_budgets())
check("designed sibling counts", [ok.expected_siblings_at(t) for t in (0, 128, 384, 640)] == [8, 4, 2, 1])
check("parse_int_list: every spelling means the same thing",
      parse_int_list("1, 2,3") == (1, 2, 3) == parse_int_list([1, 2, 3])
      == parse_int_list("[1,2,3]") == parse_int_list(" [1, 2, 3] "),
      "hydra sends =[1,2,3] as a list and ='1,2,3' as a string")
check("parse_int_list: empty spellings are all 'off'",
      parse_int_list(None) == parse_int_list("") == parse_int_list("  ") == parse_int_list([]) == ())

for label, kw in [
    ("roots*prod(factors) != n rejected", dict(n=8, response_length=64, depths="16", factors="3")),
    ("non-increasing depths rejected", dict(n=8, response_length=64, depths="32,16", factors="2,2", roots=2)),
    ("last depth >= response_length rejected", dict(n=4, response_length=64, depths="64", factors="4")),
    ("length mismatch rejected", dict(n=4, response_length=64, depths="8,16", factors="4")),
    ("zero factor rejected", dict(n=4, response_length=64, depths="8", factors="0")),
]:
    try:
        TreeRolloutConfig(**kw)
        check(label, False, "no error raised")
    except ValueError as e:
        check(label, True, str(e).split(";")[0][:70])

flat = TreeRolloutConfig(n=8, response_length=64)
check("no depths => flat, one stage", (not flat.enabled) and flat.num_stages == 1 and flat.roots == 8)

# ----------------------------------------------------------------------
print("\n[2] sibling_support_stats == pairwise first_divergence reference")
rng = random.Random(7)
cases = {
    "identical rollouts": [[[1, 2, 3]] * 4],
    "all distinct at t=0": [[[i, 9, 9] for i in range(4)]],
    "ragged / early death": [[[1, 2, 3, 4], [1, 2], [1, 2, 3, 4], [1, 5]]],
    "shared trunk then split": [[[1, 1, 1, 2, 7], [1, 1, 1, 2, 8], [1, 1, 1, 3, 7], [1, 1, 1, 3, 9]]],
    "random ragged": [
        [[rng.randrange(3) for _ in range(rng.randrange(1, 12))] for _ in range(5)] for _ in range(4)
    ],
    "single rollout": [[[4, 4, 4]]],
}
for name, groups in cases.items():
    mine = sibling_support_stats(groups)["support_frac"]
    ref = reference_support_frac(groups)
    check(f"matches reference: {name}", abs(mine - ref) < 1e-12, f"{mine:.6f} vs {ref:.6f}")

check("empty input is not a crash", sibling_support_stats([])["support_frac"] == 0.0)

# ----------------------------------------------------------------------
print("\n[3] flat sampling reproduces the support collapse")
P, N, T = 12, 8, 96
eng = MarkovEngine(seed=1)
flat_res = generate_tree(eng, [[5] for _ in range(P)], TreeRolloutConfig(n=N, response_length=T))
flat_sup = flat_res.stats["support_frac"]
check("flat used exactly one engine call", flat_res.stats["num_engine_calls"] == 1)
check("flat returns n per prompt", all(len(r) == N for r in flat_res.responses))
check("flat support is near zero", flat_sup < 0.05, f"support_frac={flat_sup:.4f} (training measured 0.003)")

# ----------------------------------------------------------------------
print("\n[4] the tree restores it, on the same engine and token budget")
cfg = TreeRolloutConfig(n=N, response_length=T, depths="12,36,60", factors="2,2,2", roots=1)
eng2 = MarkovEngine(seed=1)
res = generate_tree(eng2, [[5] for _ in range(P)], cfg, collect_logprobs=True)
sup = res.stats["support_frac"]
check("tree used L+1 engine calls", res.stats["num_engine_calls"] == cfg.num_stages,
      f"{res.stats['num_engine_calls']} calls, {cfg.describe()}")
check("tree returns exactly n per prompt", all(len(r) == N for r in res.responses))
check("no refills when nothing stops early", res.stats["num_refilled"] == 0)
check("support_frac jumps", sup > 0.6 and sup > 12 * flat_sup, f"{flat_sup:.4f} -> {sup:.4f}")
want = cfg.expected_mean_siblings()
check("mean siblings matches the design", abs(res.stats["mean_siblings"] - want) < 0.25,
      f"measured {res.stats['mean_siblings']:.3f} vs designed {want:.3f}")
check("support is front-loaded but not degenerate",
      res.stats["support_frac_by_decile"][0] == 1.0 and res.stats["support_frac_by_decile"][5] > 0.5,
      [round(x, 2) for x in res.stats["support_frac_by_decile"]])
check("diagnostic agrees with reference on real output",
      abs(sup - reference_support_frac(res.responses)) < 1e-12)

# the structural promise: siblings share their trunk exactly
shared = all(
    res.responses[p][a][: cfg.depths[0]] == res.responses[p][b][: cfg.depths[0]]
    for p in range(P) for a in range(N) for b in range(N)
)
check("all 8 rollouts share the first cut depth", shared)
half = all(
    res.responses[p][a][: cfg.depths[1]] == res.responses[p][a ^ 1][: cfg.depths[1]]
    for p in range(P) for a in range(N)
)
check("pairs share through the second cut depth", half)
check("logprobs line up with tokens",
      all(len(lp) == len(tok) for gp, gl in zip(res.responses, res.logprobs) for tok, lp in zip(gp, gl)))
check("paths are unique within a prompt", all(len(set(pp)) == N for pp in res.paths))

# ----------------------------------------------------------------------
print("\n[5] early stops refill instead of duplicating")
eng3 = MarkovEngine(stop_prob=0.03, seed=3)
res3 = generate_tree(eng3, [[5] for _ in range(P)], cfg)
check("still exactly n per prompt", all(len(r) == N for r in res3.responses))
check("some trunks died and were refilled", res3.stats["num_refilled"] > 0,
      f"refill_frac={res3.stats['refill_frac']:.3f}")
check("one extra engine call for the refills", res3.stats["num_engine_calls"] == cfg.num_stages + 1)
dupes = sum(
    1 for grp in res3.responses
    for i in range(N) for j in range(i + 1, N)
    if grp[i] == grp[j]
)
check("no rollout is a copy of a sibling", dupes == 0, f"{dupes} identical pairs")
check("refilled rollouts are marked", any(pp == (-1,) for g in res3.paths for pp in g))
check("support survives the deaths", res3.stats["support_frac"] > 0.4,
      f"{res3.stats['support_frac']:.4f}")

# ----------------------------------------------------------------------
print("\n[6] token budget is not inflated")
eng4 = MarkovEngine(seed=1)
generate_tree(eng4, [[5] for _ in range(P)], TreeRolloutConfig(n=N, response_length=T))
flat_tokens = eng4.generated_tokens
eng5 = MarkovEngine(seed=1)
generate_tree(eng5, [[5] for _ in range(P)], cfg)
tree_tokens = eng5.generated_tokens
check("tree generates no more tokens than flat", tree_tokens <= flat_tokens,
      f"flat {flat_tokens} vs tree {tree_tokens} ({tree_tokens / flat_tokens:.2f}x) -- "
      "shared trunks are generated once and reused, so the decode budget drops; "
      "the prefill of those trunks is re-paid unless the engine caches prefixes")

# ----------------------------------------------------------------------
print("\n[7] a lying engine is caught, not absorbed")
def short_group(prompts, n, max_tokens):
    return [[TreeSample([1], False)] * (n - 1) for _ in prompts]

def over_long(prompts, n, max_tokens):
    return [[TreeSample([1] * (max_tokens + 1), False) for _ in range(n)] for _ in prompts]

for label, fn in [("wrong sample count rejected", short_group), ("over-long sample rejected", over_long)]:
    try:
        generate_tree(fn, [[1]], TreeRolloutConfig(n=2, response_length=8))
        check(label, False, "no error raised")
    except RuntimeError as e:
        check(label, True, str(e)[:60])

try:
    TreeSample([1, 2], False, logprobs=[0.1])
    check("mismatched logprobs rejected", False, "no error raised")
except ValueError:
    check("mismatched logprobs rejected", True)

# ----------------------------------------------------------------------
print("\n[8] cross-check against the trainer's own sibling_support")
try:
    import torch  # noqa: F401
    from steer_f.entropy_forecast import sibling_support
except Exception as exc:  # pragma: no cover - depends on which tree this runs in
    print(f"  SKIP  steer_f.entropy_forecast.sibling_support unavailable ({type(exc).__name__}: {exc})")
    print("        Run this script from the TRAINING tree to exercise section [8]: it is the")
    print("        only check that compares the diagnostic to the function the trainer calls,")
    print("        rather than to a reference reimplemented in this file.")
else:
    import torch

    PAD = 10**6  # outside the fake vocabulary, so padding can never look like a token
    for label, tcfg, stop in [
        ("flat", TreeRolloutConfig(n=N, response_length=T), 0.0),
        ("tree", cfg, 0.0),
        ("tree+deaths", cfg, 0.02),
    ]:
        eng = MarkovEngine(stop_prob=stop, seed=11)
        r = generate_tree(eng, [[5] for _ in range(P)], tcfg)
        rows, msk, uid = [], [], []
        for pi, grp in enumerate(r.responses):
            for seq in grp:
                rows.append(list(seq) + [PAD] * (T - len(seq)))
                msk.append([1] * len(seq) + [0] * (T - len(seq)))
                uid.append(pi)
        resp, m = torch.tensor(rows), torch.tensor(msk)
        trainer = (sibling_support(resp, m, uid) & m.bool()).sum().item() / m.sum().item()
        check(f"trainer agrees: {label}", abs(trainer - r.stats["support_frac"]) < 1e-12,
              f"trainer={trainer:.6f} diagnostic={r.stats['support_frac']:.6f} "
              f"refill={r.stats['refill_frac']:.3f}")

# ----------------------------------------------------------------------
print()
if fails:
    print(f"FAILED {len(fails)}: " + "; ".join(fails))
    raise SystemExit(1)
print("all tree-rollout smoke checks passed")
