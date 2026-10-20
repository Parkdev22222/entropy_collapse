# Measured two-channel decomposition of entropy collapse

Registered 2026-10-20, before any sample was drawn.
Code: `scripts/channel_decomp.py`, launcher `run/channel_decomp.sh`.

## Why

The paper derives that the gradient of trajectory entropy has a local and a
visitation channel and that token-level entropy control sees only the first.
It never measures how much of the entropy that real training loses goes
through each channel. This measures it, on checkpoints that already exist.

## The identity (exact, no approximation)

For a policy pi, H(pi) = E_{y~pi}[ sum_t H(pi(.|s_t)) ], the trajectory
entropy in the paper's chain-rule form. For two checkpoints a (earlier) and b
(later), write E_x[S_m] for the mean, over responses sampled from x, of the sum
of model m's next-token entropies along the response. Then

    dH = E_b[S_b] - E_a[S_a] = L + V
    L  = E_b[S_b] - E_b[S_a]     same states, sharper conditionals
    V  = E_b[S_a] - E_a[S_a]     different states visited, scored by a

The split depends on which policy's states are used. We report the mean of
this ordering and the reverse one (L' = E_a[S_b] - E_a[S_a],
V' = E_b[S_b] - E_a[S_b]); both satisfy L + V = dH exactly.

Entropies are full-vocabulary, temperature 1, from an fp32-softmax of each
model's logits. Responses are sampled at temperature 1, top-p 1, up to 3072
tokens, the training sampler.

## Data

512 training prompts (DAPO-Math-17k, fixed seed), 4 responses each, the same
prompts for every checkpoint. Uncertainty: 95% percentile bootstrap over
prompts (10,000 resamples, fixed seed), paired across checkpoints because the
prompts are shared.

## Checkpoints (all seed 1)

| label | checkpoint |
|---|---|
| base | Qwen2.5-Math-1.5B |
| grpo80, grpo100, grpo110 | `grpo-Qwen2.5-Math-1.5B-s1/global_step_{80,100,110}` |
| perm80, perm100, perm110 | `steer-f-Qwen2.5-Math-1.5B-s1-tree-rollout-permuted/global_step_{80,100,110}` |
| signed50 ... signed200 | `pilot-steer-f-1p5b-s1-150/global_step_{50,100,150,200}` |

`signed` is the campaign's seed-1 configuration re-run on four GPUs
(see `docs/manipulation_check.md`, Amendment 1).

## Questions and rules, fixed now

**Q1 (primary).** Interval base -> grpo100. Is a measurable share of the
entropy GRPO loses carried by the visitation channel? Reported: dH, L, V and
the share V/dH, each with its interval. **Rule:** the claim "a measurable
share of real collapse goes through the visitation channel" is made only if
the 95% interval of V excludes zero with the same sign as dH. Otherwise it is
reported as not established.

**Q2.** Intervals base -> perm100 and base -> signed100 against
base -> grpo100, paired on prompts. Reported: L_arm - L_grpo and
V_arm - V_grpo with intervals. The paper's analysis predicts that token-level
damping acts on the local channel. **Rule:** described as consistent with that
prediction only if the interval of L_arm - L_grpo excludes zero and
|L_arm - L_grpo| > |V_arm - V_grpo|. Reported either way.

**Secondary, without a rule:** grpo80 -> grpo110, perm80 -> perm110,
base -> signed50 -> signed100 -> signed150 -> signed200.

## Known limitations, stated in advance

- One seed. The `signed` checkpoints come from a re-run on different hardware.
- The decomposition is finite-difference, between checkpoints tens of steps
  apart; it is the integrated version of the gradient split, not the gradient.
- 512 prompts x 4 responses; a share near zero may be below resolution.

## Amendment 1 — replication on another seed (2026-10-20, before sampling)

Decided after the seed-1 result was seen, to test whether it holds on a second
seed; the rule below is fixed before any sample of that seed is drawn.

- **What:** base -> grpo100 (and base -> perm100 where that checkpoint exists)
  for seed 3 or 4, whichever has a step-100 checkpoint on the Hub; both if both
  do. Same prompts, sampler and analysis as above; the base model's samples
  and self-entropies are reused from seed 1, since they do not depend on the
  seed under study. Launcher: `SEED=<s> bash run/channel_decomp.sh`, output
  `results/decomp_s<s>/`, `results/channel_decomp_s<s>.json`.
- **Rule:** the seed-1 Q1 claim counts as replicated on seed s if the same Q1
  rule holds there (the 95% interval of V excludes zero with the sign of dH).
  The share V/dH and its interval are reported whatever they are; no threshold
  on its size is set.
- **Reported either way**, including the case where no other seed has a
  step-100 checkpoint and nothing is run.
