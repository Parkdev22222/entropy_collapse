# Manipulation check: did the A_H correction act on the policy?

Registered 2026-10-13, before any probe was built or any checkpoint scored.
Code: `scripts/manip_check.py`, launcher `run/manip_check.sh`.

## Why

The dissociation result (Section 6.1, Table `tab:entropy`, Table `tab:htraj`)
shows that changing only the A_H signal barely moves aggregate entropy. On its
own that cannot be told apart from "the correction does nothing". This check
asks whether the trained policies carry the trace the correction is built to
leave.

## What the correction does, read from the code

`steer_f/omega_tilde.py:branch_weight_correction`, mode `signed`:

    w = clamp(w_steer + lam * (w_max - w_min) * tanh(visit / rms(visit)), w_min, w_max)
    visit = dlogpi_hat * clip(A_H),   dlogpi_hat = eta * w * (1 - pi_sampled) > 0

`w` multiplies that token's reward policy gradient. So at a branch point the
sibling with the larger entropy-to-go is damped less and the one with the
smaller entropy-to-go is damped more. The correction does not push either
sibling in a fixed direction; it changes **how strongly each is learned**,
in whichever direction the reward advantage points. Weights sit at the upper
clamp almost everywhere (`twg_frac_high` about .995), so in practice the
richer sibling keeps its full step and the poorer one loses part of it.

`uniform` damps both siblings by the same amount, and `permuted` runs the same
`signed` path on A_H shuffled within the sibling set. Neither should favour
the richer sibling.

## Probe

1. From the base model (Qwen2.5-Math-1.5B), on 512 training prompts drawn with
   a fixed seed, sample one response and cut it at the tree's branch depths
   64, 192 and 384. Each cut is a branch point.
2. At each branch point, draw next tokens from the base policy until two
   distinct ones appear (at most 8 draws; a point with only one distinct
   token is dropped, and the drop rate is reported).
3. For each of the two branch tokens, sample 4 continuations from the base
   model and compute the realised entropy-to-go: the sum of full-vocabulary
   next-token entropies over every state after the branch token, averaged over
   the 4 continuations. The sibling with the larger value is `hi`, the other
   `lo`; ties are dropped.
4. For the base model and each checkpoint, compute log pi(token | prefix) for
   both branch tokens (fp32, one forward pass per prefix).

The probe is built once and shared by every checkpoint, so all comparisons are
paired on the same branch points.

## Primary statistic

Per branch point, with d = log pi_ckpt - log pi_base:

    D = |d(hi)| - |d(lo)|

Per checkpoint, the mean of D over branch points. Per seed, the contrasts

    STEER-F - uniform      STEER-F - permuted

on seeds 1, 3, 4 (the paper's seed set).

**Rule, fixed now:** the check counts as passed when both contrasts are
positive on all three seeds. Anything else is reported as not passed. The
result is reported either way, with the per-seed values.

## Secondary, reported without a rule

- The same contrasts on the half of branch points with the larger
  |H_togo(hi) - H_togo(lo)|, where the signal is strongest.
- The directional quantity S = d(hi) - d(lo).
- The contrasts against GRPO and STEER, and STEER - GRPO, for context.

## Known limitations, stated in advance

- Each run kept one checkpoint, its AIME24 argmax, so the scored steps differ
  across arms. The step of every checkpoint is reported.
- The training runs ranked siblings with the MTP forecast under the current
  policy; the probe ranks them by realised entropy under the base policy. The
  two agree imperfectly, which dilutes the effect but cannot reverse its sign.
- The probe states come from the base policy, which the trained policies visit
  less as training proceeds.
- The correction touches about 0.1-0.2% of tokens with a small weight change,
  so the effect may be below what 512 prompts can resolve. A null here means
  "not detected at this resolution", not "absent".
