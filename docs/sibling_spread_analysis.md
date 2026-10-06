# Sibling-spread analysis — plan fixed before computing (2026-10-06)

Written and committed **before** any of the numbers below were computed or looked
at. The script that computes them (`scripts/sibling_spread.py`) is committed after
this file.

## Why

The zero-sum property of `A_H` (Section 4, "The correction cannot move the mean")
says a correction built on it cannot change the *average* weight at branch tokens;
its positive prediction is that the *spread across siblings* widens. The
pre-registered test of that (an entropy split restricted to `A_H != 0`) was
instrumented only after the five-arm campaign and no run carries it. This analysis
asks the closest question the existing logs can answer. It is **exploratory and
post hoc** with respect to the pre-registration, and will be reported as such
whatever it shows.

## Quantity

From the training logs of the three tree arms (STEER-F = `signed`, `uniform`,
`permuted`), at every logged training step `t`:

    S_t = steerf/a_h_abs_mean_t * (G * L_t) / steerf/n_branch_points_t

* `a_h_abs_mean` — mean of `|A_H|` over all valid response tokens
  (`verl/trainer/ppo/ray_trainer.py`); `A_H` is the sibling-differenced forecast of
  `H_togo`, so `|A_H|` at a branch point is how far that branch's forecast to-go
  entropy sits from its siblings' mean.
* `n_branch_points` — number of sibling divergence points in the batch
  (`branch_recall_at_k`, true branch points, independent of the advantage).
* `G * L_t` — valid tokens in the batch: `G = 512 * 8 = 4096` responses times
  `response_length/mean`.

So `S_t` is the mean `|A_H|` per branch point (up to the constant number of
columns each branch point occupies, which is the same for every arm and cancels in
the ratios below). It is the **forecast** spread, not the realized-entropy spread.

`branch_corr_frac_strict` is not used as the denominator: it is defined on
`visit = dlogpi * clip(A_H)`, which is zero wherever the GRPO advantage is zero,
and the zero-advantage fraction differs by arm.

## Statistic and contrasts

* Per run: mean of `S_t` over training steps 40–110 inclusive (every logged step
  in the converged window, not only validation steps).
* Runs: the log `results/per_seed.tsv` assigns to each (arm, seed), seeds 1, 3, 4
  — the seed set of every other statistic in the paper.
* Contrasts, paired within seed, as relative differences:
  `S(STEER-F) / S(uniform) - 1` and `S(STEER-F) / S(permuted) - 1`.
* Reported: per-seed values, the mean, the paired t (df = 2), sign counts.

## Verdict rule (fixed now)

* **Consistent with widening:** both contrasts have a positive mean **and** are
  positive on at least 2 of 3 seeds each.
* **Against:** both contrasts have a negative mean.
* **Inconclusive:** anything else.

## Caveats stated in advance

1. Forecast spread, not realized entropy: a wider `|A_H|` can reflect the MTP
   heads' view of the policy rather than the policy itself.
2. Three seeds; the t has 2 degrees of freedom.
3. `permuted` preserves the multiset of `A_H` within each sibling set at the step
   it is computed, so any difference from STEER-F arises only through how training
   changed later rollouts — which is the effect asked about. `uniform` changes the
   weights at every supported position by a larger magnitude (`omega_tilde.py`
   notes ~15x mean |delta|), so the two controls are not equivalent and both are
   reported.
