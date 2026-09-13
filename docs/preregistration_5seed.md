# Pre-registration: the 5-seed, 6-benchmark campaign

Written **before any run of this campaign has been launched**, so that the analysis
cannot be chosen to fit the result. Commit this, then start the runs.

Everything here supersedes the statistics used in the single-seed manuscript.

---

## 1. What is being run

| arm | intervention | seeds |
|---|---|---|
| GRPO | none (`loss_mode=vanilla`) | 1–5 |
| STEER | local `Ω` only (`λ=0`, plain rollout) | 1–5 |
| **STEER-F** | `A_H` as derived (`λ=.25`, tree) | 1–5 |
| uniform | tree + damping, `A_H`'s value discarded | 1–5 |
| permuted | tree + damping, `A_H` shuffled among siblings | 1–5 |

> **개정 2026-09-13.** 원안은 uniform/permuted를 3시드로 잡았다. 실제로 실행되는
> `run/run_campaign.sh`는 `CAMPAIGN_ARMS`(5개) × `SEEDS="2 3 4 5"`로 **5 arm 전부를
> 5시드**씩 돌린다. 두 통제군 대조의 검정력이 올라가는 변경이고, 캠페인 결과를 보기
> **전에** 이루어졌다. 조용히 흡수하지 않고 여기와 논문 부록에 기록한다.

Qwen2.5-Math-1.5B, DAPO-Math-17k, 110 steps, `n=8`, all other hyperparameters as in
the manuscript. **Seeds are matched across arms** (`data.seed = s` in every arm), so
contrasts are paired by seed.

`data.seed` fixes the data order only. Rollout sampling is not seeded across
processes, so two runs with the same `data.seed` still diverge; a seed here labels an
**independent replicate**, not a reproducibility handle. The paper must say this.

## 2. Primary analysis — fixed in advance

- **Unit of analysis: the seed.** Not the validation step.
- **Primary endpoint:** mean `acc@32` on AIME24 over the converged window, steps
  40–110 (8 validation points), averaged within a run to a single number per seed.
- **Primary contrast:** STEER-F − GRPO, paired by seed, one-sample *t* on the 5
  paired differences.
- **Secondary contrasts, in this order:** STEER-F − STEER, STEER-F − uniform,
  STEER-F − permuted. Reported with the same test; contrasts against the 3-seed arms
  use 3 pairs.
- **Secondary endpoints:** `maj@32`; `uplift = maj@32 − acc@32`; the MATH-6 average
  from the post-hoc evaluation.

We report effect sizes with confidence intervals, not significance stars, and we
report all five arms whatever the outcome.

### Demoted

The manuscript's paired *t* over the 8 window steps is a **within-run stability**
measure. It stays in an appendix and is never presented as evidence that a gap
generalizes.

### Not computed from summary statistics

The across-problem standard error of a *paired* difference cancels per-problem
difficulty and is the right scale for generalizing to new problems. It needs
per-problem scores, which is why `validation_data_dir` is switched on for this
campaign (see `run/instrument_campaign.sh`). If for any reason those dumps are
missing, we say so rather than substituting the unpaired error.

## 3. Stopping and exclusion rules

- Every run goes to step 110. No run is stopped early because it looks good or bad.
- A run is excluded **only** for a mechanical failure: OOM, a crash, a corrupted
  checkpoint, or a step-0 validation score outside `.030–.055` (which would mean a
  different initial checkpoint). Exclusions are listed in the paper with the reason.
- A failed run is re-launched with the **same** seed, not a new one.
- No seed is added after the results are seen. If the campaign is extended, the
  extension is reported as a separate, later set.

## 4. Checkpoint selection, fixed in advance

Two rules are computed and **both** are reported:

1. **argmax** — the checkpoint with the highest AIME24 `acc@32` in that run
   (`save_best_only=True`).
2. **fixed** — step 110.

The post-hoc 6-benchmark evaluation uses rule 1. Rule 2 exists so the paper can state
whether the arm ordering depends on the selection rule. Selecting a maximum over 12
noisy checkpoints is biased upward; in the single-seed data that bias was +.011 to
+.017 per arm, which is why the primary endpoint is a window mean and not a maximum.

## 5. Post-hoc evaluation

Six math benchmarks (AIME24, AIME25, AMC23, MATH500, Minerva, OlympiadBench) via
`run/eval_steerf.sh`, one evaluation per run at the rule-1 checkpoint.

- Logs must be named `eval-<arm>-s<seed>.log`; `scripts/collect_results.py` silently
  skips anything else.
- **Before trusting any of it**, confirm the runs used different weights:
  `grep -h MODEL_PATH logs/experiments/eval-*.log | sort -u` must return one line per
  run. A previous attempt produced five logs that all pointed at the same checkpoint.

**Direction consistency** is the headline output: in how many of the six benchmarks
does `STEER-F > {uniform, permuted}` hold? Declared before seeing the results: 4/6 or
better is support, 3/6 is equivocal, 2/6 or worse is evidence against.

## 6. What would falsify the paper's claim

Stated now so it cannot be renegotiated later.

- **STEER-F − uniform** straddling zero across seeds would mean the forecast's value
  does not matter and the apparatus explains the effect. That is the central claim.
- **STEER-F − permuted** straddling zero would mean the pairing between a branch and
  its score does not matter, which the theory says it must.
- Aggregate entropy ordering the arms in the same order as accuracy would undercut
  the dissociation argument, which depends on the two orderings coming apart.
- Mean token weight departing from ≈0.999 in the STEER-F arm would mean `A_H` is not
  sum-zero over sibling sets as derived, and the implementation is wrong.

## 7. Instrumentation this campaign depends on

Verify with `bash run/instrument_campaign.sh --check` on every pod **before**
launching. All three are additive and none changes a gradient:

1. `validation_data_dir` set — per-problem scores (item 2 above depends on it).
2. `save_best_only` env-driven — checkpoint rule 1.
3. `seq_entropy_agg` uncommented — logs `actor/seq_entropy`, the true trajectory
   entropy rather than entropy divided by mean length.

A bit-equivalence test must pass before any change to the training path (for example
a masked MTP forward) is allowed into the campaign.
