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
- **An OOM re-launch may set `OFFLOAD=1`, and such a run is kept, not excluded.**
  Added 2026-09-15, after the seed-2 STEER arm hit a CUDA OOM in
  `update_policy`. The alternative fix — shrinking
  `ppo_micro_batch_size_per_gpu` — is *not* available to us at any price: that
  group is STEER's min–max pool, so a seed rescued that way would be a
  different treatment from the seeds beside it. `OFFLOAD=1` moves the
  parameters and optimizer to the CPU and leaves the objective, the pool and
  every hyper-parameter untouched. It is not bit-identical (the AdamW update
  then runs in CPU fp32), so the run is **marked**: `scripts/analyze_seeds.py`
  reads `param_offload` out of each log and reports it as the `stack` column of
  `per_seed.tsv`, and any table drawn from a mixed set says so. A re-launch
  that also resumes from a mid-run checkpoint is likewise kept — the queue
  writes optimizer state precisely so a crash costs steps rather than a run.
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

---

## 8. The backbone rows (added 2026-09-14, before any backbone run)

Three further backbones — Qwen2.5-Math-7B, Llama-3.1-8B, Mistral-7B-v0.3 — each
carry **three arms** (GRPO / STEER / STEER-F) at **seed 1**. The five-arm
decomposition is not repeated: it answers whether the gain comes from the
apparatus or from the forecast's value, and that is a question about the method,
settled once on the primary backbone. A second backbone answers something
narrower.

### Primary statistic, fixed now

**The sign of `STEER-F − GRPO`, within a backbone, on that backbone's own
validation set, over the same converged window (step 40–110).** The reported
result is **how many of the four backbones share that sign** — a count out of 4,
not an average.

### What is not computed

- **No column averages, and no comparison of absolute accuracy across
  backbones.** The rows differ in scale, in family, in whether mathematics was
  specialized for, and in which benchmark selects the checkpoint. An average
  down that column would be a number with no referent.
- **No claim about effect size from a single backbone row.** One seed per cell
  carries the same uncertainty as one seed on the primary backbone
  (paired-difference SD ≈ .0088). No individual row is decisive.

### Validation set, fixed now

| backbone | validates and selects on | why |
|---|---|---|
| Qwen2.5-Math-1.5B / -7B | AIME24, `acc/mean@32` | as the main campaign |
| Llama-3.1-8B, Mistral-7B-v0.3 | **MATH500**, `acc/mean@1` | AIME24 is 30 problems, `SE ≈ .027`; a backbone scoring near zero puts all three arms inside one SE **and** makes `save_best_only` draw the checkpoint from noise. MATH500 is 500 problems, ≈4× tighter |

This is decided before the runs, not after seeing which benchmark is kinder.
`run/_arms.sh:backbone_profile()` carries the pairing so it cannot be changed
per-run without changing the code.

### What would falsify the generality claim

A backbone on which the sign **reverses** (`STEER-F < GRPO`) is evidence against
the account, not noise to be averaged away: our claim is that the visitation
channel matters wherever the rollout distribution has branch points that differ
in what they lead to, which is a property of the task rather than of the
pre-training corpus. A **split** result (2 of 4) is reported as a limit on
generality.

---

## 9. The branch-point entropy reading (added 2026-09-21, **after** the aggregate test failed)

Everything above §8 was written before the runs it governs. **This section was
not.** It is added after §6's aggregate-entropy condition was checked against the
completed seeds and **came out against us**, and it is marked as such rather than
folded into the text above.

What that means concretely: this section **justifies nothing about the seeds
already collected**. It binds only the seeds not yet in hand. The manuscript
reports the branch-level reading as exploratory for exactly this reason
(§12.6), and that does not change if the remaining seeds agree with it.

### What actually happened to §6's third bullet

The pre-registration's unit is the seed, with arms paired within a seed, so the
"orders the arms the same way" condition is settled by the paired contrasts, not
by ranking five arm means:

| aggregate contrast | n | Δentropy | Δacc | |
|---|---|---|---|---|
| `STEER − GRPO` | 3 | −.0487 | −.0101 | same direction |
| `STEER-F − STEER` | 2 | +.0101 | +.0135 | same direction |
| `STEER-F − GRPO` | **1** | −.0281 | +.0145 | dissociated |

Two of three move together; the one that comes apart has a single seed under it.
**The dissociation claim as pre-registered is recorded as failed.**

Ranking the arm means does not rescue it. The rank correlation between converged
entropy and converged accuracy comes out with a *different sign* depending on
whether each arm is averaged over the seeds common to all five arms
(`--balanced`, what the manuscript's Table 1 shows) or over its own seeds, which
differ in number mid-campaign. A statistic whose sign turns on that choice is not
one a falsification condition can be settled with at this seed count.

### Registered now, before the remaining seeds

- **Quantity:** `steerf/branch_entropy` — entropy at the positions where `A_H`
  can be non-zero. The trainer already logs it; `scripts/analyze_seeds.py` emits
  `brentropy`, `nbrentropy`, `brgap` per seed.
- **Primary contrast:** `STEER-F − permuted`, **paired by seed**. That pair shares
  λ, λ_min, the sampler and the *magnitudes* of `A_H`, and differs only in whether
  a branch is paired with its own score.
- **Secondary:** `STEER-F − uniform`, paired by seed.
- **The narrow claim:** a branch-point entropy contrast near zero can accompany an
  accuracy contrast away from zero — what separates the arms is *which sibling*
  the budget is spent on, not how large the budget is.

### What would falsify the narrow claim

Stated before the remaining seeds land, so it cannot be renegotiated after.

1. If the branch-point contrast **tracks** the accuracy contrast in sign and
   magnitude across seeds the way the aggregate contrasts above do, the
   branch-level split is the aggregate statistic in a smaller window and the
   narrow claim fails with the wide one.
2. If STEER-F keeps the **lowest** branch-point entropy and the **smallest**
   branch-to-non-branch gap of the three tree arms — which is what the seeds in
   hand already show (`.1446`/`.0231` against `.1512`/`.0314` for permuted and
   `.1777`/`.0399` for uniform) — then reading this method as "preserving
   diversity where it matters" is **refused outright**, and the manuscript says
   so in those words rather than softening the claim to fit.
