# Pre-registration: diversity of correct reasoning traces (2026-10-03)

Committed before any trace-diversity number has been computed for any arm.
Everything below is fixed now; a change after the first result is an
amendment, dated and listed in section 9, and the paper reports both the rule
and the amendment.

Code: `scripts/trace_diversity.py` (tests: `tests/test_trace_diversity.py`).

## 1. Question

Among the traces an arm gets **right**, are its traces more varied than
another arm's on the same problems? Accuracy, pass@k and token entropy do not
answer this: two arms can have the same accuracy with one of them reaching
the answer by the same route every time.

The hypothesis is two-sided. Nothing below assumes the direction, and the
result is reported whichever way it comes out.

What is already known, and disclosed because it shaped the question: during
training, the two-root tree pilot (`steerf-r2-1p5b-s1-200`) showed a higher
bootstrap maj@32 and a similar or slightly lower best@32 on AIME24 than the
one-root pilot. That points, if anywhere, towards *less* diverse correct
traces for the two-root run. For that reason the roots contrast in section 3
is exploratory; only the contrasts fixed before that observation are
confirmatory.

## 2. Data

- The six-benchmark evaluation dumps (`VAL_DATA_DIR`, written by
  `run/eval_steerf.sh` / `run/eval_lora.sh`), **last-step weights only**, the
  same rule as every other result in the paper.
- The 32-sample pass: AIME24, AIME25, AMC23 (temperature 1.0, top-p 0.7, the
  registered evaluation settings). The 1-sample pass carries no diversity
  information and is not used.
- No new generations are made for the primary analysis.

## 3. Contrasts

Confirmatory, in this order:

1. STEER-F (`signed`) - GRPO. **Primary.**
2. STEER-F - `uniform`, STEER-F - `permuted` (the paper's one-change
   contrasts).

Exploratory (section 1): two-root tree - one-root tree, full fine-tuning
pilot, seed 1.

Each contrast is computed **per seed**, on the seeds both arms share.

## 4. Unit, filter and confound control

- **Unit:** a problem, matched across arms on (benchmark, prompt text). A
  problem missing from an arm is dropped and counted.
- **Filter:** a problem enters only if every arm in the comparison has at
  least k correct traces on it. k = 4 is primary; k = 2 is reported as a
  sensitivity analysis. The number of eligible problems per benchmark is
  reported with every result.
- **Count:** each arm contributes exactly k correct traces per problem (drawn
  at random, 50 draws for lexical metrics; exact rarefaction for the judge
  metric). An arm is never compared using "all its correct traces".
- **Length:** lexical metrics are reported on traces cut to the shortest
  trace in each draw. The untruncated value is printed beside it but is not
  the registered figure.
- **Sampling settings:** identical for every arm (section 2).

## 5. Metrics

**Primary - solution strategies.** For each eligible problem, up to 8
correct traces per arm are pooled, shuffled and shown to a judge with no arm
information (`export-judge`). The judge partitions them into groups that use
the same solution strategy (section 7). The metric is the expected number of
distinct strategies among k = 4 correct traces of an arm, computed exactly by
rarefaction (`import-judge`). Range 1 to 4.

**Secondary - lexical** (`auto`), all on 4-grams of a word/symbol
tokenisation, length-truncated:

- `vendi_trunc`: effective number of distinct traces among k (1 to k);
- `pair_dist_trunc`: mean pairwise 1 - Jaccard;
- `distinct_trunc`: unique / total 4-grams.

Lexical metrics cannot tell a new idea from a paraphrase; they are reported,
not used for the claim.

## 6. Decision rule

For each contrast and seed: the mean over eligible problems of the paired
difference A - B, with a 95% percentile bootstrap interval over problems
(2000 resamples, seed 0).

- "A's correct traces are more diverse than B's" is claimed only if, on the
  **primary** metric, the interval excludes zero **in the same direction in
  every seed analysed**.
- The mirror statement uses the mirror rule.
- Anything else is reported as "no difference detected", with the intervals.
- Secondary metrics never upgrade or overturn the primary verdict; their
  agreement or disagreement with it is reported as such.

## 7. Judge protocol and validity checks

**Judge.** A fixed LLM and a fixed prompt, chosen before export. The judge
sees the problem and the shuffled traces with opaque ids (`T01`, ...), never
the arm. Output, one line per task:

```
{"task_id": "P0001", "clusters": {"T01": "s1", "T02": "s1", "T03": "s2", ...}}
```

Prompt (fixed):

> Below is a math problem and several correct solutions to it. Group the
> solutions by the method they use to reach the answer. Two solutions are in
> the same group if they rely on the same key idea (for example: the same
> substitution, the same theorem, the same decomposition of cases), even if
> the wording, notation or order of steps differs. Two solutions are in
> different groups if the key idea differs. Do not consider correctness,
> length, or style. Return JSON mapping every solution id to a group label.

Every task must be labelled. `import-judge` refuses a partial set: a subset
chosen after seeing labels is not this sample.

**Where it may run.** Exporting traces to an external API is uploading them.
It is not done from an H200 machine. Where the judge runs is decided by the
user and recorded in section 9 before export.

**Validity checks, run before any arm contrast is looked at.** A metric that
fails its check is not used as evidence; the failure is reported.

1. *Positive control (all metrics).* One checkpoint sampled at temperature
   0.6, 1.0 and 1.2 on the same 32-sample sets: the metric must increase
   monotonically with temperature on the pooled problems. The base model
   must score at least as high as the trained one at temperature 1.0.
2. *Paraphrase control (judge).* For 30 problems, take one correct trace and
   a paraphrase of it that keeps the method; the judge must put the pair in
   one group in at least 90% of cases.
3. *Human agreement (judge).* A person labels 30 tasks blind; the adjusted
   Rand index between human and judge partitions, averaged over tasks, must
   be at least 0.6.

## 8. Known limits, stated now

- Power: 100 problems in the 32-sample pass, and AIME accuracy near 15%, so
  few AIME problems will have four correct traces in every arm; AMC23 will
  carry most of the sample. If the eligible count at k = 4 is below 20, that
  is reported, and evaluating MATH500 at 32 samples becomes an amendment the
  user decides on before seeing any contrast.
- Seeds: three per LoRA arm at most, one for the full-fine-tuning pilots.
  The every-seed rule in section 6 is deliberately conservative for that
  reason.
- Diversity of correct traces is not the same as coverage. pass@k at large k
  is a separate, secondary question and is not substituted for this one.

## 9. Amendments

None yet. Judge model and location: to be fixed here before the first export.
