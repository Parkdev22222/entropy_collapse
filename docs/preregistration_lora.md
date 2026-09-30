# Pre-registration: the LoRA campaign (2026-09-29)

Committed before any LoRA training run exists. Everything below is fixed now;
a change after the first result is an amendment, dated and listed at the end,
and the paper reports both the rule and the amendment.

## 1. Why this campaign, and what it replaces

The first version of the paper trains every arm with LoRA. The reason is
compute: it lets every arm run at three seeds, with the ablations, inside the
submission window. The comparisons in the paper are between arms trained
identically, so the claims are relative claims under LoRA; whether they carry
over to full fine-tuning is left to a second paper and is not claimed.

Full fine-tuning pilots exist (branches `claude/3b-text-generation-models-thz2vl`
and `paper`). They are not pooled with anything here and no claim rests on
them. One of them informed a design choice below and is disclosed in §4.

## 2. Setting (identical for every arm)

| | |
|---|---|
| Model | Qwen2.5-Math-1.5B |
| Training data | DAPO-Math-17k, batch 512 prompts x 8 rollouts, mini-batch 32, micro-batch 8 |
| Steps | 150 for every run (about 4.4 epochs), lr constant, no warm-up |
| LoRA | rank 64, alpha 32, all linear layers, lr 1e-5 (`run/_lora_defaults.sh`) |
| Policy loss | clip 0.2 / 0.28 / c 10, no KL, no entropy bonus |
| Token weights | [0.7, 1.0] for every weighted arm: STEER's published value (lambda_min 0.7, exponential map capped at 1.0); STEER-F's min-max map uses the same range |
| Validation | AIME24, 32 samples, every 10 steps, temperature 1.0, top-p 0.7 |
| Checkpoints | every 50 steps, the last step, and the AIME24-best (`verl/trainer/ppo/ckpt_policy.py`) |
| Seeds | 1, 2, 3 (core); 1 (follow-ups) |
| Topology | 1.5B runs: one run per H100 80GB, four side by side, vLLM with CUDA graphs (`TOPOLOGY=1gpu ROLLOUT_EAGER=0`, locked in `logs/lora/campaign_settings`). Backbone runs: the whole four-GPU node, same vLLM setting. Every arm of a table shares its layout. Recorded in every log |

## 3. Arms

Core, seeds 1-3: `grpo`, `steer` (lambda 0), `uniform`, `permuted`, `signed`
(the method, STEER-V), `mtp` (STEER-V with the MTP forecaster). Follow-ups, seed 1: `lam0.1`, `lam0.5`, `lam0-tree`,
`wmin-steer` (STEER at the released script's 0.8), `xclip-signed`, `xclip-steer`,
`rloo-signed`, `rloo-steer`, `opo-signed`, `opo-steer`. There is no separate long
GRPO run: the compute-matched point is read off the regular GRPO run
(censored, and said so, if GRPO is the cheaper arm). Definitions:
`run/_lora_arms.sh`.

**The method reads H_togo from the realised entropy of the rollout**
(`STEERF_FORECAST=oracle`) -- no MTP heads, no Phase 1, no extra forward pass.
`uniform` and `permuted` read the same quantity, so each registered contrast
changes one thing.

## 4. The forecaster choice, and the arm that tests it

The realised-entropy variant was chosen for cost (19% less time per step in
the full fine-tuning pilot) and because it removes Phase 1 and the head
calibration. In the one pilot that compared the two at the same seed (full
fine-tuning, seed 4, n = 1), the MTP forecaster led by +.0112 plateau
accuracy. One run per side does not measure a difference, but it points the
other way, so the choice is tested rather than assumed: `mtp` runs at
every core seed, and the paper reports `signed - mtp` whatever it is.

Reading, fixed now (paired six-benchmark difference at the last step, pooled
MATH500/OlympiadBench/Minerva, across-problem SE over the three seeds):
- |diff| < 2 SE: the two are not distinguished at this resolution; the paper
  says that, not "equivalent".
- mtp ahead by >= 2 SE: the paper says the forecaster carries accuracy
  that the realised value does not, and presents the method as the cheaper
  variant with that cost stated.
- signed ahead by >= 2 SE: reported as measured.

## 5. Endpoints and decision rules

**Primary.** Six-benchmark accuracy at 32 samples per problem (AIME24, AIME25,
AMC23, MATH500, Minerva, OlympiadBench; GSM8K reported only) **at the last
training step** of each run. No checkpoint selection, so AIME24 is as held-out
as the rest.

- Direction consistency, per registered contrast, seed-averaged: >= 4 of 6
  benchmarks favouring the treatment = support, 3 = equivocal, <= 2 = against.
- Paired across-problem difference and SE (`scripts/eval_paired_se.py`),
  pooled over MATH500/OlympiadBench/Minerva and over seeds.

**Secondary.** (a) The same evaluation at the AIME24-best checkpoint, labelled
in-sample for AIME24. (b) AIME24 plateau accuracy, mean over validation steps
40-150, paired within seed (`scripts/analyze_seeds.py`).

**Registered contrasts.** signed - grpo, signed - steer, signed - uniform,
signed - permuted, steer - grpo, uniform - steer, signed - mtp.
signed - uniform and signed - permuted are the two that change exactly one
thing; they carry the claim.

**Refutation conditions** (any one is reported as refuting the claim it names):
1. `signed - uniform` straddles zero across seeds on the primary endpoint's
   pooled paired difference, or is equivocal/against by direction consistency:
   the sign of the weighting is not what helps.
2. The same for `signed - permuted`: the forecast's information is not what helps.
3. **The dissociation fails.** The construction predicts that the damping
   apparatus moves the aggregate entropy and A_H's contents do not: the
   apparatus contrasts (steer - grpo, uniform - steer) should move converged
   `actor/entropy` and the A_H contrasts (signed - uniform, signed - permuted)
   should not. If the |t| of the two groups overlap, the claim fails.
4. Mean token weight leaves [0.99, 1.0] for `signed`: the arm is a learning-rate
   change, not a reweighting.

## 6. Does entropy collapse under LoRA at all?

A reviewer's first question: LoRA's low-rank constraint may itself slow the
sharpening that collapse describes, leaving the method nothing to fix. So,
fixed now: for `grpo`, report `actor/entropy` at step 10 and at the last step
(150) per seed. **Collapse** = the last-step value below half the step-10
value in all three seeds (the full fine-tuning pilot went .311 -> .146 over
110 steps). If GRPO does not
collapse by this definition, the paper says so and does not motivate the
method by collapse under LoRA.

## 7. Backbones (added 2026-09-29, before any backbone run)

Qwen2.5-Math-7B (scale, same family) and Llama-3.2-3B-Instruct (a different
pre-training family), each with `grpo`, `steer` and `signed` (STEER-V) at
seed 1, the same LoRA configuration and 150 steps. No MTP arm and no Phase 1:
STEER-V needs no heads. The 7B validates and selects on AIME24 like the 1.5B;
Llama validates on MATH500 (500 problems, one sample), because a model that
scores near zero on AIME24 cannot separate arms or checkpoints there, and it
is the instruction-tuned checkpoint because the base one answered almost no
training problem under this protocol. Every backbone run uses the whole box.

Reported, and nothing more: per backbone, the sign of STEER-V - GRPO and
STEER-V - STEER on the plateau window and on the six benchmarks at the last
step, and how many backbones share the 1.5B's sign. One seed per cell carries
a sign, not a size; no accuracy is compared across backbones.

## 8. Not in this campaign

Full fine-tuning (second paper), Mistral-7B, pass@k, code benchmarks.

## Amendments

One after the first run, (4) below. Three changes before it, recorded because
this file had already been committed. (1) 2026-09-29: training length set to 150
steps for every run (was 110, with a 200-step compute-matched GRPO), the
plateau window to 40-150, and the separate long GRPO run dropped in favour
of reading the compute-matched point off the regular GRPO run. (3) 2026-09-29: the backbone section (7) added, before any backbone run.
(2) One correction, 2026-09-29, recorded
because this file had already been committed: condition 3 first read "aggregate
entropy orders the arms the same way accuracy does", which the full
fine-tuning manuscript had retired as the wrong instrument (it treats the
aggregate as a signal about A_H, which Section "The correction cannot move the
mean" says it cannot be). It was replaced by the dissociation test above, the
one that manuscript registered instead, before any LoRA run existed.

(4) 2026-09-30, after the first run had started. The first campaign run
(STEER-V, seed 1) was stopped at about step 12 and discarded, and the campaign
restarted from scratch with a different parallel layout. The decision used
step time and GPU memory only; no accuracy, entropy or validation value
entered it. Measured on the method's own step (`run/bench_lora.sh`,
`BENCH_ARM=signed`, median of steps 2-3), four H100s:

| layout | runs at once | s/step per run | peak memory | node steps/hour |
|---|---|---|---|---|
| one run on four GPUs (tp=4), eager vLLM | 1 | 848 | 62 GB | 4.24 |
| one run per GPU, vLLM CUDA graphs | 4 | 1063 | 72 GB | 13.55 |

The second is 3.2x the throughput and became the campaign's layout before any
kept run began. The stopped run's checkpoints and logs are archived, not
deleted, and are not used.
