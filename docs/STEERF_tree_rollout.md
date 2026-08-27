# Tree-structured rollouts

## The problem

`entropy_forecast.sibling_prefix_baseline` defines rollout `j` to be a sibling
of rollout `i` at position `t` iff both come from the same prompt and
`responses[i, :t] == responses[j, :t]`. `A_H` is the deviation of a rollout's
forecast from that sibling mean, so where a rollout is its own only sibling the
baseline **is** its own value and

```
A_H(s_t, y_t) = H_togo(s_t + y_t) - H_bar_togo(s_t) = 0     exactly
```

and the whole visitation term `v = eta * w * (1 - pi) * clip(A_H, -c, c)` is
multiplied by zero.

verl's stock rollout draws the `n` samples of a prompt i.i.d. from the policy.
They part company within a handful of tokens and never meet again. Measured on
`train-math-Qwen2.5-Math-1.5B-s1`:

| quantity | value |
|---|---|
| `steerf/branch_corr_frac` | 0.003 |
| positions with `A_H == 0` at `T = 1024` | 99.4% |

The forecast term is not weak in that region. It is undefined there, and the
"forecast contributes nothing" reading of the training logs is a statement
about the sampler, not about the MTP heads.

The offline Part A protocol (`scripts/phase1_sibling_spread.py`) does not have
the problem: it branches `K_mc` continuations off one shared prefix and gets
**28%** support. Same model, same definition of sibling, two orders of
magnitude apart — because it branches and training does not.

## The design

Sample the `n` rollouts of a prompt as a tree. With `roots` independent trunks,
cut depths `d_1 < ... < d_L` and branch factors `f_1, ..., f_L` obeying
`roots * prod(f) == n`:

```
stage 0    prompt                     -> roots sequences of length d_1
stage i    each sequence, n = f_i     -> ... of length d_{i+1}
stage L    each sequence, n = f_L     -> completion to the length budget
```

Every rollout below a cut point shares its prefix with `f_i * ... * f_L - 1`
others *by construction*. For `n=8, roots=1, depths=(128,384,640),
factors=(2,2,2)` at `T=1024`:

| position | siblings |
|---|---|
| `t < 128` | 8 |
| `128 <= t < 384` | 4 |
| `384 <= t < 640` | 2 |
| `t >= 640` | 1 |

62.5% of the response inside the support, against 0.3% today.

Implementation: `steer_f/tree_rollout.py` (engine-agnostic, driven by a
`generate(prompts, n, max_tokens)` callable), wired to vLLM in
`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`, exercised on CPU
against a fake engine by `scripts/smoke_tree_rollout_cpu.py`.

### Trunks that stop early

A trunk that emits EOS before its cut depth is a *finished* rollout and cannot
branch. Its unused sibling slots are refilled with fresh i.i.d. samples from
the prompt — what the stock sampler would have produced — and the fraction
filled that way is reported as `refill_frac`.

They are deliberately **not** filled by copying the finished sequence. That
would be free support and a lie twice over: the copies carry identical rewards
so GRPO's group advantage for them is exactly zero, and every sibling statistic
would be inflated by rollouts that were never independently sampled.

## Installing it

Three brand-new files and one edited file. The new ones come across whole, so
nothing can conflict with local edits:

```bash
# from the root of the TRAINING tree
git fetch origin claude/3b-text-generation-models-thz2vl
git checkout origin/claude/3b-text-generation-models-thz2vl -- \
    steer_f/tree_rollout.py \
    scripts/smoke_tree_rollout_cpu.py \
    docs/STEERF_tree_rollout.md \
    patches/steerf_tree_rollout.patch

git apply patches/steerf_tree_rollout.patch   # touches only vllm_rollout_spmd.py
python3 scripts/smoke_tree_rollout_cpu.py     # [1]-[8] must all PASS
```

If `git apply` reports a context mismatch, the local `vllm_rollout_spmd.py` has
drifted; `git apply -3 patches/steerf_tree_rollout.patch` resolves it via the
object store. `run/run_steerf.sh` is deliberately **not** patched -- it forwards
`"$@"` to hydra, so the tree is configured from the command line instead and
local edits to that script cannot collide with this change.

## Running it

Three hydra overrides, appended to whatever `run_steerf.sh` invocation is
already in use. Omitting them is bit-identical to every run so far.

```bash
MODEL_PATH=Qwen/Qwen2.5-Math-1.5B SEED=1 \
RUN_NAME=math-Qwen2.5-Math-1.5B-s1-tree-rollout \
bash run/run_steerf.sh \
  "++actor_rollout_ref.rollout.steerf_tree_depths=[64,192,384]" \
  "++actor_rollout_ref.rollout.steerf_tree_factors=[2,2,2]" \
  "++actor_rollout_ref.rollout.steerf_tree_roots=1"
```

The brackets matter. Bare `64,192,384` is a hydra *sweep*, which `main_ppo`
rejects because it is not a multirun; `[64,192,384]` is a list and
`'64,192,384'` is a string, and `parse_int_list` accepts either. The outer
double quotes are for the shell, so the brackets survive globbing.

`roots * prod(factors)` must equal `actor_rollout_ref.rollout.n`; it is checked
at startup rather than discovered as a shape error mid-run.

### Choosing the depths

The deepest cut has to sit inside the bulk of the response-length distribution,
or most trunks finish before they can branch. Measured on the 1.5B run at
`RESP_LEN=3072`: `response_length/mean ~= 1010`, `response_length/min ~= 165`
over a 4096-rollout batch. `64,192,384` is the conservative starting point;
push the last cut deeper only after reading `refill_frac` off step 1.

Rules of thumb:

* keep `refill_frac < 0.1` -- above that the tree is quietly degrading into the
  flat sampler for a large minority of the batch;
* the support profile is a **step function** of the cuts, so `A_H` is still
  exactly zero beyond `d_L`. More, shallower cuts buy coverage; more, deeper
  ones buy sibling *count* where the coverage already exists.

### What to watch

| metric | where | expected |
|---|---|---|
| `steerf/branch_corr_frac` | training log | 0.003 -> order 0.5 |
| `support_frac` | `[steerf-tree]` rollout log line | matches `expected_mean_siblings` design |
| `refill_frac` | same line | < 0.1 |
| `oov_dropped` | same line | small; see below |
| `steerf/adv_zero_frac` | training log | **watch for an increase** (see below) |
| `response_length/mean` | training log | unchanged; the tree does not shorten responses |

### Tokens the tokenizer cannot represent

Qwen2.5-Math-1.5B has `vocab_size = 151936` against a tokenizer whose largest
id is well below that, and at temperature 1.0 the unused lm_head rows do get
sampled. Flat rollouts never notice: the id lands in the response tensor,
decodes to nothing, and the rollout scores as wrong. The tree feeds prefixes
back in as prompts, where the same id is a hard

```
ValueError: Token id 151878 is out of vocabulary
```

from vLLM's input validation. A trunk carrying such a token is therefore
dropped and its slots go through the ordinary refill path -- the rollout was
garbage either way, and dropping the node rather than patching the token keeps
the response identical to the prefix it was actually sampled under.

Only sequences that will be fed back in are checked. The last stage generates
most of the tokens and goes straight to the output, so policing it would
discard whole rollouts to fix a problem nothing downstream has.

The bound is `max(tokenizer.get_vocab().values())`, which is how
`vllm.transformers_utils.tokenizer.get_cached_tokenizer` computes the
`max_token_id` the v1 processor validates against. If it cannot be read the
rollout logs a warning and the check is skipped.

## Caveats, stated plainly

**1. The rollouts in a group are no longer independent.** Each leaf still has
the exactly correct marginal — a trunk is drawn from `pi`, a continuation from
`pi(. | trunk)`, so the leaf is distributed as `pi(. | prompt)` — which keeps
GRPO's group-mean baseline unbiased. But the group *variance* shrinks, because
siblings share a trunk and therefore share much of whatever the trunk
determined. Two consequences:

* the normalised advantage `(r - mean) / std` is inflated by the smaller `std`;
* groups where every rollout gets the same reward become more likely, and those
  contribute exactly zero gradient. `adv_zero_frac` was already 0.52 on the
  flat sampler. If the tree pushes it materially higher, the support gain is
  being paid for in dead batches and the cuts are too deep.

**2. This changes the sampler, not just the loss.** Any comparison against the
existing STEER baseline now differs in two places at once. The control arm that
attributes a gain to the forecast rather than to the tree is

```bash
STEERF_LAM=0 STEERF_MAPPING=minmax bash run/run_steerf.sh \
  "++actor_rollout_ref.rollout.steerf_tree_depths=[64,192,384]" \
  "++actor_rollout_ref.rollout.steerf_tree_factors=[2,2,2]" \
  "++actor_rollout_ref.rollout.steerf_tree_roots=1"
```

i.e. stock STEER *with* tree rollouts. Without it, a win is unattributable —
the same defect the current A3 arm has by changing `lam` and `mapping`
together.

**3. Decode tokens go down, prefill goes up.** Shared trunks are generated once
and reused by every rollout beneath them, so the decode budget drops (measured
0.58x in the CPU smoke at `n=8` with three cuts). Against that, each branch
stage re-prefills `prompt + trunk`. `enable_prefix_caching=True` is already set
in `vLLMRollout.__init__`, which makes that re-prefill nearly free; with prefix
caching off the tree is slower, and the rollout logs a warning saying so.

**4. Validation is untouched.** The tree runs only when
`do_sample and not is_validate`. Validation neither computes `A_H` nor may have
its sampling distribution altered, so `acc/mean@32`, `pass@32` and `maj@32`
remain comparable to every run in the table.

**5. Not supported: multi-modal prompts and LoRA.** A branch re-conditions on
`prompt + trunk` token ids, which cannot carry an image payload; and verl
builds one `LoRARequest` per row of the original batch, while the tree's stages
have different batch sizes. Both raise `NotImplementedError` at rollout time
rather than producing quietly wrong data.
