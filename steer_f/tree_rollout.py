# Copyright 2026 STEER-F authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tree-structured rollouts: give ``A_H``'s sibling baseline an actual support.

The problem this exists to fix
------------------------------
``entropy_forecast.sibling_prefix_baseline`` defines rollout ``j`` to be a
sibling of rollout ``i`` at position ``t`` iff both come from the same prompt
and ``responses[i, :t] == responses[j, :t]``.  ``A_H`` is the deviation of a
rollout's forecast from that sibling mean, so wherever a rollout is its own
only sibling the baseline equals its own value and ``A_H`` is **exactly zero**
-- not small, zero.

Under verl's stock rollout the ``n`` samples of a prompt are drawn i.i.d. from
the policy, so they part company within a handful of tokens and never meet
again.  Measured on the real training run: ``branch_corr_frac = 0.003``.  At
``T = 1024`` that is 99.4% of positions where the visitation term is
identically zero, which is why the forecast contribution reads as numerically
dead in the logs.  The forecast is not weak there; it is undefined.

The offline Part A protocol (``scripts/phase1_sibling_spread.py``) does not
have this problem -- it *branches* ``K_mc`` continuations off one shared prefix
and gets 28% support -- so the gap is a property of how rollouts are sampled,
not of the method.

What this module does
---------------------
Sample the ``n`` rollouts of a prompt as a tree instead of a flat i.i.d. batch.
With ``roots`` independent trunks, cut depths ``d_1 < ... < d_L`` and branch
factors ``f_1, ..., f_L`` such that ``roots * prod(f) == n``:

    stage 0   prompt                      -> roots  sequences of length d_1
    stage i   each sequence, n = f_i      -> ... of length d_{i+1}
    stage L   each sequence, n = f_L      -> completion to the length budget

Every rollout below a cut point shares its prefix with ``f_i * ... * f_L - 1``
others *by construction*, so the support is designed in rather than hoped for.
For ``n = 8, roots = 1, depths = (128, 384, 640), factors = (2, 2, 2)`` the
sibling count is 8 for ``t <= 128``, 4 up to 384, 2 up to 640 -- 62.5% of a
1024-token response inside the support, against today's 0.3%.

The cost is ``L + 1`` engine calls per step instead of one, over the same total
token budget; prefix caching makes the re-prefill of the shared trunks cheap,
and the branch stages are the same tokens the flat sampler would have drawn
anyway.

What it deliberately does not do
--------------------------------
A trunk that emits EOS before reaching its cut depth is a *finished* rollout
and cannot branch.  Copying it into its unused sibling slots would be free
support and a lie twice over: the duplicates carry identical rewards (GRPO's
group advantage for them is exactly zero) and they would inflate every sibling
statistic with rollouts that were never independently sampled.  Instead the
unused slots are refilled with fresh i.i.d. samples from the prompt -- exactly
what the stock sampler would have produced -- and the fraction of slots filled
that way is reported as ``refill_frac`` so it can be read off rather than
guessed at.

This module is engine-agnostic on purpose: it drives a ``generate`` callable
with the signature described in :class:`GenerateFn`, so it is exercised on CPU
against a fake engine in ``scripts/smoke_tree_rollout_cpu.py`` and wired to
vLLM by ``patches/steerf_tree_rollout.patch``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

__all__ = [
    "TreeSample",
    "TreeRolloutConfig",
    "TreeRolloutResult",
    "generate_tree",
    "sibling_support_stats",
    "parse_int_list",
]


# ----------------------------------------------------------------------
# engine interface
# ----------------------------------------------------------------------
@dataclass
class TreeSample:
    """One continuation returned by the generation engine.

    Attributes:
        token_ids: the newly generated tokens only, not the conditioning prefix.
        finished: ``True`` when generation stopped on its own (EOS / stop
            string), ``False`` when it stopped because it hit ``max_tokens``.
            Only ``finished=False`` samples may be branched further -- a
            sequence that already emitted EOS has no future to branch into.
        logprobs: optional per-token sampling log-probabilities, same length as
            ``token_ids``.  Carried through so the caller can reconstruct
            ``rollout_log_probs`` across stages.
    """

    token_ids: list[int]
    finished: bool
    logprobs: Optional[list[float]] = None

    def __post_init__(self):
        if self.logprobs is not None and len(self.logprobs) != len(self.token_ids):
            raise ValueError(
                f"logprobs has length {len(self.logprobs)} but token_ids has "
                f"{len(self.token_ids)}; they must line up token for token"
            )


# ``generate(prompts, n, max_tokens) -> [len(prompts)][n] TreeSample``.
#
# `prompts` are full token-id sequences (task prompt + whatever trunk has been
# generated so far), because that is the only thing every inference engine
# accepts.  The driver never assumes the engine caches those prefixes; it is
# merely much faster when it does.
GenerateFn = Callable[[list[list[int]], int, int], list[list[TreeSample]]]


# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
def parse_int_list(spec: str | Sequence[int] | None) -> tuple[int, ...]:
    """``"128,384,640"`` -> ``(128, 384, 640)``.  Empty / None -> ``()``.

    Every spelling this value can arrive in has to mean the same thing:
    ``rollout.steerf_tree_depths=[128,384,640]`` reaches the config as an
    OmegaConf list, the same override written ``='128,384,640'`` reaches it as
    a string, and a shell launcher may hand over either.  Brackets are stripped
    rather than rejected so a string that kept them round-trips too.
    """
    if spec is None:
        return ()
    if isinstance(spec, str):
        spec = spec.strip().strip("[]()")
        spec = [p for p in spec.replace(" ", "").split(",") if p]
    return tuple(int(x) for x in spec)


@dataclass
class TreeRolloutConfig:
    """Shape of the rollout tree.

    Args:
        n: rollouts per prompt.  Must equal verl's ``rollout.n`` -- everything
            downstream (GRPO grouping, ``uid`` repetition, the reward tensor's
            batch dimension) is built on that number, so the tree has to hand
            back exactly as many rollouts as the flat sampler would have.
        response_length: the total generation budget per rollout, i.e. verl's
            ``data.max_response_length``.  Cut depths are positions inside it.
        depths: strictly increasing cut depths in *response* tokens.  Empty
            disables the tree entirely and the driver degrades to one flat
            call, which is bit-equivalent to the stock sampler.
        factors: branch factor applied at each depth; same length as ``depths``.
        roots: independent trunks per prompt.

    Invariant: ``roots * prod(factors) == n``.  Violating it either starves or
    overfills the group, and both corrupt GRPO silently, so it is checked here
    rather than discovered as a shape error twenty minutes into a run.
    """

    n: int
    response_length: int
    depths: tuple[int, ...] = ()
    factors: tuple[int, ...] = ()
    roots: int = 1

    def __post_init__(self):
        self.depths = parse_int_list(self.depths)
        self.factors = parse_int_list(self.factors)
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        if self.response_length < 1:
            raise ValueError(f"response_length must be >= 1, got {self.response_length}")
        if len(self.depths) != len(self.factors):
            raise ValueError(
                f"depths has {len(self.depths)} entries but factors has "
                f"{len(self.factors)}; each cut depth needs its own branch factor"
            )
        if not self.enabled:
            if self.roots not in (1, self.n):
                raise ValueError(f"with no depths, roots must be 1 or n={self.n}, got {self.roots}")
            self.roots = self.n
            return
        if self.roots < 1:
            raise ValueError(f"roots must be >= 1, got {self.roots}")
        if any(f < 1 for f in self.factors):
            raise ValueError(f"every branch factor must be >= 1, got {self.factors}")
        if any(b <= a for a, b in zip(self.depths, self.depths[1:])):
            raise ValueError(f"depths must be strictly increasing, got {self.depths}")
        if self.depths[0] < 1:
            raise ValueError(f"the first cut depth must be >= 1, got {self.depths[0]}")
        if self.depths[-1] >= self.response_length:
            raise ValueError(
                f"the last cut depth {self.depths[-1]} must leave room to finish inside "
                f"response_length={self.response_length}; a branch with a zero-token "
                "budget produces empty rollouts"
            )
        total = self.roots * math.prod(self.factors)
        if total != self.n:
            raise ValueError(
                f"roots({self.roots}) * prod(factors){self.factors} = {total}, but n={self.n}. "
                "The tree must yield exactly n rollouts per prompt."
            )

    @property
    def enabled(self) -> bool:
        return len(self.depths) > 0

    @property
    def num_stages(self) -> int:
        """Engine calls per prompt, refills excluded."""
        return len(self.depths) + 1

    def stage_budgets(self) -> list[int]:
        """Tokens generated by each stage: ``d1, d2-d1, ..., response_length-dL``."""
        if not self.enabled:
            return [self.response_length]
        edges = (0,) + self.depths + (self.response_length,)
        return [b - a for a, b in zip(edges, edges[1:])]

    def expected_siblings_at(self, t: int) -> int:
        """How many rollouts share a prefix through position ``t``, if none died.

        The design-time number the empirical ``sibling_support_stats`` is
        checked against.  ``t`` is 0-based; a rollout at ``t < depths[0]`` is
        still on its trunk and shares it with everything below that trunk.
        """
        if not self.enabled:
            return 1
        remaining = math.prod(self.factors)
        for d, f in zip(self.depths, self.factors):
            if t < d:
                return remaining
            remaining //= f
        return 1

    def expected_mean_siblings(self) -> float:
        """Design-time mean sibling count over a full-length response.

        The number ``sibling_support_stats(...)["mean_siblings"]`` converges to
        when no rollout dies early, so the two can be compared instead of a
        threshold being guessed at.
        """
        return sum(self.expected_siblings_at(t) for t in range(self.response_length)) / self.response_length

    def describe(self) -> str:
        if not self.enabled:
            return f"tree disabled (flat n={self.n})"
        parts = [f"roots={self.roots}"] + [f"@{d}x{f}" for d, f in zip(self.depths, self.factors)]
        cov = self.depths[-1] / self.response_length
        return (
            f"tree {' '.join(parts)} -> n={self.n}, {self.num_stages} stages, "
            f"budgets={self.stage_budgets()}, designed support <= t={self.depths[-1]} "
            f"({cov:.1%} of {self.response_length})"
        )


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------
@dataclass
class _Node:
    """A sequence under construction inside one prompt's tree."""

    tokens: list[int]
    logprobs: list[float]
    finished: bool
    path: tuple[int, ...]           # branch index taken at each stage so far


@dataclass
class TreeRolloutResult:
    """Rollouts laid out as ``[prompt][n]``, plus what the tree actually did.

    ``responses[p][j]`` is the full response token id list for rollout ``j`` of
    prompt ``p``: trunk and every continuation concatenated, so it is
    indistinguishable in shape from what a flat sampler returns.
    """

    responses: list[list[list[int]]]
    logprobs: list[list[list[float]]]
    paths: list[list[tuple[int, ...]]]
    stats: dict = field(default_factory=dict)


def _flat_generate(generate: GenerateFn, prompts, n, max_tokens):
    """Call the engine and check the shape it promised, loudly."""
    if not prompts:
        return []
    out = generate(prompts, n, max_tokens)
    if len(out) != len(prompts):
        raise RuntimeError(f"generate returned {len(out)} groups for {len(prompts)} prompts")
    for i, group in enumerate(out):
        if len(group) != n:
            raise RuntimeError(f"generate returned {len(group)} samples for prompt {i}, expected n={n}")
        for s in group:
            if len(s.token_ids) > max_tokens:
                raise RuntimeError(
                    f"generate returned {len(s.token_ids)} tokens for a max_tokens={max_tokens} request"
                )
    return out


def generate_tree(
    generate: GenerateFn,
    prompt_token_ids: Sequence[Sequence[int]],
    config: TreeRolloutConfig,
    collect_logprobs: bool = False,
    max_token_id: Optional[int] = None,
) -> TreeRolloutResult:
    """Sample ``config.n`` tree-structured rollouts for every prompt.

    All prompts advance through the stages together -- one engine call per
    stage for the whole batch, not per prompt -- because an inference engine
    only reaches its throughput with a full batch, and a per-prompt loop would
    turn ``L+1`` calls into ``L+1`` times the number of prompts.

    Args:
        generate: see :data:`GenerateFn`.
        prompt_token_ids: ``[P]`` task prompts, already tokenised.
        config: validated tree shape.
        collect_logprobs: keep per-token sampling logprobs.  Requires the
            engine to populate ``TreeSample.logprobs``.
        max_token_id: largest token id the engine will accept back as input.
            A model whose ``vocab_size`` exceeds its tokenizer -- Qwen2.5-Math
            has 151936 output rows against a tokenizer that stops well below
            that -- can sample one of the unused rows.  Flat sampling does not
            care: the id lands in the response tensor, decodes to nothing and
            scores as wrong.  The tree feeds prefixes back in as prompts, where
            the same id is a hard ``Token id N is out of vocabulary`` from the
            engine.  Given this bound, such a trunk is dropped and its slots go
            through the ordinary refill path, since the rollout was garbage
            either way; left ``None`` no check is made.

    Returns:
        :class:`TreeRolloutResult` with exactly ``config.n`` responses per
        prompt, in depth-first tree order so siblings are adjacent.
    """
    prompts = [list(p) for p in prompt_token_ids]
    n_prompts = len(prompts)
    if n_prompts == 0:
        return TreeRolloutResult([], [], [], {"num_engine_calls": 0})

    # nodes[p] is the frontier of prompt p's tree.
    nodes: list[list[_Node]] = [[] for _ in range(n_prompts)]
    done: list[list[_Node]] = [[] for _ in range(n_prompts)]
    calls = 0

    budgets = config.stage_budgets()
    factors = (config.roots,) + config.factors

    n_oov = 0

    def _in_vocab(token_ids) -> bool:
        return max_token_id is None or not token_ids or max(token_ids) <= max_token_id

    last_stage = len(factors) - 1

    for stage, (fan, budget) in enumerate(zip(factors, budgets)):
        if stage == 0:
            conds = prompts
            owners = list(range(n_prompts))
            parents: list[Optional[_Node]] = [None] * n_prompts
        else:
            conds, owners, parents = [], [], []
            for p in range(n_prompts):
                for node in nodes[p]:
                    conds.append(prompts[p] + node.tokens)
                    owners.append(p)
                    parents.append(node)
            nodes = [[] for _ in range(n_prompts)]

        if not conds:
            break
        out = _flat_generate(generate, conds, fan, budget)
        calls += 1

        for group, p, parent in zip(out, owners, parents):
            for j, s in enumerate(group):
                # Only what will be fed back in as a prompt is checked:
                # sequences from the last stage go straight to the output, and
                # an unusable id there is exactly as harmless as it is under
                # flat sampling.  Checking them anyway would discard whole
                # rollouts -- most of the generated tokens live past the last
                # cut -- to fix a problem nothing downstream has.
                #
                # New tokens only: the parent's were checked when it was
                # created and the task prompt came from the dataset.  Dropping
                # the node rather than the offending token keeps the response
                # and the prefix it was sampled under identical, which
                # truncating or patching would not.
                if stage < last_stage and not _in_vocab(s.token_ids):
                    n_oov += 1
                    continue
                base_tokens = parent.tokens if parent else []
                base_lp = parent.logprobs if parent else []
                child = _Node(
                    tokens=base_tokens + list(s.token_ids),
                    logprobs=(base_lp + list(s.logprobs or [])) if collect_logprobs else [],
                    finished=s.finished,
                    path=(parent.path if parent else ()) + (j,),
                )
                # A node that stopped on its own, or that has already used its
                # whole budget, cannot be branched further.
                if child.finished or len(child.tokens) >= config.response_length:
                    done[p].append(child)
                else:
                    nodes[p].append(child)

    # Anything still on the frontier after the last stage is a complete rollout.
    for p in range(n_prompts):
        done[p].extend(nodes[p])
        nodes[p] = []

    # ---- refill the slots that early-finished trunks could not branch into --
    refill_conds, refill_owner = [], []
    for p in range(n_prompts):
        have = len(done[p])
        missing = config.n - have
        if missing < 0:
            raise RuntimeError(f"prompt {p} produced {have} rollouts, more than n={config.n}")
        for _ in range(missing):
            refill_conds.append(prompts[p])
            refill_owner.append(p)

    n_refill = len(refill_conds)
    if n_refill:
        out = _flat_generate(generate, refill_conds, 1, config.response_length)
        calls += 1
        for group, p in zip(out, refill_owner):
            s = group[0]
            done[p].append(
                _Node(
                    tokens=list(s.token_ids),
                    logprobs=list(s.logprobs or []) if collect_logprobs else [],
                    finished=s.finished,
                    path=(-1,),  # -1 marks "not part of the tree"
                )
            )

    responses, logprobs, paths = [], [], []
    for p in range(n_prompts):
        if len(done[p]) != config.n:
            raise RuntimeError(f"prompt {p} produced {len(done[p])} rollouts, expected n={config.n}")
        ordered = sorted(done[p], key=lambda nd: nd.path)
        responses.append([nd.tokens for nd in ordered])
        logprobs.append([nd.logprobs for nd in ordered])
        paths.append([nd.path for nd in ordered])

    stats = {
        "num_engine_calls": calls,
        "num_refilled": n_refill,
        "refill_frac": n_refill / (n_prompts * config.n),
        "num_oov_dropped": n_oov,
        "config": config.describe(),
    }
    stats.update(sibling_support_stats(responses))
    return TreeRolloutResult(responses, logprobs, paths, stats)


# ----------------------------------------------------------------------
# diagnostics
# ----------------------------------------------------------------------
def sibling_support_stats(groups: Sequence[Sequence[Sequence[int]]]) -> dict:
    """The number this whole module exists to move.

    Reproduces :func:`steer_f.entropy_forecast.sibling_support` offline, on
    ragged token id lists, without a GPU: rollout ``j`` is a sibling of ``i`` at
    ``t`` iff both are alive at ``t`` and agree on every token before ``t``.
    ``A_H`` is identically zero wherever the count is 1, so
    ``support_frac`` -- the fraction of alive positions with at least one other
    sibling -- is the fraction of the response where the forecast term is even
    defined.  Training measured 0.003 of it.

    Implementation note: rather than the ``[G, G, T]`` pairwise divergence
    tensor, this refines a partition.  Sequences that agree on ``[0, t)`` sit in
    one bucket; at each step the dead ones leave (dying *is* a divergence, which
    is why they must leave rather than be masked in place) and the survivors
    split by their token at ``t``.  Same answer, ``O(G*T)`` and no torch.

    Args:
        groups: ``[P][G]`` ragged token id lists, one group per prompt.

    Returns:
        ``support_frac``, ``mean_siblings`` (over alive positions),
        ``alive_positions``, and ``support_frac_by_decile`` -- the profile that
        shows where in the response the support actually lives, since a single
        mean hides "all of it is in the first 20 tokens".
    """
    total_alive = 0
    total_supported = 0
    sibling_sum = 0
    decile_alive = [0] * 10
    decile_supported = [0] * 10

    for group in groups:
        g = len(group)
        if g == 0:
            continue
        lengths = [len(r) for r in group]
        t_max = max(lengths)
        if t_max == 0:
            continue
        buckets = [list(range(g))]
        for t in range(t_max):
            dec = min(9, (t * 10) // t_max)
            nxt = []
            for bucket in buckets:
                alive = [i for i in bucket if lengths[i] > t]
                if not alive:
                    continue
                cnt = len(alive)
                total_alive += cnt
                sibling_sum += cnt * cnt
                decile_alive[dec] += cnt
                if cnt > 1:
                    total_supported += cnt
                    decile_supported[dec] += cnt
                if cnt == 1:
                    nxt.append(alive)
                    continue
                split: dict[int, list[int]] = {}
                for i in alive:
                    split.setdefault(group[i][t], []).append(i)
                nxt.extend(split.values())
            buckets = nxt

    if total_alive == 0:
        return {"support_frac": 0.0, "mean_siblings": 0.0, "alive_positions": 0,
                "support_frac_by_decile": [0.0] * 10}
    return {
        "support_frac": total_supported / total_alive,
        "mean_siblings": sibling_sum / total_alive,
        "alive_positions": total_alive,
        "support_frac_by_decile": [
            (s / a if a else 0.0) for s, a in zip(decile_supported, decile_alive)
        ],
    }
