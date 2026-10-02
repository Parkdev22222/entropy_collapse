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

Where to cut: fixed depths, or the first high-entropy token
-----------------------------------------------------------
``cut_mode="fixed"`` cuts every trunk at its design depth, whatever token sits
there.  If that token is near-deterministic the children all draw it, carry
on down the same road and score the same, and GRPO gets a tied group.

``cut_mode="entropy"`` keeps the tree's shape and moves each cut into a window
just past the fixed depth: level ``i`` generates its usual gap ``g_i`` (the
fixed design's stage budget) plus a window of ``w_i`` tokens, and is cut at
the **first** position in ``[g_i, g_i + w_i]`` whose next-token entropy is at
least ``tau_i``; if none is, at the window's end.  The token at the cut and
everything after it is thrown away and every child redraws it.

Why the *first* crossing and not the window's maximum: the first crossing is a
stopping time.  Whether the trunk is cut at ``k`` depends only on the entropies
at ``<= k``, which are functions of the tokens before ``k``, and the token at
``k`` itself is redrawn.  So every rollout is still, token for token, a draw
from the policy given its own prefix -- the tree changes which draws are
shared, not their distribution.  The window maximum looks at tokens after the
cut and would break that.  ``tau_i`` is the ``cut_quantile`` of level ``i``'s
window entropies on the *previous* call (``next_tau`` in the stats); the first
call has no previous one and takes the quantile of its own batch, a dependence
of one sequence on itself diluted over the whole batch, reported as
``tau_source="batch"``.

Entropies come from the engine's top-``cut_topk`` log-probs
(:func:`topk_entropy`), a lower bound on the true entropy.  With
``cut_mode="fixed"`` and ``cut_topk > 0`` the fixed cuts are kept but the
entropy at each cut is measured (the trunk draws one extra token and drops it),
so a fixed and an entropy tree can be compared on the same number.

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
    "topk_entropy",
]

CUT_MODES = ("fixed", "entropy")


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
        entropies: optional next-token entropy at each position (the entropy
            of the distribution ``token_ids[i]`` was drawn from), same length
            as ``token_ids``.  Filled only when the driver asks for ``topk``.
    """

    token_ids: list[int]
    finished: bool
    logprobs: Optional[list[float]] = None
    entropies: Optional[list[float]] = None

    def __post_init__(self):
        for name in ("logprobs", "entropies"):
            v = getattr(self, name)
            if v is not None and len(v) != len(self.token_ids):
                raise ValueError(
                    f"{name} has length {len(v)} but token_ids has "
                    f"{len(self.token_ids)}; they must line up token for token"
                )


# ``generate(prompts, n, max_tokens[, topk=K]) -> [len(prompts)][n] TreeSample``.
#
# `prompts` are full token-id sequences (task prompt + whatever trunk has been
# generated so far), because that is the only thing every inference engine
# accepts.  The driver never assumes the engine caches those prefixes; it is
# merely much faster when it does.  `max_tokens` is an int, or -- only under
# ``cut_mode="entropy"``, whose cuts differ per node -- one int per prompt.
# `topk` is passed only when entropies are wanted; the engine must then fill
# ``TreeSample.entropies``.
GenerateFn = Callable[..., list[list[TreeSample]]]


def topk_entropy(logprobs: Sequence[float]) -> float:
    """Entropy (nats) of a distribution known through its top-k log-probs.

    The mass outside the top k is lumped into one outcome, which can only
    lower the entropy, so this is a lower bound; at the near-deterministic
    positions that decide whether a cut is worth making it is close to exact.
    """
    ps = [math.exp(lp) for lp in logprobs]
    h = -sum(p * math.log(p) for p in ps if p > 0.0)
    rest = 1.0 - sum(ps)
    if rest > 1e-12:
        h -= rest * math.log(rest)
    return max(h, 0.0)


def _quantile(values: Sequence[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


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
        cut_mode: ``"fixed"`` (cut at ``depths``) or ``"entropy"`` (cut at the
            first token past each fixed depth whose entropy reaches the level's
            threshold; see the module docstring).
        cut_widths: ``"entropy"`` only -- the window each level may move its
            cut by, one per depth.
        cut_quantile: ``"entropy"`` only -- the threshold is this quantile of
            the level's window entropies on the previous call.
        cut_topk: top-k log-probs the entropy is computed from.  Required by
            ``"entropy"``; with ``"fixed"``, ``> 0`` measures the entropy at the
            fixed cuts without moving them.

    Invariant: ``roots * prod(factors) == n``.  Violating it either starves or
    overfills the group, and both corrupt GRPO silently, so it is checked here
    rather than discovered as a shape error twenty minutes into a run.
    """

    n: int
    response_length: int
    depths: tuple[int, ...] = ()
    factors: tuple[int, ...] = ()
    roots: int = 1
    cut_mode: str = "fixed"
    cut_widths: tuple[int, ...] = ()
    cut_quantile: float = 0.9
    cut_topk: int = 0

    def __post_init__(self):
        self.depths = parse_int_list(self.depths)
        self.factors = parse_int_list(self.factors)
        self.cut_widths = parse_int_list(self.cut_widths)
        self.cut_mode = str(self.cut_mode).strip().lower()
        self.cut_quantile = float(self.cut_quantile)
        self.cut_topk = int(self.cut_topk)
        if self.cut_mode not in CUT_MODES:
            raise ValueError(f"cut_mode must be one of {CUT_MODES}, got {self.cut_mode!r}")
        if self.cut_topk < 0:
            raise ValueError(f"cut_topk must be >= 0, got {self.cut_topk}")
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
            if self.cut_mode != "fixed" or self.cut_widths or self.cut_topk:
                raise ValueError("cut_mode / cut_widths / cut_topk need a tree; set depths too")
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
        if self.cut_mode == "fixed":
            if self.cut_widths:
                raise ValueError("cut_widths only applies to cut_mode='entropy'")
            return
        if len(self.cut_widths) != len(self.depths):
            raise ValueError(
                f"cut_mode='entropy' needs one window width per depth: depths "
                f"{self.depths}, cut_widths {self.cut_widths}"
            )
        if any(w < 1 for w in self.cut_widths):
            raise ValueError(f"every cut width must be >= 1, got {self.cut_widths}")
        if self.cut_topk < 1:
            raise ValueError("cut_mode='entropy' needs cut_topk >= 1 to compute entropies")
        if not 0.0 < self.cut_quantile < 1.0:
            raise ValueError(f"cut_quantile must be in (0, 1), got {self.cut_quantile}")
        reach = sum(self.cut_gaps) + sum(self.cut_widths)
        if reach >= self.response_length:
            raise ValueError(
                f"the deepest possible cut, {reach} (gaps {self.cut_gaps} + widths "
                f"{self.cut_widths}), must leave room inside response_length={self.response_length}"
            )

    @property
    def enabled(self) -> bool:
        return len(self.depths) > 0

    @property
    def measuring(self) -> bool:
        """Whether trunk stages ask the engine for entropies."""
        return self.enabled and (self.cut_mode == "entropy" or self.cut_topk > 0)

    @property
    def cut_gaps(self) -> tuple[int, ...]:
        """Tokens each level generates before its window: the fixed stage budgets."""
        return tuple(self.stage_budgets()[: len(self.depths)])

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
        cut = ""
        if self.cut_mode == "entropy":
            cut = (f", cut=entropy windows={list(self.cut_widths)} "
                   f"q={self.cut_quantile} topk={self.cut_topk}")
        elif self.cut_topk:
            cut = f", cut=fixed (measuring cut entropy, topk={self.cut_topk})"
        return (
            f"tree {' '.join(parts)} -> n={self.n}, {self.num_stages} stages, "
            f"budgets={self.stage_budgets()}, designed support <= t={self.depths[-1]} "
            f"({cov:.1%} of {self.response_length}){cut}"
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


def _flat_generate(generate: GenerateFn, prompts, n, max_tokens, topk=0):
    """Call the engine and check the shape it promised, loudly.

    ``topk`` is passed only when non-zero, so an engine written before
    entropies existed keeps working for every tree that does not ask for them.
    """
    if not prompts:
        return []
    out = generate(prompts, n, max_tokens, topk=topk) if topk else generate(prompts, n, max_tokens)
    if len(out) != len(prompts):
        raise RuntimeError(f"generate returned {len(out)} groups for {len(prompts)} prompts")
    limits = max_tokens if isinstance(max_tokens, (list, tuple)) else [max_tokens] * len(prompts)
    for i, (group, limit) in enumerate(zip(out, limits)):
        if len(group) != n:
            raise RuntimeError(f"generate returned {len(group)} samples for prompt {i}, expected n={n}")
        for s in group:
            if len(s.token_ids) > limit:
                raise RuntimeError(
                    f"generate returned {len(s.token_ids)} tokens for a max_tokens={limit} request"
                )
            if topk and (s.entropies is None or len(s.entropies) != len(s.token_ids)):
                raise RuntimeError(f"generate was asked for topk={topk} but returned no per-token entropies")
    return out


def _find_cut(sample: TreeSample, gap: int, width: int, tau: float):
    """Where a trunk stage's output is cut, and why.

    Returns ``(cut, kind)``: ``kind`` is ``"cross"`` (first position in
    ``[gap, gap + width]`` with entropy >= ``tau``), ``"fallback"`` (no crossing,
    cut at ``gap + width``) or ``"done"`` (the policy stopped before either; the
    rollout is complete and ``cut`` is its length).  The scan stops at the
    first crossing -- see the module docstring for why that matters.
    """
    ids, ent = sample.token_ids, sample.entropies
    end = gap + width
    for k in range(gap, min(end, len(ids) - 1) + 1):
        if ent[k] >= tau:
            return k, "cross"
    if len(ids) > end:
        return end, "fallback"
    if sample.finished:
        return len(ids), "done"
    # Neither stopped nor long enough: only an engine that under-delivers
    # gets here.  Keep what there is, as the fixed path would.
    return len(ids), "fallback"


def generate_tree(
    generate: GenerateFn,
    prompt_token_ids: Sequence[Sequence[int]],
    config: TreeRolloutConfig,
    collect_logprobs: bool = False,
    max_token_id: Optional[int] = None,
    tau: Optional[Sequence[Optional[float]]] = None,
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
        tau: ``cut_mode="entropy"`` only -- one threshold per level, normally
            the previous call's ``stats["next_tau"]``.  ``None`` (or a ``None``
            entry) takes the quantile of this call's own window entropies.

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

    entropy_cut = config.enabled and config.cut_mode == "entropy"
    measuring = config.measuring
    n_levels = len(config.depths)
    tau_in = list(tau) if tau is not None else [None] * n_levels
    if len(tau_in) != n_levels:
        raise ValueError(f"tau has {len(tau_in)} entries for {n_levels} cut levels")
    tau_used: list[Optional[float]] = [None] * n_levels
    next_tau: list[Optional[float]] = list(tau_in)
    tau_source = "prev" if all(t is not None for t in tau_in) else "batch"
    cut_depths: list[list[int]] = [[] for _ in range(n_levels)]
    cut_entropies: list[float] = []
    window_entropies_all: list[float] = []
    n_cuts = n_fallback = 0
    generated = discarded = 0

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
        trunk = stage < last_stage
        if trunk and measuring:
            # One token past the window so the entropy at its last position is
            # known; whatever lies at or after the cut is redrawn by the children.
            gap = budget
            width = config.cut_widths[stage] if entropy_cut else 0
            out = _flat_generate(generate, conds, fan, gap + width + 1, topk=config.cut_topk)
            level_tau = math.inf  # fixed cuts: measure, never move
            if entropy_cut:
                window = [e for group in out for s in group for e in s.entropies[gap:gap + width + 1]]
                window_entropies_all.extend(window)
                if window:
                    next_tau[stage] = _quantile(window, config.cut_quantile)
                level_tau = tau_in[stage] if tau_in[stage] is not None else next_tau[stage]
                if level_tau is None:  # nothing reached the window: nothing to cut
                    level_tau = math.inf
            tau_used[stage] = level_tau
        elif not trunk and entropy_cut:
            # Cuts differ per node, so the last stage's budgets do too.
            out = _flat_generate(generate, conds, fan,
                                 [config.response_length - len(parent.tokens) for parent in parents])
        else:
            out = _flat_generate(generate, conds, fan, budget)
        calls += 1
        generated += sum(len(s.token_ids) for group in out for s in group)

        for group, p, parent in zip(out, owners, parents):
            for j, s in enumerate(group):
                cut_kind = None
                if trunk and measuring:
                    cut, cut_kind = _find_cut(s, gap, width, level_tau)
                    if cut_kind != "done":
                        n_cuts += 1
                        n_fallback += cut_kind == "fallback"
                        if cut < len(s.token_ids):
                            cut_entropies.append(s.entropies[cut])
                    discarded += len(s.token_ids) - cut
                    s = TreeSample(
                        token_ids=list(s.token_ids[:cut]),
                        finished=cut_kind == "done",
                        logprobs=None if s.logprobs is None else list(s.logprobs[:cut]),
                    )
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
                if cut_kind in ("cross", "fallback"):
                    cut_depths[stage].append(len(child.tokens))
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
        generated += sum(len(group[0].token_ids) for group in out)
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
    if measuring:
        def _mean(xs):
            return sum(xs) / len(xs) if xs else float("nan")

        stats.update({
            "cut_mode": config.cut_mode,
            "cut_entropy_mean": _mean(cut_entropies),
            "cut_depth_mean": [_mean(d) for d in cut_depths],
            "wasted_tok_frac": discarded / generated if generated else 0.0,
        })
        if entropy_cut:
            stats.update({
                "tau": tau_used,
                "tau_source": tau_source,
                "next_tau": next_tau,
                "window_entropy_mean": _mean(window_entropies_all),
                "fallback_frac": n_fallback / n_cuts if n_cuts else float("nan"),
            })
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
