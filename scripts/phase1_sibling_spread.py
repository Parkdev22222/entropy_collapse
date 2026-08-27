#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Measure the two things C5 corollary 1 and C4 make load-bearing but nobody has measured.

Why this script exists
----------------------
`phase1_validate.py` measures `gt_branch_div` as **one scalar per cut point**:
the mean pairwise distance among *all* K_mc continuations sampled from one
prefix.  That answers

    "how much do futures fan out *from this position*?"        (a property of the position)

which is what C1 needs.  It does not answer

    "at this branch point, whose future is wider -- sibling a's or sibling b's?"
                                                              (a comparison *across* siblings)

and that second question is the one `A_H` exists to answer.  C5 corollary 1
states the consequence precisely:

    갈래들의 미래가 똑같이 다양하면 (A_H === 0) 이 항은 정확히 0이고 Omega 가
    전부를 설명한다. 기존 방법이 실패하는 정도는 갈래별 미래 다양성의
    *불균등*에 정확히 비례한다.

So "sibling futures are unequally diverse" is a *precondition* for STEER-F to
have anything to correct.  Training logs report `a_h_std = 0.009` against
`h_togo_mean = 0.351` (2.6%), which is consistent with two very different
worlds and cannot distinguish them:

    (A) sibling futures really are near-equally diverse
        -> corollary 1 fires, the visitation term is *correctly* ~0,
           and no amount of head/support/horizon work changes that.
    (B) they differ, but the forecast cannot see it
        -> a fixable estimation problem (C6 shrinks the effective horizon
           to ~1 step: H_togo = 0.721 H_1 + 0.064 H_2 + 0.106).

Part A below separates them by measuring the ground-truth spread directly.

Part B then asks, given real spread, whether either estimator tracks it:
the MTP forecast (what training uses) and the oracle realised entropy (what
`STEERF_FORECAST=oracle` would use).  Three outcomes, three different papers.

Part C is M1 from `STEERF_claims.md` §3, which C4 needs to go from [계산] to
[실측].  C4 tabulates `E|Omega|/|A|` and the visitation prefactor `E[1-pi_a]`
over an *analytic* distribution family and notes:

    [미검증] ... **실제 롤아웃에서 두 인자의 곱을 잰 적이 없다.**

Here both are computed on real rollouts and regressed against how branchy the
position actually is (measured by resampling, not inferred from entropy).

Note on the two Omega forms: `local_omega_signed` implements the released
STEER form `-(A/pi_old) * x(1-x)(log x + H)`, while C2's identity form is
`A (1-pi_a)(I_a - H)`.  At a *pre-update* point `pi_theta == pi_old`, so

    -(1/pi) * pi(1-pi)(log pi + H) = (1-pi)(-log pi - H) = (1-pi)(I_a - H)

the two coincide exactly and the `use_ratio` ambiguity does not arise here.

Cost
----
Dominated by `--n-continuations` generations per cut point.  Defaults target a
few GPU-hours on a 1.5B; nothing is trained.  Oracle entropies need one forward
per measured continuation, so they are capped by `--measure-per-branch`.

    python scripts/phase1_sibling_spread.py \
        --model Qwen/Qwen2.5-Math-1.5B \
        --heads checkpoints/mtp_heads_Qwen2.5-Math-1.5B-paper.pt \
        --calib checkpoints/mtp_calibration_Qwen2.5-Math-1.5B-paper.json \
        --problems datasets/math500.parquet \
        --embed-model sentence-transformers/all-MiniLM-L6-v2 \
        --out docs/phase1_sibling_spread_Qwen2.5-Math-1.5B.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import torch
except ModuleNotFoundError:  # --self-test exercises only the pure-stdlib analysis
    torch = None

_EPS = 1e-8


def _no_grad(fn):
    """`torch.no_grad()` applied at call time, so the module imports without torch."""
    def wrapper(*a, **kw):
        with torch.no_grad():
            return fn(*a, **kw)
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def load_steer_f():
    """Import `steer_f` late, and refuse the tree this script cannot drive.

    Two incompatible `steer_f` trees exist across branches.  This script targets
    the one that produced `docs/phase1_results_*.json` -- the tree carrying
    `MTPHeads.forecast_entropy` and a three-field `HeadCalibration`.  The other
    has neither, and binding to it fails deep inside a generation loop, after
    the sampling budget has already been spent.
    """
    from steer_f.entropy_forecast import HeadCalibration, h_togo
    from steer_f.mtp_heads import MTPHeads, entropy_from_logits

    problems = []
    if not hasattr(MTPHeads, "forecast_entropy"):
        problems.append("MTPHeads.forecast_entropy is missing")
    if "temperature" not in getattr(HeadCalibration, "__annotations__", {}):
        problems.append("HeadCalibration has no `temperature` field")
    if problems:
        raise SystemExit(
            "[spread] incompatible steer_f: " + "; ".join(problems) + ".\n"
            "         This script needs the tree from the branch that ran Phase 1\n"
            "         (the one with MTPHeads.forecast_entropy). Check out that\n"
            "         branch\'s steer_f/ before running."
        )
    return HeadCalibration, h_togo, MTPHeads, entropy_from_logits


# ----------------------------------------------------------------------
# rank statistics
#
# Implemented here rather than imported: `steer_f.validation.spearman` and
# `within_problem_spearman` exist on both branches but with different argument
# orders and return types (`(pred, target, ids) -> dict` vs
# `(ids, pred, target) -> tuple`), so an import silently computes the wrong
# thing on one of them.  These are pure-stdlib and unit-tested by `--self-test`.
# ----------------------------------------------------------------------
def _average_ranks(values):
    """0-based ranks with ties resolved to their average."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a, b):
    """Spearman rho over the pairs where both entries are finite.  nan if < 3."""
    pairs = [(x, y) for x, y in zip(a, b)
             if isinstance(x, (int, float)) and isinstance(y, (int, float))
             and math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    rx = _average_ranks([p[0] for p in pairs])
    ry = _average_ranks([p[1] for p in pairs])
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((u - mx) * (v - my) for u, v in zip(rx, ry))
    dx = math.sqrt(sum((u - mx) ** 2 for u in rx))
    dy = math.sqrt(sum((v - my) ** 2 for v in ry))
    if dx < 1e-12 or dy < 1e-12:  # one side constant -- rho undefined, not 0
        return float("nan")
    return num / (dx * dy)


def fisher_z_mean(rhos):
    """Average correlations in z space, the plan's §3.3 aggregation."""
    vals = [r for r in rhos if r is not None and math.isfinite(r)]
    if not vals:
        return float("nan")
    z = [math.atanh(min(max(r, -0.999999), 0.999999)) for r in vals]
    return math.tanh(statistics.fmean(z))


def within_group_spearman(pred, target, group_ids, min_per_group=3):
    """Fisher-z mean of per-group Spearman.

    Taken *within* a cut point on purpose: a shared per-position level would
    otherwise inflate the pooled correlation, and `A_H` is a within-position
    deviation by construction (C5 corollary 2).
    """
    groups = {}
    for p, t, g in zip(pred, target, group_ids):
        groups.setdefault(g, ([], []))
        groups[g][0].append(p)
        groups[g][1].append(t)
    rhos = [spearman(p, t) for p, t in groups.values() if len(p) >= min_per_group]
    return {"rho_mean": fisher_z_mean(rhos), "n_groups": len(rhos)}


# ----------------------------------------------------------------------
# sampling / measurement helpers
#
# Lifted from `scripts/phase1_validate.py` so the protocol is identical to the
# one that produced `docs/phase1_results_*.json`.  They are vendored rather
# than imported because that file has diverged into two incompatible shapes
# across branches -- one exposes `sample_continuations`/`branch_diversity`, the
# other a staged `stage_ground_truth`/`stage_forecast` pipeline with no name in
# common -- so an import would silently bind to whichever happens to be on
# disk.  Only `steer_f` is depended on, and every symbol used from it exists in
# both versions.
# ----------------------------------------------------------------------
def render_prompt(raw, tokenizer):
    """Turn one parquet `prompt` cell into the string the policy actually sees.

    Datasets store the prompt in chat form; verl renders it with
    `apply_chat_template(..., add_generation_prompt=True)`.  Rendering it any
    other way (notably `str()`, which yields a Python repr) puts this script
    off the training distribution.
    """
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], dict):
        messages = [{"role": m["role"], "content": m["content"]} for m in raw]
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    if isinstance(raw, dict):
        return render_prompt([raw], tokenizer)
    return str(raw)


@_no_grad
def sample_continuations(model, tokenizer, prefix_ids, n, max_new_tokens,
                         temperature, top_p, device):
    """Sample `n` continuations from one prefix.  Returns a list of id tensors.

    Each returned tensor stops at its own EOS.  `generate` right-pads sequences
    that finished early, and the padding would otherwise be read downstream as
    real policy states, so trimming here is load-bearing rather than cosmetic.
    """
    batch = prefix_ids.unsqueeze(0).expand(n, -1).to(device)
    out = model.generate(
        input_ids=batch,
        attention_mask=torch.ones_like(batch),
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    eos_ids = {tokenizer.eos_token_id}
    if getattr(model, "generation_config", None) is not None:
        extra = model.generation_config.eos_token_id
        if isinstance(extra, int):
            eos_ids.add(extra)
        elif extra:
            eos_ids.update(extra)
    eos_ids.discard(None)

    conts = []
    for i in range(n):
        c = out[i, batch.shape[1]:]
        # Keep the EOS itself -- the distribution that produced it is a real
        # policy state -- and drop only what follows, which is padding.
        hit = [j for j, tok in enumerate(c.tolist()) if tok in eos_ids]
        if hit:
            c = c[: hit[0] + 1]
        conts.append(c)
    return conts


@_no_grad
def measured_future_entropy(model, prefix_ids, continuations, kappa_max, device,
                            entropy_from_logits):
    """Per-offset entropy of the policy along each continuation.  `[n_cont, kappa_max]`.

    Entry `[c, k]` is the entropy of the policy's distribution at offset `+k+1`
    from the prefix along continuation `c` -- the quantity head `k` forecasts,
    so the comparison is like-for-like.  Continuations that ended before the
    horizon contribute NaN rather than padding entropy.
    """
    rows = []
    for cont in continuations:
        ids = torch.cat([prefix_ids.to(device), cont.to(device)]).unsqueeze(0)
        logits = model(input_ids=ids, use_cache=False).logits[0]
        start = prefix_ids.shape[0] - 1  # distribution that produced the first new token
        take = min(kappa_max, logits.shape[0] - start)
        ent = entropy_from_logits(logits[start:start + take].float())
        if take < kappa_max:
            ent = torch.cat(
                [ent, torch.full((kappa_max - take,), float("nan"), device=ent.device)]
            )
        rows.append(ent.cpu())
    return torch.stack(rows)


@_no_grad
def forecast_head_entropies(model, heads, prefix_ids, device, temperature):
    """Per-head forecast entropy at the prefix's final position.  `[K]`."""
    ids = prefix_ids.unsqueeze(0).to(device)
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    last_hidden = out.hidden_states[-1][:, -1:, :]  # [1, 1, H]
    ent = heads.forecast_entropy(
        last_hidden, model.get_output_embeddings(), temperature=temperature
    )
    return ent[:, 0, 0].cpu()


def branch_diversity(continuations, embed_model=None, texts=None):
    """How different a set of continuations is -- `GT_branch_div` of plan §3.3.

    Sentence embeddings when available (mean pairwise cosine distance),
    otherwise first-divergence depth negated, so larger is more diverse under
    both back-ends.
    """
    if embed_model is not None and texts:
        emb = embed_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        sim = emb @ emb.T
        n = sim.shape[0]
        off = (sim.sum() - sim.diagonal().sum()) / max(1, n * (n - 1))
        return float(1.0 - off)

    depths = []
    for i in range(len(continuations)):
        for j in range(i + 1, len(continuations)):
            a, b = continuations[i], continuations[j]
            m = min(len(a), len(b))
            d = m
            for t in range(m):
                if int(a[t]) != int(b[t]):
                    d = t
                    break
            depths.append(d)
    if not depths:
        return 0.0
    return float(-sum(depths) / len(depths))  # shallower divergence == more diverse


def build_prefixes(response_ids, fractions):
    """Cut one trajectory at the requested fractions of its length."""
    out = []
    n = len(response_ids)
    for f in fractions:
        cut = int(n * f)
        if 4 <= cut < n:
            out.append(cut)
    return out


# ----------------------------------------------------------------------
# Part C helpers -- C4's two factors at one position
# ----------------------------------------------------------------------
@_no_grad
def position_local_stats(model, prefix_ids, taken_token, device, entropy_from_logits):
    """`Omega/A`, the visitation prefactor, and H at the prefix's last position.

    `Omega = A (1 - pi_a)(I_a - H)` with `I_a = -log pi_a` (C2), evaluated for
    the token the original trajectory actually took.  `A` is a training-time
    quantity and is deliberately factored out: C4's table reports `E|Omega|/|A|`
    for exactly this reason, so the offline number is directly comparable.

    Returns a dict with:
        omega_over_a   signed `(1 - pi_a)(I_a - H)`
        visit_prefac   `1 - pi_a`, the visitation term's prefactor
        entropy        `H(pi(.|s_t))` in nats
        pi_a           probability of the taken token
        eff_support    `exp(H)`, the perplexity-style effective branch count
    """
    ids = prefix_ids.unsqueeze(0).to(device)
    logits = model(input_ids=ids, use_cache=False).logits[0, -1].float()
    logp = torch.log_softmax(logits, dim=-1)
    ent = float(entropy_from_logits(logits.unsqueeze(0))[0])

    log_pi_a = float(logp[int(taken_token)])
    pi_a = math.exp(log_pi_a)
    surprisal = -log_pi_a
    return {
        "omega_over_a": (1.0 - pi_a) * (surprisal - ent),
        "visit_prefac": 1.0 - pi_a,
        "entropy": ent,
        "pi_a": pi_a,
        "eff_support": math.exp(ent),
    }


def group_by_first_token(conts, min_group):
    """Split continuations into sibling branches by their first sampled token.

    This is the operational definition of "siblings at `s_t`" that
    `sibling_prefix_baseline` uses at training time -- rollouts agreeing on
    everything before `t` and free to differ at `t` -- reproduced offline by
    resampling from one prefix.
    """
    buckets = defaultdict(list)
    for c in conts:
        if len(c) == 0:
            continue
        buckets[int(c[0])].append(c)
    return {tok: g for tok, g in buckets.items() if len(g) >= min_group}


def spread_stats(values):
    """Dispersion of one cut point's per-branch values."""
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    sd = statistics.pstdev(values)
    return {
        "n": len(values),
        "mean": mean,
        "std": sd,
        "range": max(values) - min(values),
        # Scale-free so branches, positions and problems stay comparable.
        # `branch_diversity`'s token fallback returns *negative* depths, so the
        # absolute value is what makes this meaningful for both back-ends.
        "rel_std": sd / (abs(mean) + _EPS),
    }


def build_argparser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model")
    p.add_argument("--heads", help="checkpoint from phase1_warmup_heads.py")
    p.add_argument("--problems", help="parquet with a prompt column")
    p.add_argument("--out", help="JSON results path")
    p.add_argument("--calib", default=None, help="mtp_calibration_*.json; strongly recommended")
    p.add_argument("--prompt-column", default="prompt")
    p.add_argument("--n-problems", type=int, default=25)
    p.add_argument("--n-trajectories", type=int, default=8)
    p.add_argument("--n-continuations", type=int, default=64,
                   help="K_mc. Higher than phase1_validate's 16 on purpose: these get "
                        "split across branches, and a 3-member branch gives a noisy "
                        "pairwise-distance estimate")
    p.add_argument("--min-group", type=int, default=4,
                   help="drop branches with fewer members than this")
    p.add_argument("--min-branches", type=int, default=2,
                   help="skip cut points with fewer surviving branches; below 2 there "
                        "is no sibling comparison to make")
    p.add_argument("--measure-per-branch", type=int, default=6,
                   help="continuations per branch scored for oracle H_togo (one forward "
                        "each). 0 disables Part B's oracle column")
    p.add_argument("--fractions", default="0.2,0.4,0.6,0.8")
    p.add_argument("--max-prefixes", type=int, default=300)
    p.add_argument("--max-prefixes-per-problem", type=int, default=None)
    p.add_argument("--traj-max-tokens", type=int, default=1024)
    p.add_argument("--cont-max-tokens", type=int, default=64)
    p.add_argument("--traj-temperature", type=float, default=1.0)
    p.add_argument("--cont-temperature", type=float, default=1.0,
                   help="1.0, not phase1_validate's 0.7: siblings in training come from "
                        "rollouts at temperature 1.0, and top-p/temperature truncation "
                        "changes how often branches survive at all")
    p.add_argument("--cont-top-p", type=float, default=1.0)
    p.add_argument("--forecast-temperature", type=float, default=1.0)
    p.add_argument("--kappa", type=int, default=2, help="Phase 1's chosen horizon")
    p.add_argument("--gamma-h", type=float, default=0.7)
    p.add_argument("--embed-model", default=None,
                   help="e.g. sentence-transformers/all-MiniLM-L6-v2. Strongly "
                        "recommended: the token-depth fallback measures first-divergence "
                        "depth, and every continuation inside a branch shares its first "
                        "token by construction, so the fallback is biased for Part A")
    p.add_argument("--self-test", action="store_true",
                   help="run the analysis on synthetic records and exit; needs no GPU, model or steer_f")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None,
                   help="default: cuda when available")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    if args.self_test:
        return _self_test()
    for req in ("model", "heads", "problems", "out"):
        if getattr(args, req) is None:
            raise SystemExit(f"[spread] --{req} is required (omit only with --self-test)")
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    HeadCalibration, h_togo, MTPHeads, entropy_from_logits = load_steer_f()
    torch.manual_seed(args.seed)

    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    ).to(args.device).eval()

    ckpt = torch.load(args.heads, map_location="cpu", weights_only=False)
    if ckpt.get("untrained"):
        print("[spread] NOTE: --heads is an UNTRAINED checkpoint. Part A and Part C are "
              "head-free and stay valid; Part B's MTP column is a chance floor.")
    heads = MTPHeads(**ckpt["config"]).to(args.device, dtype=dtype)
    heads.load_state_dict(ckpt["state_dict"])
    heads.eval()
    kappa_max = ckpt["config"]["num_heads"]
    if args.kappa > kappa_max:
        raise SystemExit(f"[spread] kappa={args.kappa} > trained heads K={kappa_max}")

    calib = None
    if args.calib and Path(args.calib).exists():
        d = json.loads(Path(args.calib).read_text())
        calib = HeadCalibration(
            scale=torch.tensor(d["scale"]),
            bias=torch.tensor(d["bias"]),
            temperature=torch.tensor(d.get("temperature", [1.0] * len(d["scale"]))),
        )
        print(f"[spread] calibration loaded from {args.calib}")
    else:
        print("[spread] WARNING: no calibration. C6 says distant heads over-estimate "
              "entropy monotonically in k, so an uncalibrated H_togo is dominated by "
              "that bias rather than by branch differences.")

    embed_model = None
    if args.embed_model:
        from sentence_transformers import SentenceTransformer

        embed_model = SentenceTransformer(args.embed_model, device=args.device)
    else:
        print("[spread] WARNING: no --embed-model; falling back to first-divergence "
              "depth. Every continuation in a branch shares its first token, so the "
              "fallback floors at depth 1 and understates Part A's spread.")

    df = pd.read_parquet(args.problems)
    prompts = [render_prompt(x, tokenizer)
               for x in df[args.prompt_column].tolist()[: args.n_problems]]
    fractions = [float(x) for x in args.fractions.split(",")]

    per_problem = args.max_prefixes_per_problem
    if per_problem is None:
        per_problem = max(1, args.max_prefixes // max(1, len(prompts)))
    print(f"[spread] {len(prompts)} problems, <= {per_problem} prefixes each, "
          f"K_mc={args.n_continuations}, kappa={args.kappa}, gamma_h={args.gamma_h}")

    records = []
    skipped_no_branch = 0

    for pi, prompt in enumerate(prompts):
        problem_start = len(records)
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        trajectories = sample_continuations(
            model, tokenizer, prompt_ids, args.n_trajectories, args.traj_max_tokens,
            args.traj_temperature, 1.0, args.device,
        )

        stop = False
        for ti, traj in enumerate(trajectories):
            for cut in build_prefixes(traj, fractions):
                if len(records) >= args.max_prefixes or \
                        len(records) - problem_start >= per_problem:
                    stop = True
                    break

                prefix_ids = torch.cat([prompt_ids, traj[:cut].cpu()])
                conts = sample_continuations(
                    model, tokenizer, prefix_ids, args.n_continuations,
                    args.cont_max_tokens, args.cont_temperature, args.cont_top_p,
                    args.device,
                )
                branches = group_by_first_token(conts, args.min_group)
                if len(branches) < args.min_branches:
                    skipped_no_branch += 1
                    continue

                # ---- Part C: C4's two factors at this position (head-free) ----
                local = position_local_stats(
                    model, prefix_ids, traj[cut], args.device, entropy_from_logits
                )
                # Branchiness measured by resampling rather than inferred from H.
                # `n_branches` counts branches that survived --min-group; the
                # normalised entropy of the first-token histogram is the finer
                # reading and does not depend on that threshold.
                first_counts = defaultdict(int)
                for c in conts:
                    if len(c):
                        first_counts[int(c[0])] += 1
                total_first = sum(first_counts.values())
                probs = [v / total_first for v in first_counts.values()]
                branch_entropy = -sum(p * math.log(p) for p in probs if p > 0)

                # ---- per-branch quantities ----
                per_branch = []
                for tok, group in sorted(branches.items()):
                    texts = ([tokenizer.decode(c, skip_special_tokens=True) for c in group]
                             if embed_model else None)
                    div = branch_diversity(group, embed_model, texts)

                    # MTP forecast for *this* branch: H_togo(s_t + y_t) per
                    # sibling_prefix_baseline's docstring, so the prefix is
                    # extended by the branch's own first token before forecasting.
                    branch_prefix = torch.cat(
                        [prefix_ids, torch.tensor([tok], dtype=prefix_ids.dtype)]
                    )
                    fc = forecast_head_entropies(
                        model, heads, branch_prefix, args.device, args.forecast_temperature
                    )
                    h_mtp = float(h_togo(fc.view(-1, 1, 1), kappa=args.kappa,
                                         gamma_h=args.gamma_h, calib=calib))

                    h_oracle = None
                    if args.measure_per_branch > 0:
                        sub = group[: args.measure_per_branch]
                        measured = measured_future_entropy(
                            model, branch_prefix, sub, args.kappa, args.device,
                            entropy_from_logits,
                        )  # [n_sub, kappa]
                        per_offset = torch.nanmean(measured, dim=0)
                        if torch.isfinite(per_offset).all():
                            w = torch.tensor(
                                [args.gamma_h ** (k + 1) for k in range(args.kappa)]
                            )
                            h_oracle = float((w * per_offset).sum())

                    per_branch.append({
                        "first_token": tok,
                        "n_members": len(group),
                        "gt_div": div,
                        "h_togo_mtp": h_mtp,
                        "h_togo_oracle": h_oracle,
                    })

                records.append({
                    "problem": pi,
                    "trajectory": ti,
                    "cut": cut,
                    "n_branches": len(branches),
                    "first_token_entropy": branch_entropy,
                    "local": local,
                    "branches": per_branch,
                })
            if stop:
                break
        print(f"[spread] problem {pi+1}/{len(prompts)}: +{len(records)-problem_start} "
              f"(total {len(records)}, skipped {skipped_no_branch})", flush=True)
        if len(records) >= args.max_prefixes:
            break

    if not records:
        raise SystemExit(
            "[spread] no usable cut points. Every prefix produced fewer than "
            f"--min-branches={args.min_branches} branches of >= --min-group="
            f"{args.min_group}. That is itself a finding -- report it -- but try "
            "--n-continuations higher or --min-group lower first."
        )

    summary = summarise(records, args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"args": vars(args), "n_cut_points": len(records),
         "skipped_no_branch": skipped_no_branch,
         "summary": summary, "records": records},
        indent=2,
    ))
    print(f"\n[spread] wrote {args.out}")
    print(render_report(summary))


# ----------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------
def summarise(records, args):
    """Reduce per-cut-point records to the three verdicts."""
    # ---- Part A: is sibling future diversity unequal at all? ----
    within, cut_means = [], []
    for r in records:
        divs = [b["gt_div"] for b in r["branches"]]
        st = spread_stats(divs)
        if st:
            within.append(st)
            cut_means.append(st["mean"])

    across_std = statistics.pstdev(cut_means) if len(cut_means) > 1 else float("nan")
    med_within_std = statistics.median([s["std"] for s in within]) if within else float("nan")
    med_rel_std = statistics.median([s["rel_std"] for s in within]) if within else float("nan")
    # The number that matters. C1 established the *position* signal is strong;
    # this says how the *sibling* signal compares on the same scale.
    ratio = med_within_std / across_std if across_std > 0 else float("nan")

    # The median alone cannot separate the two worlds that matter here:
    #
    #   world 1  every branch point has near-equal siblings
    #            -> corollary 1 fires everywhere, the visitation term is
    #               correctly inert, and no forecast work changes that;
    #   world 2  most branch points are equal but a small tail is wildly
    #            unequal -> the method lives in that tail, and "3% of branch
    #            points carry the correction" is a sharper claim than the
    #            original one, not a weaker one.
    #
    # Both report ratio ~ 0 on the median. The upper quantiles and the
    # exceedance fractions below are what tell them apart, so they are
    # first-class outputs rather than diagnostics.
    stds = sorted(s["std"] for s in within)
    rels = sorted(s["rel_std"] for s in within)

    def q(vals, frac):
        if not vals:
            return float("nan")
        i = min(len(vals) - 1, max(0, int(round(frac * (len(vals) - 1)))))
        return vals[i]

    # Exceedance is measured against the position-level scale, so "large"
    # means large compared with the signal C1 already established rather
    # than against an arbitrary constant.
    def frac_above(mult):
        if not (across_std > 0):
            return float("nan")
        thr = mult * across_std
        return sum(1 for v in stds if v > thr) / len(stds)

    part_a = {
        "n_cut_points": len(within),
        "median_within_cutpoint_std": med_within_std,
        "median_within_cutpoint_rel_std": med_rel_std,
        "across_cutpoint_std_of_means": across_std,
        "sibling_to_position_ratio": ratio,
        "mean_branches_per_cutpoint": statistics.fmean(
            [r["n_branches"] for r in records]
        ),
        "within_cutpoint_std_quantiles": {
            "p10": q(stds, 0.10), "p25": q(stds, 0.25), "p50": q(stds, 0.50),
            "p75": q(stds, 0.75), "p90": q(stds, 0.90), "p99": q(stds, 0.99),
            "max": stds[-1] if stds else float("nan"),
        },
        "within_cutpoint_rel_std_quantiles": {
            "p50": q(rels, 0.50), "p90": q(rels, 0.90), "p99": q(rels, 0.99),
        },
        "tail_frac_std_above_0.5x_position": frac_above(0.5),
        "tail_frac_std_above_1x_position": frac_above(1.0),
        "tail_frac_std_above_2x_position": frac_above(2.0),
    }

    # ---- Part B: does either estimator track the true sibling deviation? ----
    # Deviations are taken within a cut point, exactly as `A_H` is: the
    # sibling mean is subtracted before anything is correlated, so a shared
    # per-position level cannot inflate the correlation.
    a_true, a_mtp, a_oracle, gids = [], [], [], []
    for i, r in enumerate(records):
        bs = r["branches"]
        if len(bs) < 2:
            continue
        mt = statistics.fmean([b["gt_div"] for b in bs])
        mm = statistics.fmean([b["h_togo_mtp"] for b in bs])
        ors = [b["h_togo_oracle"] for b in bs]
        mo = statistics.fmean(ors) if all(o is not None for o in ors) else None
        for b in bs:
            a_true.append(b["gt_div"] - mt)
            a_mtp.append(b["h_togo_mtp"] - mm)
            a_oracle.append(b["h_togo_oracle"] - mo if mo is not None else float("nan"))
            gids.append(i)

    ok = [j for j, v in enumerate(a_oracle) if math.isfinite(v)]
    _wc = (within_group_spearman(a_mtp, a_true, gids) if len(a_true) > 2
           else {"rho_mean": float("nan"), "n_groups": 0})
    _wo = (within_group_spearman([a_oracle[j] for j in ok], [a_true[j] for j in ok],
                                 [gids[j] for j in ok]) if len(ok) > 2
           else {"rho_mean": float("nan"), "n_groups": 0})
    part_b = {
        "n_branch_observations": len(a_true),
        "pooled_rho_mtp_vs_true": spearman(a_mtp, a_true) if len(a_true) > 2 else float("nan"),
        # Cut points with only 2 branches cannot yield a Spearman and are
        # dropped here, so `n_groups` is what the within-cut number rests on.
        # It is routinely far below `n_cut_points` -- read the pooled row too.
        "within_cutpoint_rho_mtp_vs_true": _wc["rho_mean"],
        "within_cutpoint_n_groups": _wc["n_groups"],
        "n_oracle_observations": len(ok),
        "pooled_rho_oracle_vs_true": spearman(
            [a_oracle[j] for j in ok], [a_true[j] for j in ok]
        ) if len(ok) > 2 else float("nan"),
        "within_cutpoint_rho_oracle_vs_true": _wo["rho_mean"],
        "within_cutpoint_oracle_n_groups": _wo["n_groups"],
    }

    # ---- Part C: M1 -- C4 on real rollouts ----
    om = [abs(r["local"]["omega_over_a"]) for r in records]
    vp = [r["local"]["visit_prefac"] for r in records]
    br = [r["first_token_entropy"] for r in records]

    # C4's table is NOT monotone in branchiness -- |Omega|/|A| reads
    # 0.092 -> 1.129 (max) -> 0.457 -> 0.000 -> 0.000 as the distribution
    # flattens, an inverted U whose peak sits at "confident with a long tail".
    # A single Spearman over that shape lands near zero and says nothing, so
    # the claim is tested where it actually bites: the branchy tail, where C2
    # says Omega vanishes *exactly*. The visitation prefactor IS monotone
    # (0.020 -> 0.990), so a rank correlation is the right statistic there.
    order = sorted(range(len(records)), key=lambda i: br[i])
    n = len(order)
    bins = []
    for b in range(5):
        lo, hi = b * n // 5, (b + 1) * n // 5
        idx = order[lo:hi]
        if not idx:
            continue
        bins.append({
            "quintile": b + 1,
            "n": len(idx),
            "branchiness_mean": statistics.fmean([br[i] for i in idx]),
            "abs_omega_over_a_mean": statistics.fmean([om[i] for i in idx]),
            "visit_prefactor_mean": statistics.fmean([vp[i] for i in idx]),
        })
    top_idx = order[max(0, n - max(1, n // 10)):]
    top_om = statistics.fmean([om[i] for i in top_idx])
    med_om = statistics.median(om)
    part_c = {
        "n_positions": len(records),
        "mean_abs_omega_over_a": statistics.fmean(om),
        "mean_visit_prefactor": statistics.fmean(vp),
        "mean_product": statistics.fmean([o * v for o, v in zip(om, vp)]),
        "mean_policy_entropy": statistics.fmean([r["local"]["entropy"] for r in records]),
        "by_branchiness_quintile": bins,
        # C4's core prediction, as a ratio: << 1 means Omega really does go
        # quiet exactly where the branching happens.
        "top_decile_abs_omega": top_om,
        "median_abs_omega": med_om,
        "top_decile_over_median": top_om / med_om if med_om > 0 else float("nan"),
        "rho_visitprefac_vs_branchiness": spearman(vp, br) if len(vp) > 2 else float("nan"),
    }

    return {"part_a_sibling_spread": part_a,
            "part_b_estimator_tracking": part_b,
            "part_c_m1_omega_at_branch_points": part_c}


def render_report(s):
    a, b, c = (s["part_a_sibling_spread"], s["part_b_estimator_tracking"],
               s["part_c_m1_omega_at_branch_points"])
    L = []
    add = L.append
    add("=" * 74)
    add("PART A -- is sibling future diversity unequal?  (C5 corollary 1)")
    add("=" * 74)
    add(f"  cut points usable                  {a['n_cut_points']}")
    add(f"  mean branches per cut point        {a['mean_branches_per_cutpoint']:.2f}")
    add(f"  median within-cutpoint std(div)    {a['median_within_cutpoint_std']:.4f}")
    add(f"  median within-cutpoint rel std     {a['median_within_cutpoint_rel_std']:.4f}")
    add(f"  across-cutpoint std of means       {a['across_cutpoint_std_of_means']:.4f}")
    add(f"  >> sibling / position ratio        {a['sibling_to_position_ratio']:.4f}")
    add("")
    qs = a["within_cutpoint_std_quantiles"]
    add("  within-cutpoint std(div) distribution")
    add("    p10      p25      p50      p75      p90      p99      max")
    add("  " + " ".join(f"{qs[k]:8.4f}" for k in
                        ("p10", "p25", "p50", "p75", "p90", "p99", "max")))
    add("")
    add(f"  >> frac of cut points with std > 0.5x position scale "
        f"{a['tail_frac_std_above_0.5x_position']:.4f}")
    add(f"  >> frac with std > 1x position scale               "
        f"{a['tail_frac_std_above_1x_position']:.4f}")
    add(f"  >> frac with std > 2x position scale               "
        f"{a['tail_frac_std_above_2x_position']:.4f}")
    add("")
    add("  Reading: the ratio puts the sibling signal on the same scale as the")
    add("  position signal C1 already established. But the median cannot tell")
    add("  'uniform everywhere' from 'uniform except a consequential tail', and")
    add("  those are different papers -- read the quantiles and the exceedance")
    add("  fractions together with it:")
    add("")
    add("    ratio ~ 0 AND p90/p99 also ~ 0   -> corollary 1 fires everywhere.")
    add("        A_H is 0 because the futures really are equally diverse and the")
    add("        visitation term is CORRECTLY inert. No forecast, horizon or")
    add("        support work recovers it: with equal sibling futures the chain")
    add("        rule's second term is constant in pi, so Omega is complete.")
    add("        That is a publishable measurement about the branch structure of")
    add("        math RLVR, and it closes the remedy rather than the diagnosis.")
    add("    ratio ~ 0 BUT p99/max large       -> the method lives in the tail.")
    add("        Report what fraction of branch points carry it; a correction")
    add("        needed at 3% of positions is a sharper claim, not a weaker one.")
    add("    ratio order 1                     -> signal is broadly there;")
    add("        Part B decides which estimator can see it.")
    add("")
    add("=" * 74)
    add("PART B -- does either estimator track it?")
    add("=" * 74)
    add(f"  branch observations                {b['n_branch_observations']}")
    add(f"  MTP    rho(A_H, true)  pooled      {b['pooled_rho_mtp_vs_true']:+.4f}")
    add(f"  MTP    rho(A_H, true)  within-cut  {b['within_cutpoint_rho_mtp_vs_true']:+.4f}"
        f"   ({b['within_cutpoint_n_groups']} cut points with >=3 branches)")
    add(f"  oracle rho(A_H, true)  pooled      {b['pooled_rho_oracle_vs_true']:+.4f}"
        f"   (n={b['n_oracle_observations']})")
    add(f"  oracle rho(A_H, true)  within-cut  {b['within_cutpoint_rho_oracle_vs_true']:+.4f}")
    add("")
    add("  MTP low + oracle high  -> estimation problem. C6 shrank the effective")
    add("                            horizon to ~1 step; STEERF_FORECAST=oracle")
    add("                            is the arm that recovers it.")
    add("  both low, Part A high  -> discounted future entropy is a poor proxy")
    add("                            for subtree diversity. That is C5's choice")
    add("                            of quantity, not an implementation defect.")
    add("  both high              -> the signal exists and is estimable; the")
    add("                            0.3% branch_corr_frac is then a support")
    add("                            problem (baseline='sibling', long rollouts).")
    add("")
    add("=" * 74)
    add("PART C -- M1: C4's two factors on real rollouts")
    add("=" * 74)
    add(f"  positions                          {c['n_positions']}")
    add(f"  mean |Omega|/|A|                   {c['mean_abs_omega_over_a']:.4f}")
    add(f"  mean visitation prefactor 1-pi_a   {c['mean_visit_prefactor']:.4f}")
    add(f"  mean product                       {c['mean_product']:.4f}")
    add(f"  mean policy entropy H              {c['mean_policy_entropy']:.4f}")
    add("")
    add("  quintile  branchiness   |Omega|/|A|   1-pi_a")
    for q in c["by_branchiness_quintile"]:
        add(f"     {q['quintile']}         {q['branchiness_mean']:7.4f}     "
            f"{q['abs_omega_over_a_mean']:8.4f}   {q['visit_prefactor_mean']:6.4f}"
            f"   (n={q['n']})")
    add("")
    add(f"  >> top-decile |Omega| / median     {c['top_decile_over_median']:.4f}")
    add(f"  >> rho(1-pi_a, branchiness)        {c['rho_visitprefac_vs_branchiness']:+.4f}")
    add("")
    add("  C4 predicts the quintile column rises then COLLAPSES: |Omega|/|A|")
    add("  peaks at 'confident with a long tail' and goes to exactly 0 at a")
    add("  flat distribution, because there every I_a equals H (C2). So the")
    add("  ratio should be well BELOW 1 and the rank correlation POSITIVE.")
    add("  A ratio near or above 1 rejects C4 -- Omega would then be loudest")
    add("  exactly where the paper says it is silent -- and puts the whole")
    add("  C2-C4 argument back on the bench.")
    add("=" * 74)
    return "\n".join(L)




# ----------------------------------------------------------------------
# self-test -- exercises the analysis on synthetic records, no GPU needed
# ----------------------------------------------------------------------
def _synthetic(n_cuts, sibling_spread, mtp_tracks, oracle_tracks, seed=0,
               c4_holds=True):
    """Records whose ground truth is known, so the verdicts can be checked.

    `sibling_spread` scales how unequal sibling diversities are within a cut
    point; `mtp_tracks` / `oracle_tracks` in [0, 1] set how faithfully each
    estimator follows that deviation.

    `c4_holds` picks which world the branchy positions live in, and the two are
    genuinely different claims about real policies:

      True  -- branchy positions are near-*uniform*. Every `I_a` equals `H`,
               so `Omega` vanishes there (C2) and C4 holds.
      False -- branchy positions are "confident with a long tail": high `H`
               reached by spreading a little mass over many tokens. C4's own
               table puts the |Omega| *maximum* (1.129) on exactly that shape,
               so Omega is loudest where the paper says it is silent.

    C4's table only ever reaches `Omega = 0` at exact uniformity, which real
    LLM next-token distributions essentially never are -- which is why this
    has to be measured rather than assumed. The self-test asserts only that
    the statistic separates the two worlds, not that either is true.
    """
    rnd = __import__("random").Random(seed)
    recs = []
    for i in range(n_cuts):
        level = 0.5 + 0.30 * rnd.random()          # per-position level (C1's signal)
        n_b = rnd.choice([2, 3, 4])
        branches = []
        for b in range(n_b):
            dev = sibling_spread * (rnd.random() - 0.5)
            noise_m = (1 - mtp_tracks) * 0.5 * (rnd.random() - 0.5)
            noise_o = (1 - oracle_tracks) * 0.5 * (rnd.random() - 0.5)
            branches.append({
                "first_token": 1000 + b,
                "n_members": 6,
                "gt_div": level + dev,
                "h_togo_mtp": 0.35 + mtp_tracks * dev + noise_m,
                "h_togo_oracle": 0.35 + oracle_tracks * dev + noise_o,
            })
        # C4's analytic family verbatim: top token `q`, the remaining `1-q`
        # spread uniformly over `m`. Sampling `pi_a` from that distribution --
        # rather than picking it independently of `H` -- is what reproduces
        # C4's shape: at `q -> 1/(m+1)` the distribution is uniform, every
        # `I_a` equals `H`, and `Omega` vanishes *exactly* (C2's corollary).
        m = rnd.choice([3, 10, 50])
        q_unif = 1.0 / (m + 1)
        if c4_holds:
            flat = rnd.random()  # 0 = confident, 1 = uniform
            q = q_unif + (0.99 - q_unif) * (1 - flat) ** 3
        else:
            q = q_unif + (0.99 - q_unif) * rnd.random()
        tail = (1.0 - q) / m
        ent = -(q * math.log(q) + m * tail * math.log(tail))
        pi_a = q if rnd.random() < q else tail
        recs.append({
            "problem": i // 4, "trajectory": i % 4, "cut": 10 + i,
            "n_branches": n_b, "first_token_entropy": ent,
            "local": {
                "omega_over_a": (1 - pi_a) * (-math.log(pi_a) - ent),
                "visit_prefac": 1 - pi_a, "entropy": ent, "pi_a": pi_a,
                "eff_support": math.exp(ent),
            },
            "branches": branches,
        })
    return recs


def _self_test():
    class _A:
        pass

    fails = []

    def check(label, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  -- ' + detail) if detail else ''}")
        if not cond:
            fails.append(label)

    print("rank statistics")
    check("spearman monotone == 1", abs(spearman([1, 2, 3, 4], [2, 4, 6, 9]) - 1) < 1e-9)
    check("spearman reversed == -1", abs(spearman([1, 2, 3, 4], [9, 6, 4, 2]) + 1) < 1e-9)
    check("spearman constant is nan", math.isnan(spearman([1, 1, 1, 1], [1, 2, 3, 4])))
    check("spearman n<3 is nan", math.isnan(spearman([1, 2], [1, 2])))
    check("spearman ignores nan pairs",
          abs(spearman([1, 2, 3, 4, float("nan")], [1, 2, 3, 4, 99]) - 1) < 1e-9)
    check("ties get average ranks", _average_ranks([5, 5, 9]) == [0.5, 0.5, 2.0])
    check("fisher_z_mean(0.5,0.5) == 0.5", abs(fisher_z_mean([0.5, 0.5]) - 0.5) < 1e-9)
    check("fisher_z_mean skips nan", abs(fisher_z_mean([0.5, float("nan"), 0.5]) - 0.5) < 1e-9)

    print("\ngrouping")
    g = group_by_first_token([[7, 1], [7, 2], [7, 3], [8, 1], []], min_group=2)
    check("keeps groups >= min_group", set(g) == {7}, f"got {sorted(g)}")
    check("drops empty continuations", sum(len(v) for v in g.values()) == 3)
    check("spread_stats needs 2+", spread_stats([1.0]) is None)
    st = spread_stats([1.0, 3.0])
    check("spread_stats std/range", abs(st["std"] - 1.0) < 1e-9 and abs(st["range"] - 2.0) < 1e-9)

    print("\nscenario (A): siblings equally diverse -> corollary 1 fires")
    a = _A(); a.kappa = 2; a.gamma_h = 0.7
    s_a = summarise(_synthetic(60, sibling_spread=0.001, mtp_tracks=0.9,
                               oracle_tracks=0.9, seed=1), a)
    ratio_a = s_a["part_a_sibling_spread"]["sibling_to_position_ratio"]
    check("sibling/position ratio is small", ratio_a < 0.05, f"ratio={ratio_a:.4f}")

    print("\nscenario (B): real spread, MTP blind, oracle sighted")
    s_b = summarise(_synthetic(60, sibling_spread=0.40, mtp_tracks=0.0,
                               oracle_tracks=1.0, seed=2), a)
    ratio_b = s_b["part_a_sibling_spread"]["sibling_to_position_ratio"]
    r_mtp = s_b["part_b_estimator_tracking"]["pooled_rho_mtp_vs_true"]
    r_orc = s_b["part_b_estimator_tracking"]["pooled_rho_oracle_vs_true"]
    check("ratio is large", ratio_b > 0.5, f"ratio={ratio_b:.4f}")
    check("oracle beats MTP", r_orc > 0.9 > r_mtp, f"mtp={r_mtp:+.3f} oracle={r_orc:+.3f}")

    print("\nscenario (C): both estimators track")
    s_c = summarise(_synthetic(60, sibling_spread=0.40, mtp_tracks=1.0,
                               oracle_tracks=1.0, seed=3), a)
    check("both rho high",
          min(s_c["part_b_estimator_tracking"]["pooled_rho_mtp_vs_true"],
              s_c["part_b_estimator_tracking"]["pooled_rho_oracle_vs_true"]) > 0.9)

    print("\nPart C: does the statistic separate a C4 world from a non-C4 one?")
    pc_yes = summarise(_synthetic(200, 0.4, 1.0, 1.0, seed=4, c4_holds=True),
                       a)["part_c_m1_omega_at_branch_points"]
    pc_no = summarise(_synthetic(200, 0.4, 1.0, 1.0, seed=4, c4_holds=False),
                      a)["part_c_m1_omega_at_branch_points"]
    r_yes, r_no = pc_yes["top_decile_over_median"], pc_no["top_decile_over_median"]
    check("C4 world -> ratio well below 1", r_yes < 0.5, f"ratio={r_yes:.3f}")
    check("non-C4 world -> ratio not below 1", r_no >= 1.0, f"ratio={r_no:.3f}")
    check("the two worlds are separated", r_no > 3 * r_yes,
          f"{r_yes:.3f} vs {r_no:.3f}")
    qs = [q["abs_omega_over_a_mean"] for q in pc_yes["by_branchiness_quintile"]]
    check("C4 world quintiles collapse at the top", qs[-1] < max(qs),
          f"{['%.3f' % q for q in qs]}")
    check("rho(1-pi_a, branchiness) > 0 in both",
          min(pc_yes["rho_visitprefac_vs_branchiness"],
              pc_no["rho_visitprefac_vs_branchiness"]) > 0)

    print("\nreport renders")
    txt = render_report(s_b)
    check("report non-empty", len(txt.splitlines()) > 25)

    print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
