#!/usr/bin/env python3
# Copyright 2026 STEER-F authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Phase 1 — Monte-Carlo validation of the entropy forecast, and (kappa, gamma_H).

Plan §3.3, adapted from ExTra's Appendix-C protocol to entropy:

1. pick problems the model gets both right and wrong (the regime where
   branching actually matters);
2. sample 32 full trajectories per problem at temperature 1.0;
3. cut each at 20/40/60/80% of its length -> a pool of prefixes;
4. from each prefix sample K_mc continuations and *measure* the future
   entropy the policy really has there;
5. forecast the same quantity with the MTP heads over a (kappa, gamma_H) grid;
6. score with within-problem Spearman, and test branch recall against chance.

The gate rule itself lives in `steer_f.validation` and is unit-tested, so a
G1 verdict does not depend on anything written only in this file.

    python scripts/phase1_validate.py \
        --model Qwen/Qwen2.5-Math-7B \
        --heads checkpoints/mtp_heads_qwen7b.pt \
        --problems datasets/math500.parquet \
        --out docs/phase1_results.json

Ground truth, precisely
-----------------------
`H_togo^kappa` is a *discounted* sum over the next kappa steps, so the honest
comparison is a discounted sum of the *measured* per-position entropies over
the same horizon — recomputed per (kappa, gamma_H) cell.  The plan's plain
"mean of summed token entropies" is also reported as `gt_sum_h`, but scoring a
discounted forecast against an undiscounted truth would penalise every gamma_H
below 1.0 for a reason that has nothing to do with forecast quality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from steer_f.entropy_forecast import HeadCalibration, fit_head_calibration, h_togo  # noqa: E402
from steer_f.mtp_heads import MTPHeads, entropy_from_logits  # noqa: E402
from steer_f.validation import (  # noqa: E402
    evaluate_gate_g1,
    select_kappa_gamma,
    within_problem_spearman,
)

GAMMA_GRID = [0.7, 0.85, 1.0]


def render_prompt(raw, tokenizer):
    """Turn one parquet `prompt` cell into the string the policy actually sees.

    STEER's datasets store the prompt in chat form -- a list of
    ``{"role", "content"}`` dicts, loaded by pyarrow as a numpy object array --
    and verl renders it with
    ``tokenizer.apply_chat_template(messages, add_generation_prompt=True)``
    (`verl/utils/dataset/rl_dataset.py`).

    Rendering it any other way puts this script off the training distribution.
    `str()` on the cell, in particular, yields the literal text
    ``[{'content': 'If $2^8=4^x$...', 'role': 'user'}]`` -- the model is then
    conditioned on a Python repr, and the trajectories it produces are not the
    ones Phase 2 will see.  A G1 FAIL reached that way would read as "the
    future term is noise" when it is really "the prompt was malformed", so the
    conversion is done here rather than left to `str()`.
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


# ----------------------------------------------------------------------
# generation helpers
# ----------------------------------------------------------------------
@torch.no_grad()
def sample_continuations(model, tokenizer, prefix_ids, n, max_new_tokens,
                         temperature, top_p, device):
    """Sample `n` continuations from one prefix.  Returns a list of id tensors.

    Each returned tensor stops at its own EOS.  `generate` runs until every
    sequence in the batch is done and right-pads the ones that finished early,
    so the raw slice would hand back real tokens followed by padding.
    `measured_future_entropy` reads those positions as policy states and only
    emits NaN when the tensor is *shorter* than the horizon — with padding it
    never is, so a continuation that ended at offset 2 would silently
    contribute the entropy of `...<eos><pad><pad>` as if it were ground truth,
    and the `isfinite` filter downstream would keep it.  Trimming here is what
    makes that NaN path work as designed.
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
        # Keep the EOS itself: the distribution that produced it is a real
        # policy state and head k is meant to forecast it. Drop only what
        # follows, which is padding rather than sampled text.
        hit = [j for j, tok in enumerate(c.tolist()) if tok in eos_ids]
        if hit:
            c = c[: hit[0] + 1]
        conts.append(c)
    return conts


@torch.no_grad()
def measured_future_entropy(model, prefix_ids, continuations, kappa_max, device):
    """Per-offset entropy of the policy along each continuation.

    Returns `[n_cont, kappa_max]` where entry `[c, k]` is the entropy of the
    policy's distribution at offset `+k+1` from the prefix, along continuation
    `c`.  This is the quantity head `k` is trying to forecast, so the
    comparison is like-for-like rather than a proxy.
    """
    rows = []
    for cont in continuations:
        ids = torch.cat([prefix_ids.to(device), cont.to(device)]).unsqueeze(0)
        logits = model(input_ids=ids, use_cache=False).logits[0]
        start = prefix_ids.shape[0] - 1  # distribution that produced the first new token
        take = min(kappa_max, logits.shape[0] - start)
        ent = entropy_from_logits(logits[start:start + take].float())
        if take < kappa_max:  # continuation hit EOS early
            ent = torch.cat([ent, torch.full((kappa_max - take,), float("nan"), device=device)])
        rows.append(ent.cpu())
    return torch.stack(rows)


@torch.no_grad()
def forecast_head_entropies(model, heads, prefix_ids, device, temperature):
    """Per-head forecast entropy at the prefix's final position.  `[K]`."""
    ids = prefix_ids.unsqueeze(0).to(device)
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    last_hidden = out.hidden_states[-1][:, -1:, :]  # [1, 1, H] — after the last prefix token
    ent = heads.forecast_entropy(
        last_hidden, model.get_output_embeddings(), temperature=temperature
    )
    return ent[:, 0, 0].cpu()


def branch_diversity(continuations, embed_model=None, texts=None):
    """How different the continuations are — the `GT_branch_div` of plan §3.3.

    Uses sentence embeddings when available (mean pairwise cosine distance);
    otherwise falls back to first-divergence depth, i.e. how early the
    continuations stop agreeing.  The fallback is coarser but needs no extra
    model, and it measures the same thing the branch score is about.
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


# ----------------------------------------------------------------------
# main protocol
# ----------------------------------------------------------------------
def build_prefixes(response_ids, fractions):
    """Cut one trajectory at the requested fractions of its length."""
    out = []
    n = len(response_ids)
    for f in fractions:
        cut = int(n * f)
        if 4 <= cut < n:
            out.append(cut)
    return out


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--heads", required=True, help="checkpoint from phase1_warmup_heads.py")
    p.add_argument("--problems", required=True, help="parquet with a prompt column")
    p.add_argument("--out", required=True, help="JSON results path")
    p.add_argument("--prompt-column", default="prompt")
    p.add_argument("--n-problems", type=int, default=25, help="plan asks for 20-30")
    p.add_argument("--n-trajectories", type=int, default=32)
    p.add_argument("--n-continuations", type=int, default=16, help="K_mc")
    p.add_argument("--fractions", default="0.2,0.4,0.6,0.8")
    p.add_argument("--max-prefixes", type=int, default=600, help="plan targets ~600")
    p.add_argument("--max-prefixes-per-problem", type=int, default=None,
                   help="cap per problem; default spreads --max-prefixes evenly over "
                        "--n-problems. Without a cap the budget is consumed by the "
                        "first few problems and the within-problem average is taken "
                        "over far fewer problems than requested")
    p.add_argument("--traj-max-tokens", type=int, default=1024)
    p.add_argument("--cont-max-tokens", type=int, default=64,
                   help="only the first kappa_max offsets are scored")
    p.add_argument("--traj-temperature", type=float, default=1.0)
    p.add_argument("--cont-temperature", type=float, default=0.7)
    p.add_argument("--cont-top-p", type=float, default=0.9)
    p.add_argument("--forecast-temperature", type=float, default=1.0,
                   help="must match the temperature verl uses for `entropys`")
    p.add_argument("--embed-model", default=None,
                   help="e.g. sentence-transformers/all-MiniLM-L6-v2; omitted -> token fallback")
    p.add_argument("--calibrate", action="store_true",
                   help="fit per-head scale/bias on the measured entropies")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
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

    ckpt = torch.load(args.heads, map_location="cpu")
    heads = MTPHeads(**ckpt["config"]).to(args.device, dtype=dtype)
    heads.load_state_dict(ckpt["state_dict"])
    heads.eval()
    kappa_max = ckpt["config"]["num_heads"]
    print(f"[validate] heads K={kappa_max} from {args.heads}")

    embed_model = None
    if args.embed_model:
        from sentence_transformers import SentenceTransformer

        embed_model = SentenceTransformer(args.embed_model, device=args.device)

    df = pd.read_parquet(args.problems)
    prompts = [render_prompt(x, tokenizer)
               for x in df[args.prompt_column].tolist()[: args.n_problems]]
    fractions = [float(x) for x in args.fractions.split(",")]
    print(f"[validate] {len(prompts)} problems, fractions={fractions}")

    # The primary metric is a *within-problem* correlation averaged over
    # problems, so problem coverage is what gives it its resolution. Collecting
    # greedily, one problem at a time, until --max-prefixes is reached spends
    # the whole budget on the first few: with 32 trajectories x 4 cuts = 128
    # prefixes per problem, a 600-prefix budget was exhausted after 5 of 25
    # problems, and rho was a Fisher-z average over 5 values.
    per_problem = args.max_prefixes_per_problem
    if per_problem is None:
        per_problem = max(1, args.max_prefixes // max(1, len(prompts)))
    print(f"[validate] budget: {args.max_prefixes} prefixes, <= {per_problem} per problem")

    records = []
    for pi, prompt in enumerate(prompts):
        problem_start = len(records)
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        trajectories = sample_continuations(
            model, tokenizer, prompt_ids, args.n_trajectories, args.traj_max_tokens,
            args.traj_temperature, 1.0, args.device,
        )

        for ti, traj in enumerate(trajectories):
            for cut in build_prefixes(traj, fractions):
                if len(records) >= args.max_prefixes:
                    break
                if len(records) - problem_start >= per_problem:
                    break
                prefix_ids = torch.cat([prompt_ids, traj[:cut].cpu()])

                conts = sample_continuations(
                    model, tokenizer, prefix_ids, args.n_continuations,
                    args.cont_max_tokens, args.cont_temperature, args.cont_top_p, args.device,
                )
                measured = measured_future_entropy(
                    model, prefix_ids, conts, kappa_max, args.device
                )  # [n_cont, kappa_max]
                gt_per_offset = torch.nanmean(measured, dim=0)  # [kappa_max]

                forecast = forecast_head_entropies(
                    model, heads, prefix_ids, args.device, args.forecast_temperature
                )  # [kappa_max]

                texts = [tokenizer.decode(c, skip_special_tokens=True) for c in conts] \
                    if embed_model else None
                records.append({
                    "problem": pi,
                    "trajectory": ti,
                    "cut": cut,
                    "forecast_per_head": forecast.tolist(),
                    "gt_per_offset": gt_per_offset.tolist(),
                    "gt_sum_h": float(torch.nansum(gt_per_offset)),
                    "gt_branch_div": branch_diversity(conts, embed_model, texts),
                })
            if len(records) >= args.max_prefixes:
                break
            if len(records) - problem_start >= per_problem:
                break
        print(f"[validate] problem {pi+1}/{len(prompts)}: "
              f"+{len(records)-problem_start} (total {len(records)})", flush=True)
        if len(records) >= args.max_prefixes:
            break

    if not records:
        raise SystemExit("[validate] no prefixes collected — check --fractions / lengths")

    # ---- optional calibration, fit on the measured per-offset entropies ----
    calib = None
    if args.calibrate:
        f_mat = torch.tensor([r["forecast_per_head"] for r in records]).T   # [K, N]
        m_mat = torch.tensor([r["gt_per_offset"] for r in records]).T       # [K, N]
        keep = torch.isfinite(m_mat).all(dim=0)
        calib = fit_head_calibration(f_mat[:, keep], m_mat[:, keep])
        print(f"[validate] calibration scale={['%.3f' % s for s in calib.scale]}")
        print(f"[validate]             bias ={['%.3f' % b for b in calib.bias]}")

    # ---- (kappa, gamma_H) grid ----
    grid = []
    for gamma in GAMMA_GRID:
        for kappa in range(1, kappa_max + 1):
            forecasts, truths, pids = [], [], []
            for r in records:
                f = h_togo(
                    torch.tensor(r["forecast_per_head"]).view(-1, 1, 1),
                    kappa=kappa, gamma_h=gamma, calib=calib,
                )
                gt_offsets = torch.tensor(r["gt_per_offset"])[:kappa]
                if not torch.isfinite(gt_offsets).all():
                    continue  # continuation ended before the horizon
                weights = torch.tensor([gamma ** (k + 1) for k in range(kappa)])
                forecasts.append(float(f))
                truths.append(float((weights * gt_offsets).sum()))
                pids.append(r["problem"])

            stats = within_problem_spearman(forecasts, truths, pids)
            grid.append({
                "kappa": kappa, "gamma_h": gamma,
                "rho_mean": stats["rho_mean"], "rho_pooled": stats["rho_pooled"],
                "n_problems": stats["n_problems"], "n_observations": stats["n_observations"],
            })
            print(f"[validate] kappa={kappa} gamma_H={gamma}: "
                  f"rho_within={stats['rho_mean']:.4f} (pooled {stats['rho_pooled']:.4f}, "
                  f"{stats['n_observations']} obs)")

    chosen = select_kappa_gamma(grid)
    print(f"\n[validate] selected kappa={chosen['kappa']} gamma_H={chosen['gamma_h']} "
          f"(rho={chosen['rho_mean']:.4f}, best available {chosen['rho_best']:.4f}, "
          f"giving up {chosen['rho_giveup']:.4f})")

    # ---- secondary: does H_togo track branch diversity? ----
    div_forecasts, div_truths, div_pids = [], [], []
    for r in records:
        f = h_togo(
            torch.tensor(r["forecast_per_head"]).view(-1, 1, 1),
            kappa=chosen["kappa"], gamma_h=chosen["gamma_h"], calib=calib,
        )
        div_forecasts.append(float(f))
        div_truths.append(r["gt_branch_div"])
        div_pids.append(r["problem"])
    div_stats = within_problem_spearman(div_forecasts, div_truths, div_pids)
    print(f"[validate] rho(H_togo, branch diversity) = {div_stats['rho_mean']:.4f}")

    result = {
        "args": vars(args),
        "n_records": len(records),
        "grid": grid,
        "chosen": chosen,
        "branch_diversity_rho": div_stats["rho_mean"],
        "calibration": calib.to_dict() if calib else None,
        "records": records,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[validate] wrote {out_path}")

    print(
        "\n[validate] NOTE: gate G1 also needs branch recall@10%, which is measured on a\n"
        "           real rollout group (steer_f.monitors.branch_recall_at_k) rather than on\n"
        "           these independently-sampled prefixes. Run it on a Phase-0 rollout dump,\n"
        "           then feed both numbers to steer_f.validation.evaluate_gate_g1."
    )
    partial = evaluate_gate_g1(
        rho_mean=chosen["rho_mean"], recall=float("nan"),
        n_branch=0, n_hit=0, selection_rate=0.1,
    )
    print(partial.summary())


if __name__ == "__main__":
    main()
