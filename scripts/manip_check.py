#!/usr/bin/env python3
"""Manipulation check for the A_H correction. Registered in docs/manipulation_check.md.

The signed correction damps the sibling with the larger entropy-to-go less and
the one with the smaller entropy-to-go more, so after training the richer
sibling's log-probability should have moved further from the base model's than
the poorer one's -- in STEER-F, and not in the uniform or permuted controls.

Subcommands, one GPU process each so vLLM and HF never share memory:

    build    sample branch points and continuations from the base model (vLLM)
    togo     realised entropy-to-go of every continuation (HF, full vocabulary)
    score    log pi of both branch tokens at every branch point (HF, fp32)
    fetch    download one run's checkpoint from the Hub, print its directory
    analyze  per-checkpoint means and per-seed contrasts (CPU only)

`analyze` imports nothing heavier than the standard library, so the statistics
are testable without a GPU (tests/test_manip_check.py).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEPTHS = (64, 192, 384)      # the tree's branch depths (run_uniform_ablation.sh)
PRIMARY = [("signed", "uniform"), ("signed", "permuted")]
CONTEXT = [("signed", "grpo"), ("signed", "steer"), ("steer", "grpo"),
           ("uniform", "steer")]


# ------------------------------------------------------------------ helpers
def stop_ids(base: str) -> set[int]:
    from transformers import AutoTokenizer, GenerationConfig
    ids = set()
    tok = AutoTokenizer.from_pretrained(base)
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    try:
        e = GenerationConfig.from_pretrained(base).eos_token_id
        ids.update([e] if isinstance(e, int) else (e or []))
    except Exception:
        pass
    return ids


def load_prompts(parquet: str, n: int, seed: int, base: str) -> list[list[int]]:
    """Chat-templated prompt ids, as verl's RLHFDataset builds them."""
    import pandas as pd
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    df = pd.read_parquet(parquet)
    idx = random.Random(seed).sample(range(len(df)), n)
    out = []
    for i in idx:
        msgs = [dict(m) for m in df.iloc[i]["prompt"]]
        out.append(tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           tokenize=True))
    return out


# -------------------------------------------------------------------- build
def cmd_build(a) -> None:
    from vllm import LLM, SamplingParams
    probe = Path(a.probe); probe.mkdir(parents=True, exist_ok=True)
    stops = stop_ids(a.base)
    prompts = load_prompts(a.data, a.n_prompts, a.seed, a.base)
    llm = LLM(model=a.base, dtype="bfloat16", seed=a.seed,
              gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True)

    def gen(ids_list, **kw):
        sp = SamplingParams(temperature=1.0, top_p=1.0, seed=None, **kw)
        return llm.generate([{"prompt_token_ids": x} for x in ids_list], sp,
                            use_tqdm=True)

    depths = [int(d) for d in a.depths.split(",")]
    first = gen(prompts, max_tokens=max(depths) + 1)
    points = []
    for pi, (p, o) in enumerate(zip(prompts, first)):
        resp = list(o.outputs[0].token_ids)
        for d in depths:
            # The prefix must be d tokens of a response still running at d.
            if len(resp) <= d or any(t in stops for t in resp[:d]):
                continue
            points.append({"id": f"{pi}-{d}", "prompt": pi, "depth": d,
                           "ctx": p + resp[:d]})
    print(f"[build] {len(points)} branch points from {len(prompts)} prompts")

    draws = gen([pt["ctx"] for pt in points], n=a.draws, max_tokens=1)
    kept, one_token = [], 0
    for pt, o in zip(points, draws):
        seen = []
        for c in o.outputs:
            t = c.token_ids[0] if c.token_ids else None
            if t is not None and t not in seen:
                seen.append(t)
        if len(seen) < 2:
            one_token += 1
            continue
        pt["tok"] = seen[:2]
        kept.append(pt)
    print(f"[build] dropped {one_token} points with one distinct token in {a.draws} draws")

    for pt in kept:
        # A branch token that ends the response has nothing to go: H_togo = 0.
        pt["conts"] = [[[] for _ in range(a.k)], [[] for _ in range(a.k)]]
    # The response budget is the training one (3072), so a continuation from
    # depth d gets what is left of it -- one call per depth.
    for d in depths:
        jobs = [(i, j) for i, pt in enumerate(kept) if pt["depth"] == d
                for j in range(2) if pt["tok"][j] not in stops]
        if not jobs:
            continue
        outs = gen([kept[i]["ctx"] + [kept[i]["tok"][j]] for i, j in jobs],
                   n=a.k, max_tokens=a.max_len - d - 1)
        for (i, j), o in zip(jobs, outs):
            kept[i]["conts"][j] = [list(c.token_ids) for c in o.outputs]

    with (probe / "probe.jsonl").open("w") as fh:
        for pt in kept:
            fh.write(json.dumps(pt) + "\n")
    meta = {"base": a.base, "data": a.data, "n_prompts": a.n_prompts,
            "seed": a.seed, "depths": depths, "draws": a.draws, "k": a.k,
            "max_len": a.max_len, "points_cut": len(points),
            "points_kept": len(kept), "dropped_one_token": one_token}
    (probe / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[build] wrote {len(kept)} branch points to {probe}/probe.jsonl")


def read_probe(probe: Path) -> list[dict]:
    with (probe / "probe.jsonl").open() as fh:
        return [json.loads(line) for line in fh]


# --------------------------------------------------------------------- togo
def cmd_togo(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM
    probe = Path(a.probe)
    pts = read_probe(probe)
    model = AutoModelForCausalLM.from_pretrained(
        a.base, torch_dtype=torch.bfloat16).cuda().eval()
    head = model.get_output_embeddings()

    @torch.no_grad()
    def entropy_sum(ctx: list[int], cont: list[int]) -> float:
        """Sum of next-token entropies over the states that emit `cont`."""
        if not cont:
            return 0.0
        ids = torch.tensor([ctx + cont], device="cuda")
        hid = model.model(input_ids=ids).last_hidden_state[0]
        # positions len(ctx)-1 .. len(ctx)+len(cont)-2 predict cont[0..]
        h = hid[len(ctx) - 1: len(ctx) + len(cont) - 1]
        tot = 0.0
        for s in range(0, h.shape[0], 512):
            lp = torch.log_softmax(head(h[s:s + 512]).float(), dim=-1)
            tot += float(-(lp.exp() * lp).sum())
        return tot

    out = {}
    for n, pt in enumerate(pts):
        out[pt["id"]] = [[entropy_sum(pt["ctx"] + [pt["tok"][j]], c)
                          for c in pt["conts"][j]] for j in range(2)]
        if n % 100 == 0:
            print(f"[togo] {n}/{len(pts)}", flush=True)
    (probe / "togo.json").write_text(json.dumps(out))
    print(f"[togo] wrote {probe}/togo.json")


# -------------------------------------------------------------------- score
def cmd_score(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM
    probe = Path(a.probe)
    pts = read_probe(probe)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float32).cuda().eval()
    lp = {}
    with torch.no_grad():
        for s in range(0, len(pts), a.batch):
            chunk = pts[s:s + a.batch]
            L = max(len(p["ctx"]) for p in chunk)
            ids = torch.zeros(len(chunk), L, dtype=torch.long)
            mask = torch.zeros(len(chunk), L, dtype=torch.long)
            for r, p in enumerate(chunk):
                ids[r, :len(p["ctx"])] = torch.tensor(p["ctx"])
                mask[r, :len(p["ctx"])] = 1
            logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits
            for r, p in enumerate(chunk):
                row = torch.log_softmax(logits[r, len(p["ctx"]) - 1].float(), -1)
                lp[p["id"]] = [float(row[p["tok"][0]]), float(row[p["tok"][1]])]
    m = re.search(r"global_step_(\d+)", a.model)
    rec = {"label": a.label, "model": a.model,
           "step": int(m.group(1)) if m else None, "lp": lp}
    d = probe / "scores"; d.mkdir(exist_ok=True)
    (d / f"{a.label.replace(':', '-s')}.json").write_text(json.dumps(rec))
    print(f"[score] {a.label}: {len(lp)} branch points")


# -------------------------------------------------------------------- fetch
def cmd_fetch(a) -> None:
    """Same Hub layout as run_eval_all.sh:hub_step/resolve_ckpt: <run>/global_step_<N>/."""
    from huggingface_hub import HfApi, snapshot_download
    for repo in a.repos.split():
        steps = set()
        try:
            for e in HfApi().list_repo_tree(repo, path_in_repo=a.run,
                                            recursive=True, repo_type="model"):
                m = re.search(r"/global_step_(\d+)/", "/" + e.path + "/")
                if m:
                    steps.add(int(m.group(1)))
        except Exception:
            continue
        if not steps or (a.step is not None and a.step not in steps):
            continue
        step = a.step if a.step is not None else max(steps)
        if a.dry:
            print(f"hub:{repo}/{a.run}/global_step_{step}")
            return
        local = snapshot_download(repo, repo_type="model", local_dir=a.dest,
                                  allow_patterns=[f"{a.run}/global_step_{step}/*"])
        print(os.path.join(local, a.run, f"global_step_{step}"))
        return


# ------------------------------------------------------------------ analyze
def point_stats(togo: dict, base: dict, model: dict) -> list[dict]:
    """Per branch point: which sibling is richer, and how far each moved."""
    rows = []
    for pid, (ha, hb) in togo.items():
        if pid not in base or pid not in model or not ha or not hb:
            continue
        ma, mb = sum(ha) / len(ha), sum(hb) / len(hb)
        if ma == mb:
            continue
        hi, lo = (0, 1) if ma > mb else (1, 0)
        d = [model[pid][j] - base[pid][j] for j in (0, 1)]
        rows.append({"id": pid, "gap": abs(ma - mb),
                     "D": abs(d[hi]) - abs(d[lo]), "S": d[hi] - d[lo]})
    return rows


def mean_se(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def summarise(rows: list[dict], gap_cut: float | None) -> dict:
    sel = rows if gap_cut is None else [r for r in rows if r["gap"] >= gap_cut]
    mD, seD = mean_se([r["D"] for r in sel])
    mS, seS = mean_se([r["S"] for r in sel])
    return {"n": len(sel), "D": mD, "D_se": seD, "S": mS, "S_se": seS}


def verdict(per_seed: dict[str, list[float]]) -> str:
    """Registered rule: both primary contrasts positive on every seed."""
    ok = all(v and all(x > 0 for x in v) for v in per_seed.values())
    return "passed" if ok and len(per_seed) == len(PRIMARY) else "not passed"


def point_contrast(rows_x: list[dict], rows_y: list[dict], stat: str = "D",
                   n_boot: int = 10000, seed: int = 0) -> dict:
    """Mean of the per-point difference x - y over shared branch points, with a
    percentile bootstrap interval over points. Both checkpoints were scored on
    the same probe, so the points pair exactly (Amendment 1)."""
    ys = {r["id"]: r[stat] for r in rows_y}
    d = [r[stat] - ys[r["id"]] for r in rows_x if r["id"] in ys]
    n = len(d)
    if n == 0:
        return {"n": 0}
    m = sum(d) / n
    rng = random.Random(seed)
    boots = sorted(sum(rng.choices(d, k=n)) / n for _ in range(n_boot))
    return {"n": n, "mean": m, "ci_lo": boots[int(0.025 * n_boot)],
            "ci_hi": boots[int(0.975 * n_boot) - 1]}


def verdict_amended(cs: dict[str, dict]) -> str:
    """Amendment 1: every primary contrast that can be formed has its 95%
    interval above zero, and at least one can be formed."""
    have = [c for c in cs.values() if c.get("n")]
    if not have:
        return "not applicable"
    return "passed" if all(c["ci_lo"] > 0 for c in have) else "not passed"


def analyze(probe: Path) -> dict:
    from analyze_seeds import paired
    togo = json.loads((probe / "togo.json").read_text())
    scores = {}
    for f in sorted((probe / "scores").glob("*.json")):
        rec = json.loads(f.read_text())
        scores[rec["label"]] = rec
    base = scores.pop("base")["lp"]

    # One gap threshold for every checkpoint: the median over the probe, so
    # the "strong half" is the same set of branch points everywhere.
    gaps = sorted(abs(sum(a) / len(a) - sum(b) / len(b))
                  for a, b in togo.values() if a and b)
    cut = gaps[len(gaps) // 2] if gaps else None

    per_model, point_rows = {}, {}
    for label, rec in scores.items():
        rows = point_stats(togo, base, rec["lp"])
        point_rows[label] = rows
        per_model[label] = {"step": rec.get("step"),
                            "all": summarise(rows, None),
                            "strong": summarise(rows, cut)}

    def arm_seed(label):
        arm, seed = label.split(":")
        return arm, int(seed)

    by = {arm_seed(k): v for k, v in per_model.items()}
    seeds = sorted({s for _, s in by})
    contrasts = {}
    for x, y in PRIMARY + CONTEXT:
        for subset in ("all", "strong"):
            for stat in ("D", "S"):
                diffs = [by[(x, s)][subset][stat] - by[(y, s)][subset][stat]
                         for s in seeds if (x, s) in by and (y, s) in by]
                if not diffs:
                    continue
                contrasts[f"{x}-{y}:{subset}:{stat}"] = {
                    "per_seed": diffs, **paired(diffs)}
    primary = {f"{x}-{y}": contrasts.get(f"{x}-{y}:all:D", {}).get("per_seed", [])
               for x, y in PRIMARY}
    # Point-level contrasts within each seed (Amendment 1). Labels are kept
    # as given, so a checkpoint from a differently named run is paired by the
    # arm and seed it is scored under.
    points = {}
    for x, y in PRIMARY + CONTEXT:
        for s in seeds:
            lx, ly = f"{x}:{s}", f"{y}:{s}"
            if lx not in point_rows or ly not in point_rows:
                continue
            strong_x = [r for r in point_rows[lx] if cut is not None and r["gap"] >= cut]
            points[f"{x}-{y}:s{s}:all:D"] = point_contrast(point_rows[lx], point_rows[ly])
            points[f"{x}-{y}:s{s}:strong:D"] = point_contrast(strong_x, point_rows[ly])
            points[f"{x}-{y}:s{s}:all:S"] = point_contrast(point_rows[lx], point_rows[ly], "S")
    amended = {k: v for k, v in points.items()
               if k.endswith(":all:D") and k.split(":")[0] in
               {f"{x}-{y}" for x, y in PRIMARY}}
    n_seeds = min((len(v) for v in primary.values()), default=0)
    return {"gap_cut": cut, "per_model": per_model, "contrasts": contrasts,
            "primary": primary,
            "verdict": verdict(primary) if n_seeds >= 3
            else f"not applicable ({n_seeds} seed(s); see Amendment 1)",
            "points": points, "verdict_amended": verdict_amended(amended)}


def cmd_analyze(a) -> None:
    res = analyze(Path(a.probe))
    Path(a.out).write_text(json.dumps(res, indent=2, default=str))
    print(f"{'checkpoint':16s} {'step':>5s} {'n':>5s} {'D':>9s} {'(se)':>8s}"
          f" {'S':>9s} {'(se)':>8s}")
    for label, v in sorted(res["per_model"].items()):
        s = v["all"]
        print(f"{label:16s} {str(v['step']):>5s} {s['n']:5d} {s['D']:+9.4f} "
              f"{s['D_se']:8.4f} {s['S']:+9.4f} {s['S_se']:8.4f}")
    print()
    for key, c in res["contrasts"].items():
        if ":all:D" in key or ":strong:D" in key:
            print(f"{key:28s} mean {c['mean']:+.4f}  t {c['t']:+.2f}  "
                  f"seeds {' '.join(f'{x:+.4f}' for x in c['per_seed'])}")
    print()
    for key, c in res["points"].items():
        if c.get("n"):
            print(f"{key:30s} n {c['n']:5d}  mean {c['mean']:+.4f}  "
                  f"95% CI [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]")
    print(f"\nregistered rule (both primary contrasts > 0 on every seed): "
          f"{res['verdict']}")
    print(f"amended rule (point-level 95% CI above zero): "
          f"{res['verdict_amended']}")
    print(f"wrote {a.out}")


# ---------------------------------------------------------------------- cli
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--base", required=True)
    b.add_argument("--data", required=True)
    b.add_argument("--probe", required=True)
    b.add_argument("--n-prompts", type=int, default=512)
    b.add_argument("--depths", default=",".join(map(str, DEPTHS)))
    b.add_argument("--draws", type=int, default=8)
    b.add_argument("--k", type=int, default=4)
    b.add_argument("--max-len", type=int, default=3072)
    b.add_argument("--seed", type=int, default=20261013)
    b.add_argument("--gpu-mem", type=float, default=0.85)

    t = sub.add_parser("togo")
    t.add_argument("--base", required=True)
    t.add_argument("--probe", required=True)

    s = sub.add_parser("score")
    s.add_argument("--model", required=True)
    s.add_argument("--label", required=True, help="'base' or '<arm>:<seed>'")
    s.add_argument("--probe", required=True)
    s.add_argument("--batch", type=int, default=16)

    f = sub.add_parser("fetch")
    f.add_argument("--repos", required=True)
    f.add_argument("--run", required=True)
    f.add_argument("--dest", required=True)
    f.add_argument("--step", type=int, default=None,
                   help="a specific global_step; default the highest")
    f.add_argument("--dry", action="store_true")

    z = sub.add_parser("analyze")
    z.add_argument("--probe", required=True)
    z.add_argument("--out", required=True)

    a = ap.parse_args(argv)
    {"build": cmd_build, "togo": cmd_togo, "score": cmd_score,
     "fetch": cmd_fetch, "analyze": cmd_analyze}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
