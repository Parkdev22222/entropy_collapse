#!/usr/bin/env python3
"""Two-channel decomposition of measured entropy change between checkpoints.
Registered in docs/channel_decomposition.md.

For checkpoints a -> b, with E_x[S_m] the mean over responses sampled from x
of the summed next-token entropies of model m along the response:

    dH = E_b[S_b] - E_a[S_a] = L + V
    L  = E_b[S_b] - E_b[S_a]   (local: same states, sharper conditionals)
    V  = E_b[S_a] - E_a[S_a]   (visitation: different states visited)

averaged with the reverse ordering. Exact; no approximation.

Subcommands, one GPU process each:
    sample   responses of one checkpoint on the shared prompt set (vLLM)
    entropy  summed entropies of one sample set under one model (HF)
    analyze  decomposition, contrasts and bootstrap intervals (CPU only)
    macros   the manuscript's \\num{} values, recomputed from a git ref (CPU only)
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PRIMARY = [("base", "grpo100"), ("base", "perm100"), ("base", "signed100")]
SECONDARY = [("grpo80", "grpo110"), ("perm80", "perm110"), ("base", "signed50"),
             ("signed50", "signed100"), ("signed100", "signed150"),
             ("signed150", "signed200")]


# ------------------------------------------------------------------- sample
def cmd_sample(a) -> None:
    from vllm import LLM, SamplingParams
    from manip_check import load_prompts
    out = Path(a.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    pf = out / "prompts.json"
    if pf.exists():
        prompts = json.loads(pf.read_text())
    else:
        prompts = load_prompts(a.data, a.n_prompts, a.seed, a.base)
        pf.write_text(json.dumps(prompts))
    llm = LLM(model=a.model, tokenizer=a.base, dtype="bfloat16", seed=a.seed,
              gpu_memory_utilization=a.gpu_mem)
    sp = SamplingParams(temperature=1.0, top_p=1.0, n=a.k, max_tokens=a.max_len)
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], sp, use_tqdm=True)
    rec = {"label": a.label, "model": a.model, "k": a.k,
           "responses": [[list(c.token_ids) for c in o.outputs] for o in outs]}
    (out / "samples" / f"{a.label}.json").write_text(json.dumps(rec))
    n = sum(len(r) for r in rec["responses"])
    print(f"[sample] {a.label}: {n} responses over {len(prompts)} prompts")


# ------------------------------------------------------------------ entropy
def cmd_entropy(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM
    out = Path(a.out)
    prompts = json.loads((out / "prompts.json").read_text())
    rec = json.loads((out / "samples" / f"{a.samples}.json").read_text())
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16).cuda().eval()
    head = model.get_output_embeddings()

    @torch.no_grad()
    def entropy_sum(ctx: list[int], resp: list[int]) -> float:
        """Sum of next-token entropies over the states that emit `resp`."""
        if not resp:
            return 0.0
        ids = torch.tensor([ctx + resp], device="cuda")
        hid = model.model(input_ids=ids).last_hidden_state[0]
        h = hid[len(ctx) - 1: len(ctx) + len(resp) - 1]
        tot = 0.0
        for s in range(0, h.shape[0], 512):
            lp = torch.log_softmax(head(h[s:s + 512]).float(), dim=-1)
            tot += float(-(lp.exp() * lp).sum())
        return tot

    sums = []
    for i, (p, rs) in enumerate(zip(prompts, rec["responses"])):
        sums.append([entropy_sum(p, r) for r in rs])
        if i % 64 == 0:
            print(f"[entropy] {a.samples} under {a.label}: {i}/{len(prompts)}", flush=True)
    d = out / "ent"; d.mkdir(exist_ok=True)
    (d / f"{a.samples}__{a.label}.json").write_text(json.dumps(
        {"samples": a.samples, "under": a.label, "model": a.model,
         "samples_model": rec["model"], "sums": sums}))
    print(f"[entropy] wrote {a.samples} under {a.label}")


# ------------------------------------------------------------------ analyze
def per_prompt(sums: list[list[float]]) -> list[float]:
    return [sum(s) / len(s) if s else float("nan") for s in sums]


def interval_terms(E: dict, a: str, b: str) -> dict[str, list[float]]:
    """Per-prompt dH, L, V (mean of both orderings). E[(x, m)] = per-prompt
    mean of S_m over samples from x."""
    aa, ab, ba, bb = E[(a, a)], E[(a, b)], E[(b, a)], E[(b, b)]
    n = len(aa)
    dH = [bb[i] - aa[i] for i in range(n)]
    L1 = [bb[i] - ba[i] for i in range(n)]      # b's states
    V1 = [ba[i] - aa[i] for i in range(n)]
    L2 = [ab[i] - aa[i] for i in range(n)]      # a's states
    V2 = [bb[i] - ab[i] for i in range(n)]
    L = [(L1[i] + L2[i]) / 2 for i in range(n)]
    V = [(V1[i] + V2[i]) / 2 for i in range(n)]
    return {"dH": dH, "L": L, "V": V, "L_b_states": L1, "L_a_states": L2}


def boot(stat, data: list[list[float]], n_boot: int = 10000, seed: int = 0):
    """Percentile interval of stat(*resampled lists), resampling prompts
    jointly so paired quantities stay paired."""
    n = len(data[0])
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.choices(range(n), k=n)
        vals.append(stat(*[[d[i] for i in idx] for d in data]))
    vals.sort()
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]


def mean(x): return sum(x) / len(x)


def analyze(out: Path, n_boot: int = 10000) -> dict:
    E = {}
    for f in sorted((out / "ent").glob("*.json")):
        r = json.loads(f.read_text())
        E[(r["samples"], r["under"])] = per_prompt(r["sums"])
    res = {"intervals": {}, "contrasts": {}}
    terms = {}
    for a, b in PRIMARY + SECONDARY:
        if not all(k in E for k in [(a, a), (a, b), (b, a), (b, b)]):
            continue
        t = interval_terms(E, a, b)
        terms[(a, b)] = t
        row = {}
        for k in ("dH", "L", "V"):
            lo, hi = boot(mean, [t[k]], n_boot)
            row[k] = {"mean": mean(t[k]), "ci": [lo, hi]}
        lo, hi = boot(lambda v, d: mean(v) / mean(d), [t["V"], t["dH"]], n_boot)
        row["V_share"] = {"mean": mean(t["V"]) / mean(t["dH"]), "ci": [lo, hi]}
        row["H_a"], row["H_b"] = mean(E[(a, a)]), mean(E[(b, b)])
        row["n_prompts"] = len(t["dH"])
        row["L_ordering_gap"] = mean(t["L_b_states"]) - mean(t["L_a_states"])
        res["intervals"][f"{a}->{b}"] = row

    ref = ("base", "grpo100")
    for arm in ("perm100", "signed100"):
        if ref not in terms or ("base", arm) not in terms:
            continue
        c = {}
        for k in ("L", "V"):
            d = [x - y for x, y in zip(terms[("base", arm)][k], terms[ref][k])]
            lo, hi = boot(mean, [d], n_boot)
            c[k] = {"mean": mean(d), "ci": [lo, hi]}
        c["consistent_with_local"] = bool(
            (c["L"]["ci"][0] > 0 or c["L"]["ci"][1] < 0)
            and abs(c["L"]["mean"]) > abs(c["V"]["mean"]))
        res["contrasts"][f"{arm}-grpo100"] = c

    q1 = res["intervals"].get("base->grpo100")
    if q1:
        v, d = q1["V"], q1["dH"]["mean"]
        excl = v["ci"][0] > 0 or v["ci"][1] < 0
        same = (v["mean"] > 0) == (d > 0)
        res["q1_verdict"] = ("visitation share established" if excl and same
                             else "not established")
    return res


def robust(out: Path, max_len: int = 3072, n_boot: int = 4000) -> dict:
    """Exploratory, added after the registered result was seen: the primary
    intervals recomputed (i) without responses truncated at max_len, whose
    rambling could carry the visitation term on its own, and (ii) per token,
    sum of entropies over sum of lengths, so a change in length cannot."""
    samples = {}
    for f in (out / "samples").glob("*.json"):
        r = json.loads(f.read_text())
        samples[r["label"]] = [[len(x) for x in rs] for rs in r["responses"]]
    S = {}
    for f in (out / "ent").glob("*.json"):
        r = json.loads(f.read_text())
        S[(r["samples"], r["under"])] = r["sums"]
    res = {}
    for name, keep, per_token in [("drop_truncated", lambda n: n < max_len, False),
                                  ("per_token", lambda n: True, True),
                                  ("drop_truncated_per_token", lambda n: n < max_len, True)]:
        res[name] = {}
        for a, b in PRIMARY:
            if not all(k in S for k in [(a, a), (a, b), (b, a), (b, b)]):
                continue
            rows = []
            for i in range(len(samples[a])):
                def E(x, m):
                    v = [(e, n) for e, n in zip(S[(x, m)][i], samples[x][i]) if keep(n)]
                    if not v:
                        return None
                    return (sum(e for e, _ in v) / sum(n for _, n in v) if per_token
                            else sum(e for e, _ in v) / len(v))
                aa, ab, ba, bb = E(a, a), E(a, b), E(b, a), E(b, b)
                if None in (aa, ab, ba, bb):
                    continue
                dH, Lc = bb - aa, ((bb - ba) + (ab - aa)) / 2
                rows.append((dH, Lc, dH - Lc))
            dH, V = [r[0] for r in rows], [r[2] for r in rows]
            lo, hi = boot(lambda v, d: mean(v) / mean(d), [V, dH], n_boot)
            res[name][f"{a}->{b}"] = {"n": len(rows), "dH": mean(dH),
                                      "L": mean([r[1] for r in rows]), "V": mean(V),
                                      "V_share": mean(V) / mean(dH), "ci": [lo, hi]}
    return res


def cmd_analyze(a) -> None:
    res = analyze(Path(a.out), a.n_boot)
    if (Path(a.out) / "samples").is_dir():
        res["exploratory_robustness"] = robust(Path(a.out))
    Path(a.result).write_text(json.dumps(res, indent=2))
    f = lambda r: f"{r['mean']:+9.2f} [{r['ci'][0]:+8.2f}, {r['ci'][1]:+8.2f}]"
    print(f"{'interval':22s} {'H_a':>7s} {'H_b':>7s}   {'dH':>27s}   {'L (local)':>27s}"
          f"   {'V (visitation)':>27s}   V/dH")
    for k, r in res["intervals"].items():
        s = r["V_share"]
        print(f"{k:22s} {r['H_a']:7.1f} {r['H_b']:7.1f}   {f(r['dH'])}   {f(r['L'])}"
              f"   {f(r['V'])}   {s['mean']:+.2f} [{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}]")
    print()
    for k, c in res["contrasts"].items():
        print(f"{k:18s} dL {f(c['L'])}   dV {f(c['V'])}   "
              f"consistent with local-channel prediction: {c['consistent_with_local']}")
    print(f"\nQ1 (base->grpo100): {res.get('q1_verdict', 'not computed')}")
    for name, rows in res.get("exploratory_robustness", {}).items():
        for k, r in rows.items():
            print(f"[exploratory {name}] {k}: V/dH {r['V_share']:.2f} "
                  f"[{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]  n={r['n']}")
    print(f"wrote {a.result}")


# ------------------------------------------------------------------- macros
# macro name -> (path into the analyze() result, format). tests/test_paper.py
# reads names() to decide which slots a run can fill, as it does for the other
# emitters, so a name the manuscript asks for has to be here.
_I = {"Grpo": "base->grpo100", "Perm": "base->perm100", "Signed": "base->signed100"}
MACROS = {
    "DecGrpoHa":        (("intervals", _I["Grpo"], "H_a"), "1f"),
    "DecGrpoHb":        (("intervals", _I["Grpo"], "H_b"), "1f"),
    "DecGrpoDH":        (("intervals", _I["Grpo"], "dH", "mean"), "1f"),
    "DecGrpoDHLo":      (("intervals", _I["Grpo"], "dH", "ci", 0), "1f"),
    "DecGrpoDHHi":      (("intervals", _I["Grpo"], "dH", "ci", 1), "1f"),
    "DecGrpoL":         (("intervals", _I["Grpo"], "L", "mean"), "1f"),
    "DecGrpoLLo":       (("intervals", _I["Grpo"], "L", "ci", 0), "1f"),
    "DecGrpoLHi":       (("intervals", _I["Grpo"], "L", "ci", 1), "1f"),
    "DecGrpoV":         (("intervals", _I["Grpo"], "V", "mean"), "1f"),
    "DecGrpoVLo":       (("intervals", _I["Grpo"], "V", "ci", 0), "1f"),
    "DecGrpoVHi":       (("intervals", _I["Grpo"], "V", "ci", 1), "1f"),
    "DecGrpoShare":     (("intervals", _I["Grpo"], "V_share", "mean"), "2f"),
    "DecGrpoShareLo":   (("intervals", _I["Grpo"], "V_share", "ci", 0), "2f"),
    "DecGrpoShareHi":   (("intervals", _I["Grpo"], "V_share", "ci", 1), "2f"),
    "DecPermHb":        (("intervals", _I["Perm"], "H_b"), "1f"),
    "DecPermDH":        (("intervals", _I["Perm"], "dH", "mean"), "1f"),
    "DecPermL":         (("intervals", _I["Perm"], "L", "mean"), "1f"),
    "DecPermV":         (("intervals", _I["Perm"], "V", "mean"), "1f"),
    "DecPermShare":     (("intervals", _I["Perm"], "V_share", "mean"), "2f"),
    "DecPermShareLo":   (("intervals", _I["Perm"], "V_share", "ci", 0), "2f"),
    "DecPermShareHi":   (("intervals", _I["Perm"], "V_share", "ci", 1), "2f"),
    "DecSignedHb":      (("intervals", _I["Signed"], "H_b"), "1f"),
    "DecSignedDH":      (("intervals", _I["Signed"], "dH", "mean"), "1f"),
    "DecSignedL":       (("intervals", _I["Signed"], "L", "mean"), "1f"),
    "DecSignedV":       (("intervals", _I["Signed"], "V", "mean"), "1f"),
    "DecSignedShare":   (("intervals", _I["Signed"], "V_share", "mean"), "2f"),
    "DecSignedShareLo": (("intervals", _I["Signed"], "V_share", "ci", 0), "2f"),
    "DecSignedShareHi": (("intervals", _I["Signed"], "V_share", "ci", 1), "2f"),
    "DecPermdL":        (("contrasts", "perm100-grpo100", "L", "mean"), "1f"),
    "DecPermdLLo":      (("contrasts", "perm100-grpo100", "L", "ci", 0), "1f"),
    "DecPermdLHi":      (("contrasts", "perm100-grpo100", "L", "ci", 1), "1f"),
    "DecPermdV":        (("contrasts", "perm100-grpo100", "V", "mean"), "1f"),
    "DecPermdVLo":      (("contrasts", "perm100-grpo100", "V", "ci", 0), "1f"),
    "DecPermdVHi":      (("contrasts", "perm100-grpo100", "V", "ci", 1), "1f"),
    "DecSigneddL":      (("contrasts", "signed100-grpo100", "L", "mean"), "1f"),
    "DecSigneddLLo":    (("contrasts", "signed100-grpo100", "L", "ci", 0), "1f"),
    "DecSigneddLHi":    (("contrasts", "signed100-grpo100", "L", "ci", 1), "1f"),
    "DecSigneddV":      (("contrasts", "signed100-grpo100", "V", "mean"), "1f"),
    "DecSigneddVLo":    (("contrasts", "signed100-grpo100", "V", "ci", 0), "1f"),
    "DecSigneddVHi":    (("contrasts", "signed100-grpo100", "V", "ci", 1), "1f"),
    "DecGrpoLgap":      (("intervals", _I["Grpo"], "L_ordering_gap"), "1f"),
}
# The exploratory robustness rows (robust()), one share and one prompt count
# per variant and interval.
_R = {"Drop": "drop_truncated", "Tok": "per_token", "Both": "drop_truncated_per_token"}
for _v, _rk in _R.items():
    for _arm, _ik in _I.items():
        MACROS[f"DecRob{_v}{_arm}"] = (("exploratory_robustness", _rk, _ik, "V_share"), "2f")
        MACROS[f"DecRob{_v}{_arm}Lo"] = (("exploratory_robustness", _rk, _ik, "ci", 0), "2f")
        MACROS[f"DecRob{_v}{_arm}Hi"] = (("exploratory_robustness", _rk, _ik, "ci", 1), "2f")
        MACROS[f"DecRob{_v}{_arm}N"] = (("exploratory_robustness", _rk, _ik, "n"), "0f")


def names() -> set[str]:
    """Every macro emit() can write; tests/test_paper.py reads this."""
    return set(MACROS) | {"DecNprompts", "DecRobustMin", "DecRobustMax"}


def _get(doc, path):
    for k in path:
        doc = doc[k]
    return doc


def emit(res: dict) -> dict[str, str]:
    """macro name -> rendered value; anything the result lacks stays \\PENDING."""
    out = {}
    for name, (path, how) in MACROS.items():
        try:
            out[name] = f"{float(_get(res, path)):.{how}}"
        except (KeyError, IndexError, TypeError):
            continue
    first = next(iter(res["intervals"].values()), None)
    if first:
        out["DecNprompts"] = str(first["n_prompts"])
    shares = [r["V_share"] for rows in res.get("exploratory_robustness", {}).values()
              for r in rows.values()]
    if shares:
        out["DecRobustMin"] = f"{min(shares):.2f}"
        out["DecRobustMax"] = f"{max(shares):.2f}"
    return out


def cmd_macros(a) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        arc = subprocess.run(["git", "archive", a.git_ref, a.path],
                             check=True, capture_output=True).stdout
        tar = Path(tmp) / "a.tar"
        tar.write_bytes(arc)
        with tarfile.open(tar) as t:
            t.extractall(tmp)
        out = Path(tmp) / a.path
        if (out / "samples.tar.gz").exists():
            with tarfile.open(out / "samples.tar.gz") as t:
                t.extractall(out)
        res = analyze(out, a.n_boot)
        if (out / "samples").is_dir():
            res["exploratory_robustness"] = robust(out)
    values = emit(res)
    lines = ["% generated by scripts/channel_decomp.py macros -- do not edit",
             f"% source: {a.git_ref}:{a.path}"]
    lines += [f"\\providecommand{{\\{k}}}{{{v}}}" for k, v in sorted(values.items())]
    Path(a.tex).parent.mkdir(parents=True, exist_ok=True)
    Path(a.tex).write_text("\n".join(lines) + "\n")
    missing = sorted(set(MACROS) - set(values))
    print(f"[decomp-macros] {len(values)} macros -> {a.tex}")
    if missing:
        print(f"[decomp-macros] not in the result, left red: {missing}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--model", required=True)
    s.add_argument("--label", required=True)
    s.add_argument("--base", required=True, help="tokenizer and chat template")
    s.add_argument("--data", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n-prompts", type=int, default=512)
    s.add_argument("--k", type=int, default=4)
    s.add_argument("--max-len", type=int, default=3072)
    s.add_argument("--seed", type=int, default=20261020)
    s.add_argument("--gpu-mem", type=float, default=0.85)
    e = sub.add_parser("entropy")
    e.add_argument("--model", required=True)
    e.add_argument("--label", required=True)
    e.add_argument("--samples", required=True)
    e.add_argument("--out", required=True)
    z = sub.add_parser("analyze")
    z.add_argument("--out", required=True)
    z.add_argument("--result", required=True)
    z.add_argument("--n-boot", type=int, default=10000)
    m = sub.add_parser("macros")
    m.add_argument("--git-ref", default="origin/paper")
    m.add_argument("--path", default="results/decomp")
    m.add_argument("--tex", default="results/numbers-decomp.tex")
    m.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args(argv)
    {"sample": cmd_sample, "entropy": cmd_entropy, "analyze": cmd_analyze,
     "macros": cmd_macros}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
