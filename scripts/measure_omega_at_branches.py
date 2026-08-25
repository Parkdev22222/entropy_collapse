#!/usr/bin/env python3
"""M1: is |Omega| small exactly where the rollouts actually fork?

``docs/STEERF_claims.md`` C4 predicts the local term and the visitation term
are anti-correlated -- that STEER's attenuation budget goes to rare-token draws
from confident distributions while the true forks, where the trajectory damage
is, get nothing.  Until now that has been a calculation over an analytic
distribution family (``docs/STEERF_verification.md`` §4, tier C).  This measures
it on the model's own rollouts.

The quantity.  At the first inner epoch the importance ratio is 1, so

    Omega = -(A / pi_old) * pi (1 - pi)(ln pi + H)  ->  A (1 - pi)(I_a - H)

with I_a = -ln pi_a the surprisal of the sampled token.  A is a GRPO outcome
advantage and is therefore *constant across the tokens of one response*: it
scales whole sequences and cannot order positions within one.  What orders
positions is the shape factor

    g = (1 - pi_a) * |I_a - H|

and that is what this script ranks.  Reported alongside is the per-token
entropy H, which C1 says should be HIGHER at forks -- so a single table shows
the dissociation the argument rests on, measured on the same positions.

A "true branch point" is the repository's own definition, imported rather than
reimplemented: position t of rollout i where some sibling sharing the prefix
[0, t) chose a different token at t (``steer_f.entropy_forecast``).

COST AND ISOLATION.  Runs on CPU only -- it never calls .cuda(), never imports
vllm, and asserts no CUDA tensor is created.  It does not read or write any
checkpoint, dataset or state file a training run touches.  Its footprint is one
model copy in RAM (about 6 GB in fp32 for a 1.5B model) plus one [1, T, V]
logit tensor at a time.  On a box that is training, that RAM and those cores
are shared -- decide with `--groups` whether you want to spend them.

    # validate the pipeline with no weights and no network (seconds)
    python3 scripts/measure_omega_at_branches.py --self-test

    # the real measurement, on a box that already has the weights cached
    python3 scripts/measure_omega_at_branches.py \
        --model Qwen/Qwen2.5-Math-1.5B \
        --data datasets/DAPO-Math-17k.parquet \
        --groups 16 --rollouts 8 --max-new 384 --out docs/omega_at_branches.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # belt and braces: no GPU

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

torch.set_grad_enabled(False)


# ------------------------------------------------------------------ metric
def per_token_stats(logits: torch.Tensor, taken: torch.Tensor,
                    chunk: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """Entropy H and surprisal I of the taken token, position by position.

    ``logits`` is [T, V] already aligned so that row t scores ``taken[t]``.
    Chunked over positions because [T, V] in fp32 is the memory peak here.
    """
    n = logits.shape[0]
    H = torch.empty(n, dtype=torch.float64)
    I = torch.empty(n, dtype=torch.float64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        lg = logits[s:e].float()
        lse = torch.logsumexp(lg, dim=-1)
        p = torch.softmax(lg, dim=-1)
        H[s:e] = (lse - (p * lg).sum(dim=-1)).double()
        I[s:e] = (lse - lg.gather(1, taken[s:e].view(-1, 1)).squeeze(1)).double()
    return H, I


def shape_factor(H: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
    """g = (1 - pi_a) |I_a - H| -- the part of |Omega| that orders positions."""
    pi = torch.exp(-I)
    return (1.0 - pi) * (I - H).abs()


def average_ranks(x: torch.Tensor) -> torch.Tensor:
    """Tie-corrected 0-based ranks, normalised to [0, 1]."""
    n = x.numel()
    if n < 2:
        return torch.zeros(n, dtype=torch.float64)
    order = torch.argsort(x)
    pos = torch.empty(n, dtype=torch.float64)
    pos[order] = torch.arange(n, dtype=torch.float64)
    vals, inv = torch.unique(x, return_inverse=True)
    tot = torch.zeros(vals.numel(), dtype=torch.float64).scatter_add_(0, inv, pos)
    cnt = torch.zeros(vals.numel(), dtype=torch.float64).scatter_add_(
        0, inv, torch.ones(n, dtype=torch.float64))
    return (tot / cnt)[inv] / (n - 1)


def branch_mask(responses: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """True branch points, by the repository's own definition."""
    from steer_f.entropy_forecast import first_divergence

    g, T = responses.shape
    div = first_divergence(responses, mask)                      # [g, g]
    positions = torch.arange(T).view(1, 1, T)
    hits = (div.unsqueeze(-1) == positions).any(dim=1)           # [g, T]
    return hits & mask.bool()


# ------------------------------------------------------------------ stats
def welch(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """Difference of means, its standard error, and t."""
    ma, mb = st.mean(a), st.mean(b)
    va = st.variance(a) / len(a) if len(a) > 1 else 0.0
    vb = st.variance(b) / len(b) if len(b) > 1 else 0.0
    se = math.sqrt(va + vb)
    d = ma - mb
    return d, se, (d / se if se > 0 else float("nan"))


# ------------------------------------------------------------------ driver
def analyse(groups: list[dict]) -> dict:
    """Pool per-group results into the numbers the claim turns on."""
    rank_b, rank_n, ent_b, ent_n, g_b, g_n = [], [], [], [], [], []
    per_group, per_group_ent = [], []
    for grp in groups:
        rb = [v for v in grp["rank_branch"]]
        rn = [v for v in grp["rank_other"]]
        if not rb or not rn:
            continue
        rank_b += rb
        rank_n += rn
        ent_b += grp["ent_branch"]
        ent_n += grp["ent_other"]
        g_b += grp["g_branch"]
        g_n += grp["g_other"]
        per_group.append(st.mean(rb) - st.mean(rn))
        per_group_ent.append(st.mean(grp["ent_branch"]) - st.mean(grp["ent_other"]))

    d_rank, se_rank, t_rank = welch(rank_b, rank_n)
    d_ent, se_ent, t_ent = welch(ent_b, ent_n)
    # Group means are the unit the design actually randomises over; the pooled
    # token-level test treats correlated tokens as independent and overstates n.
    def by_group(v: list[float]) -> tuple[float, float]:
        if len(v) < 2:
            return float("nan"), float("nan")
        return st.mean(v), st.stdev(v) / math.sqrt(len(v))

    gm, gse = by_group(per_group)
    em, ese = by_group(per_group_ent)
    return {
        "n_groups": len(per_group),
        "n_branch_tokens": len(rank_b),
        "n_other_tokens": len(rank_n),
        "rank_branch": st.mean(rank_b), "rank_other": st.mean(rank_n),
        "rank_diff": d_rank, "rank_se": se_rank, "rank_t": t_rank,
        "rank_diff_bygroup": gm, "rank_se_bygroup": gse,
        "rank_t_bygroup": gm / gse if gse and gse == gse and gse > 0 else float("nan"),
        "entropy_branch": st.mean(ent_b), "entropy_other": st.mean(ent_n),
        "entropy_diff": d_ent, "entropy_se": se_ent, "entropy_t": t_ent,
        "entropy_diff_bygroup": em, "entropy_se_bygroup": ese,
        "entropy_t_bygroup": em / ese if ese and ese == ese and ese > 0 else float("nan"),
        "g_branch": st.mean(g_b), "g_other": st.mean(g_n),
    }


def report(res: dict) -> None:
    print("\n" + "=" * 74)
    print("M1 — 진짜 분기점에서 |Omega| 의 순위")
    print("=" * 74)
    print(f"  그룹 {res['n_groups']}개 | 분기 토큰 {res['n_branch_tokens']} | "
          f"비분기 토큰 {res['n_other_tokens']}\n")
    print(f"  {'':22}{'분기점':>10}{'비분기':>10}{'차이':>10}{'t':>8}")
    print(f"  {'|Omega| 정규화 순위':22}{res['rank_branch']:>10.4f}{res['rank_other']:>10.4f}"
          f"{res['rank_diff']:>10.4f}{res['rank_t_bygroup']:>8.1f}")
    print(f"  {'per-token 엔트로피 H':22}{res['entropy_branch']:>10.4f}{res['entropy_other']:>10.4f}"
          f"{res['entropy_diff']:>10.4f}{res['entropy_t_bygroup']:>8.1f}")
    print("\n  t 는 그룹 평균 기준. 토큰 단위로 풀링하면 상관된 토큰을 독립으로 세어 n 을 부풀린다")
    print(f"    순위차   {res['rank_diff_bygroup']:+.4f} ± {res['rank_se_bygroup']:.4f}"
          f"   (토큰 풀링 t={res['rank_t']:+.1f} — 참고용, 신뢰 금지)")
    print(f"    엔트로피차 {res['entropy_diff_bygroup']:+.4f} ± {res['entropy_se_bygroup']:.4f}"
          f"   (토큰 풀링 t={res['entropy_t']:+.1f} — 참고용, 신뢰 금지)")
    print("\n  판정:")
    t = res["rank_t_bygroup"]
    if t == t and t < -2:
        print("    C4 지지 — 분기점의 |Omega| 순위가 유의하게 낮다 (감쇠를 덜 받는다)")
    elif t == t and t > 2:
        print("    C4 기각 — 분기점의 |Omega| 순위가 오히려 높다. C2-C4 논증 재검토 필요")
    else:
        print("    판정 불가 — 차이가 유의하지 않다. 표본(--groups)을 늘리거나 C4를 약하게 서술")
    if res["entropy_t_bygroup"] > 2:
        print("    C1 재확인 — 같은 위치에서 엔트로피는 더 높다 (분기점 = 고엔트로피 자리)")


# ------------------------------------------------------------------ models
def load_real(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, trust_remote_code=True).eval()
    return tok, model


def load_tiny():
    """A randomly initialised Qwen2 -- no weights, no network. Pipeline only."""
    from transformers import AutoModelForCausalLM, Qwen2Config
    cfg = Qwen2Config(vocab_size=512, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=512)
    torch.manual_seed(0)
    return None, AutoModelForCausalLM.from_config(cfg).eval()


def prompts_from_parquet(path: str, n: int, tok, seed: int) -> list[list[int]]:
    import random
    import pandas as pd
    df = pd.read_parquet(path)
    col = next((c for c in ("prompt", "question", "problem", "input") if c in df.columns), None)
    if col is None:
        raise SystemExit(f"프롬프트 열을 찾지 못했습니다: {list(df.columns)}")
    rng = random.Random(seed)
    rows = rng.sample(range(len(df)), min(n, len(df)))
    out = []
    for i in rows:
        v = df[col].iloc[i]
        if isinstance(v, (list, tuple)) or hasattr(v, "tolist"):
            v = list(v)
            v = v[-1]["content"] if v and isinstance(v[-1], dict) else str(v)
        out.append(tok(str(v), return_tensors="pt").input_ids[0].tolist())
    return out


def run_group(model, prompt_ids: list[int], g: int, max_new: int,
              temperature: float, seed: int) -> dict | None:
    torch.manual_seed(seed)
    prompt = torch.tensor(prompt_ids).view(1, -1)
    gen = model.generate(prompt.expand(g, -1), do_sample=True,
                         temperature=temperature, top_p=1.0, top_k=0,
                         max_new_tokens=max_new, min_new_tokens=8,
                         pad_token_id=getattr(model.config, "eos_token_id", 0) or 0)
    P = prompt.shape[1]
    resp = gen[:, P:]                                    # [g, R]
    R = resp.shape[1]
    if R < 2:
        return None

    eos = getattr(model.config, "eos_token_id", None)
    mask = torch.ones_like(resp)
    if eos is not None:
        for i in range(g):
            hit = (resp[i] == eos).nonzero()
            if hit.numel():
                mask[i, int(hit[0]) + 1:] = 0
    if int(mask.sum()) < g * 2:
        return None

    Hs, Gs = [], []
    for i in range(g):
        logits = model(gen[i:i + 1]).logits[0]            # [P+R, V]
        H, I = per_token_stats(logits[P - 1:P - 1 + R], resp[i])
        Hs.append(H)
        Gs.append(shape_factor(H, I))
    Hm = torch.stack(Hs)
    Gm = torch.stack(Gs)

    br = branch_mask(resp, mask)
    valid = mask.bool()
    if not (br & valid).any() or not ((~br) & valid).any():
        return None

    # Rank within the group: a micro-batch is exactly one prompt's rollouts
    # (ppo_micro_batch_size_per_gpu 8 = rollout.n 8), which is the population
    # STEER's mapping actually ranks over.
    flat_g = Gm[valid]
    ranks = torch.empty_like(Gm, dtype=torch.float64)
    ranks[valid] = average_ranks(flat_g)
    b, o = br & valid, (~br) & valid
    return {
        "rank_branch": ranks[b].tolist(), "rank_other": ranks[o].tolist(),
        "ent_branch": Hm[b].tolist(), "ent_other": Hm[o].tolist(),
        "g_branch": Gm[b].tolist(), "g_other": Gm[o].tolist(),
        "n_branch": int(b.sum()), "n_valid": int(valid.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    ap.add_argument("--data", default=os.path.join(REPO, "datasets", "DAPO-Math-17k.parquet"))
    ap.add_argument("--groups", type=int, default=16)
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", action="store_true",
                    help="가중치도 네트워크도 없이 파이프라인만 검증")
    args = ap.parse_args()

    if args.self_test:
        print("[self-test] 랜덤 초기화 tiny Qwen2 — 숫자는 무의미, 파이프라인만 검증")
        tok, model = load_tiny()
        prompts = [[7, 11, 23, 5] for _ in range(max(4, args.groups))]
        max_new = min(args.max_new, 48)
    else:
        print(f"[m1] CPU 전용. 모델 로드: {args.model}")
        tok, model = load_real(args.model)
        prompts = prompts_from_parquet(args.data, args.groups, tok, args.seed)
        max_new = args.max_new

    assert not torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") == "", \
        "이 스크립트는 CPU 전용입니다"

    out = []
    for gi, pid in enumerate(prompts):
        r = run_group(model, pid, args.rollouts, max_new, args.temperature, args.seed + gi)
        if r:
            out.append(r)
        print(f"  group {gi+1}/{len(prompts)}  "
              f"{'분기 %d/%d' % (r['n_branch'], r['n_valid']) if r else '건너뜀(분기 없음/너무 짧음)'}",
              flush=True)
    if not out:
        print("사용 가능한 그룹이 없습니다 — --max-new 를 늘리거나 --groups 를 늘리세요")
        return 1

    res = analyse(out)
    report(res)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"config": vars(args), "result": res}, f, indent=1)
        print(f"\n  저장: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
