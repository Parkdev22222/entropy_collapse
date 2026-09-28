#!/usr/bin/env python3
"""Phase 1: MTP 예보의 Monte Carlo 검증 + (κ, γ_H) 결정 + 게이트 G1 판정.

프로토콜 (계획서 §3.3, ExTra Appendix C를 엔트로피용으로 변형):
  1. 모델이 정답·오답을 모두 내는 문제 20~30개 선정 (난이도 4~5).
  2. 문제당 temp 1.0으로 전체 궤적 32개 샘플링.
  3. 각 궤적을 추론 스텝의 20/40/60/80%에서 절단 → prefix 풀 (~600개).
  4. 각 prefix에서 K_mc=16개 continuation 샘플링 (temp 0.7, top-p 0.9) → 실측치.
  5. 같은 prefix에서 MTP 예보 H_togo^κ 계산 (κ=1..8, γ_H 그리드).
  6. 문제-내 Spearman ρ (Fisher z 평균), 분기 토큰 recall@10%.

**측정 호라이즌 주의**: GT_future_entropy를 continuation 전체 길이에 대해 합하면
길이와 교락된다 (긴 continuation = 큰 합). 기본값은 `--gt-horizon` 토큰의 고정
창에서 합하고, 길이 정규화 평균도 함께 기록해 둘 다 보고한다.

사용 예:
    python scripts/phase1_validate.py \\
        --model Qwen/Qwen2.5-Math-7B \\
        --heads artifacts/mtp_heads_qwen7b.pt \\
        --problems datasets/math500.parquet \\
        --workdir artifacts/phase1 \\
        --n-problems 24 --n-trajectories 32 --n-mc 16
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from dataclasses import asdict, dataclass
from typing import Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts._common import (  # noqa: E402
    GenerationBackend,
    apply_chat_template,
    get_last_hidden_states,
    get_lm_head,
    load_policy,
    load_prompts_from_parquet,
)
from steer_f.entropy_forecast import HeadCalibration, entropy_advantage, h_togo  # noqa: E402
from steer_f.mtp_heads import MTPHeads, entropy_from_logits  # noqa: E402
from steer_f.validation import (  # noqa: E402
    GridResult,
    branch_recall_at_k,
    evaluate_gate_g1,
    select_kappa_gamma,
    within_problem_spearman,
)

TRUNC_FRACS = (0.2, 0.4, 0.6, 0.8)


# ----------------------------------------------------------------------
# 정답 판정 (문제 선정용 — 엄밀한 채점기가 아니라 혼합 난이도 필터)
# ----------------------------------------------------------------------
_BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_answer(text: str) -> Optional[str]:
    matches = _BOXED.findall(text)
    return matches[-1].strip() if matches else None


def answers_match(pred: Optional[str], gt: Optional[str]) -> bool:
    if pred is None or gt is None:
        return False
    norm = lambda s: re.sub(r"[\s,$]|\\left|\\right|\\!|\\,", "", str(s)).rstrip(".")
    return norm(pred) == norm(gt)


# ----------------------------------------------------------------------
# 헤드 로딩
# ----------------------------------------------------------------------
def load_heads(path: str, device: str):
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    heads = MTPHeads(
        hidden_size=cfg["hidden_size"],
        num_heads=cfg["num_heads"],
        head_hidden=cfg["head_hidden"],
        vocab_size=cfg.get("vocab_size"),
        residual=cfg.get("residual", True),
    )
    heads.load_state_dict(ckpt["state_dict"])
    return heads.to(device).eval(), cfg


# ----------------------------------------------------------------------
# 단계 1-2: 문제 선정 + 전체 궤적
# ----------------------------------------------------------------------
def stage_trajectories(args, tok, backend, workdir: pathlib.Path) -> list[dict]:
    cache = workdir / "trajectories.json"
    if cache.exists() and not args.force:
        print(f"[stage1] cache hit {cache}")
        return json.loads(cache.read_text())

    problems = load_prompts_from_parquet(args.problems, limit=args.problem_pool)
    texts = [apply_chat_template(tok, p["messages"]) for p in problems]
    print(f"[stage1] {len(problems)} candidate problems, sampling {args.n_trajectories} trajectories each")

    outs = backend.generate(
        texts,
        n=args.n_trajectories,
        temperature=1.0,
        top_p=1.0,
        max_tokens=args.max_response_length,
        seed=args.seed,
    )

    kept = []
    for p, prompt_text, completions in zip(problems, texts, outs):
        correct = [answers_match(extract_answer(c), p["ground_truth"]) for c in completions]
        rate = sum(correct) / max(len(correct), 1)
        # 정답·오답을 모두 내는 문제만 (계획서 §3.3 step 1)
        if not (args.min_pass_rate <= rate <= args.max_pass_rate):
            continue
        kept.append(
            {
                "problem_id": p["problem_id"],
                "prompt": prompt_text,
                "ground_truth": p["ground_truth"],
                "pass_rate": rate,
                "trajectories": completions,
                "correct": correct,
            }
        )
        if len(kept) >= args.n_problems:
            break

    print(f"[stage1] kept {len(kept)} mixed-outcome problems")
    if len(kept) < args.n_problems:
        print(f"[stage1] WARNING: 목표 {args.n_problems}개에 미달 — --problem-pool 을 늘릴 것")
    cache.write_text(json.dumps(kept))
    return kept


# ----------------------------------------------------------------------
# 단계 3: prefix 풀
# ----------------------------------------------------------------------
def split_steps(text: str) -> list[str]:
    """추론 '스텝' 근사: 빈 줄이 아닌 줄 단위."""
    return [ln for ln in text.split("\n") if ln.strip()]


def stage_prefixes(problems: list[dict], workdir: pathlib.Path, force: bool) -> list[dict]:
    cache = workdir / "prefixes.json"
    if cache.exists() and not force:
        print(f"[stage2] cache hit {cache}")
        return json.loads(cache.read_text())

    prefixes = []
    for prob in problems:
        for traj_idx, traj in enumerate(prob["trajectories"]):
            steps = split_steps(traj)
            if len(steps) < len(TRUNC_FRACS):
                continue
            for frac in TRUNC_FRACS:
                n_keep = max(1, int(round(len(steps) * frac)))
                if n_keep >= len(steps):
                    continue
                prefixes.append(
                    {
                        "problem_id": prob["problem_id"],
                        "prompt": prob["prompt"],
                        "traj_idx": traj_idx,
                        "frac": frac,
                        "prefix": "\n".join(steps[:n_keep]) + "\n",
                    }
                )
    print(f"[stage2] {len(prefixes)} prefixes")
    cache.write_text(json.dumps(prefixes))
    return prefixes


# ----------------------------------------------------------------------
# 단계 4: MC 실측치
# ----------------------------------------------------------------------
def stage_ground_truth(args, model, tok, backend, prefixes, workdir: pathlib.Path) -> list[dict]:
    import torch

    cache = workdir / "ground_truth.json"
    if cache.exists() and not args.force:
        print(f"[stage3] cache hit {cache}")
        return json.loads(cache.read_text())

    full_prompts = [p["prompt"] + p["prefix"] for p in prefixes]
    print(f"[stage3] sampling {args.n_mc} continuations for {len(prefixes)} prefixes")
    conts = backend.generate(
        full_prompts,
        n=args.n_mc,
        temperature=args.mc_temperature,
        top_p=args.mc_top_p,
        max_tokens=args.mc_max_tokens,
        seed=args.seed,
    )

    embedder = None
    if args.embed_model:
        try:
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer(args.embed_model)
        except Exception as exc:
            print(f"[stage3] 임베딩 모델 로드 실패 ({exc}) — GT_branch_div 생략")

    results = []
    for i, (pref, completions) in enumerate(zip(prefixes, conts)):
        # 실측 엔트로피: continuation을 teacher-forcing해 위치별 토큰 엔트로피를 잰다.
        ent_sums, ent_means = [], []
        ctx_ids = tok(full_prompts[i], add_special_tokens=False)["input_ids"]
        for c in completions:
            c_ids = tok(c, add_special_tokens=False)["input_ids"][: args.gt_horizon]
            if not c_ids:
                continue
            ids = torch.tensor([ctx_ids + c_ids], device=model.device)
            attn = torch.ones_like(ids)
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=attn, use_cache=False)
            # 위치 (len(ctx)-1 .. end-1)의 로짓이 continuation 토큰들을 예측한다.
            logits = out.logits[0, len(ctx_ids) - 1 : -1, :]
            ent = entropy_from_logits(logits)
            ent_sums.append(float(ent.sum()))
            ent_means.append(float(ent.mean()))

        row = {
            "problem_id": pref["problem_id"],
            "traj_idx": pref["traj_idx"],
            "frac": pref["frac"],
            "gt_future_entropy": float(sum(ent_sums) / len(ent_sums)) if ent_sums else float("nan"),
            "gt_future_entropy_mean": float(sum(ent_means) / len(ent_means)) if ent_means else float("nan"),
            "n_continuations": len(ent_sums),
        }

        if embedder is not None and len(completions) > 1:
            import numpy as np

            emb = embedder.encode(completions, normalize_embeddings=True)
            centroid = emb.mean(axis=0)
            row["gt_branch_div"] = float(np.mean(((emb - centroid) ** 2).sum(axis=1)))

        results.append(row)
        if (i + 1) % 50 == 0:
            print(f"[stage3] {i + 1}/{len(prefixes)}")

    cache.write_text(json.dumps(results))
    return results


# ----------------------------------------------------------------------
# 단계 5: MTP 예보 (헤드별 원시 엔트로피를 저장 → 그리드는 사후 계산)
# ----------------------------------------------------------------------
def stage_forecast(args, model, tok, heads, prefixes, workdir: pathlib.Path) -> list[list[float]]:
    import torch

    cache = workdir / "forecast.json"
    if cache.exists() and not args.force:
        print(f"[stage4] cache hit {cache}")
        return json.loads(cache.read_text())

    lm_head = get_lm_head(model)
    out = []
    for i, pref in enumerate(prefixes):
        ids = tok(pref["prompt"] + pref["prefix"], add_special_tokens=False)["input_ids"]
        ids_t = torch.tensor([ids], device=model.device)
        attn = torch.ones_like(ids_t)
        with torch.no_grad():
            hidden, _ = get_last_hidden_states(model, ids_t, attn)
            # prefix 마지막 위치의 히든에서 +1..+K 예보
            ent = heads.forward_entropy(hidden[:, -1:, :].to(next(heads.parameters()).dtype), lm_head)
        out.append(ent[:, 0, 0].float().cpu().tolist())
        if (i + 1) % 100 == 0:
            print(f"[stage4] {i + 1}/{len(prefixes)}")

    cache.write_text(json.dumps(out))
    return out


# ----------------------------------------------------------------------
# 단계 6: 분기 토큰 recall (sibling 궤적에서 실제 분기 위치 라벨링)
# ----------------------------------------------------------------------
def stage_branch_recall(args, model, tok, heads, problems, workdir: pathlib.Path, kappa: int, gamma_h: float) -> dict:
    import torch

    cache = workdir / f"branch_recall_k{kappa}_g{gamma_h}.json"
    if cache.exists() and not args.force:
        print(f"[stage5] cache hit {cache}")
        return json.loads(cache.read_text())

    lm_head = get_lm_head(model)
    scores, labels = [], []

    for prob in problems:
        group = prob["trajectories"][: args.branch_group_size]
        if len(group) < 2:
            continue
        prompt_ids = tok(prob["prompt"], add_special_tokens=False)["input_ids"]
        resp_ids = [tok(t, add_special_tokens=False)["input_ids"][: args.branch_max_len] for t in group]
        t_max = max(len(r) for r in resp_ids)
        if t_max < 2:
            continue

        pad = tok.pad_token_id or 0
        g = len(resp_ids)
        resp = torch.full((g, t_max), pad, dtype=torch.long)
        mask = torch.zeros((g, t_max), dtype=torch.long)
        h_all = torch.zeros((g, t_max), dtype=torch.float32)

        for i, r in enumerate(resp_ids):
            resp[i, : len(r)] = torch.tensor(r, dtype=torch.long)
            mask[i, : len(r)] = 1
            ids = torch.tensor([prompt_ids + r], device=model.device)
            with torch.no_grad():
                hidden, _ = get_last_hidden_states(model, ids, torch.ones_like(ids))
                hid = hidden[:, len(prompt_ids) - 1 : -1, :].to(next(heads.parameters()).dtype)
                head_ent = heads.forward_entropy(hid, lm_head)  # (K, 1, len(r))
            h_all[i, : len(r)] = h_togo(head_ent, kappa=kappa, gamma_h=gamma_h)[0].cpu()

        a_h, _ = entropy_advantage(
            h_all, response_ids=resp, group_size=g, response_mask=mask.float(), baseline="sibling"
        )

        # 실제 분기 위치: 같은 prefix를 공유하던 sibling들의 다음 토큰이 갈라지는 자리
        from steer_f.entropy_forecast import sibling_mask

        sib = sibling_mask(resp, group_size=g, response_mask=mask.float())[0]  # (G, G, T)
        same_next = resp.unsqueeze(1) == resp.unsqueeze(0)  # (G, G, T)
        for i in range(g):
            for t in range(len(resp_ids[i])):
                partners = sib[i, :, t].clone()
                partners[i] = False
                if not partners.any():
                    continue  # sibling이 자기 자신뿐 → A_H = 0, 판정 대상 아님
                scores.append(float(a_h[i, t]))
                labels.append(bool((~same_next[i, partners, t]).any()))

    stats = branch_recall_at_k(scores, labels, top_frac=args.branch_top_frac, seed=args.seed)
    stats["n_positions"] = len(scores)
    cache.write_text(json.dumps(stats))
    return stats


# ----------------------------------------------------------------------
@dataclass
class Phase1Report:
    kappa: int
    gamma_h: float
    rho: float
    rho_gt_mean: float
    grid: list[dict]
    recall: dict
    calibration: dict
    gate_passed: bool
    gate_summary: str
    n_prefixes: int
    n_problems: int


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--heads", required=True, help="phase1_warmup_heads.py train 산출물")
    ap.add_argument("--problems", required=True, help="math500.parquet 등")
    ap.add_argument("--workdir", default="artifacts/phase1")
    ap.add_argument("--report", default="docs/phase1_report.md")

    ap.add_argument("--n-problems", type=int, default=24)
    ap.add_argument("--problem-pool", type=int, default=120)
    ap.add_argument("--n-trajectories", type=int, default=32)
    ap.add_argument("--n-mc", type=int, default=16)
    ap.add_argument("--min-pass-rate", type=float, default=0.15)
    ap.add_argument("--max-pass-rate", type=float, default=0.85)
    ap.add_argument("--max-response-length", type=int, default=3072)
    ap.add_argument("--mc-temperature", type=float, default=0.7)
    ap.add_argument("--mc-top-p", type=float, default=0.9)
    ap.add_argument("--mc-max-tokens", type=int, default=512)
    ap.add_argument("--gt-horizon", type=int, default=64, help="실측 엔트로피 합의 고정 창 (길이 교락 방지)")

    ap.add_argument("--kappa-grid", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--gamma-grid", default="0.7,0.85,1.0")
    ap.add_argument("--elbow-tol", type=float, default=0.01)
    ap.add_argument("--calibrate", action="store_true", help="헤드별 아핀 보정을 적합해 적용")

    ap.add_argument("--branch-group-size", type=int, default=8)
    ap.add_argument("--branch-max-len", type=int, default=512)
    ap.add_argument("--branch-top-frac", type=float, default=0.1)

    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--no-vllm", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="캐시 무시하고 재계산")
    args = ap.parse_args()

    import torch

    workdir = pathlib.Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    model, tok = load_policy(args.model, dtype=args.dtype, device=args.device)
    heads, _ = load_heads(args.heads, args.device)
    backend = GenerationBackend(args.model, prefer_vllm=not args.no_vllm, tensor_parallel_size=args.tp)

    problems = stage_trajectories(args, tok, backend, workdir)
    prefixes = stage_prefixes(problems, workdir, args.force)
    gt = stage_ground_truth(args, model, tok, backend, prefixes, workdir)
    fc = stage_forecast(args, model, tok, heads, prefixes, workdir)

    head_ent = torch.tensor(fc).T.unsqueeze(-1)  # (K, N, 1)
    pids = [r["problem_id"] for r in gt]
    targets = [r["gt_future_entropy"] for r in gt]

    calib = None
    if args.calibrate:
        # 헤드별로 실측 '평균' 엔트로피에 맞춘다 (합은 호라이즌 길이에 비례하므로 부적절).
        tgt = torch.tensor([r["gt_future_entropy_mean"] for r in gt]).unsqueeze(0).expand(head_ent.shape[0], -1)
        calib = HeadCalibration.fit(head_ent[:, :, 0], tgt)
        print(f"[calib] scale={['%.3f' % s for s in calib.scale]} bias={['%.3f' % b for b in calib.bias]}")

    grid: list[GridResult] = []
    for kappa in [int(x) for x in args.kappa_grid.split(",")]:
        if kappa > head_ent.shape[0]:
            continue
        for gamma in [float(x) for x in args.gamma_grid.split(",")]:
            vals = h_togo(head_ent, kappa=kappa, gamma_h=gamma, calib=calib)[:, 0].tolist()
            rho, per = within_problem_spearman(pids, vals, targets)
            grid.append(GridResult(kappa=kappa, gamma_h=gamma, rho=rho, per_problem=per))
            print(f"[grid] kappa={kappa} gamma_h={gamma} rho={rho:.4f} ({len(per)} problems)")

    best = select_kappa_gamma(grid, elbow_tol=args.elbow_tol)
    print(f"[grid] selected kappa={best.kappa} gamma_h={best.gamma_h} rho={best.rho:.4f}")

    best_vals = h_togo(head_ent, kappa=best.kappa, gamma_h=best.gamma_h, calib=calib)[:, 0].tolist()
    rho_mean, _ = within_problem_spearman(pids, best_vals, [r["gt_future_entropy_mean"] for r in gt])

    recall = stage_branch_recall(args, model, tok, heads, problems, workdir, best.kappa, best.gamma_h)
    print(f"[recall] {recall}")

    gate = evaluate_gate_g1(best.rho, recall)
    print(gate.summary())

    report = Phase1Report(
        kappa=best.kappa,
        gamma_h=best.gamma_h,
        rho=best.rho,
        rho_gt_mean=rho_mean,
        grid=[g.as_row() for g in grid],
        recall=recall,
        calibration=calib.to_dict() if calib else {},
        gate_passed=gate.passed,
        gate_summary=gate.summary(),
        n_prefixes=len(prefixes),
        n_problems=len(problems),
    )
    (workdir / "phase1_result.json").write_text(json.dumps(asdict(report), indent=2))
    write_report(report, pathlib.Path(args.report), args)
    _maybe_plot(grid, workdir)
    return 0 if gate.passed else 2


def write_report(rep: Phase1Report, path: pathlib.Path, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 1 리포트 — MTP 예보 검증 및 (κ, γ_H) 결정",
        "",
        f"- 모델: `{args.model}`",
        f"- 헤드: `{args.heads}`",
        f"- 문제 {rep.n_problems}개, prefix {rep.n_prefixes}개, 문제당 궤적 {args.n_trajectories}, prefix당 MC {args.n_mc}",
        f"- 실측 호라이즌: {args.gt_horizon} 토큰 (길이 교락 방지용 고정 창)",
        "",
        "## 결정",
        "",
        f"| κ | γ_H | within-problem ρ |",
        f"|---|---|---|",
        f"| **{rep.kappa}** | **{rep.gamma_h}** | **{rep.rho:.4f}** |",
        "",
        f"보조: 길이정규화 실측(mean) 기준 ρ = {rep.rho_gt_mean:.4f}",
        "",
        "## (κ, γ_H) 그리드",
        "",
        "| κ | γ_H | ρ | n_problems |",
        "|---|---|---|---|",
    ]
    for row in rep.grid:
        lines.append(f"| {row['kappa']} | {row['gamma_h']} | {row['rho']:.4f} | {row['n_problems']} |")

    lines += [
        "",
        "## 분기 토큰 recall@10%",
        "",
        "| recall | 랜덤 baseline | lift | p-value | 위치 수 |",
        "|---|---|---|---|---|",
        f"| {rep.recall.get('recall', float('nan')):.4f} | {rep.recall.get('baseline', float('nan')):.4f} "
        f"| {rep.recall.get('lift', float('nan')):.2f}× | {rep.recall.get('p_value', float('nan')):.4g} "
        f"| {rep.recall.get('n_positions', 0)} |",
        "",
        "## 캘리브레이션",
        "",
        "```json",
        json.dumps(rep.calibration, indent=2),
        "```",
        "",
        "## 게이트 G1",
        "",
        "```",
        rep.gate_summary,
        "```",
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"[report] wrote {path}")


def _maybe_plot(grid, workdir: pathlib.Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[plot] matplotlib 없음 — 곡선 생략")
        return

    gammas = sorted({g.gamma_h for g in grid})
    fig, ax = plt.subplots(figsize=(6, 4))
    for gamma in gammas:
        pts = sorted([(g.kappa, g.rho) for g in grid if g.gamma_h == gamma])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"γ_H={gamma}")
    ax.axhline(0.2, ls="--", c="gray", label="G1 threshold 0.2")
    ax.set_xlabel("κ (forecast horizon)")
    ax.set_ylabel("within-problem Spearman ρ")
    ax.legend()
    fig.tight_layout()
    out = workdir / "kappa_gamma_curve.png"
    fig.savefig(out, dpi=150)
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
