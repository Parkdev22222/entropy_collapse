"""Phase 1 검증 지표 — GPU 없이 단위 테스트 가능한 순수 계산부.

`scripts/phase1_validate.py`는 샘플링/포워드(I/O·GPU)만 담당하고, 지표 계산은
전부 여기로 위임한다. 게이트 G1 판정도 여기서 한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "spearman",
    "fisher_z_mean",
    "within_problem_spearman",
    "branch_recall_at_k",
    "GridResult",
    "select_kappa_gamma",
    "GateG1",
    "evaluate_gate_g1",
]


# ----------------------------------------------------------------------
# 상관
# ----------------------------------------------------------------------
def _rank(x: np.ndarray) -> np.ndarray:
    """평균 순위 (동점 처리)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # 동점 그룹에 평균 순위 부여
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman ρ. 표본 < 3개이거나 한쪽이 상수면 nan."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def fisher_z_mean(rhos: Iterable[float]) -> float:
    """상관계수들의 Fisher z 평균 (계획서 §3.3 주 지표 집계 방식)."""
    vals = [r for r in rhos if r is not None and math.isfinite(r)]
    if not vals:
        return float("nan")
    clipped = np.clip(np.asarray(vals, dtype=float), -0.999999, 0.999999)
    return float(np.tanh(np.arctanh(clipped).mean()))


def within_problem_spearman(
    problem_ids: Sequence,
    pred: Sequence[float],
    target: Sequence[float],
    min_prefixes: int = 5,
) -> tuple[float, dict]:
    """문제-내 Spearman을 문제별로 구하고 Fisher z로 평균.

    문제 간 난이도 차이가 상관을 부풀리는 것을 막기 위해 반드시 문제 내에서 잰다.

    Returns:
        (fisher_z_mean_rho, per_problem_rho_dict)
    """
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    pids = np.asarray(problem_ids)

    per_problem: dict = {}
    for pid in dict.fromkeys(pids.tolist()):
        sel = pids == pid
        if sel.sum() < min_prefixes:
            continue
        rho = spearman(pred[sel], target[sel])
        if math.isfinite(rho):
            per_problem[pid] = rho
    return fisher_z_mean(per_problem.values()), per_problem


# ----------------------------------------------------------------------
# 분기 토큰 회수율
# ----------------------------------------------------------------------
def branch_recall_at_k(
    scores: Sequence[float],
    is_branch: Sequence[bool],
    top_frac: float = 0.1,
    n_permutations: int = 2000,
    seed: int = 0,
) -> dict:
    """A_H 상위 top_frac 위치가 실제 분기 위치와 겹치는 비율 + 순열검정 p값.

    귀무가설: A_H 순위가 분기 여부와 무관 (랜덤 선택). 순열검정으로 p를 낸다.

    Returns:
        {"recall", "baseline", "lift", "p_value", "n_selected", "n_branch"}
    """
    s = np.asarray(scores, dtype=float)
    b = np.asarray(is_branch, dtype=bool)
    ok = np.isfinite(s)
    s, b = s[ok], b[ok]
    n = len(s)
    n_branch = int(b.sum())
    if n == 0 or n_branch == 0:
        return {"recall": float("nan"), "baseline": float("nan"), "lift": float("nan"),
                "p_value": float("nan"), "n_selected": 0, "n_branch": n_branch}

    n_sel = max(1, int(round(n * top_frac)))
    top_idx = np.argsort(-s, kind="mergesort")[:n_sel]
    recall = float(b[top_idx].sum() / n_branch)
    baseline = n_sel / n  # 랜덤 선택의 기대 recall

    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n)[:n_sel]
        if float(b[perm].sum() / n_branch) >= recall:
            hits += 1
    p_value = (hits + 1) / (n_permutations + 1)

    return {
        "recall": recall,
        "baseline": baseline,
        "lift": recall / baseline if baseline > 0 else float("nan"),
        "p_value": float(p_value),
        "n_selected": n_sel,
        "n_branch": n_branch,
    }


# ----------------------------------------------------------------------
# (κ, γ_H) 그리드
# ----------------------------------------------------------------------
@dataclass
class GridResult:
    kappa: int
    gamma_h: float
    rho: float
    per_problem: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return {"kappa": self.kappa, "gamma_h": self.gamma_h, "rho": self.rho,
                "n_problems": len(self.per_problem)}


def select_kappa_gamma(results: Sequence[GridResult], elbow_tol: float = 0.01) -> GridResult:
    """팔꿈치 규칙으로 (κ, γ_H) 선택 (계획서 §3.4).

    최고 ρ의 `elbow_tol` 이내인 후보 중 **가장 작은 κ**를 고른다 — κ가 크면 먼 헤드의
    과대추정 편향과 계산 비용이 함께 커지므로, 성능이 사실상 같다면 짧은 호라이즌이 낫다.
    동률이면 ρ가 높은 쪽.
    """
    finite = [r for r in results if math.isfinite(r.rho)]
    if not finite:
        raise ValueError("no finite rho in grid results")
    best = max(r.rho for r in finite)
    near = [r for r in finite if r.rho >= best - elbow_tol]
    min_kappa = min(r.kappa for r in near)
    return max([r for r in near if r.kappa == min_kappa], key=lambda r: r.rho)


# ----------------------------------------------------------------------
# 게이트 G1
# ----------------------------------------------------------------------
@dataclass
class GateG1:
    passed: bool
    rho: float
    rho_threshold: float
    recall_p: float
    recall_lift: float
    reasons: list[str]

    def summary(self) -> str:
        head = "PASS" if self.passed else "FAIL"
        lines = [
            f"[G1 {head}]",
            f"  within-problem rho = {self.rho:.4f} (threshold {self.rho_threshold})",
            f"  branch recall@10%: lift x{self.recall_lift:.2f}, p = {self.recall_p:.4g}",
        ]
        lines += [f"  - {r}" for r in self.reasons]
        return "\n".join(lines)


def evaluate_gate_g1(
    rho: float,
    recall_stats: dict,
    rho_threshold: float = 0.2,
    alpha: float = 0.05,
) -> GateG1:
    """계획서 §3 게이트 G1 판정.

    - within-problem ρ(H_togo, GT_future_entropy) >= 0.2
    - 분기 토큰 recall@10%가 랜덤 대비 유의하게 높음 (p < 0.05, lift > 1)
    """
    reasons: list[str] = []
    ok = True

    if not math.isfinite(rho):
        ok = False
        reasons.append("rho is not finite (표본 부족 또는 상수 예보)")
    elif rho < rho_threshold:
        ok = False
        reasons.append(f"rho {rho:.4f} < {rho_threshold}")

    p = recall_stats.get("p_value", float("nan"))
    lift = recall_stats.get("lift", float("nan"))
    if not math.isfinite(p) or not math.isfinite(lift):
        ok = False
        reasons.append("recall 통계가 비유한 (분기 라벨 없음?)")
    else:
        if p >= alpha:
            ok = False
            reasons.append(f"recall p-value {p:.4g} >= {alpha}")
        if lift <= 1.0:
            ok = False
            reasons.append(f"recall lift {lift:.2f} <= 1.0 (랜덤보다 낫지 않음)")

    if ok:
        reasons.append("두 조건 모두 충족 — Phase 2 진행 가능")
    else:
        reasons.append("계획서 §3 실패 절차: 워밍업 증량 → head_hidden 증가 → 순차형 MTP, 그래도 실패면 중단·보고")
    return GateG1(ok, rho, rho_threshold, p, lift, reasons)
