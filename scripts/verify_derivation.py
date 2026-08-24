#!/usr/bin/env python3
"""Numerically verify STEER-F's derivation before spending GPU-days on it.

Every proposition in ``docs/STEERF_derivation.md`` that can be checked without
training is checked here.  Two tiers:

  TIER A  pure mathematics -- softmax entropy derivatives, the two-channel
          identity, baseline invariance.  No data, no model, runs in a second.
          A failure here means the derivation is wrong, full stop.

  TIER B  the model's own warmup rollouts (``docs/phase1_results_*.json``).
          Checks the one information-theoretic prediction that involves the
          MTP heads.  A failure here means the forecast does not behave the
          way the derivation says it must.

What is NOT here: anything that needs a training run, and the one check that
needs a forward pass over stored rollouts (M1 of ``docs/STEERF_claims.md`` --
the |Omega| rank at true branch points).  Those are named at the end.

Usage:
    python3 scripts/verify_derivation.py            # both tiers
    python3 scripts/verify_derivation.py --tier a   # maths only, no data
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics as st
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PHASE1 = os.path.join(REPO, "docs", "phase1_results_Qwen2.5-Math-1.5B-paper.json")

# Finite differences on a smooth function of O(1) magnitude: the central first
# difference is O(h^2 + eps/h), the mixed second is O(h^2 + eps/h^2).  These
# tolerances sit an order of magnitude above the floor each one reaches.
TOL_D1 = 1e-8
TOL_D2 = 1e-6
TOL_EXACT = 1e-12


class Report:
    """Collects pass/fail so one failure does not hide the rest."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, bool, str]] = []

    def check(self, prop: str, what: str, passed: bool, detail: str) -> None:
        self.rows.append((prop, what, passed, detail))
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {what}\n         {detail}")

    def summary(self) -> int:
        bad = [r for r in self.rows if not r[2]]
        print("\n" + "=" * 74)
        print(f"{len(self.rows) - len(bad)}/{len(self.rows)} 통과")
        for prop, what, _, detail in bad:
            print(f"  FAIL  {prop}  {what}\n        {detail}")
        print("=" * 74)
        return 1 if bad else 0


# ---------------------------------------------------------------- softmax
def softmax(z: list[float]) -> list[float]:
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def entropy(z: list[float]) -> float:
    return -sum(p * math.log(p) for p in softmax(z) if p > 0)


def grad_analytic(z: list[float]) -> list[float]:
    """dH/dz_a = -p_a (ln p_a + H)."""
    p = softmax(z)
    h = -sum(x * math.log(x) for x in p if x > 0)
    return [-p[a] * (math.log(p[a]) + h) for a in range(len(z))]


def hess_analytic(z: list[float]) -> list[list[float]]:
    """d2H/dz_a dz_c = -p_a (delta_ac - p_c)(u_a + 1) + p_a p_c u_c,  u = ln p + H."""
    p = softmax(z)
    h = -sum(x * math.log(x) for x in p if x > 0)
    u = [math.log(x) + h for x in p]
    k = len(z)
    return [
        [-p[a] * ((1.0 if a == c else 0.0) - p[c]) * (u[a] + 1.0) + p[a] * p[c] * u[c]
         for c in range(k)]
        for a in range(k)
    ]


# ---------------------------------------------------------------- TIER A
def tier_a(rep: Report, seed: int = 0, trials: int = 6) -> None:
    rng = random.Random(seed)
    print("\nTIER A — 순수 수학 (데이터 불필요)")
    print("-" * 74)

    # P1: first derivative
    worst = 0.0
    for _ in range(trials):
        k = rng.randint(3, 9)
        z = [rng.gauss(0, 2) for _ in range(k)]
        ana = grad_analytic(z)
        for a in range(k):
            h = 1e-6
            zp, zm = z[:], z[:]
            zp[a] += h
            zm[a] -= h
            worst = max(worst, abs(ana[a] - (entropy(zp) - entropy(zm)) / (2 * h)))
    rep.check("P1", "dH/dz_a = -p_a (ln p_a + H)", worst < TOL_D1,
              f"{trials}개 랜덤 분포 전 좌표, 중심차분 대비 최대오차 {worst:.2e} (허용 {TOL_D1:.0e})")

    # P2: Hessian
    worst = 0.0
    for _ in range(trials):
        k = rng.randint(3, 6)
        z = [rng.gauss(0, 1.5) for _ in range(k)]
        ana = hess_analytic(z)
        for a in range(k):
            for c in range(k):
                h = 1e-4

                def shift(da: float, dc: float) -> float:
                    zz = z[:]
                    zz[a] += da
                    zz[c] += dc
                    return entropy(zz)

                num = (shift(h, h) - shift(h, -h) - shift(-h, h) + shift(-h, -h)) / (4 * h * h)
                worst = max(worst, abs(ana[a][c] - num))
    rep.check("P2", "d2H/dz_a dz_c = -p_a(d_ac - p_c)(u_a+1) + p_a p_c u_c", worst < TOL_D2,
              f"{trials}개 랜덤 분포 전 (a,c), 혼합차분 대비 최대오차 {worst:.2e} (허용 {TOL_D2:.0e})")

    # P3: Hessian at the uniform point is -(1/k)(I - J/k)
    worst = 0.0
    for k in (2, 3, 5, 10, 50):
        ana = hess_analytic([0.0] * k)
        for a in range(k):
            for c in range(k):
                want = -(1.0 / k) * ((1.0 if a == c else 0.0) - 1.0 / k)
                worst = max(worst, abs(ana[a][c] - want))
    rep.check("P3", "균등점 헤시안 = -(1/k)(I - J/k), 고유값 -1/k (중복 k-1) 및 0",
              worst < TOL_EXACT,
              f"k in 2,3,5,10,50 전 성분 최대오차 {worst:.2e}. 음반정부호 -> 균등점은 엔트로피 최대점; "
              f"0 고유벡터는 로짓 전체 평행이동(분포 불변)")

    # P4: closed form for a single-logit push away from uniform
    worst = 0.0
    for k in (2, 5, 10, 100):
        for d in (-2.0, -0.5, -0.1, 0.1, 0.5, 2.0):
            z = [0.0] * k
            z[0] += d
            S = math.exp(d) + k - 1
            worst = max(worst, abs(entropy(z) - (math.log(S) - d * math.exp(d) / S)))
    rep.check("P4", "균등점에서 H(d) = ln S - d e^d / S,  S = e^d + k - 1", worst < 1e-13,
              f"k in 2,5,10,100 x d in +-{{0.1,0.5,2}} 최대오차 {worst:.2e}. "
              f"H'(d) = -(k-1) d e^d / S^2 이므로 부호는 -sign(d): 어느 방향이든 엔트로피는 감소, "
              f"단 전적으로 2차항 (H'(0)=0)")

    # P5 / P6 / P7: the two-channel identity on an exactly-enumerable tree.
    # Root distribution is theta (= logits); each branch leads to a subtree of
    # fixed, theta-independent entropy H_b.  Then
    #   H_traj(theta) = H(p(theta)) + sum_b p_b(theta) H_b
    # is differentiable in closed form and by finite difference.
    k = 4
    z = [0.3, -0.8, 1.1, 0.0]
    Hb = [math.log(100), math.log(2), math.log(20), math.log(5)]

    def h_traj(zz: list[float]) -> float:
        p = softmax(zz)
        return -sum(x * math.log(x) for x in p) + sum(pi * hb for pi, hb in zip(p, Hb))

    p = softmax(z)
    h_root = -sum(x * math.log(x) for x in p)
    hbar = sum(pi * hb for pi, hb in zip(p, Hb))

    def visit(a: int, base: float) -> float:
        return sum(p[b] * ((1.0 if a == b else 0.0) - p[a]) * (Hb[b] - base) for b in range(k))

    worst = 0.0
    for a in range(k):
        h = 1e-6
        zp, zm = z[:], z[:]
        zp[a] += h
        zm[a] -= h
        num = (h_traj(zp) - h_traj(zm)) / (2 * h)
        local = -p[a] * (math.log(p[a]) + h_root)
        worst = max(worst, abs(num - (local + visit(a, hbar))))
    rep.check("P5", "grad H_traj = (L) 로컬 + (V) 방문  [항등식]", worst < TOL_D1,
              f"4갈래 트리, 갈래별 미래 H_togo 불균등. 중심차분 대비 최대오차 {worst:.2e}")

    spread = 0.0
    for a in range(k):
        vals = [visit(a, c) for c in (0.0, hbar, -3.656, 12.7)]
        spread = max(spread, max(vals) - min(vals))
    rep.check("P6", "베이스라인 무편향: sum_b dp_b/dz_a = 0 이므로 (V)는 뺀 상수에 무관",
              spread < TOL_EXACT,
              f"상수 c in {{0, H_bar, -3.656, 12.7}} 전체에서 (V) 최대편차 {spread:.2e}. "
              f"c = H_bar 로 두면 A_H = H_togo - H_bar 가 나온다")

    flat = max(abs(sum(p[b] * ((1.0 if a == b else 0.0) - p[a]) * (1.5 - 1.5) for b in range(k)))
               for a in range(k))
    rep.check("P7", "갈래별 미래가 모두 동일하면 (V) = 0 -- Omega 만으로 충분한 유일한 경우",
              flat < TOL_EXACT,
              f"H_b 를 상수로 두면 (V) 최대 {flat:.2e}. 기존 방법의 실패 정도는 "
              f"갈래별 미래 다양성의 '불균등'에 정확히 비례한다")


# ---------------------------------------------------------------- TIER B
def tier_b(rep: Report, path: str) -> None:
    print("\nTIER B — 이 모델의 워밍업 롤아웃 (학습 불필요)")
    print("-" * 74)
    if not os.path.exists(path):
        rep.check("P8", "MTP 상한 부등식", False, f"데이터 없음: {path}")
        return
    blob = json.load(open(path))
    R = blob["records"]
    n = len(R)
    n_heads = len(R[0]["forecast_per_head"])

    # P8: forecast_k - realised_k = I(Y_{u+k}; Y_{u+1..u+k-1} | S_u) >= 0,
    # and it must be ~0 at k=1 where there is no intervening token.
    gaps = []
    for k in range(n_heads):
        d = [r["forecast_per_head"][k] - r["gt_per_offset"][k] for r in R]
        gaps.append((st.mean(d), st.stdev(d) / math.sqrt(n)))

    m1, se1 = gaps[0]
    rep.check("P8a", "k=1 에서 격차 = 0 (중간 토큰이 없으므로 상호정보량 0)",
              abs(m1 / se1) < 3.0,
              f"격차 {m1:+.4f} +- {se1:.4f} (t={m1/se1:+.1f}), n={n}. "
              f"|t|<3 이므로 0 과 구별되지 않는다 -- 유도가 예측한 값")

    bad = [k + 1 for k in range(1, n_heads) if gaps[k][0] / gaps[k][1] <= 3.0]
    rep.check("P8b", "k>=2 에서 격차 > 0 (예보는 항상 과대추정)", not bad,
              "모든 k 에서 t>3: " + ", ".join(
                  f"k={k+1} {gaps[k][0]:+.2f}(t={gaps[k][0]/gaps[k][1]:.0f})"
                  for k in range(1, n_heads))
              if not bad else f"위배: k={bad}")

    drops = []
    for k in range(1, n_heads - 1):
        d = [(R[i]["forecast_per_head"][k + 1] - R[i]["gt_per_offset"][k + 1])
             - (R[i]["forecast_per_head"][k] - R[i]["gt_per_offset"][k]) for i in range(n)]
        m, se = st.mean(d), st.stdev(d) / math.sqrt(n)
        if m / se < -2.0:
            drops.append((k + 1, k + 2, m, m / se))
    rep.check("P8c", "격차가 k 에 단조 증가 (중간 토큰이 늘수록 상호정보량 증가)",
              not drops,
              "유의하게 감소하는 구간 없음 (일부 구간은 평평하나 t>-2)"
              if not drops else f"유의한 감소: {drops}")

    # P9: the independently-fitted calibration must mirror P8's gaps.
    scale = blob["calibration"]["scale"]
    a1_ok = 0.8 <= scale[0] <= 1.25
    rest_ok = all(s < 0.5 for s in scale[1:])
    rep.check("P9", "캘리브레이션 계수가 P8의 격차와 독립적으로 일치",
              a1_ok and rest_ok,
              f"a_1 = {scale[0]:.3f} (격차~0 이므로 1 근처여야 함), "
              f"a_(k>=2) = {[round(s, 3) for s in scale[1:]]} (큰 양의 격차를 흡수하려면 << 1). "
              f"두 값은 서로 다른 절차에서 나왔으므로 이 일치는 비자명하다")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=["a", "b", "both"], default="both")
    ap.add_argument("--phase1", default=DEFAULT_PHASE1,
                    help="Phase 1 결과 JSON (TIER B)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 74)
    print("STEER-F 유도 검증 — 학습 전에 반증 가능한 것 전부")
    print("=" * 74)

    rep = Report()
    if args.tier in ("a", "both"):
        tier_a(rep, seed=args.seed)
    if args.tier in ("b", "both"):
        tier_b(rep, args.phase1)

    code = rep.summary()
    print("\n여기서 검증되지 '않는' 것 (docs/STEERF_claims.md §3):")
    print("  M1  진짜 분기점에서 |Omega| 의 순위 -- 저장된 롤아웃에 forward 1회 필요")
    print("  M2  궤적 수준 지표 (분기 생존율, 분기 후 의미 다양성) -- 체크포인트별 재롤아웃 필요")
    print("  M3  대조군 lam=0 / apply=branch -- 학습 필요 (arm 당 약 5일, 1 GPU)")
    return code


if __name__ == "__main__":
    sys.exit(main())
