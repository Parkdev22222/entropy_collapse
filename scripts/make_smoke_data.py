#!/usr/bin/env python3
"""스모크용 합성 문제 파일 생성 (parquet/pandas 불필요).

STEER 동봉 parquet 과 **같은 스키마**의 jsonl 을 쓴다 — `_common.load_prompts_from_parquet`
가 확장자를 보고 분기하므로 두 Phase 1 스크립트가 그대로 읽는다.

수학 문제처럼 보이지만 실제 난이도는 없다. 목적은 배관 검증뿐이다
(`tests/test_smoke_pipeline.py`, `run/run_smoke_cpu.sh`).

사용 예:
    python scripts/make_smoke_data.py --out artifacts/smoke/problems.jsonl --n-problems 8
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random

TEMPLATE = (
    "Solve step by step.\n"
    "Compute {a} * {b} + {c}.\n"
    "Put the final answer in \\boxed{{}}."
)


def build_problems(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        a, b, c = rng.randint(2, 19), rng.randint(2, 19), rng.randint(1, 50)
        rows.append(
            {
                "data_source": "smoke",
                "prompt": [{"role": "user", "content": TEMPLATE.format(a=a, b=b, c=c)}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(a * b + c)},
                "extra_info": {"index": f"smoke-{i:03d}"},
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-problems", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = build_problems(args.n_problems, args.seed)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"[make_smoke_data] wrote {len(rows)} problems -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
