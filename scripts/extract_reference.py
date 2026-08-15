#!/usr/bin/env python3
"""STEER 원본 `compute_token_weights`를 verbatim 추출해 tests/reference_steer.py 를 만든다.

STEER 레포를 업데이트했을 때 참조 사본이 낡지 않도록 재실행할 것.

    python scripts/extract_reference.py --steer-root /path/to/STEER
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

HEADER = '''"""STEER 원본 `compute_token_weights` 의 검증용 사본.

verl/trainer/ppo/core_algos.py:{start}-{end} 에서 **자동 추출**한 것이다 (수작업 편집 금지).
tests/test_lambda_zero_equiv.py가 STEER-F 구현과 이 사본을 대조해 λ=0 동치성을 강제한다.

재추출:
    python scripts/extract_reference.py --steer-root <STEER clone> --out tests/reference_steer.py
"""

import torch


'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steer-root", required=True, type=pathlib.Path)
    ap.add_argument("--out", default=pathlib.Path("tests/reference_steer.py"), type=pathlib.Path)
    args = ap.parse_args()

    src_path = args.steer_root / "verl" / "trainer" / "ppo" / "core_algos.py"
    if not src_path.exists():
        print(f"error: {src_path} not found", file=sys.stderr)
        return 1

    lines = src_path.read_text().splitlines(keepends=True)

    start = None
    for i, line in enumerate(lines):
        if re.match(r"^def compute_token_weights\(", line):
            start = i
            break
    if start is None:
        print("error: compute_token_weights not found", file=sys.stderr)
        return 1

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if re.match(r"^(def |@|class )", lines[j]):
            end = j
            break
    while end > start and lines[end - 1].strip() == "":
        end -= 1

    body = "".join(lines[start:end])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(HEADER.format(start=start + 1, end=end) + body)
    print(f"wrote {args.out} from {src_path}:{start + 1}-{end}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
