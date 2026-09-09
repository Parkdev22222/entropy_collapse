"""`phase1_validate.py` 의 빈 풀 가드.

**발견 경위**: CPU 스모크 실행(`tests/test_smoke_pipeline.py`)에서 stage2 가
`0 prefixes` 를 낸 뒤에도 stage3/4/5 가 조용히 진행되어, 4단계 뒤 전혀 무관한
`ValueError: no finite rho in grid results` 로 죽었다. GPU 노드에서 같은 일이
벌어지면 원인 추적에 시간과 GPU 시간을 모두 버린다.

가드는 **원인이 발생한 스테이지에서** 실패해야 하고, 어떤 인자를 고칠지 말해야 한다.
종료코드는 게이트 판정(0/2)과 구분되는 3을 쓴다 — 판정이 아니라 설정/데이터 문제이므로.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts.phase1_validate import EMPTY_POOL_EXIT, EmptyPoolError, check_pools  # noqa: E402


def test_check_pools_passes_when_both_populated():
    check_pools([{"problem_id": "a"}], [{"prefix": "x"}])  # 예외 없음


def test_empty_problem_pool_raises_with_actionable_knobs():
    with pytest.raises(EmptyPoolError) as exc:
        check_pools([], [{"prefix": "x"}])
    msg = str(exc.value)
    assert "--problem-pool" in msg
    assert "--min-pass-rate" in msg and "--max-pass-rate" in msg


def test_empty_prefix_pool_names_the_step_count_requirement():
    """prefix 가 0인 원인은 거의 항상 '궤적의 줄 수 < 절단 지점 수'다."""
    with pytest.raises(EmptyPoolError) as exc:
        check_pools([{"problem_id": "a"}], [])
    msg = str(exc.value)
    assert "--max-response-length" in msg
    assert "4" in msg  # TRUNC_FRACS 개수 = 필요한 최소 스텝 수


def test_empty_problem_pool_is_reported_before_prefix_pool():
    """둘 다 비면 근본 원인(문제 선정)을 먼저 알려야 한다."""
    with pytest.raises(EmptyPoolError) as exc:
        check_pools([], [])
    assert "--problem-pool" in str(exc.value)


def test_exit_code_is_distinct_from_gate_verdicts():
    """0=G1 통과, 2=G1 실패. 설정 오류가 이 둘과 섞이면 자동화가 오판한다."""
    assert EMPTY_POOL_EXIT not in (0, 2)
