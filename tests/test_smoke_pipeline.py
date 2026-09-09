"""Phase 1 파이프라인 end-to-end 스모크 (CPU, 의존성 없음).

지금까지 `steer_f/` 모듈은 단위 테스트로 덮여 있었지만 `scripts/` 의 두 스크립트는
**단 한 번도 실행된 적이 없다** (docs/experiment_log.md — 개발 컨테이너에 GPU 부재).
따라서 GPU 노드의 첫 실행에서 배관 문제로 깨질 위험이 가장 컸다.

이 테스트는 합성 스택으로 `phase1_warmup_heads.py train` → `phase1_validate.py` 를
서브프로세스로 실제 완주시켜 다음을 확인한다:

- CLI 파싱, 스테이지 캐시, 체크포인트 저장/로드 형상
- 리포트 파일과 `phase1_result.json` 생성
- 게이트 종료코드 규약 (0=통과, 2=실패) — **판정 내용이 아니라 판정이 내려지는지**

합성 모델의 ρ/recall 값 자체는 의미가 없다 (난수 가중치). 여기서 실패하면
그것은 배관 버그이지 연구 결과가 아니다.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SMOKE_MODEL = "smoke:h16,l2,seed0"


def _run(args: list[str], cwd: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args], cwd=cwd, capture_output=True, text=True, timeout=900
    )


@pytest.fixture(scope="module")
def smoke_problems(tmp_path_factory) -> pathlib.Path:
    """`scripts/make_smoke_data.py` 로 합성 문제 파일을 만든다."""
    out = tmp_path_factory.mktemp("data") / "problems.jsonl"
    proc = _run(["scripts/make_smoke_data.py", "--out", str(out), "--n-problems", "6", "--seed", "0"], ROOT)
    assert proc.returncode == 0, proc.stderr
    return out


def test_make_smoke_data_writes_loadable_problems(smoke_problems):
    from scripts._common import load_prompts_from_parquet

    problems = load_prompts_from_parquet(smoke_problems)
    assert len(problems) == 6
    assert all(p["messages"] and p["ground_truth"] is not None for p in problems)


@pytest.fixture(scope="module")
def trained_heads(tmp_path_factory, smoke_problems) -> pathlib.Path:
    """generate → train 을 실제로 돌려 헤드 체크포인트를 만든다."""
    work = tmp_path_factory.mktemp("warmup")
    rollouts = work / "rollouts.jsonl"
    gen = _run(
        [
            "scripts/phase1_warmup_heads.py", "generate",
            "--model", SMOKE_MODEL, "--prompts", str(smoke_problems), "--out", str(rollouts),
            "--n-prompts", "6", "--n-samples", "2", "--max-response-length", "48",
            "--gen-batch", "4", "--no-vllm", "--seed", "0",
        ],
        ROOT,
    )
    assert gen.returncode == 0, gen.stderr
    assert rollouts.exists()

    ckpt = work / "mtp_heads.pt"
    train = _run(
        [
            "scripts/phase1_warmup_heads.py", "train",
            "--model", SMOKE_MODEL, "--rollouts", str(rollouts), "--out", str(ckpt),
            "--num-heads", "3", "--head-hidden", "16", "--batch-size", "4",
            "--max-len", "128", "--log-every", "1", "--dtype", "float32", "--device", "cpu",
        ],
        ROOT,
    )
    assert train.returncode == 0, train.stderr
    return ckpt


def test_generate_stage_writes_one_rollout_per_sample(trained_heads):
    from scripts._common import load_rollouts_jsonl

    rollouts = load_rollouts_jsonl(trained_heads.parent / "rollouts.jsonl")
    assert len(rollouts) == 12  # 6 prompts × 2 samples
    assert all(r.prompt and r.problem_id for r in rollouts)


def test_trained_heads_checkpoint_loads_through_verl_helper(trained_heads):
    """`verl_integration.load_heads_checkpoint` 가 워밍업 산출물을 그대로 읽어야 한다.

    체크포인트 스키마가 두 모듈 사이에서 어긋나면 GPU 런의 첫 스텝에서 죽는다.
    """
    from steer_f.verl_integration import load_heads_checkpoint

    heads, cfg, calib = load_heads_checkpoint(trained_heads)
    assert cfg["num_heads"] == 3 and cfg["hidden_size"] == 16
    assert heads.num_heads == 3
    assert calib is None  # 워밍업은 캘리브레이션을 적합하지 않는다


def test_training_log_is_written_and_finite(trained_heads):
    log = json.loads((trained_heads.with_suffix(".log.json")).read_text())
    assert log, "학습 로그가 비어 있다 — 스텝이 한 번도 돌지 않았다"
    assert all(row["loss"] == row["loss"] for row in log)  # NaN 아님


@pytest.fixture(scope="module")
def validate_run(tmp_path_factory, smoke_problems, trained_heads):
    work = tmp_path_factory.mktemp("validate")
    report = work / "phase1_report.md"
    proc = _run(
        [
            "scripts/phase1_validate.py",
            "--model", SMOKE_MODEL, "--heads", str(trained_heads), "--problems", str(smoke_problems),
            "--workdir", str(work), "--report", str(report),
            "--n-problems", "3", "--problem-pool", "6", "--n-trajectories", "4", "--n-mc", "2",
            "--min-pass-rate", "0.0", "--max-pass-rate", "1.0",
            "--max-response-length", "48", "--mc-max-tokens", "16", "--gt-horizon", "8",
            "--kappa-grid", "1,2,3", "--gamma-grid", "0.85,1.0",
            "--branch-group-size", "3", "--branch-max-len", "24",
            "--embed-model", "", "--dtype", "float32", "--device", "cpu", "--no-vllm",
            "--calibrate", "--seed", "0",
        ],
        ROOT,
    )
    return proc, work, report


def test_validate_exits_with_gate_verdict_code(validate_run):
    """0=G1 통과, 2=실패. 그 외 코드는 배관이 깨졌다는 뜻이다."""
    proc, _, _ = validate_run
    assert proc.returncode in (0, 2), f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def test_validate_writes_report_and_result_json(validate_run):
    _, work, report = validate_run
    assert report.exists(), "리포트가 생성되지 않았다"
    result = json.loads((work / "phase1_result.json").read_text())
    assert result["kappa"] in (1, 2, 3)
    assert result["gamma_h"] in (0.85, 1.0)
    assert "gate_summary" in result


def test_validate_grid_covers_full_kappa_gamma_product(validate_run):
    _, work, _ = validate_run
    result = json.loads((work / "phase1_result.json").read_text())
    assert len(result["grid"]) == 3 * 2, "그리드 일부가 조용히 건너뛰어졌다"


def test_validate_report_contains_gate_section(validate_run):
    _, _, report = validate_run
    text = report.read_text()
    assert "게이트 G1" in text and "분기 토큰 recall" in text


def test_validate_caches_stages_and_second_run_is_faster(validate_run):
    """스테이지 캐시가 실제로 재사용되는지 — GPU 런에서 재개 가능해야 한다."""
    proc, work, report = validate_run
    assert proc.returncode in (0, 2)
    for name in ("trajectories.json", "prefixes.json", "ground_truth.json", "forecast.json"):
        assert (work / name).exists(), f"{name} 캐시가 없다"
