"""scripts/emit_r2_numbers.py: what it reads, and what it refuses to read."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "emit_r2_numbers", ROOT / "scripts" / "emit_r2_numbers.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

REF = ROOT / "docs" / "paper_reference.tsv"
OTHER = {"aime_2025_dapo_boxed": ("mean@32", 0.092),
         "amc2023_dapo_boxed": ("mean@32", 0.571),
         "math500": ("mean@1", 0.740),
         "minerva_math": ("mean@1", 0.246),
         "olympiadbench": ("mean@1", 0.348),
         "gsm8k_test": ("mean@1", 0.829)}


def step_line(step, acc, reward):
    return (f"(TaskRunner pid=1) step:{step} - global_seqlen/min:1.0 - "
            f"val-aux/aime_2024_dapo_boxed/reward/mean@32:{reward:.3f} - "
            f"val-core/aime_2024_dapo_boxed/acc/mean@32:{acc:.3f} - x:1\n")


def train_log(best_step, best_k, last_k, tracker=None, acc_override=None):
    best = best_k / 960
    tracked = best if tracker is None else tracker
    acc = best if acc_override is None else acc_override
    return ("Training from scratch\n"
            + step_line(best_step, acc, 2 * best - 1)
            + f"New best val-core/aime_2024_dapo_boxed/acc/mean@32: {tracked:.4f} "
              f"at step {best_step} (previous: 0.1000)\n"
            + step_line(200, last_k / 960, 2 * last_k / 960 - 1))


def eval_log(run, step, aime=0.169):
    parts = [f"actor_rollout_ref.model.path=checkpoints/STEER-F/{run}/"
             f"global_step_{step}/actor/huggingface trainer.val_only=True\n",
             "step:0 - "]
    sets = dict(OTHER, aime_2024_dapo_boxed=("mean@32", aime))
    for ds, (k, v) in sets.items():
        parts.append(f"val-aux/{ds}/reward/{k}:{2 * v - 1:.3f} - "
                     f"val-core/{ds}/acc/{k}:{v:.3f} - ")
    return "".join(parts) + "\n"


def write_all(tmp, s1=None, s2=None, evals=True):
    d = tmp / "logs"
    d.mkdir()
    s1 = s1 or train_log(150, 173, 169)
    s2 = s2 or train_log(190, 162, 155)
    (d / "train-steerf-r2-1p5b-s1-200.log").write_text(s1)
    (d / "train-steerf-r2-1p5b-s2-200.log").write_text(s2)
    if evals:
        for seed, steps in ((1, (150, 200)), (2, (190, 200))):
            run = f"steerf-r2-1p5b-s{seed}-200"
            for st in steps:
                (d / f"eval-r2-s{seed}-step{st}.log").write_text(eval_log(run, st))
    return d


def test_full_set_and_exact_two_run_mean(tmp_path):
    d = write_all(tmp_path)
    values, notes = mod.collect(None, str(d), REF)
    assert notes == []
    assert values["ProtobestOneAime"] == "18.0"      # 173/960, the training value
    assert values["ProtobestTwoAime"] == "16.9"      # 162/960
    # (173 + 162) / 1920 = .17448 -> 17.4; the three-decimal values would give 17.45
    assert values["ProtobestMeanAime"] == "17.4"
    assert values["ProtostepBestOne"] == "150" and values["ProtostepBestTwo"] == "190"
    assert values["ProtobestMeanMinerva"] == "24.6"
    assert "ProtobestMeanAvg" in values and "ProtolastMeanAvg" in values
    assert values["ProtorefSteerAime"] == "17.4"
    assert set(values) <= mod.names()


def test_aime_comes_from_training_not_from_evaluation(tmp_path):
    d = write_all(tmp_path)
    run = "steerf-r2-1p5b-s1-200"
    (d / "eval-r2-s1-step150.log").write_text(eval_log(run, 150, aime=0.150))
    values, _ = mod.collect(None, str(d), REF)
    assert values["ProtobestOneAime"] == "18.0"


def test_value_that_disagrees_with_its_reward_is_refused(tmp_path):
    # the 2026-10-05 case: 0.181 printed beside a reward that says 0.169
    d = write_all(tmp_path, s2=train_log(190, 162, 155, acc_override=0.181))
    values, notes = mod.collect(None, str(d), REF)
    assert "ProtobestTwoAime" not in values
    assert "ProtobestMeanAime" not in values          # the mean needs both runs
    assert "ProtobestOneAime" in values
    assert any("disagrees with its reward" in n for n in notes)


def test_value_that_disagrees_with_the_trainer_is_refused(tmp_path):
    d = write_all(tmp_path, s2=train_log(190, 162, 155, tracker=0.1810))
    values, notes = mod.collect(None, str(d), REF)
    assert "ProtobestTwoAime" not in values
    assert any("New best" in n for n in notes)


def test_evaluation_of_the_wrong_checkpoint_is_refused(tmp_path):
    d = write_all(tmp_path)
    (d / "eval-r2-s2-step190.log").write_text(
        eval_log("steerf-r2-1p5b-s2-200", 200))
    values, notes = mod.collect(None, str(d), REF)
    assert "ProtobestTwoAmc" not in values
    assert "ProtobestTwoAime" in values               # training value unaffected
    assert any("model path" in n for n in notes)


def test_incoherent_evaluation_is_refused(tmp_path):
    d = write_all(tmp_path)
    p = d / "eval-r2-s1-step150.log"
    p.write_text(p.read_text().replace("math500/acc/mean@1:0.740",
                                       "math500/acc/mean@1:0.790"))
    values, notes = mod.collect(None, str(d), REF)
    assert "ProtobestOneMathfive" not in values
    assert any("incoherent" in n for n in notes)


def test_missing_evaluations_leave_only_those_cells_red(tmp_path):
    d = write_all(tmp_path, evals=False)
    values, notes = mod.collect(None, str(d), REF)
    assert values["ProtobestMeanAime"] == "17.4"
    assert "ProtobestMeanAmc" not in values
    assert sum("no evaluation log" in n for n in notes) == 4


def test_unfinished_evaluation_is_reported_not_skipped(tmp_path):
    # the 2026-10-06 s2 step-190 log: model path right, stopped before any metric
    d = write_all(tmp_path)
    (d / "eval-r2-s2-step190.log").write_text(
        "actor_rollout_ref.model.path=checkpoints/STEER-F/steerf-r2-1p5b-s2-200/"
        "global_step_190/actor/huggingface\n[score] 1.0\n")
    values, notes = mod.collect(None, str(d), REF)
    assert "ProtobestTwoAmc" not in values
    assert "ProtobestTwoAime" in values
    assert any("did not finish" in n for n in notes)
