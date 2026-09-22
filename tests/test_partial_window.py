"""A run that died inside the plateau window is not a shorter version of one.

On 2026-09-22 signed s4 died at training step 91. Its plateau mean covered six
validation points where every other run covered eight, and analyze_seeds.py
paired the six-point mean against the eight-point ones with no gate and no
warning. That alone moved the headline contrast from +.0145 to about zero.

docs/preregistration_5seed.md fixes the endpoint as the mean over steps 40-110,
eight validation points, and excludes a run "only for mechanical failure -- out
of memory, a crash, a corrupted checkpoint", relaunching it with the same seed.
The gate implements that; these tests keep it implemented.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_seeds.py"

ACC = "val-core/aime_2024_dapo_boxed/acc/mean@32"
MAJ = "val-core/aime_2024_dapo_boxed/acc/maj@32/mean"


def _log(path: Path, last_step: int, acc: float):
    """A log with a validation point every ten steps up to last_step."""
    out = ["'loss_mode': 'vanilla'", "'n_gpus_per_node': 4"]
    for step in range(0, last_step + 1):
        row = [f"step:{step} - global_seqlen/mean:1.0", "actor/entropy:0.120",
               "response_length/mean:900.0", "perf/time_per_step:800.0"]
        if step % 10 == 0:
            row += [f"{ACC}:{acc:.4f}", f"{MAJ}:{acc + 0.06:.4f}",
                    "val-aux/aime_2024_dapo_boxed/acc/std@32:0.150"]
        out.append(" - ".join(row))
    path.write_text("\n".join(out) + "\n")


def _tree(tmp_path, seeds):
    """seeds: {(arm, seed): last_step}. Accuracy is seeded off the arm."""
    logs = tmp_path / "logs" / "experiments"
    logs.mkdir(parents=True)
    names = {"grpo": "grpo-T-s{}", "steer": "steer-T-s{}",
             "signed": "steer-f-T-s{}-tree-rollout",
             "uniform": "steer-f-T-s{}-tree-rollout-uniform",
             "permuted": "steer-f-T-s{}-tree-rollout-permuted"}
    base = {"grpo": 0.13, "steer": 0.13, "signed": 0.15,
            "uniform": 0.13, "permuted": 0.13}
    for (arm, seed), last in seeds.items():
        _log(logs / f"train-{names[arm].format(seed)}.log", last, base[arm])
    return logs


def _run(logs, out, *extra):
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--logs", str(logs), "--out", str(out),
         "--model-tag", "T", "--plateau", "40:110", "--steps", "110", *extra],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def _seeds_of(out, contrast, metric="acc"):
    for line in (out / "contrasts.tsv").read_text().splitlines()[1:]:
        f = line.split("\t")
        if f[0] == contrast and f[1] == metric:
            return f[2], f[8], f[9]          # n, seeds, held_out
    return None


def test_a_short_run_is_held_out_and_said_so(tmp_path):
    logs = _tree(tmp_path, {("grpo", 1): 110, ("grpo", 4): 110,
                            ("signed", 1): 110, ("signed", 4): 91})
    out = tmp_path / "res"
    stdout = _run(logs, out)

    assert "did not reach step 110" in stdout
    assert "signed    s4" in stdout and "HELD OUT" in stdout

    n, seeds, held = _seeds_of(out, "signed - grpo")
    assert n == "1", f"the six-point run is still in the contrast (n={n})"
    assert seeds == "1"
    assert held == "4", "the held-out seed is not named in contrasts.tsv"


def test_the_record_keeps_it(tmp_path):
    """Held out of the statistics, never deleted from per_seed.tsv."""
    logs = _tree(tmp_path, {("grpo", 1): 110, ("signed", 1): 110,
                            ("signed", 4): 91})
    out = tmp_path / "res"
    _run(logs, out)
    rows = (out / "per_seed.tsv").read_text().splitlines()
    head = rows[0].split("\t")
    got = {(f[0], f[1]): f for f in (r.split("\t") for r in rows[1:])}
    assert ("signed", "4") in got, "the short run vanished from the record"
    assert got[("signed", "4")][head.index("partial")] == "1.0000"
    assert got[("signed", "1")][head.index("partial")] == "0.0000"


def test_allow_partial_puts_it_back(tmp_path):
    logs = _tree(tmp_path, {("grpo", 1): 110, ("grpo", 4): 110,
                            ("signed", 1): 110, ("signed", 4): 91})
    out = tmp_path / "res"
    stdout = _run(logs, out, "--allow-partial")
    assert "included (--allow-partial)" in stdout
    n, seeds, _ = _seeds_of(out, "signed - grpo")
    assert n == "2" and seeds == "1,4"


def test_a_narrower_window_readmits_it(tmp_path):
    """The gate asks whether validation reached the top of the window.

    A run that died at 91 has every point of 40-90, so it belongs in that
    window -- which is why the 2026-09-22 recompute on 40-90 legitimately
    included it and still showed the contrast at about zero.
    """
    logs = _tree(tmp_path, {("grpo", 1): 110, ("grpo", 4): 110,
                            ("signed", 1): 110, ("signed", 4): 91})
    out = tmp_path / "res"
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--logs", str(logs), "--out", str(out),
         "--model-tag", "T", "--plateau", "40:90", "--steps", "110"],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "did not reach step 90" not in r.stdout
    n, seeds, _ = _seeds_of(out, "signed - grpo")
    assert n == "2" and seeds == "1,4"


def test_full_runs_are_untouched(tmp_path):
    """The regression guard: the gate must not shrink a healthy campaign."""
    logs = _tree(tmp_path, {(a, s): 110
                            for a in ("grpo", "steer", "signed", "uniform", "permuted")
                            for s in (1, 2, 3)})
    out = tmp_path / "res"
    stdout = _run(logs, out)
    assert "HELD OUT" not in stdout
    for c in ("signed - grpo", "signed - steer", "signed - uniform",
              "signed - permuted", "steer - grpo", "uniform - steer"):
        n, seeds, held = _seeds_of(out, c)
        assert n == "3" and seeds == "1,2,3" and held == "-", c


# ------------------------- the file name is a label, the config dump is not
def _identified(path, seed, name, gpus=4):
    """Prepend the config dump verl writes near the top of every run."""
    head = ("{'data': {'seed': %d}, 'trainer': {'experiment_name': '%s', "
            "'n_gpus_per_node': %d, 'tensor_model_parallel_size': %d}}\n"
            % (seed, name, gpus, gpus))
    path.write_text(head + path.read_text())


def test_a_log_whose_dump_names_another_seed_is_excluded(tmp_path):
    """2026-09-22: a push put the seed-2 run into train-grpo-<tag>-s4.log.

    Nothing here reads the file's contents for identity -- the seed comes from
    the NAME and the numbers from the metric lines -- so GRPO would have taken
    seed 2's numbers twice, one of them carrying an A100 topology into the H100
    set. A later edit fixed the two-line banner at the top of the file and left
    the dump alone, so a human skim saw s4 while the dump still said s2.
    """
    logs = _tree(tmp_path, {("grpo", 1): 110, ("grpo", 4): 110,
                            ("signed", 1): 110, ("signed", 4): 110})
    _identified(logs / "train-grpo-T-s1.log", 1, "grpo-T-s1")
    # the s4 file actually holds the seed-2 run, on two GPUs
    _identified(logs / "train-grpo-T-s4.log", 2, "grpo-T-s2", gpus=2)
    out = tmp_path / "res"
    stdout = _run(logs, out)

    assert "EXCLUDED" in stdout, stdout
    assert "train-grpo-T-s4.log" in stdout
    assert "seed 2" in stdout

    n, seeds, _ = _seeds_of(out, "signed - grpo")
    assert n == "1" and seeds == "1", f"the wrong-seed run reached the contrast (n={n})"


def test_an_older_experiment_name_is_kept(tmp_path):
    """The first version of this check rejected on experiment_name and dropped
    a real STEER seed: the early runs were called math-<tag>-steer-s1 before
    the campaign settled on steer-<tag>-s1. The seed is what indexes the
    analysis, so the seed is what has to agree."""
    logs = _tree(tmp_path, {("steer", 1): 110, ("signed", 1): 110})
    _identified(logs / "train-steer-T-s1.log", 1, "math-T-steer-s1")
    out = tmp_path / "res"
    stdout = _run(logs, out)

    assert "EXCLUDED" not in stdout, stdout
    assert "older experiment_name" in stdout
    rows = (out / "per_seed.tsv").read_text()
    assert "steer\t1\t" in rows, "a renamed but genuine run was dropped"


def test_mixed_gpu_counts_are_reported(tmp_path):
    """Seed-paired contrasts cancel a box effect; arm means over different
    seed sets do not, and the mapping is invisible in the numbers."""
    logs = _tree(tmp_path, {("grpo", 1): 110, ("grpo", 4): 110,
                            ("signed", 1): 110})
    _identified(logs / "train-grpo-T-s1.log", 1, "grpo-T-s1", gpus=2)
    _identified(logs / "train-grpo-T-s4.log", 4, "grpo-T-s4", gpus=4)
    out = tmp_path / "res"
    stdout = _run(logs, out)
    assert "did not all run on the same GPU count" in stdout, stdout
    assert "s1=2" in stdout and "s4=4" in stdout
