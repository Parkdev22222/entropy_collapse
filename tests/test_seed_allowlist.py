"""One seed set for the whole paper, enforced in one place.

The manuscript carried a mixture: --balanced restricted the means table to the
seeds all five arms share while every contrast paired on its own two arms'
shared seeds, so two rows were n=4, two row differences did not equal their
contrasts, and seven sentences existed to explain that away.

--seeds is the single gate. It is applied where per_seed, per_seed_partial and
topo are fed, so the contrasts, the run-to-run spread, the machine effect, the
within-run appendix, the compute-matched control and the follow-up topology
check all inherit it -- six separate restrictions would be six chances to miss
one, which is how the mixture happened.
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = "Qwen2.5-Math-1.5B"
BOX = {1: 2, 2: 2, 3: 4, 4: 4}          # a seed index also names a machine
RUN = {"grpo": "grpo-%s-s%d", "steer": "steer-%s-s%d",
       "signed": "steer-f-%s-s%d-tree-rollout",
       "uniform": "steer-f-%s-s%d-tree-rollout-uniform",
       "permuted": "steer-f-%s-s%d-tree-rollout-permuted"}


def campaign(tmp_path: Path, acc) -> Path:
    """The campaign's shape: five arms x four seeds, except signed has no seed 2.

    That asymmetry is the whole reason the manuscript carried a mixture -- the
    shared set is {1,3,4} while four arms also have a fourth seed -- so a
    fixture without it cannot exercise what the gate is for.
    """
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for arm, pat in RUN.items():
        for seed in (1, 2, 3, 4):
            if arm == "signed" and seed == 2:
                continue
            body = ["{'n_gpus_per_node': %d, 'seed': %d}" % (BOX[seed], seed)]
            for step in range(1, 111):
                line = f"step:{step} - global_seqlen: 1 - perf/time_per_step:500.0"
                if step % 10 == 0:
                    a = acc(arm, seed)
                    line += (f" - val-core/aime_2024_dapo_boxed/acc/mean@32:{a:.4f}"
                             f" - val-core/aime_2024_dapo_boxed/acc/maj@32/mean:{a:.4f}"
                             f" - actor/entropy:{a:.4f}")
                body.append(line)
            (logs / f"train-{pat % (TAG, seed)}.log").write_text("\n".join(body) + "\n")
    return logs


def run(tmp_path, logs, *extra):
    out = tmp_path / ("out" + "".join(extra).replace(",", "").replace("-", ""))
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_seeds.py"),
                        "--balanced", "--logs", str(logs), "--out", str(out), *extra],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert p.returncode == 0, p.stderr
    txt = (out / "numbers.tex").read_text()
    mac = {m.group(1): m.group(2) for m in
           re.finditer(r"\\providecommand\{\\([A-Za-z]+)\}\{([^}]*)\}", txt)}
    rows = list(csv.DictReader((out / "contrasts.tsv").open(), delimiter="\t"))
    seeds = list(csv.DictReader((out / "per_seed.tsv").open(), delimiter="\t"))
    return mac, rows, seeds, p.stdout


# a box effect (+.01 on the four-GPU seeds) and a per-arm level
def ACC(arm, seed):
    base = {"grpo": .140, "steer": .137, "uniform": .142,
            "permuted": .141, "signed": .149}[arm]
    return base + (.01 if BOX[seed] == 4 else 0) + (.002 if seed == 2 else 0)


def test_every_contrast_is_on_the_same_seeds(tmp_path):
    mac, rows, _, _ = run(tmp_path, campaign(tmp_path, ACC), "--seeds", "1,3,4")
    ns = {r["contrast"]: r["n_seeds"] for r in rows if r["metric"] == "acc"}
    assert len(ns) == 6, ns
    assert set(ns.values()) == {"3"}, ns
    # the two the mixture used to leave at four
    assert mac["Nsteergrpo"] == "3" and mac["Nuniformsteer"] == "3"
    assert mac["Nsteergrpo"] == mac["Nsignedgrpo"]


def test_the_seed_set_is_emitted_not_described(tmp_path):
    mac, _, _, _ = run(tmp_path, campaign(tmp_path, ACC), "--seeds", "1,3,4")
    assert mac["Seedset"] == "1, 3, 4", mac["Seedset"]
    assert mac["Nseedsused"] == "3"


def test_the_machine_effect_inherits_the_gate(tmp_path):
    """The cost the user accepted: one run on the smaller box, not two."""
    mac, _, _, _ = run(tmp_path, campaign(tmp_path, ACC), "--seeds", "1,3,4")
    assert mac["Boxnlo"] == "1", mac["Boxnlo"]
    assert mac["Boxnhi"] == "2", mac["Boxnhi"]
    # and without the gate it is two against two
    mac2, _, _, _ = run(tmp_path, campaign(tmp_path, ACC))
    assert mac2["Boxnlo"] == "2", mac2["Boxnlo"]


def test_an_excluded_seed_stays_on_the_record(tmp_path):
    """per_seed_partial's rule: never deleted, only kept out of the statistics."""
    _, _, seeds, out = run(tmp_path, campaign(tmp_path, ACC), "--seeds", "1,3,4")
    s2 = [r for r in seeds if r["seed"] == "2"]
    # four arms have a seed 2; signed does not (see campaign()'s docstring)
    assert len(s2) == 4, f"every seed-2 run must still be listed, got {len(s2)}"
    assert all(r["excluded"] == "1" for r in s2), s2
    assert all(r["excluded"] == "0" for r in seeds if r["seed"] != "2")
    # and it is reported once, so the exclusion is not inferred from an absence
    assert "held out of every statistic" in out, out


def test_without_the_flag_nothing_is_held_out(tmp_path):
    _, rows, seeds, out = run(tmp_path, campaign(tmp_path, ACC))
    assert all(r["excluded"] == "0" for r in seeds)
    assert "held out of every statistic" not in out
    ns = {r["n_seeds"] for r in rows if r["metric"] == "acc"}
    assert ns == {"3", "4"}, f"the mixture is what the default still produces: {ns}"


def test_row_differences_equal_the_contrasts_for_all_six(tmp_path):
    logs = campaign(tmp_path, ACC)
    mac, rows, _, _ = run(tmp_path, logs, "--seeds", "1,3,4")
    out = tmp_path / "outseeds134"
    means = {r["arm"]: float(r["acc"])
             for r in csv.DictReader((out / "arm_means.tsv").open(), delimiter="\t")}
    for r in rows:
        if r["metric"] != "acc":
            continue
        a, b = r["contrast"].split(" - ")
        assert abs((means[a] - means[b]) - float(r["mean_diff"])) < 1e-4, r
