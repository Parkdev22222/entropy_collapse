"""scripts/analyze_seeds.py --lora: the LoRA campaign's arms, labels and the
registered collapse check (docs/preregistration_lora.md section 6)."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = "Qwen2.5-Math-1.5B"
NAMES = {"grpo": "grpo-{t}-s{s}", "steer": "steer-{t}-s{s}",
         "signed": "steer-f-{t}-s{s}-tree-rollout",
         "uniform": "steer-f-{t}-s{s}-tree-rollout-uniform",
         "permuted": "steer-f-{t}-s{s}-tree-rollout-permuted",
         "mtp": "steer-f-{t}-s{s}-tree-rollout-mtp"}


def write_log(path, acc, ent0, ent1, last=110):
    body = []
    for step in range(1, last + 1):
        ent = ent0 + (ent1 - ent0) * (step - 1) / (last - 1)
        line = f"step:{step} - global_seqlen: 1 - actor/entropy:{ent:.4f}"
        if step % 10 == 0:
            line += (f" - val-core/aime_2024_dapo_boxed/acc/mean@32:{acc}"
                     f" - val-core/aime_2024_dapo_boxed/acc/maj@32/mean:{acc}")
        body.append(line)
    path.write_text("\n".join(body) + "\n")


def campaign(tmp_path, grpo_end=0.10):
    logs = tmp_path / "logs"
    logs.mkdir()
    for arm, pat in NAMES.items():
        for s in (1, 2, 3):
            end = grpo_end if arm == "grpo" else 0.3
            acc = 0.15 + (0.01 if arm == "signed" else 0) + 0.001 * s
            write_log(logs / f"train-lora-{pat.format(t=TAG, s=s)}.log", f"{acc:.3f}", 0.6, end)
    return logs


def analyse(tmp_path, logs, *extra):
    mac = tmp_path / "m.tex"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/analyze_seeds.py"),
                        "--logs", str(logs), "--out", str(tmp_path / "out"),
                        "--tex-macros", str(mac), "--run-prefix", "lora-",
                        "--seeds", "1,2,3", "--balanced", *extra],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr + r.stdout
    return {m.group(1): m.group(2) for m in
            re.finditer(r"\\providecommand\{\\([A-Za-z]+)\}\{([^}]*)\}", mac.read_text())}


def test_lora_mode_adds_the_mtp_arm_and_its_contrast(tmp_path):
    m = analyse(tmp_path, campaign(tmp_path), "--lora")
    assert m["Nmtp"] == "3"
    assert m["Csignedmtpacc"].startswith("+")
    assert m["Nsigned"] == "3"


def test_collapse_is_read_from_grpo_step_10_and_the_last_step(tmp_path):
    m = analyse(tmp_path, campaign(tmp_path, grpo_end=0.10), "--lora")
    assert m["Collapsenseeds"] == "3" and m["Collapsencollapsed"] == "3"
    assert m["Collapseverdict"] == "collapses"


def test_no_collapse_is_said_so(tmp_path):
    m = analyse(tmp_path, campaign(tmp_path, grpo_end=0.45), "--lora")
    assert m["Collapsencollapsed"] == "0"
    assert m["Collapseverdict"] == "does not collapse"


def test_without_the_flag_nothing_changes(tmp_path):
    m = analyse(tmp_path, campaign(tmp_path))
    assert "Nmtp" not in m and "Collapseverdict" not in m


def test_verdict_words_are_emitted(tmp_path):
    m = analyse(tmp_path, campaign(tmp_path), "--lora")
    for k in ("Dissocverdict", "Matchcensored", "Collapseverdict"):
        assert k in m, k


def test_acc_key_reads_a_math500_backbone(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    tag = "Llama-3.2-3B-Instruct"
    for arm, acc in (("grpo", 0.30), ("steer", 0.29), ("signed", 0.32)):
        name = NAMES[arm].format(t=tag, s=1)
        body = [f"step:{k} - global_seqlen: 1"
                + (f" - val-core/math500/acc/mean@1:{acc}" if k % 10 == 0 else "")
                for k in range(1, 151)]
        (logs / f"train-lora-{name}.log").write_text("\n".join(body) + "\n")
    mac = tmp_path / "b.tex"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/analyze_seeds.py"),
                        "--logs", str(logs), "--out", str(tmp_path / "o"), "--tex-macros", str(mac),
                        "--run-prefix", "lora-", "--model-tag", tag, "--steps", "150",
                        "--plateau", "40:150", "--acc-key", "val-core/math500/acc/mean@1",
                        "--macro-prefix", "Bllama"], capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr + r.stdout
    m = {x.group(1): x.group(2) for x in
         re.finditer(r"\\providecommand\{\\([A-Za-z]+)\}\{([^}]*)\}", mac.read_text())}
    assert m["BllamaRsignedacc"] == ".3200"
    assert m["BllamaCsignedgrpoacc"] == "+.0200"
