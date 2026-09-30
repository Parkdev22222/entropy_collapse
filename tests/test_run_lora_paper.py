"""run/run_lora_paper.sh end to end against fake launchers: what it launches,
with which settings, and that a second invocation resumes instead of redoing."""
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FAKE = r'''#!/usr/bin/env bash
{ printf 'CALL %s' "$(basename "$0")"
  for v in SEED RUN_NAME STEPS KEEP_EVERY ARM STEERF_FORECAST STEERF_LAM TOKEN_WEIGHT_MIN MODEL_PATH \
           CUDA_VISIBLE_DEVICES RAY_STOP RESUME SAVE_BEST_ONLY SAVE_AFTER_OVERRIDE ROLLOUT_EAGER; do
      [ -n "${!v:-}" ] && printf ' %s=%s' "$v" "${!v}"
  done
  printf ' ARGS=%s\n' "$*"; } >> "${FAKE_CALLS}"
steps=${STEPS}
for a in "$@"; do case "$a" in ++trainer.total_training_steps=*) steps=${a#*=} ;; esac; done
line="step:${steps} - global_seqlen:1"
case "$(basename "$0")" in
    run_steerf.sh) echo "${line}" ;;                                  # the runner redirects stdout
    run_grpo.sh)   echo "${line}" > "${LOG}" ;;
    *)             echo "${line}" > "${LOG_DIR}/train-${RUN_NAME}.log" ;;
esac
d="${FAKE_CKPT}/${RUN_NAME}/global_step_${steps}/actor/lora_adapter"
mkdir -p "$d"; : > "$d/adapter_model.safetensors"
'''


def tree(tmp_path):
    root = tmp_path / "repo"
    (root / "run").mkdir(parents=True)
    for f in ("_arms.sh", "_lora_arms.sh", "run_lora_paper.sh", "eval_lora.sh"):
        shutil.copy(ROOT / "run" / f, root / "run" / f)
    for f in ("run_grpo.sh", "run_steerf.sh", "run_uniform_ablation.sh"):
        (root / "run" / f).write_text(FAKE)
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "nvidia-smi").write_text(
        '#!/bin/sh\ncase "$1" in -L) echo "GPU 0: x"; echo "GPU 1: y" ;; esac\n')
    (bin_ / "nvidia-smi").chmod(0o755)
    return root, bin_


def run(root, bin_, tmp_path, **env):
    e = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}",
             FAKE_CALLS=str(tmp_path / "calls.log"), FAKE_CKPT=str(root / "checkpoints/STEER-F"),
             N_GPUS="2", TP_SIZE="2", LORA_SEEDS="1 2", LORA_CORE_ARMS="grpo steer signed",
             LORA_FOLLOWUP_ARMS="wmin-steer xclip-signed grpo-long",
             LORA_STAGGER="0", LORA_POLL="1")
    e.update({"STAGES": "core followups", **env})
    return subprocess.run(["bash", str(root / "run/run_lora_paper.sh")],
                          capture_output=True, text=True, env=e, timeout=120)


def calls(tmp_path):
    p = tmp_path / "calls.log"
    return p.read_text().splitlines() if p.exists() else []


def test_tp4_launches_every_run_with_the_campaign_settings(tmp_path):
    root, bin_ = tree(tmp_path)
    r = run(root, bin_, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    c = calls(tmp_path)
    assert len(c) == 2 * 3 + 3
    assert all("KEEP_EVERY=50" in x and "SAVE_BEST_ONLY=False" in x
               and "SAVE_AFTER_OVERRIDE=0" in x for x in c)
    assert all("RUN_NAME=lora-" in x for x in c)
    signed = [x for x in c if "tree-rollout " in x + " " and "SEED=1" in x and "xclip" not in x]
    assert signed and "STEERF_FORECAST=oracle" in signed[0] and "ARM=signed" in signed[0]
    long_ = [x for x in c if "-long" in x][0]
    assert "STEPS=200" in long_ and "run_grpo.sh" in long_
    xclip = [x for x in c if "tree-rollout-xclip" in x][0]
    assert "clip_ratio_high=5" in xclip
    wmin = [x for x in c if "wmin08" in x][0]
    assert "TOKEN_WEIGHT_MIN=0.8" in wmin and "total_training_steps=150" in wmin
    assert all("STEPS=150" in x for x in c if "-long" not in x)
    assert r.stdout.count("[lora] OK") == 9


def test_second_invocation_resumes_nothing_left(tmp_path):
    root, bin_ = tree(tmp_path)
    run(root, bin_, tmp_path)
    n = len(calls(tmp_path))
    r = run(root, bin_, tmp_path)
    assert r.returncode == 0
    assert len(calls(tmp_path)) == n
    assert r.stdout.count("  done  ") == 9


def test_1gpu_pins_one_card_per_run_and_skips_ray_stop(tmp_path):
    root, bin_ = tree(tmp_path)
    r = run(root, bin_, tmp_path, TOPOLOGY="1gpu", STAGES="core")
    assert r.returncode == 0, r.stdout + r.stderr
    c = calls(tmp_path)
    assert len(c) == 6
    assert all("RAY_STOP=0" in x for x in c)
    assert {x.split("CUDA_VISIBLE_DEVICES=")[1].split()[0] for x in c} == {"0", "1"}


def test_shards_split_the_runs(tmp_path):
    root, bin_ = tree(tmp_path)
    run(root, bin_, tmp_path, SHARD="0/2")
    a = set(calls(tmp_path))
    (tmp_path / "calls.log").unlink()
    run(root, bin_, tmp_path, SHARD="1/2")
    b = set(calls(tmp_path))
    assert len(a) + len(b) == 9 and not (a & b)


def test_backbones_train_their_own_model_on_the_whole_box(tmp_path):
    root, bin_ = tree(tmp_path)
    r = run(root, bin_, tmp_path, STAGES="backbones", TOPOLOGY="1gpu")
    assert r.returncode == 0, r.stdout + r.stderr
    c = calls(tmp_path)
    assert len(c) == 6
    qwen = [x for x in c if "Qwen2.5-Math-7B" in x]
    llama = [x for x in c if "Llama-3.2-3B-Instruct" in x]
    assert len(qwen) == 3 and len(llama) == 3
    assert all("CUDA_VISIBLE_DEVICES" not in x for x in c)          # never pinned
    assert all("math500.parquet" in x and "math500/acc/mean@1" in x for x in llama)
    assert all("aime24.parquet" in x for x in qwen)
    # a second call finds them done
    n = len(c)
    run(root, bin_, tmp_path, STAGES="backbones")
    assert len(calls(tmp_path)) == n


def test_rollout_eager_reaches_every_launcher(tmp_path):
    root, bin_ = tree(tmp_path)
    r = run(root, bin_, tmp_path, ROLLOUT_EAGER="0")
    assert r.returncode == 0, r.stdout + r.stderr
    c = calls(tmp_path)
    assert c and all("ROLLOUT_EAGER=0" in x for x in c)
    assert "rollout_eager 0" in r.stdout


def test_campaign_settings_are_locked_across_invocations(tmp_path):
    root, bin_ = tree(tmp_path)
    settings = root / "logs/lora/campaign_settings"
    r = run(root, bin_, tmp_path, STAGES="core", LORA_CORE_ARMS="grpo")
    assert r.returncode == 0, r.stdout + r.stderr
    assert settings.read_text().split() == ["topology=tp4", "rollout_eager=1"]
    n = len(calls(tmp_path))
    for env in ({"TOPOLOGY": "1gpu"}, {"ROLLOUT_EAGER": "0"}):
        r = run(root, bin_, tmp_path, STAGES="core", LORA_CORE_ARMS="grpo steer", **env)
        assert r.returncode == 2 and "REFUSE" in r.stderr, env
        assert len(calls(tmp_path)) == n                  # nothing launched
    r = run(root, bin_, tmp_path, STAGES="core", LORA_CORE_ARMS="grpo steer",
            TOPOLOGY="1gpu", CAMPAIGN_SETTINGS_OVERRIDE="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(calls(tmp_path)) > n
    assert settings.read_text().split() == ["topology=tp4", "rollout_eager=1"]   # not rewritten


def test_backbones_lock_only_the_engine_setting(tmp_path):
    root, bin_ = tree(tmp_path)
    run(root, bin_, tmp_path, STAGES="core", LORA_CORE_ARMS="grpo", TOPOLOGY="1gpu")
    r = run(root, bin_, tmp_path, STAGES="backbones")      # tp4 default: topology not theirs
    assert r.returncode == 0, r.stdout + r.stderr
    r = run(root, bin_, tmp_path, STAGES="backbones", ROLLOUT_EAGER="0")
    assert r.returncode == 2


def test_dry_run_records_nothing(tmp_path):
    root, bin_ = tree(tmp_path)
    r = run(root, bin_, tmp_path, DRY="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert not (root / "logs/lora/campaign_settings").exists()
