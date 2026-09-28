"""instrument_phase2.sh change 4: the second eval pass takes AVG1_N.

eval_steerf.sh lives on the pod and not on this branch, so the change is made
the way the other three are: check, apply with a .bak, re-check. What has to
hold is that the default is the original protocol, that another n gets a tag
of its own (the tag names the dump directory, and "avg32" is pass A's), and
that applying twice changes nothing the second time.
"""
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The shape of the pod's launcher around the two passes, with change 1 in place.
EVAL_SH = r'''#!/usr/bin/env bash
run_eval () {
    local test_files="$1"; local val_n="$2"; local tag="$3"
    echo "files=$test_files n=$val_n tag=$tag"
    : \
        ${VAL_DATA_DIR:+++trainer.validation_data_dir=${VAL_DATA_DIR}/${tag}} \
        trainer.val_only=True \
        trainer.resume_mode=disable
}
files_at32="A32"
files_at1="B1"
if [ "${CODE:-0}" = "1" ]; then
    run_eval "code" 4 "code-avg4"
else
    run_eval "$files_at32" 1 "avg32"
    run_eval "$files_at1"   1 "avg1"
fi
'''


def pod(tmp_path):
    r = tmp_path / "pod"
    (r / "run").mkdir(parents=True)
    (r / "verl/trainer/ppo").mkdir(parents=True)
    shutil.copy(ROOT / "run/instrument_phase2.sh", r / "run/")
    (r / "run/eval_steerf.sh").write_text(EVAL_SH)
    (r / "run/run_uniform_ablation.sh").write_text(
        '#!/bin/bash\nbash run_steerf.sh" "${ARGS[@]}" "$@" > /dev/null\n')
    (r / "verl/trainer/ppo/ray_trainer.py").write_text("dump_infos = {}\n")
    return r


def run(r, mode):
    return subprocess.run(["bash", str(r / "run/instrument_phase2.sh"), mode],
                          capture_output=True, text=True)


def passes(r, env):
    out = subprocess.run(["bash", str(r / "run/eval_steerf.sh")], capture_output=True,
                         text=True, env={"PATH": "/usr/bin:/bin", **env})
    return [l for l in out.stdout.splitlines() if l.startswith("files=")]


def test_check_reports_the_change_and_apply_makes_it_once(tmp_path):
    r = pod(tmp_path)
    c = run(r, "--check")
    assert c.returncode == 1 and "AVG1_N" in c.stdout, c.stdout
    a = run(r, "--apply")
    assert a.returncode == 0 and "second pass reads AVG1_N" in a.stdout, a.stdout + a.stderr
    assert list((r / "run").glob("eval_steerf.sh.bak.*")), "no backup was left"
    before = (r / "run/eval_steerf.sh").read_text()
    assert run(r, "--apply").returncode == 0
    assert (r / "run/eval_steerf.sh").read_text() == before, "a second apply changed the file"


def test_the_default_is_the_original_protocol(tmp_path):
    r = pod(tmp_path)
    run(r, "--apply")
    assert passes(r, {}) == ["files=A32 n=1 tag=avg32", "files=B1 n=1 tag=avg1"]


def test_another_n_gets_its_own_tag(tmp_path):
    """Pass A is tagged avg32; a second pass at 32 must not write there too."""
    r = pod(tmp_path)
    run(r, "--apply")
    assert passes(r, {"AVG1_N": "32"}) == ["files=A32 n=1 tag=avg32",
                                           "files=B1 n=32 tag=avg1x32"]
