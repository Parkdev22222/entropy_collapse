"""The evaluation queue: name the runs, and do not take the GPUs from a trainer.

Two failures this pins, both silent.

The queue built an ``arms x seeds`` cross product, which assumes every arm
finished the same seeds.  A campaign split across boxes does not leave a
rectangle, and there was no way to ask for one arm at one seed and another at a
different one -- the shape a half-lost campaign actually has.

And it was the one queue with no busy guard.  The three training queues have
refused to start on a busy box since September; this one is the one most likely
to be run by hand *while* a campaign is mid-flight, because checkpoints only
become interesting once runs finish.  Starting there takes the GPUs out from
under the trainer, which for a tree arm is up to sixteen hours of work.
"""
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BANNER = "### run_steerf.sh  run_name={run}  seed={seed}\n"
STEP = "step:{n} - global_seqlen/min:1.0 - actor/entropy:0.1\n"


def train_log(log_dir: Path, run: str, last_step: int) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    body = "".join(STEP.format(n=n) for n in range(1, last_step + 1))
    (log_dir / f"train-{run}.log").write_text(body)


def queue(tmp_path: Path, env: dict | None = None, dry: str = "1"):
    e = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": "/tmp",
        "DRY": dry,
        "LOG_DIR": str(tmp_path / "logs"),
        "CKPT_ROOT": str(tmp_path / "ckpt"),
        **(env or {}),
    }
    return subprocess.run(["bash", str(ROOT / "run" / "run_eval_all.sh")],
                          capture_output=True, text=True, cwd=str(ROOT), env=e)


def finished_seed1(tmp_path: Path) -> None:
    """All five arms of seed 1, each at the final step."""
    logs = tmp_path / "logs"
    for run in ("grpo-Qwen2.5-Math-1.5B-s1",
                "steer-Qwen2.5-Math-1.5B-s1",
                "steer-f-Qwen2.5-Math-1.5B-s1-tree-rollout",
                "steer-f-Qwen2.5-Math-1.5B-s1-tree-rollout-uniform",
                "steer-f-Qwen2.5-Math-1.5B-s1-tree-rollout-permuted"):
        train_log(logs, run, 110)


# --- naming the runs --------------------------------------------------------

def test_named_pairs_queue_in_order(tmp_path):
    finished_seed1(tmp_path)
    p = queue(tmp_path, {"EVAL_RUNS": "grpo:1 signed:1"})
    queued = [ln.split()[1] for ln in p.stdout.splitlines() if ln.startswith("  QUEUE")]
    assert queued == ["grpo", "signed"], p.stdout


def test_pairs_need_not_share_a_seed(tmp_path):
    """The shape the cross product could not express."""
    logs = tmp_path / "logs"
    train_log(logs, "grpo-Qwen2.5-Math-1.5B-s1", 110)
    train_log(logs, "steer-Qwen2.5-Math-1.5B-s2", 110)
    p = queue(tmp_path, {"EVAL_RUNS": "grpo:1 steer:2"})
    assert "2 eval(s) queued" in p.stdout, p.stdout


def test_commas_group_the_printout(tmp_path):
    finished_seed1(tmp_path)
    p = queue(tmp_path, {"EVAL_RUNS": "grpo:1 steer:1, uniform:1 permuted:1"})
    assert "-- group 1" in p.stdout and "-- group 2" in p.stdout, p.stdout
    assert "4 eval(s) queued" in p.stdout, p.stdout


def test_named_pairs_replace_the_cross_product(tmp_path):
    """SEEDS must not smuggle extra runs in beside EVAL_RUNS."""
    finished_seed1(tmp_path)
    p = queue(tmp_path, {"EVAL_RUNS": "grpo:1", "SEEDS": "1 2 3 4 5"})
    assert "1 eval(s) queued" in p.stdout, p.stdout


def test_a_malformed_pair_is_fatal(tmp_path):
    """Not silently dropped: a typo that shrinks the queue is the whole hazard."""
    finished_seed1(tmp_path)
    p = queue(tmp_path, {"EVAL_RUNS": "grpo:1 steer"})
    assert p.returncode == 2
    assert "is not <arm>:<seed>" in p.stderr, p.stderr


def test_an_unfinished_run_is_skipped_with_its_reason(tmp_path):
    logs = tmp_path / "logs"
    train_log(logs, "grpo-Qwen2.5-Math-1.5B-s1", 110)
    train_log(logs, "steer-Qwen2.5-Math-1.5B-s1", 40)      # died early
    p = queue(tmp_path, {"EVAL_RUNS": "grpo:1 steer:1"})
    assert "1 eval(s) queued" in p.stdout, p.stdout
    assert "never reached step 110" in p.stdout, p.stdout


# --- not taking the box from a trainer --------------------------------------

def busy_marker(tmp_path: Path):
    """A live process whose command line a custom BUSY_RE matches."""
    marker = tmp_path / "pretend_trainer_xyzzy"
    marker.write_text("#!/bin/sh\nsleep 60\n")
    marker.chmod(0o755)
    return subprocess.Popen(["/bin/sh", str(marker)])


def test_the_queue_refuses_while_a_trainer_holds_the_box(tmp_path):
    finished_seed1(tmp_path)
    proc = busy_marker(tmp_path)
    try:
        time.sleep(0.3)
        p = queue(tmp_path, {"EVAL_RUNS": "grpo:1",
                             "BUSY_RE": "[p]retend_trainer_xyzzy"}, dry="0")
        assert p.returncode == 2, (p.stdout, p.stderr)
        assert "training process is already running" in p.stderr, p.stderr
        assert "WAIT=1" in p.stderr, p.stderr
    finally:
        proc.kill()
        proc.wait()


def test_a_dry_run_still_works_on_a_busy_box(tmp_path):
    """Planning while a campaign runs is the normal case, not a mistake."""
    finished_seed1(tmp_path)
    proc = busy_marker(tmp_path)
    try:
        time.sleep(0.3)
        p = queue(tmp_path, {"EVAL_RUNS": "grpo:1",
                             "BUSY_RE": "[p]retend_trainer_xyzzy"})
        assert p.returncode == 0, (p.stdout, p.stderr)
        assert "1 eval(s) queued" in p.stdout, p.stdout
    finally:
        proc.kill()
        proc.wait()


def test_an_idle_box_gets_past_the_busy_guard(tmp_path):
    """The guard must not be the thing that stops every evaluation."""
    finished_seed1(tmp_path)
    p = queue(tmp_path, {"EVAL_RUNS": "grpo:1",
                         "BUSY_RE": "[n]othing_matches_this_xyzzy"}, dry="0")
    assert "training process is already running" not in p.stderr, p.stderr


# --- the checkpoint has to hold weights -------------------------------------

def test_checkpoint_selection_asks_for_weights():
    """fsdp_checkpoint_manager writes huggingface/ with config and tokenizer
    always and the weights only when hf_model is in save_contents, so the
    directory existing says nothing. Regression detector for the eval path,
    which tested exactly that until 2026-09-23."""
    body = (ROOT / "run" / "run_eval_all.sh").read_text()
    picker = body[body.index("resolve_ckpt ()"):body.index("# ------------------------------------------------------------------- run")]
    assert "hf_weights_present" in picker
    assert '[ -d "${CKPT_ROOT}/${cand}/global_step_${d}/actor/huggingface" ]' not in picker


# --- the checkpoint may be in either repo -----------------------------------

def test_the_hub_search_covers_every_repo_and_downloads_from_the_one_it_found():
    """The campaign's checkpoints are split across two Hub repos.

    resolve_ckpt read a single REPO, so every arm whose checkpoint lived in the
    other one resolved to nothing and was reported exactly like a run that had
    never been trained. Two things have to hold: the probe iterates EVAL_REPOS,
    and the download uses the repo that answered rather than the variable that
    started the search -- fetching from the wrong one of two repos is the
    failure this replaced, wearing a different hat.
    """
    body = (ROOT / "run" / "run_eval_all.sh").read_text()
    picker = body[body.index("resolve_ckpt ()"):
                  body.index("# ------------------------------------------------------------------- run")]
    assert "for repo in ${EVAL_REPOS}; do" in picker
    assert 'download "${found}"' in picker, "the download must use the repo that answered"
    assert 'download "${REPO}"' not in picker
    # EVAL_REPOS defaults to REPO, so no existing invocation changes meaning.
    assert "EVAL_REPOS=${EVAL_REPOS:-${REPO}}" in body
    # REPO keeps its single-valued upload meaning in the queues that upload.
    for q in ("run_campaign.sh", "run_backbones.sh"):
        assert "EVAL_REPOS" not in (ROOT / "run" / q).read_text(), q


# --- DRY says where each checkpoint would come from -------------------------

def test_dry_reports_local_hub_and_missing_apart(tmp_path):
    """DRY=1 printed the queue and not whether the queue could run.

    resolve_ckpt reports "no checkpoint" identically for a run that was never
    trained, a run whose weights only exist on another box, and a directory that
    exists and holds no model -- and it reports it one arm at a time, an hour
    into the queue. The three are different problems with different fixes, so
    the probe names them apart before any GPU time is spent.
    """
    finished_seed1(tmp_path)
    ck = tmp_path / "ckpt"
    # weights present
    d = ck / "steer-f-Qwen2.5-Math-1.5B-s1-tree-rollout/global_step_100/actor/huggingface"
    d.mkdir(parents=True)
    (d / "model.safetensors").write_text("")
    # the directory exists and holds no model: config and tokenizer are written
    # whatever save_contents says, the weights only when hf_model is in it
    h = ck / "steer-f-Qwen2.5-Math-1.5B-s1-tree-rollout-uniform/global_step_90/actor/huggingface"
    h.mkdir(parents=True)
    (h / "config.json").write_text("{}")

    p = queue(tmp_path, {"EVAL_RUNS": "signed:1 uniform:1 permuted:1"})
    assert p.returncode == 0, p.stderr
    out = p.stdout
    assert "LOCAL  global_step_100" in out, out
    assert "WITHOUT WEIGHTS" in out, out
    # permuted has a finished log and nothing on disk: absent, not hollow
    perm = [l for l in out.splitlines() if l.strip().startswith("permuted:1")][0]
    assert "MISSING" in perm and "WITHOUT WEIGHTS" not in perm, perm
    assert "2 of 3 run(s) have no usable checkpoint" in out, out


def test_the_probe_does_not_download(tmp_path):
    """It is a report, not a fetch: DRY=1 must not touch the stage directory."""
    finished_seed1(tmp_path)
    p = queue(tmp_path, {"EVAL_RUNS": "signed:1"})
    assert p.returncode == 0, p.stderr
    assert "hf download" not in p.stdout
    body = (ROOT / "run" / "run_eval_all.sh").read_text()
    probe = body[body.index("probe_ckpt ()"):body.index("add_run ()")]
    for forbidden in ("HF_CLI", "rm -rf", "mkdir"):
        assert forbidden not in probe, f"probe_ckpt must not {forbidden}"


# --- the box has to be able to make a CUDA context ---------------------------

def fake_nvidia_smi(tmp_path: Path, holder: bool) -> str:
    """A stub nvidia-smi on PATH. Returns the bin dir to prepend."""
    b = tmp_path / "fakebin"
    b.mkdir(exist_ok=True)
    body = ('#!/bin/sh\ncase "$*" in\n  *query-compute-apps*) '
            + ('echo "999999, 41234 MiB"' if holder else ':')
            + ' ;;\n  *) exit 0 ;;\nesac\n')
    f = b / "nvidia-smi"
    f.write_text(body)
    f.chmod(0o755)
    return str(b)


def test_held_vram_is_refused_before_any_benchmark(tmp_path):
    """The failure this gate exists for, and the one it had.

    is_busy only greps main_ppo off a command line. A run that died leaves
    ray::WorkerDict and vLLM engine processes holding CUDA contexts, which it
    does not match, so the queue called the box idle and started -- and NCCL
    died at FSDP init with "Cuda failure 401", before a single benchmark. On
    2026-09-28 that happened twice in one invocation, the second pass dying on
    the first pass's leftovers because nothing waited in between.
    """
    finished_seed1(tmp_path)
    bin_dir = fake_nvidia_smi(tmp_path, holder=True)
    p = queue(tmp_path, {"EVAL_RUNS": "signed:1", "GPU_WAIT": "0",
                         "PATH": bin_dir + ":" + os.environ.get("PATH", "/usr/bin:/bin")},
              dry="0")
    assert p.returncode == 2, (p.returncode, p.stdout[-500:], p.stderr[-500:])
    assert "VRAM is still held" in p.stderr, p.stderr
    # and it says so before spending anything on a checkpoint
    assert "MISSING checkpoint" not in p.stdout


def test_the_vram_refusal_has_an_escape_hatch(tmp_path):
    finished_seed1(tmp_path)
    bin_dir = fake_nvidia_smi(tmp_path, holder=True)
    p = queue(tmp_path, {"EVAL_RUNS": "signed:1", "GPU_WAIT": "0",
                         "ALLOW_BUSY_GPUS": "1",
                         "PATH": bin_dir + ":" + os.environ.get("PATH", "/usr/bin:/bin")},
              dry="0")
    assert "ALLOW_BUSY_GPUS=1" in p.stdout, p.stdout[-400:]
    assert "VRAM is still held" not in p.stderr


def test_an_idle_box_passes_the_vram_gate(tmp_path):
    """Regression: the gate must not refuse a box that is actually free."""
    finished_seed1(tmp_path)
    bin_dir = fake_nvidia_smi(tmp_path, holder=False)
    p = queue(tmp_path, {"EVAL_RUNS": "signed:1", "GPU_WAIT": "0",
                         "PATH": bin_dir + ":" + os.environ.get("PATH", "/usr/bin:/bin")})
    assert p.returncode == 0, p.stderr
    assert "GPUs: no other process holds VRAM" in p.stdout, p.stdout[-400:]


def test_the_queue_waits_between_runs_like_the_training_queues(tmp_path):
    """Each eval is two verl passes; a dead pass leaves its workers holding VRAM.

    run_campaign.sh and run_followups.sh call await_gpus both before the queue
    and between runs. This one called it in neither place.
    """
    body = (ROOT / "run" / "run_eval_all.sh").read_text()
    assert body.count("await_gpus") >= 2, "await_gpus belongs in the guard AND the loop"
    run_loop = body[body.index("for item in \"${QUEUE[@]}\""):]
    assert "await_gpus" in run_loop, "nothing waits between runs"
    # the fixed sleep it replaced was a guess at the same wait
    assert "sleep 30" not in body
