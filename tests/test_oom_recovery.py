"""A run that trained for hours and then died needs a different question asked.

diagnose_startup_failure answers "did it ever reach step 1", which is the wrong
question for a CUDA OOM at step 47 -- and until 2026-09-15 the queues asked no
other, so such a run printed nothing and the next pass re-ran it into the same
wall. These tests pin the classification and, more importantly, pin the two
remedies that must NEVER be suggested:

  ppo_micro_batch_size_per_gpu is STEER's min-max pool. Shrinking it fixes the
  OOM and changes the method, which is worse than the crash.

  expandable_segments:True is what the CUDA error itself recommends, and vLLM's
  sleep mode asserts against it at engine construction.
"""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE_ENV = {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": "/tmp"}

OOM_TAIL = """  File "/workspace/entropy_collapse/verl/workers/fsdp_workers.py", line 623, in update_actor
    metrics, clip_positions = self.actor.update_policy(data=data)
  File "/workspace/entropy_collapse/verl/workers/actor/dp_actor.py", line 862, in update_policy
    loss.backward()
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.81 GiB. GPU 0 has a total capacity of 79.25 GiB of which 2.39 GiB is free. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.
"""
LOGP_TAIL = """  File "/workspace/entropy_collapse/verl/workers/fsdp_workers.py", line 500, in compute_log_prob
    output = self.actor.compute_log_prob(data=data)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 8.03 GiB.
"""


def arms(snippet: str, **env):
    e = dict(BASE_ENV)
    e.update({k: str(v) for k, v in env.items()})
    return subprocess.run(["bash", "-c", f'. "{ROOT}/run/_arms.sh"\n{snippet}'],
                          capture_output=True, text=True, cwd=str(ROOT), env=e)


def make_log(tmp_path, name, last_step, tail=""):
    body = "".join(f"step:{i} - global_seqlen: 100\n" for i in range(1, last_step + 1))
    f = tmp_path / name
    f.write_text(body + tail)
    return f


def diagnose(tmp_path, log, steps=110):
    return arms(f'diagnose_run_failure "{log}" "an arm" {steps}; echo "RC=$?"')


def recommended(out):
    """Only what the diagnosis PROPOSES.

    The block above it quotes the log's last lines, and the CUDA error there
    recommends expandable_segments itself -- that is the evidence, not our
    advice, and the "NOT safe" section below exists to rebut it.
    """
    return out.split("What is safe to change")[1].split("What is NOT safe")[0]


# ------------------------------------------------------------ classification
def test_a_run_that_never_started_is_a_startup_failure(tmp_path):
    log = make_log(tmp_path, "train-a.log", 0, "Traceback: something\n")
    p = diagnose(tmp_path, log)
    assert "STARTUP FAILURE" in p.stdout
    assert "RC=1" in p.stdout


def test_an_oom_is_named_with_the_step_it_reached(tmp_path):
    log = make_log(tmp_path, "train-b.log", 47, OOM_TAIL)
    p = diagnose(tmp_path, log)
    assert "CUDA OOM at step 47/110" in p.stdout
    assert "STARTUP FAILURE" not in p.stdout
    assert "RC=2" in p.stdout, "the queue keys its retry on rc 2"


def test_a_plain_crash_is_neither(tmp_path):
    log = make_log(tmp_path, "train-c.log", 63, "RuntimeError: NCCL timeout\n")
    p = diagnose(tmp_path, log)
    assert "died at step 63/110" in p.stdout
    assert "OOM" not in p.stdout
    assert "RC=1" in p.stdout


# ------------------------------------------------------- the forbidden fixes
def test_the_oom_advice_never_suggests_shrinking_the_min_max_pool(tmp_path):
    """Shrinking ppo_micro_batch_size_per_gpu fixes the OOM and breaks the arm."""
    log = make_log(tmp_path, "train-d.log", 47, OOM_TAIL)
    out = diagnose(tmp_path, log).stdout
    assert "ppo_micro_batch_size_per_gpu" not in recommended(out), \
        "the min-max pool must never appear as a remedy"
    assert "ppo_micro_batch_size_per_gpu" in out, "it must appear as a warning"


def test_the_oom_advice_never_suggests_expandable_segments(tmp_path):
    """The CUDA error recommends it; vLLM sleep mode dies on it."""
    log = make_log(tmp_path, "train-e.log", 47, OOM_TAIL)
    out = diagnose(tmp_path, log).stdout
    assert "expandable_segments" not in recommended(out)
    # The log tail we quote carries the error's own suggestion, so the rebuttal
    # has to be present and has to come after it.
    assert out.index("147851") > out.index("PYTORCH_CUDA_ALLOC_CONF=expandable_segments")


def test_the_oom_advice_offers_offload(tmp_path):
    log = make_log(tmp_path, "train-f.log", 47, OOM_TAIL)
    out = diagnose(tmp_path, log).stdout
    assert "OFFLOAD=1" in out
    assert "not bit-identical" in out, "a stack change must be recorded, not hidden"


def test_the_two_oom_sites_get_different_advice(tmp_path):
    """LOGP_MBS is free to shrink; it is also irrelevant to a backward-pass OOM."""
    upd = diagnose(tmp_path, make_log(tmp_path, "train-g.log", 47, OOM_TAIL)).stdout
    lgp = diagnose(tmp_path, make_log(tmp_path, "train-h.log", 4, LOGP_TAIL)).stdout
    assert "in update_policy" in upd and "LOGP_MBS" not in recommended(upd)
    assert "in compute_log_prob" in lgp and "LOGP_MBS=2" in recommended(lgp)


def test_it_does_not_start_ray_for_a_run_that_trained(tmp_path):
    """env_preflight costs ~20 s and proves nothing here -- the run reached step 47."""
    stub = tmp_path / "bin"
    stub.mkdir()
    counter = tmp_path / "calls"
    (stub / "python3").write_text(f'#!/bin/sh\necho x >> "{counter}"\nexit 0\n')
    (stub / "python3").chmod(0o755)
    log = make_log(tmp_path, "train-i.log", 47, OOM_TAIL)
    arms(f'diagnose_run_failure "{log}" "an arm" 110',
         PATH=f"{stub}:/usr/bin:/bin")
    assert not counter.exists(), "no interpreter should have been launched"


# ------------------------------------------------------------- queue wiring
QUEUES = ["run_campaign.sh", "run_followups.sh", "run_backbones.sh"]


@pytest.mark.parametrize("q", QUEUES)
def test_every_queue_diagnoses_and_retries(q):
    body = (ROOT / "run" / q).read_text()
    assert "diagnose_run_failure" in body
    assert "OFFLOAD=1" in body, f"{q} must retry an OOM with the one safe lever"
    assert body.count("OFFLOAD=1 launch_") == 1, \
        f"{q} must retry exactly once, not loop"


@pytest.mark.parametrize("q", QUEUES)
def test_every_queue_writes_a_resumable_checkpoint(q):
    body = (ROOT / "run" / q).read_text()
    assert "'hf_model','model','optimizer','extra'" in body, \
        f"{q} without optimizer state cannot resume, whatever resume_mode says"
    assert "RESUME_MODE=auto" in body
    assert "MAX_CKPT_KEEP=1" in body, "peak disk must stay at one checkpoint"
    assert re.search(r"MIN_FREE_GB=\$\{MIN_FREE_GB:-30\}", body), \
        f"{q} must reserve room for a ~25 GiB checkpoint"


@pytest.mark.parametrize("q", QUEUES)
def test_a_queue_passes_resume_only_when_a_checkpoint_exists(q):
    body = (ROOT / "run" / q).read_text()
    assert '[ -d "${CKPT_ROOT}/${rn}" ] && resume=1' in body
    assert "${resume:+RESUME=1}" in body, \
        "the launchers refuse an existing checkpoint dir unless told this is a resume"


@pytest.mark.parametrize("q", QUEUES)
def test_no_queue_still_claims_a_resume_it_cannot_do(q):
    body = (ROOT / "run" / q).read_text()
    assert "so it can be resumed, moving on" not in body


def test_the_campaign_launcher_is_defined_once():
    """The retry must re-run what failed, not a second copy that drifts."""
    body = (ROOT / "run" / "run_campaign.sh").read_text()
    assert body.count("bash run/run_grpo.sh") == 1
    assert body.count("bash run/run_steerf.sh") == 1
    assert body.count("bash run/run_uniform_ablation.sh") == 1


# ---------------------------------------------------------------- reporting
def test_analyze_seeds_reports_which_stack_a_seed_ran_on():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from analyze_seeds import offloaded
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        gpu = Path(d) / "gpu.log"
        gpu.write_text("{'param_offload': False, 'optimizer_offload': False}\n")
        cpu = Path(d) / "cpu.log"
        cpu.write_text("{'param_offload': True}\n")
        none = Path(d) / "none.log"
        none.write_text("nothing here\n")
        assert offloaded(gpu) == "gpu"
        assert offloaded(cpu) == "cpu"
        assert offloaded(none) == "?"
