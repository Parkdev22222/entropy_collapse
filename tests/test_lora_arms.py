"""The LoRA campaign's arm table (run/_lora_arms.sh) and LoRA defaults."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sh(script: str, env_prefix: str = "") -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", f"cd {ROOT}; {env_prefix} . run/_arms.sh; . run/_lora_arms.sh; {script}"],
        capture_output=True, text=True)


def lines(script: str) -> list[str]:
    r = sh(script)
    assert r.returncode == 0, r.stderr
    return [x for x in r.stdout.splitlines() if x]


ALL = lines("echo $LORA_CORE_ARMS $LORA_FOLLOWUP_ARMS | tr ' ' '\\n'")


def test_every_arm_has_a_spec_and_a_prefixed_unique_name():
    names = []
    for a in ALL:
        assert sh(f"lora_arm_spec {a}").returncode == 0, a
        n = lines(f"lora_run_name {a} 1")[0]
        assert n.startswith("lora-"), n
        names.append(n)
    assert len(names) == len(set(names))


def test_method_arms_read_the_realised_entropy_and_only_mtp_needs_heads():
    for a in ALL:
        spec = lines(f"lora_arm_spec {a}")[0]
        if spec.startswith("tree"):
            want = "mtp" if a == "mtp" else "oracle"
            assert f"STEERF_FORECAST={want}" in spec, (a, spec)
        needs = sh(f"lora_needs_heads {a}").returncode == 0
        assert needs == (a == "mtp"), a


def test_plan_is_seed_major_then_followups():
    plan = lines("lora_plan")
    core = [p for p in plan if p.startswith("core:")]
    fol = [p for p in plan if p.startswith("followups:")]
    assert len(core) == 6 * 3 and len(fol) == 10
    assert plan == core + fol
    seeds = [p.split(":")[2] for p in core]
    assert seeds == sorted(seeds, key=int)          # whole seeds, in order
    assert all(p.endswith(":1") for p in fol)


def test_shards_partition_the_plan():
    plan = lines("lora_plan")
    got = []
    for i in range(3):
        got += lines(f"lora_plan | lora_shard {i} 3")
    assert sorted(got) == sorted(plan)


def _defaults(env: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", f"cd {ROOT}; {env} . run/_lora_defaults.sh; "
                       "printf '%s\\n' \"${LORA_ARGS[@]}\"; echo SAVE=$SAVE_CONTENTS"],
        capture_output=True, text=True)


def test_lora_is_the_default_and_full_ft_must_be_asked_for():
    r = _defaults("")
    assert r.returncode == 0
    for kv in ("lora_rank=64", "lora_alpha=32", "target_modules=all-linear",
               "optim.lr=1e-5", "rollout.load_format=safetensors"):
        assert kv in r.stdout, kv
    assert "SAVE=['model','optimizer','extra']" in r.stdout
    full = _defaults("FULL_FT=1")
    assert "lora_rank" not in full.stdout
    assert _defaults("LORA_RANK=0").returncode == 2


def test_launchers_append_lora_args_before_the_command_line():
    for f in ("run/run_grpo.sh", "run/run_steerf.sh"):
        s = (ROOT / f).read_text()
        assert '. "${SCRIPT_DIR}/_lora_defaults.sh"' in s, f
        assert "LORA_ARGS[@]" in s, f
    steerf = (ROOT / "run/run_steerf.sh").read_text()
    assert steerf.index('"${LORA_ARGS[@]}"') < steerf.rindex('"$@"')
    grpo = (ROOT / "run/run_grpo.sh").read_text()
    assert grpo.index('ARGS+=("${LORA_ARGS[@]}")') < grpo.index('main_ppo "${ARGS[@]}" "$@"')


def test_backbone_plan_is_backbone_major_and_carries_the_tag():
    plan = lines("lora_plan backbones")
    assert plan == [f"backbones:{a}:1:{b}"
                    for b in ("Qwen2.5-Math-7B", "Llama-3.2-3B-Instruct")
                    for a in ("grpo", "steer", "signed")]


def test_entering_a_backbone_replaces_the_1_5b_profile():
    out = lines("( lora_enter_backbone Qwen2.5-Math-7B >/dev/null; "
                "echo $MODEL_PATH; lora_run_name signed 1 ); "
                "( lora_enter_backbone Llama-3.2-3B-Instruct >/dev/null; "
                "echo $MODEL_PATH $VAL_PARQUET $BEST_METRIC_KEY )")
    assert out[0] == "Qwen/Qwen2.5-Math-7B"
    assert out[1] == "lora-steer-f-Qwen2.5-Math-7B-s1-tree-rollout"
    assert out[2].split() == ["meta-llama/Llama-3.2-3B-Instruct", "datasets/math500.parquet",
                              "val-core/math500/acc/mean@1"]
