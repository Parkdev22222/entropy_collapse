# STEER-F — STEER with Future-entropy Forecasting

STEER reweights each token's RLVR learning signal by a first-order prediction
of how the update will change **that token's** entropy. That estimate is
myopic: it sees the local term of the trajectory entropy and misses the
visitation term, where a policy that concentrates at a branch point silently
deletes every future state down the abandoned branch.

STEER-F adds that missing term. MTP heads forecast the entropy that lies ahead
of each decision, the forecast becomes a branch score `A_H`, and the score
extends `Ω` into `Ω̃`:

```
Ω̃   = norm(Ω) + λ · norm( Δlogπ̂ · clip(A_H, -c, c) )
Δlogπ̂ = η · w · (1 - π)
A_H   = H_togo(s_t ⊕ y_t) - H̄_togo(s_t)
H_togo = Σ_{k=1..κ} γ_H^k · H( p_MTP(y_{t+k} | s) )
```

`λ = 0` is bit-identical to stock STEER — enforced by a test against a verbatim
copy of upstream's function, not by inspection.

## Status

| Phase | Deliverable | State |
|---|---|---|
| 0 | `docs/steer_code_map.md` | done |
| 0 | STEER small-model reproduction | **not run — needs a GPU** |
| 1 | `steer_f/mtp_heads.py`, warm-up + validation scripts | code done, unrun |
| 2 | `Ω̃` integration, verl patches, λ=0 equivalence test | code done and tested; RL unrun |
| 3 | `scripts/phase3_port_model.py` | done |
| 4 | Ablations | knobs exposed; no results |

Everything that can be checked without a GPU is checked: **213 tests pass**,
and the patches apply, revert, and reproduce upstream exactly at `λ = 0`.
No training has been run, so there are no experimental results yet, and no
gate (G0–G3) has been cleared. `docs/experiment_log.md` records the state and
the exact commands to continue.

## Quickstart

```bash
pip install -r requirements.txt
bash scripts/setup_steer.sh        # clone STEER at its pinned commit, apply patches
python -m pytest tests/ -q         # 213 tests
```

Phase 0 baselines (needs GPUs):

```bash
ARM=grpo  ./run/run_steerf_small.sh     # GRPO baseline
ARM=steer ./run/run_steerf_small.sh     # STEER reproduction  -> gate G0
```

Phase 1, forecast validation:

```bash
python scripts/phase1_warmup_heads.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --rollouts rollout_data/.../rollouts.jsonl --out checkpoints/mtp_heads.pt
python scripts/phase1_validate.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --heads checkpoints/mtp_heads.pt --problems third_party/STEER/datasets/math500.parquet \
    --out docs/phase1_results.json --calibrate     # -> gate G1
```

Phase 2, the λ sweep:

```bash
for lam in 0 0.25 0.5 1.0; do
  ARM=steerf STEERF_LAM=$lam STEERF_KAPPA=4 STEERF_GAMMA_H=0.85 ./run/run_steerf_small.sh
done                                               # -> gate G2
```

## Layout

```
steer_f/
  mtp_heads.py         parallel MTP heads; forecast_entropy never builds [K,B,T,V]
  entropy_forecast.py  H_togo, per-head calibration, sibling / group baselines
  omega_tilde.py       Ω̃ and the drop-in replacement for compute_token_weights
  monitors.py          forecast-drift KL, branch entropy, λ decay controller
  validation.py        gate-G1 statistics (within-problem ρ, exact binomial)
  verl_integration.py  index alignment and batch-level A_H assembly
patches/               the only channel by which STEER-F edits verl
scripts/               setup, head warm-up, MC validation, family porting
run/                   training arms: grpo / steer / steerf
docs/                  code map, phase reports, running log
tests/                 213 tests, incl. a verbatim upstream copy for equivalence
```

## Two things worth knowing before reading the code

**The paper's band does not exist in the code.** STEER's implementation has no
`[ΔH_low, ΔH_high]` band and no discrete `α ∈ {γ, 1, 1/γ}`; it min-max rescales
`Ω` onto `[0.8, 1.2]` per micro-batch. Every design choice downstream follows
from the real mapping. `docs/steer_code_map.md` §3.

**Normalisation is RMS, not z-score.** Recentring changes STEER's mapping —
`symmetric` mode takes `abs()`, so zero is a meaningful pivot, and the
exponential mapping is not shift-invariant. Scaling without recentring is the
only normalisation that leaves all four `(mode, mapping)` combinations
untouched as `λ → 0`. `docs/steer_code_map.md` §3.3.

## Licence

Apache-2.0, matching STEER and verl. `tests/reference_steer.py` is an
unmodified excerpt of STEER, retained for the equivalence test.
