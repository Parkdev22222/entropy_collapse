"""Entropy-cut tree rollout: where it cuts, that it stays on-policy, and its wiring."""
import importlib.util
import math
import os
import random
import subprocess
import sys
from collections import Counter
from itertools import product
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# steer_f/__init__ imports torch; tree_rollout itself does not.
_spec = importlib.util.spec_from_file_location("tree_rollout_ec", ROOT / "steer_f" / "tree_rollout.py")
tr = importlib.util.module_from_spec(_spec)
sys.modules["tree_rollout_ec"] = tr
_spec.loader.exec_module(tr)
SPMD = (ROOT / "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py").read_text()


def cfg(**kw):
    base = dict(n=8, response_length=200, depths=(10, 30, 60), factors=(2, 2, 2),
                cut_mode="entropy", cut_widths=(10, 20, 30), cut_topk=20)
    base.update(kw)
    return tr.TreeRolloutConfig(**base)


# ---------------------------------------------------------------- entropy
def test_topk_entropy_is_exact_with_full_mass_and_a_lower_bound_without():
    assert tr.topk_entropy([math.log(0.25)] * 4) == pytest.approx(math.log(4))
    assert tr.topk_entropy([0.0]) == pytest.approx(0.0)
    # top-2 of a uniform-over-4: the lumped residual (0.5) gives less than log 4
    h = tr.topk_entropy([math.log(0.25)] * 2)
    assert h < math.log(4)
    assert h == pytest.approx(-2 * 0.25 * math.log(0.25) - 0.5 * math.log(0.5))


# ---------------------------------------------------------------- config
@pytest.mark.parametrize("kw, msg", [
    (dict(cut_mode="max"), "cut_mode"),
    (dict(cut_widths=(10, 20)), "one window width per depth"),
    (dict(cut_topk=0), "cut_topk >= 1"),
    (dict(cut_quantile=1.0), "cut_quantile"),
    (dict(cut_widths=(10, 20, 200)), "deepest possible cut"),
    (dict(cut_mode="fixed"), "only applies"),
])
def test_config_rejects(kw, msg):
    with pytest.raises(ValueError, match=msg):
        cfg(**kw)


def test_config_defaults_stay_fixed_and_unmeasured():
    c = tr.TreeRolloutConfig(n=8, response_length=200, depths=(10, 30, 60), factors=(2, 2, 2))
    assert c.cut_mode == "fixed" and not c.measuring
    assert cfg().cut_gaps == (10, 20, 30)


# ---------------------------------------------------------------- scripted engine
class Scripted:
    """Emits token 7 forever; the entropy of a position is ``ent(prefix_len)``.

    The prefix length is the prompt's length, which the driver grows by the
    kept trunk, so the entropy is a property of the position, as a policy's is.
    """

    def __init__(self, ent, eos_at=None):
        self.ent, self.eos_at, self.calls = ent, eos_at, []

    def __call__(self, prompts, n, max_tokens, topk=0):
        self.calls.append((len(prompts), n, max_tokens, topk))
        limits = max_tokens if isinstance(max_tokens, list) else [max_tokens] * len(prompts)
        out = []
        for p, lim in zip(prompts, limits):
            group = []
            for _ in range(n):
                k = lim
                fin = False
                if self.eos_at is not None and len(p) - 1 <= self.eos_at < len(p) - 1 + lim:
                    k, fin = self.eos_at - (len(p) - 1) + 1, True
                ents = [self.ent(len(p) - 1 + i) for i in range(k)] if topk else None
                group.append(tr.TreeSample(token_ids=[7] * k, finished=fin, entropies=ents))
            out.append(group)
        return out


def test_cuts_at_the_first_crossing_inside_each_window():
    # response position t has entropy 1.0 at t in {14, 40, 75}, else 0.0
    eng = Scripted(lambda t: 1.0 if t in (14, 40, 75) else 0.0)
    r = tr.generate_tree(eng, [[0]], cfg(), tau=[0.5, 0.5, 0.5])
    # windows: [10,20] -> 14; [14+20, 14+40] = [34,54] -> 40; [40+30, 40+60] -> 75
    assert r.stats["cut_depth_mean"] == [14, 40, 75]
    assert r.stats["fallback_frac"] == 0.0
    assert r.stats["cut_entropy_mean"] == 1.0
    assert all(len(x) == 200 for x in r.responses[0])
    assert tr.sibling_support_stats([r.responses[0][:2]])["alive_positions"] > 0


def test_never_cuts_before_the_window_and_falls_back_to_its_end():
    # a crossing before the window (t=5) must be ignored; none inside -> end
    eng = Scripted(lambda t: 1.0 if t == 5 else 0.0)
    r = tr.generate_tree(eng, [[0]], cfg(), tau=[0.5, 0.5, 0.5])
    assert r.stats["cut_depth_mean"] == [20, 60, 120]
    assert r.stats["fallback_frac"] == 1.0
    # trunk stages drew gap + width + 1 tokens and asked for entropies
    assert [c[2:] for c in eng.calls[:3]] == [(21, 20), (41, 20), (61, 20)]
    # the last stage's budgets are per node and fill the response exactly
    assert eng.calls[3][2] == [80] * 4 and eng.calls[3][3] == 0
    assert all(len(x) == 200 for x in r.responses[0])


def test_wasted_tokens_are_what_the_cuts_threw_away():
    eng = Scripted(lambda t: 0.0)
    r = tr.generate_tree(eng, [[0]], cfg(), tau=[0.5, 0.5, 0.5])
    # each trunk drew one token past its cut: 1 + 2 + 4 nodes
    assert r.stats["wasted_tok_frac"] == pytest.approx(7 / (21 + 2 * 41 + 4 * 61 + 8 * 80))


def test_eos_inside_the_window_before_a_crossing_finishes_and_is_refilled():
    eng = Scripted(lambda t: 0.0, eos_at=15)  # inside level 0's window [10, 20]
    r = tr.generate_tree(eng, [[0], [0]], cfg(), tau=[0.5, 0.5, 0.5])
    assert all(len(g) == 8 for g in r.responses)
    assert r.stats["num_refilled"] == 2 * 7


def test_eos_after_a_crossing_is_discarded_and_the_children_redraw_it():
    eng = Scripted(lambda t: 1.0 if t == 12 else 0.0, eos_at=15)
    r = tr.generate_tree(eng, [[0]], cfg(), tau=[0.5, 0.5, 0.5])
    assert r.stats["cut_depth_mean"][0] == 12


def test_tau_comes_from_the_batch_first_then_from_the_previous_call():
    eng = Scripted(lambda t: (t % 10) / 10)
    c = cfg(cut_quantile=0.5)
    r1 = tr.generate_tree(eng, [[0]] * 4, c)
    assert r1.stats["tau_source"] == "batch"
    # level 0's window is positions 10..20: entropies .0 .1 ... .9 .0, median
    window0 = sorted([(t % 10) / 10 for t in range(10, 21)] * 4)
    assert r1.stats["next_tau"][0] == window0[int(0.5 * len(window0))]
    assert r1.stats["tau"][0] == r1.stats["next_tau"][0]
    r2 = tr.generate_tree(eng, [[0]] * 4, c, tau=[9.0, 9.0, 9.0])
    assert r2.stats["tau_source"] == "prev" and r2.stats["tau"] == [9.0, 9.0, 9.0]
    assert r2.stats["fallback_frac"] == 1.0


def test_fixed_mode_can_measure_without_moving_the_cuts():
    eng = Scripted(lambda t: 0.25)
    c = tr.TreeRolloutConfig(n=8, response_length=200, depths=(10, 30, 60), factors=(2, 2, 2), cut_topk=5)
    r = tr.generate_tree(eng, [[0]], c)
    assert r.stats["cut_depth_mean"] == [10, 30, 60]
    assert r.stats["cut_entropy_mean"] == 0.25
    assert "tau" not in r.stats
    assert [x[2] for x in eng.calls] == [11, 21, 31, 140]  # last stage: one int, as before


def test_fixed_mode_without_topk_never_passes_topk():
    calls = []

    def engine(prompts, n, max_tokens):  # an engine that predates entropies
        calls.append(max_tokens)
        return [[tr.TreeSample(token_ids=[1] * max_tokens, finished=False) for _ in range(n)] for _ in prompts]

    c = tr.TreeRolloutConfig(n=8, response_length=200, depths=(10, 30, 60), factors=(2, 2, 2))
    tr.generate_tree(engine, [[0]], c)
    assert calls == [10, 20, 30, 140]


# ---------------------------------------------------------------- on-policy
P1 = {0: 0.5, 1: 0.95}  # P(next = 1 | last token)


def _h(last):
    p = P1[last]
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


class Markov:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def __call__(self, prompts, n, max_tokens, topk=0):
        limits = max_tokens if isinstance(max_tokens, list) else [max_tokens] * len(prompts)
        out = []
        for p, lim in zip(prompts, limits):
            group = []
            for _ in range(n):
                last, toks, ents = p[-1], [], []
                for _ in range(lim):
                    ents.append(_h(last))
                    last = 1 if self.rng.random() < P1[last] else 0
                    toks.append(last)
                group.append(tr.TreeSample(token_ids=toks, finished=False, entropies=ents if topk else None))
            out.append(group)
        return out


def _exact(length):
    probs = {}
    for seq in product((0, 1), repeat=length):
        last, pr = 0, 1.0
        for x in seq:
            pr *= P1[last] if x == 1 else 1 - P1[last]
            last = x
        probs[seq] = pr
    return probs


def _tv(responses, length):
    counts = Counter(tuple(r) for g in responses for r in g)
    total = sum(counts.values())
    return 0.5 * sum(abs(counts.get(s, 0) / total - p) for s, p in _exact(length).items())


def _tree_tv(monkeypatch=None, rule=None, trees=12000):
    c = tr.TreeRolloutConfig(n=2, response_length=6, depths=(1,), factors=(2,),
                             cut_mode="entropy", cut_widths=(3,), cut_topk=1)
    if rule is not None:
        monkeypatch.setattr(tr, "_find_cut", rule)
    r = tr.generate_tree(Markov(0), [[0]] * trees, c, tau=[0.5])
    return _tv(r.responses, 6)


# 24k rollouts over 64 sequences: i.i.d. draws land at TV ~0.011 from the exact
# law, first-crossing trees at 0.011-0.013, window-maximum trees at 0.063-0.072
# (seeds 0-3).
def test_first_crossing_keeps_every_rollout_a_draw_from_the_policy():
    assert _tree_tv() < 0.025


def test_cutting_at_the_window_maximum_would_not(monkeypatch):
    def window_max(sample, gap, width, tau):
        ent = sample.entropies
        k = max(range(gap, gap + width + 1), key=lambda i: (ent[i], -i))
        return k, "cross"

    assert _tree_tv(monkeypatch, window_max) > 0.04


# ---------------------------------------------------------------- wiring
def test_vllm_engine_takes_per_prompt_budgets_and_topk_logprobs():
    tree = SPMD[SPMD.index("def _generate_tree"):SPMD.index("st = result.stats")]
    assert "def engine(prompt_token_ids, n, max_tokens, topk=0):" in tree
    assert 'overrides["logprobs"] = topk' in tree
    assert "self.sampling_params.clone()" in tree
    assert "tau=self._steerf_tree_tau" in tree
    assert "self._steerf_tree_tau = None" in SPMD
    assert 'self._steerf_tree_tau = st["next_tau"]' in SPMD
    assert "lp.rank <= topk" in SPMD
    for key in ("cut_mode", "cut_widths", "cut_quantile", "cut_topk"):
        assert f'"steerf_tree_{key}"' in SPMD


def _dry(env):
    e = dict(os.environ, DRY_RUN="1", FORCE_CONCURRENT="1", ARM="signed", STEERF_FORECAST="oracle",
             RUN_NAME="dry-entropy-cut-test", **env)
    return subprocess.run(["bash", "run/run_uniform_ablation.sh"], cwd=ROOT, env=e,
                          capture_output=True, text=True)


def test_launcher_fixed_command_line_is_unchanged():
    out = _dry({}).stdout
    assert "steerf_tree_depths=[64,192,384]" in out
    assert "steerf_tree_cut" not in out


def test_launcher_entropy_passes_the_cut_settings():
    out = _dry({"TREE_CUT": "entropy"}).stdout
    for frag in ("steerf_tree_cut_mode=entropy", "steerf_tree_cut_widths=[64,128,192]",
                 "steerf_tree_cut_quantile=0.9", "steerf_tree_cut_topk=20"):
        assert frag in out


@pytest.mark.parametrize("env", [{"TREE_CUT": "max"}, {"TREE_CUT": "entropy", "TREE_CUT_TOPK": "21"},
                                 {"TREE_CUT": "entropy", "TREE_CUT_TOPK": "0"}])
def test_launcher_rejects_bad_cut_settings(env):
    assert _dry(env).returncode == 2
