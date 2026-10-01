"""verl/trainer/ppo/group_metrics.py: tied groups and prefix-shared reward agreement."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("group_metrics", ROOT / "verl/trainer/ppo/group_metrics.py")
gm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gm)

T = 600


def rollout(prefix, tail_seed, length=T):
    """A response that starts with ``prefix`` and continues with tokens no other rollout shares."""
    rng = np.random.default_rng(tail_seed)
    tail = rng.integers(1000, 2000, size=length - len(prefix)) + 1000 * tail_seed
    return np.concatenate([np.asarray(prefix, dtype=np.int64), tail])


def batch(groups):
    """groups: list of (uid, [(response, score), ...]) -> arrays for compute_group_metrics."""
    resp, scores, uids = [], [], []
    for uid, items in groups:
        for r, s in items:
            resp.append(r)
            scores.append(s)
            uids.append(uid)
    resp = np.stack(resp)
    mask = np.ones_like(resp, dtype=bool)
    adv = np.ones(resp.shape, dtype=float)
    return np.array(scores, float), np.array(uids, dtype=object), resp, mask, adv


def test_tied_groups_are_counted_and_split_by_outcome():
    g = [
        ("a", [(rollout([], k), 1.0) for k in range(4)]),        # all correct
        ("b", [(rollout([], 10 + k), -1.0) for k in range(4)]),  # all wrong
        ("c", [(rollout([], 20 + k), s) for k, s in enumerate([1, -1, -1, -1])]),
        ("d", [(rollout([], 30 + k), s) for k, s in enumerate([1, 1, -1, 1])]),
    ]
    m = gm.compute_group_metrics(*batch(g))
    assert m["group/n_groups"] == 4
    assert m["group/uniform_frac"] == pytest.approx(0.5)
    assert m["group/all_correct_frac"] == pytest.approx(0.25)
    assert m["group/all_wrong_frac"] == pytest.approx(0.25)


def test_zero_one_scores_read_the_same_way():
    g = [("a", [(rollout([], k), 0.0) for k in range(3)]), ("b", [(rollout([], 5 + k), 1.0) for k in range(3)])]
    m = gm.compute_group_metrics(*batch(g))
    assert m["group/all_wrong_frac"] == pytest.approx(0.5)
    assert m["group/all_correct_frac"] == pytest.approx(0.5)


@pytest.mark.parametrize("shared,name", [(0, "lt64"), (100, "64to192"), (200, "192to384"), (500, "ge384")])
def test_pairs_land_in_the_bin_of_the_prefix_they_share(shared, name):
    prefix = list(range(1, shared + 1))
    g = [("a", [(rollout(prefix, 1), 1.0), (rollout(prefix, 2), 1.0)])]
    m = gm.compute_group_metrics(*batch(g))
    assert m[f"group/pair_n/{name}"] == 1
    assert m[f"group/pair_agree/{name}"] == 1.0
    for other, _ in gm.PAIR_BINS:
        if other != name:
            assert m[f"group/pair_n/{other}"] == 0
            assert f"group/pair_agree/{other}" not in m  # empty bin: omitted, not NaN


def test_padding_is_not_a_shared_prefix():
    # Two empty responses: every position is padding with the same pad id.
    resp = np.zeros((2, T), dtype=np.int64)
    mask = np.zeros((2, T), dtype=bool)
    m = gm.compute_group_metrics(np.array([-1.0, -1.0]), np.array(["a", "a"], dtype=object), resp, mask, np.zeros((2, T)))
    assert m["group/pair_n/lt64"] == 1


def test_agreement_is_per_bin():
    deep = list(range(1, 401))
    g = [
        ("a", [(rollout(deep, 1), 1.0), (rollout(deep, 2), 1.0),     # ge384, agree
               (rollout([], 3), -1.0), (rollout([], 4), 1.0)]),      # independent tails
    ]
    m = gm.compute_group_metrics(*batch(g))
    assert m["group/pair_n/ge384"] == 1 and m["group/pair_agree/ge384"] == 1.0
    # the other five pairs share nothing: (deep1,r3) (deep1,r4) (deep2,r3) (deep2,r4) (r3,r4)
    assert m["group/pair_n/lt64"] == 5
    assert m["group/pair_agree/lt64"] == pytest.approx(2 / 5)


def test_zero_advantage_fraction_respects_the_mask():
    resp = np.ones((2, 4), dtype=np.int64)
    mask = np.array([[1, 1, 1, 0], [1, 0, 0, 0]], dtype=bool)
    adv = np.array([[0.0, 0.0, 0.5, 0.0], [0.3, 0.0, 0.0, 0.0]])
    m = gm.compute_group_metrics(np.array([1.0, -1.0]), np.array(["a", "a"], dtype=object), resp, mask, adv)
    assert m["group/adv_zero_tok_frac"] == pytest.approx(2 / 4)


def test_uneven_group_sizes_do_not_break_it():
    g = [("a", [(rollout([], k), 1.0) for k in range(8)]), ("b", [(rollout([], 9), -1.0)])]
    m = gm.compute_group_metrics(*batch(g))
    assert m["group/n_groups"] == 2
    assert m["group/uniform_frac"] == 1.0
    assert m["group/pair_n/lt64"] == 28  # 8 choose 2; a singleton has no pairs
