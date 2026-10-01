"""How much learning signal a step's GRPO groups carry, for any rollout scheme.

GRPO learns from the reward spread inside a prompt's group of ``n`` rollouts;
a group whose rollouts all score the same has zero advantage everywhere and
moves nothing. The tree rollout shares prefixes between a prompt's rollouts by
construction, so the worry is that it ties more groups than i.i.d. sampling
would. ``steerf/adv_zero_frac`` cannot answer that: it is logged only on the
STEER-F loss path, so the i.i.d. arms have nothing to compare against, and it
is a mean of per-micro-batch ratios. This module is called by the trainer on
every arm, on the whole batch, after the advantage is computed.

``pair_agree/<bin>`` is the difficulty-controlled view: every pair of rollouts
of the same prompt, binned by the prefix the two actually share (measured on
the tokens, so refilled tree slots land where they belong), and the fraction
of pairs in each bin that scored the same. If sharing a prefix ties rewards,
agreement rises with the shared length.

numpy only, so it is tested on a CPU without the training stack.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

# Upper edges (exclusive) of the shared-prefix bins; the last bin is open.
# They sit at the tree's cut depths (64, 192, 384), so each bin is one level.
PAIR_BINS = (("lt64", 64), ("64to192", 192), ("192to384", 384), ("ge384", None))


def _bin_name(shared: int) -> str:
    for name, upper in PAIR_BINS:
        if upper is None or shared < upper:
            return name
    raise AssertionError("unreachable: the last bin is open")


def shared_prefix_len(a: np.ndarray, b: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray) -> int:
    """Leading response tokens two rollouts have in common, padding excluded."""
    same = (a == b) & mask_a & mask_b
    if same.all():
        return int(same.size)
    return int(np.argmin(same))


def compute_group_metrics(
    scores: np.ndarray,
    uids: np.ndarray,
    responses: np.ndarray,
    response_mask: np.ndarray,
    advantages: np.ndarray,
) -> dict:
    """Group-level signal metrics for one training batch.

    Args:
        scores: (B,) sequence score of each rollout.
        uids: (B,) prompt id; rollouts of the same prompt share it.
        responses: (B, T) response token ids.
        response_mask: (B, T) 1 on real response tokens.
        advantages: (B, T) token advantages.
    """
    scores = np.asarray(scores, dtype=np.float64)
    responses = np.asarray(responses)
    mask = np.asarray(response_mask).astype(bool)
    advantages = np.asarray(advantages, dtype=np.float64)

    groups: dict = defaultdict(list)
    for i, u in enumerate(np.asarray(uids).tolist()):
        groups[u].append(i)

    n_groups = len(groups)
    uniform = all_correct = all_wrong = 0
    agree = {name: 0 for name, _ in PAIR_BINS}
    pairs = {name: 0 for name, _ in PAIR_BINS}

    for idx in groups.values():
        s = scores[idx]
        if np.all(s == s[0]):
            uniform += 1
            # Correct means a positive score: the training scorer gives +1/-1
            # and the evaluation scorers 1/0, and both read the same way.
            if s[0] > 0:
                all_correct += 1
            else:
                all_wrong += 1
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                name = _bin_name(shared_prefix_len(responses[i], responses[j], mask[i], mask[j]))
                pairs[name] += 1
                agree[name] += int(scores[i] == scores[j])

    valid = mask.sum()
    zero_adv = (mask & (advantages == 0)).sum()

    out = {
        "group/n_groups": float(n_groups),
        "group/uniform_frac": uniform / n_groups if n_groups else float("nan"),
        "group/all_correct_frac": all_correct / n_groups if n_groups else float("nan"),
        "group/all_wrong_frac": all_wrong / n_groups if n_groups else float("nan"),
        "group/adv_zero_tok_frac": float(zero_adv / valid) if valid else float("nan"),
    }
    for name, _ in PAIR_BINS:
        out[f"group/pair_n/{name}"] = float(pairs[name])
        # Omitted rather than NaN when a bin is empty: an i.i.d. arm has no
        # deep-shared pairs, and a NaN would read as a failure in the log.
        if pairs[name]:
            out[f"group/pair_agree/{name}"] = agree[name] / pairs[name]
    return out
