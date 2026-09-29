"""Which checkpoints a run keeps: every ``keep_every`` steps, the last step, and the best.

The LoRA campaign evaluates the last step as its primary endpoint and the
best-by-validation step as a secondary one, and keeps the 50-step points for
the learning curve. The trainer's own options could not express that:
``save_best_only`` keeps a new best and nothing else (not even the last step),
and ``max_actor_ckpt_to_keep`` rotates first-in-first-out, so it deletes the
best as soon as it becomes the oldest -- and forgets everything saved before a
resume. So the trainer asks :func:`plan` at every save point and does what it
says; ``max_actor_ckpt_to_keep`` is left unset.

Pure Python on purpose: the decision is tested on a CPU without the training
stack.
"""
from __future__ import annotations

import math
from typing import Optional


def validate(keep_every: int, save_freq: int, test_freq: int) -> None:
    """The trainer only reaches a save point when step % save_freq == 0 (or the
    last step), and only knows the metric on steps it validated. A keep step
    that is not a save point, or a save point with no validation, would be
    silently missed."""
    if keep_every <= 0:
        raise ValueError(f"keep_every must be positive, got {keep_every}")
    if save_freq <= 0 or keep_every % save_freq:
        raise ValueError(f"keep_every={keep_every} must be a multiple of save_freq={save_freq}")
    if test_freq <= 0 or save_freq % test_freq:
        raise ValueError(f"save_freq={save_freq} must be a multiple of test_freq={test_freq}, "
                         "or the best can only be judged on some save points")


def protected(step: int, keep_every: int, final_step: int) -> bool:
    """A step kept for its own sake, never deleted as a superseded best."""
    return step % keep_every == 0 or step >= final_step


def plan(step: int, final_step: int, metric: Optional[float], best_value: float,
         best_step: Optional[int], keep_every: int) -> dict:
    """What to do at a save point.

    Returns ``save`` (write global_step_<step>), ``new_best``, ``delete`` (steps
    whose directories to remove), and the updated ``best_value``/``best_step``.
    ``metric`` is None on a step that did not validate; NaN never wins.
    """
    is_last = step >= final_step
    new_best = metric is not None and not math.isnan(metric) and metric > best_value
    save = new_best or protected(step, keep_every, final_step)
    delete = []
    if new_best and best_step is not None and best_step != step \
            and not protected(best_step, keep_every, final_step):
        delete.append(best_step)
    return {
        "save": save,
        "new_best": new_best,
        "is_last": is_last,
        "delete": delete,
        "best_value": metric if new_best else best_value,
        "best_step": step if new_best else best_step,
    }
