"""Context budget and compaction.

The rule this module enforces: compaction is a lens, not an eraser.
When the checkpoint history grows past the watermark, the harness
shrinks what the planner is *shown*. The checkpoint list, the store,
and the trace never lose a byte. The model's context is negotiable;
the record is not.

The taxonomy entry this implements: ContextOverflow — token watermark
crossed; compact, then re-plan. The budget raises it, the executor
catches it and applies the allowed response. If compaction can't get
back under the watermark, that's an escalation, not a guess.
"""
from __future__ import annotations

import json
from typing import Callable

from .errors import ContextOverflow


def chars_over_4(checkpoints: list[dict]) -> int:
    """Default unit estimator: serialized length / 4.

    A deliberate approximation of tokens, and labeled as one. The
    budget needs a monotonic size signal, not tokenizer-exact counts;
    if you have a real tokenizer, pass it in.
    """
    return len(json.dumps(checkpoints)) // 4


class ContextBudget:
    """Watermark over the planner's view of the checkpoint history."""

    def __init__(self, max_units: int,
                 estimator: Callable[[list[dict]], int] = chars_over_4) -> None:
        self.max_units = max_units
        self.estimator = estimator

    def units(self, checkpoints: list[dict]) -> int:
        return self.estimator(checkpoints)

    def check(self, checkpoints: list[dict]) -> None:
        used = self.units(checkpoints)
        if used > self.max_units:
            raise ContextOverflow(
                f"context at {used} units, watermark is {self.max_units}"
            )


class FoldingCompactor:
    """Fold everything but the last `keep_last` checkpoints into one
    digest entry.

    The digest names the steps it swallowed and keeps a bounded scrap
    of each result, so the planner still knows the shape of what
    happened, just not every byte of it. Deterministic on purpose: a
    model-written summary can implement the same callable later, and
    the executor won't know the difference.
    """

    def __init__(self, keep_last: int = 3, result_chars: int = 80) -> None:
        if keep_last < 1:
            raise ValueError("keep_last must be >= 1")
        self.keep_last = keep_last
        self.result_chars = result_chars

    def __call__(self, checkpoints: list[dict]) -> list[dict]:
        if len(checkpoints) <= self.keep_last:
            return list(checkpoints)
        folded, kept = (checkpoints[:-self.keep_last],
                        checkpoints[-self.keep_last:])
        digest = {
            "step": folded[-1]["step"],
            "tool": "compactor",
            "result": {
                "compacted": True,
                "steps": [c["step"] for c in folded],
                "digest": [
                    {"step": c["step"], "tool": c["tool"],
                     "result": json.dumps(c["result"])[: self.result_chars]}
                    for c in folded
                ],
            },
        }
        return [digest, *kept]
