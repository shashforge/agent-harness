"""Execution states and transitions.

The design rule this module enforces: every transition is data.
See https://shashforge.dev/log/the-harness-gets-a-spine/
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class State(str, Enum):
    PLAN = "plan"
    ACT = "act"
    VERIFY = "verify"
    CHECKPOINT = "checkpoint"
    ESCALATE = "escalate"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class Transition:
    """One recorded edge of the state machine. Append-only, replayable."""

    seq: int
    step_index: int
    src: State
    dst: State
    reason: str
    detail: dict = field(default_factory=dict)
    wall_time: float = field(default_factory=time.time)

    def as_record(self) -> dict:
        return {
            "seq": self.seq,
            "step": self.step_index,
            "src": self.src.value,
            "dst": self.dst.value,
            "reason": self.reason,
            "detail": self.detail,
            "t": self.wall_time,
        }
