"""Checkpoint persistence and replay.

The trace was always data; this module makes it survive the process.
Records are written as they happen, append-only, one JSON object per
line. A run that dies mid-flight leaves a file that tells you exactly
how far it verifiably got — and a new run can pick up from there.

Replay reads the file and reconstructs what happened without calling a
single tool. If the trace can't support its own story, replay says so.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .states import State, Transition


class TraceCorruption(Exception):
    """The stored trace violates an invariant the executor guarantees."""


class JsonlStore:
    """Append-only JSONL store for transitions and checkpoints.

    One file, two record kinds:
        {"kind": "transition", ...Transition.as_record()}
        {"kind": "checkpoint", "step": int, "tool": str, "result": ...}

    Results must be JSON-serializable. That is a real constraint, and
    it is deliberate: a checkpoint you cannot serialize is a checkpoint
    you cannot resume from.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append_transition(self, t: Transition) -> None:
        self._append({"kind": "transition", **t.as_record()})

    def append_checkpoint(self, checkpoint: dict) -> None:
        self._append({"kind": "checkpoint", **checkpoint})

    def _append(self, record: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def load(self) -> tuple[list[dict], list[dict]]:
        """Returns (transition_records, checkpoints). Missing file: empty."""
        transitions: list[dict] = []
        checkpoints: list[dict] = []
        if not self.path.exists():
            return transitions, checkpoints
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                kind = record.pop("kind", None)
                if kind == "transition":
                    transitions.append(record)
                elif kind == "checkpoint":
                    checkpoints.append(record)
                else:
                    raise TraceCorruption(f"unknown record kind: {kind!r}")
        return transitions, checkpoints

    def last_seq(self) -> int:
        transitions, _ = self.load()
        return transitions[-1]["seq"] if transitions else 0


@dataclass
class ReplaySummary:
    """What a stored trace proves happened, tool-free."""

    final_state: Optional[State]     # None if the run never finished
    steps_checkpointed: int
    checkpoints: list[dict]
    transition_count: int


def replay(store: JsonlStore) -> ReplaySummary:
    """Reconstruct a run from its trace without executing anything.

    Verifies the invariants the executor promises:
    - seq strictly increasing (append-only, no rewrites)
    - every checkpoint record is preceded by a VERIFY -> CHECKPOINT
      transition for that step (nothing persisted unverified)
    """
    transitions, checkpoints = store.load()

    last_seq = 0
    for r in transitions:
        if r["seq"] <= last_seq:
            raise TraceCorruption(
                f"seq not strictly increasing at {r['seq']} (after {last_seq})"
            )
        last_seq = r["seq"]

    verified_steps = {
        r["step"]
        for r in transitions
        if r["src"] == State.VERIFY.value and r["dst"] == State.CHECKPOINT.value
    }
    for c in checkpoints:
        if c["step"] not in verified_steps:
            raise TraceCorruption(
                f"checkpoint for step {c['step']} has no verify transition"
            )

    final: Optional[State] = None
    if transitions:
        last = transitions[-1]
        if last["dst"] in (State.DONE.value, State.FAILED.value):
            final = State(last["dst"])

    return ReplaySummary(
        final_state=final,
        steps_checkpointed=len(checkpoints),
        checkpoints=checkpoints,
        transition_count=len(transitions),
    )
