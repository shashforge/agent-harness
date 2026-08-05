"""Golden traces: recorded runs as regression tests.

A golden trace is the executor's behavior, written down. Change the
retry policy, the checkpoint rule, the escalation path — any of it —
and the diff against the golden shows exactly which transition moved.
An eval harness for the harness itself.

Wall-clock time is stripped before comparison. Everything else is
behavior and counts.
"""
from __future__ import annotations

import json
from pathlib import Path

from .executor import RunResult

# every field of a transition record except wall time
_BEHAVIOR_FIELDS = ("seq", "step", "src", "dst", "reason", "detail")


class GoldenMismatch(AssertionError):
    """The run diverged from its golden trace."""


def normalize(trace: list[dict]) -> list[dict]:
    """Strip nondeterminism (wall time), keep behavior."""
    return [{k: r[k] for k in _BEHAVIOR_FIELDS if k in r} for r in trace]


def _snapshot(result: RunResult) -> dict:
    return {
        "final_state": result.final_state.value,
        "trace": normalize(result.trace),
        "checkpoints": result.checkpoints,
    }


def assert_matches_golden(result: RunResult, path: str | Path,
                          update: bool = False) -> None:
    """Compare a run against its golden trace.

    Missing golden (or update=True): record the run as the new golden
    and pass — the snapshot-test convention. Otherwise: raise
    GoldenMismatch naming the first transition that diverged.
    """
    path = Path(path)
    got = _snapshot(result)

    if update or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(got, indent=2) + "\n")
        return

    want = json.loads(path.read_text())

    if got["final_state"] != want["final_state"]:
        raise GoldenMismatch(
            f"final state {got['final_state']!r}, "
            f"golden says {want['final_state']!r}"
        )

    for i, (g, w) in enumerate(zip(got["trace"], want["trace"])):
        if g != w:
            fields = [k for k in _BEHAVIOR_FIELDS if g.get(k) != w.get(k)]
            raise GoldenMismatch(
                f"trace diverges at transition {i}: "
                f"{' ,'.join(fields)} changed — "
                f"got {[g.get(k) for k in fields]}, "
                f"golden {[w.get(k) for k in fields]}"
            )

    if len(got["trace"]) != len(want["trace"]):
        raise GoldenMismatch(
            f"trace length {len(got['trace'])}, "
            f"golden has {len(want['trace'])} "
            f"(first extra transition at index "
            f"{min(len(got['trace']), len(want['trace']))})"
        )

    if got["checkpoints"] != want["checkpoints"]:
        raise GoldenMismatch("checkpoints diverge from golden")
