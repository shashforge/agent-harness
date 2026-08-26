"""Read a trace back as a story.

    python -m agent_harness.inspect out/run.jsonl

The store is already the evidence; this module only arranges it for
reading. It is tool-free by construction: nothing here can execute
anything, and the verdict at the bottom comes from the same replay()
the tests trust. A trace that fails replay is reported as corrupt and
the exit code says so.

Exit codes: 0 valid trace, 1 usage error, 2 corrupt trace.
"""
from __future__ import annotations

import sys
from pathlib import Path

from .persistence import JsonlStore, TraceCorruption, replay
from .states import State

# what a transition means for the reader's eye, keyed by (src, dst)
_MARKS = {
    (State.ACT, State.ACT): "retry",
    (State.VERIFY, State.ACT): "retry",
    (State.ACT, State.ESCALATE): "escalation",
    (State.VERIFY, State.ESCALATE): "escalation",
    (State.PLAN, State.ESCALATE): "escalation",
    (State.VERIFY, State.CHECKPOINT): "checkpoint",
}


def _mark(record: dict) -> str:
    key = (State(record["src"]), State(record["dst"]))
    return _MARKS.get(key, "")


def render(store: JsonlStore) -> tuple[str, int]:
    """Returns (report text, exit code). Never executes a tool."""
    lines: list[str] = []
    try:
        transitions, checkpoints = store.load()
        summary = replay(store)
    except TraceCorruption as e:
        return f"{store.path}: CORRUPT TRACE\n  {e}\n", 2

    final = summary.final_state.value if summary.final_state else "unfinished"
    lines.append(f"{store.path} · {len(transitions)} transitions · "
                 f"{len(checkpoints)} checkpoints · final: {final}")
    lines.append("")

    width = max((len(r["reason"]) for r in transitions), default=0)
    for r in transitions:
        arrow = f"{r['src']} → {r['dst']}"
        mark = _mark(r)
        detail = r.get("detail") or {}
        extra = "".join(f"  {k}={v}" for k, v in detail.items())
        lines.append(f"{r['seq']:>4}  step {r['step']:<2} {arrow:<22} "
                     f"{r['reason']:<{width}}  {mark}{extra}".rstrip())

    retries = sum(1 for r in transitions if _mark(r) == "retry")
    escalations = sum(1 for r in transitions if _mark(r) == "escalation")
    lines.append("")
    lines.append(f"retries: {retries} · escalations: {escalations} · "
                 f"checkpoints: {summary.steps_checkpointed}")
    lines.append("replay: clean · seq gapless · every checkpoint verified")
    return "\n".join(lines) + "\n", 0


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m agent_harness.inspect <trace.jsonl>",
              file=sys.stderr)
        return 1
    path = Path(argv[0])
    if not path.exists():
        print(f"no such file: {path}", file=sys.stderr)
        return 1
    text, code = render(JsonlStore(path))
    print(text, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
