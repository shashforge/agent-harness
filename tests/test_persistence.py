"""Persistence and replay: the trace survives the process.

The scenario that matters: a run dies mid-flight (a real crash, not a
taxonomy failure), and a new process picks up from the last verified
checkpoint with one continuous trace.
"""
from __future__ import annotations

import json

import pytest

from agent_harness.executor import Executor, Step
from agent_harness.persistence import JsonlStore, TraceCorruption, replay
from agent_harness.states import State

from test_executor import GRANTED, make_tool, never_escalate


def want(n):
    """Planner that wants n checkpoints, then declares the goal met."""
    def plan(checkpoints):
        if len(checkpoints) >= n:
            return None
        return Step("echo", {"n": len(checkpoints)})
    return plan


def test_run_writes_trace_and_checkpoints_as_it_goes(tmp_path):
    store = JsonlStore(tmp_path / "run.jsonl")
    ex = Executor({"echo": make_tool(verifier="ok")}, {"ok": lambda r: True},
                  want(3), never_escalate, GRANTED, store=store)
    result = ex.run()
    transitions, checkpoints = store.load()
    assert len(transitions) == len(result.transitions)
    assert len(checkpoints) == 3
    assert checkpoints == result.checkpoints


def test_crash_then_resume_is_one_continuous_trace(tmp_path):
    store = JsonlStore(tmp_path / "run.jsonl")
    calls = {"n": 0}

    def dies_third_call(**kw):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("power cut")   # not in the taxonomy: a crash
        return kw

    ex = Executor({"echo": make_tool(fn=dies_third_call)}, {},
                  want(4), never_escalate, GRANTED, store=store)
    with pytest.raises(RuntimeError):
        ex.run()

    _, checkpoints = store.load()
    assert len(checkpoints) == 2              # two steps verifiably done

    ex2 = Executor({"echo": make_tool()}, {},
                   want(4), never_escalate, GRANTED,
                   store=store, resume=True)
    result = ex2.run()
    assert result.final_state is State.DONE
    transitions, checkpoints = store.load()
    assert len(checkpoints) == 4
    assert [c["step"] for c in checkpoints] == [0, 1, 2, 3]
    seqs = [t["seq"] for t in transitions]    # both runs, one sequence
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


def test_replay_reconstructs_without_executing_anything(tmp_path):
    store = JsonlStore(tmp_path / "run.jsonl")
    ex = Executor({"echo": make_tool(verifier="ok")}, {"ok": lambda r: True},
                  want(3), never_escalate, GRANTED, store=store)
    result = ex.run()

    summary = replay(store)                   # no tools, no verifiers
    assert summary.final_state is State.DONE
    assert summary.steps_checkpointed == 3
    assert summary.checkpoints == result.checkpoints
    assert summary.transition_count == len(result.transitions)


def test_replay_rejects_unverified_checkpoint(tmp_path):
    store = JsonlStore(tmp_path / "run.jsonl")
    ex = Executor({"echo": make_tool(verifier="ok")}, {"ok": lambda r: True},
                  want(2), never_escalate, GRANTED, store=store)
    ex.run()
    # forge a checkpoint no verify transition vouches for
    with store.path.open("a") as f:
        f.write(json.dumps({"kind": "checkpoint", "step": 99,
                            "tool": "echo", "result": "forged"}) + "\n")
    with pytest.raises(TraceCorruption):
        replay(store)


def test_replay_rejects_rewritten_sequence(tmp_path):
    store = JsonlStore(tmp_path / "run.jsonl")
    ex = Executor({"echo": make_tool(verifier="ok")}, {"ok": lambda r: True},
                  want(2), never_escalate, GRANTED, store=store)
    ex.run()
    transitions, _ = store.load()
    # replay a stale record: seq goes backwards, append-only is broken
    with store.path.open("a") as f:
        f.write(json.dumps({"kind": "transition",
                            **transitions[0]}) + "\n")
    with pytest.raises(TraceCorruption):
        replay(store)


def test_checkpoint_results_must_be_serializable(tmp_path):
    store = JsonlStore(tmp_path / "run.jsonl")
    ex = Executor({"echo": make_tool(fn=lambda **kw: object())}, {},
                  want(1), never_escalate, GRANTED, store=store)
    with pytest.raises(TypeError):
        ex.run()
