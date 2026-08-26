"""The inspector reads a trace back without executing anything.

These tests run real executors to produce real stores, then check
that the rendered story matches what actually happened, and that a
corrupt trace is called corrupt with a nonzero exit code.
"""
from __future__ import annotations

import json

from agent_harness.errors import TransientToolError
from agent_harness.inspect import main, render
from agent_harness.persistence import JsonlStore

from test_executor import GRANTED, make_tool, never_escalate, three_step_planner


def run_store(tmp_path, tool):
    store = JsonlStore(tmp_path / "run.jsonl")
    from agent_harness.executor import Executor
    Executor({"echo": tool}, {"ok": lambda r: True},
             three_step_planner(), never_escalate, GRANTED,
             store=store).run()
    return store


def test_a_clean_run_reads_as_a_story(tmp_path):
    store = run_store(tmp_path, make_tool(verifier="ok"))
    text, code = render(store)
    assert code == 0
    assert "final: done" in text
    assert "checkpoints: 3" in text
    assert text.count("checkpoint\n") + text.count("checkpoint") >= 3
    assert "replay: clean" in text


def test_retries_are_marked(tmp_path):
    calls = {"n": 0}

    def flaky(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TransientToolError("first call flakes")
        return {"ok": True}

    store = run_store(tmp_path, make_tool(fn=flaky))
    text, code = render(store)
    assert code == 0
    assert "retries: 1" in text
    assert "transient error, retrying" in text


def test_a_forged_checkpoint_is_called_corrupt(tmp_path):
    store = run_store(tmp_path, make_tool(verifier="ok"))
    with store.path.open("a") as f:
        f.write(json.dumps({"kind": "checkpoint", "step": 99,
                            "tool": "echo", "result": "forged"}) + "\n")
    text, code = render(store)
    assert code == 2
    assert "CORRUPT TRACE" in text


def test_main_exit_codes(tmp_path, capsys):
    assert main([]) == 1                                  # usage
    assert main([str(tmp_path / "missing.jsonl")]) == 1   # no such file
    store = run_store(tmp_path, make_tool(verifier="ok"))
    assert main([str(store.path)]) == 0
    out = capsys.readouterr().out
    assert "final: done" in out
