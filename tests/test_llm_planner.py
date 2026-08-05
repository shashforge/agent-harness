"""The LLM planner, tested at the transport seam.

These tests script the transport — the function that would hit the
Messages API — and exercise everything on our side of it: prompt
construction, the output contract, and the full executor loop driven
by scripted model decisions. What they deliberately do not test is the
model itself; that requires a key and a live run.
"""
from __future__ import annotations

import pytest

from agent_harness.errors import PlannerProtocolError
from agent_harness.executor import Executor
from agent_harness.llm_planner import LLMPlanner, anthropic_transport
from agent_harness.states import State

from test_executor import GRANTED, make_tool, never_escalate

CATALOG = {"echo": "returns its arguments unchanged"}


def scripted(replies):
    """A transport that returns canned model replies and records requests."""
    requests = []

    def transport(payload):
        requests.append(payload)
        return {"content": [{"type": "text", "text": replies[len(requests) - 1]}]}

    transport.requests = requests
    return transport


def test_full_loop_with_scripted_model():
    transport = scripted([
        '{"tool": "echo", "args": {"n": 0}}',
        '{"tool": "echo", "args": {"n": 1}}',
        '{"done": true}',
    ])
    planner = LLMPlanner("echo twice, then stop", CATALOG,
                         transport=transport)
    ex = Executor({"echo": make_tool()}, {}, planner,
                  never_escalate, GRANTED)
    result = ex.run()
    assert result.final_state is State.DONE
    assert len(result.checkpoints) == 2
    assert len(transport.requests) == 3


def test_model_is_shown_goal_catalog_and_checkpoints():
    transport = scripted([
        '{"tool": "echo", "args": {"n": 0}}',
        '{"done": true}',
    ])
    planner = LLMPlanner("the goal text", CATALOG, transport=transport)
    Executor({"echo": make_tool()}, {}, planner,
             never_escalate, GRANTED).run()

    first = transport.requests[0]["messages"][0]["content"]
    assert "the goal text" in first
    assert "echo: returns its arguments unchanged" in first

    second = transport.requests[1]["messages"][0]["content"]
    assert '"n": 0' in second        # prior verified checkpoint is visible


def test_done_means_none():
    planner = LLMPlanner("g", CATALOG, transport=scripted(['{"done": true}']))
    assert planner([]) is None


def test_fenced_json_is_tolerated():
    planner = LLMPlanner("g", CATALOG, transport=scripted(
        ['```json\n{"tool": "echo", "args": {}}\n```']))
    step = planner([])
    assert step.tool == "echo"


def test_non_json_reply_raises_protocol_error():
    planner = LLMPlanner("g", CATALOG, transport=scripted(
        ["I think we should probably use the echo tool first."]))
    with pytest.raises(PlannerProtocolError):
        planner([])


def test_tool_outside_catalog_raises_protocol_error():
    planner = LLMPlanner("g", CATALOG, transport=scripted(
        ['{"tool": "rm_rf", "args": {}}']))
    with pytest.raises(PlannerProtocolError):
        planner([])


def test_live_transport_refuses_to_run_without_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        anthropic_transport({"model": "x", "messages": []})
