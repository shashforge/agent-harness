"""The executor's dispatch is the taxonomy, literally.

After the internal refactor, every control-flow branch in
_act_and_verify is an except clause on an exception from errors.py.
These tests pin what that bought: previously-untested budget behavior,
and the widened rule that the taxonomy applies no matter who raises
it — the gate, the machinery, or the tool itself.
"""
from __future__ import annotations

import pytest

from agent_harness.errors import BudgetExhausted, VerificationFailure
from agent_harness.executor import Executor, Resolution
from agent_harness.states import State

from test_executor import GRANTED, make_tool, never_escalate, three_step_planner


def test_spent_call_budget_escalates_with_the_budget_reason():
    seen = []

    def escalate(esc):
        seen.append(esc)
        return Resolution.ABORT

    ex = Executor({"echo": make_tool(max_calls=1)}, {},
                  three_step_planner(), escalate, GRANTED)
    result = ex.run()
    assert result.final_state is State.FAILED
    assert len(result.checkpoints) == 1           # the one paid-for call
    assert "call budget spent for echo" in seen[0].reason


def test_spend_helper_raises_budget_exhausted():
    tool = make_tool(max_calls=1)
    assert tool.spend_call() is True
    with pytest.raises(BudgetExhausted, match="call budget spent"):
        Executor._spend(tool)


def test_verify_helper_raises_verification_failure():
    ex = Executor({"echo": make_tool(verifier="never")},
                  {"never": lambda r: False},
                  three_step_planner(), never_escalate, GRANTED)
    with pytest.raises(VerificationFailure, match="never"):
        ex._verify(ex.tools["echo"].contract, "anything")


def test_a_tool_raising_the_taxonomy_gets_taxonomy_treatment():
    """A tool that raises VerificationFailure itself is handled exactly
    like a failed verifier: retry within budget, then escalate. The
    taxonomy applies no matter who raises it."""
    seen = []

    def escalate(esc):
        seen.append(esc)
        return Resolution.ABORT

    def self_rejecting(**kw):
        raise VerificationFailure("i checked my own work and it is bad")

    ex = Executor({"echo": make_tool(fn=self_rejecting)}, {},
                  three_step_planner(), escalate, GRANTED)
    result = ex.run()
    assert result.final_state is State.FAILED
    assert seen[0].reason == "verification failed"
    retries = [t for t in result.transitions
               if t.reason == "verification failed, retrying"]
    assert len(retries) == 2                      # max_retries, then escalate
