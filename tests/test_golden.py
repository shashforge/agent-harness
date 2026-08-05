"""Golden traces as regression tests — including one shipped golden
that pins the executor's current behavior in the repo.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_harness.errors import TransientToolError
from agent_harness.executor import Executor
from agent_harness.golden import GoldenMismatch, assert_matches_golden
from agent_harness.states import State

from test_executor import (GRANTED, make_tool, never_escalate,
                           three_step_planner)

GOLDEN_DIR = Path(__file__).parent / "golden"


def happy_executor(fn=None):
    return Executor({"echo": make_tool(verifier="ok", fn=fn)},
                    {"ok": lambda r: True},
                    three_step_planner(), never_escalate, GRANTED)


def test_first_run_records_then_identical_run_matches(tmp_path):
    golden = tmp_path / "happy.json"
    assert_matches_golden(happy_executor().run(), golden)   # records
    assert golden.exists()
    assert_matches_golden(happy_executor().run(), golden)   # matches


def test_behavior_change_fails_and_names_the_transition(tmp_path):
    golden = tmp_path / "happy.json"
    assert_matches_golden(happy_executor().run(), golden)

    # same goal, different behavior: one transient failure, one retry
    attempts = {"n": 0}

    def flaky(**kw):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TransientToolError("timeout")
        return kw

    result = happy_executor(fn=flaky).run()
    assert result.final_state is State.DONE   # same outcome...
    with pytest.raises(GoldenMismatch) as err:
        assert_matches_golden(result, golden)  # ...different behavior
    assert "diverges at transition" in str(err.value)


def test_update_flag_rerecords(tmp_path):
    golden = tmp_path / "happy.json"
    assert_matches_golden(happy_executor().run(), golden)
    result = happy_executor().run()
    assert_matches_golden(result, golden, update=True)      # always passes
    assert_matches_golden(happy_executor().run(), golden)


def test_shipped_golden_pins_current_executor_behavior():
    """The golden committed to this repo IS the regression test.

    If a change to the executor moves any transition of the canonical
    happy path, this fails and names the transition. Recording a new
    golden is a reviewable diff, not a silent drift.
    """
    assert_matches_golden(happy_executor().run(),
                          GOLDEN_DIR / "happy_path.json")
