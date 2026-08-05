"""Behavioral tests for the executor state machine.

Each test pins one rule from the published design:
https://shashforge.dev/log/the-harness-gets-a-spine/
"""
from __future__ import annotations

from agent_harness.contract import Tool, ToolContract
from agent_harness.errors import PermanentToolError, TransientToolError
from agent_harness.executor import Escalation, Executor, Resolution, Step
from agent_harness.states import State


def make_tool(name="echo", scopes=("run:basic",), verifier=None,
              max_retries=2, max_calls=20, fn=None):
    contract = ToolContract(name=name, scopes=tuple(scopes),
                            verifier=verifier, max_retries=max_retries,
                            max_calls_per_run=max_calls)
    return Tool(contract, fn or (lambda **kw: kw))


def three_step_planner():
    """Plans 3 echo steps, then declares the goal met."""
    def plan(checkpoints):
        if len(checkpoints) >= 3:
            return None
        return Step("echo", {"n": len(checkpoints)})
    return plan


def never_escalate(esc: Escalation) -> Resolution:
    raise AssertionError(f"unexpected escalation: {esc.reason}")


GRANTED = frozenset({"run:basic"})


def test_happy_path_checkpoints_only_after_verify():
    ex = Executor({"echo": make_tool(verifier="ok")},
                  {"ok": lambda r: True},
                  three_step_planner(), never_escalate, GRANTED)
    result = ex.run()
    assert result.final_state is State.DONE
    assert len(result.checkpoints) == 3
    # verify precedes every checkpoint in the trace
    seq = [(t.src, t.dst) for t in result.transitions]
    for i, edge in enumerate(seq):
        if edge[1] is State.CHECKPOINT:
            assert edge[0] is State.VERIFY


def test_transient_errors_retry_same_step():
    attempts = {"n": 0}

    def flaky(**kw):
        attempts["n"] += 1
        if attempts["n"] < 3:            # fail twice, succeed third
            raise TransientToolError("timeout")
        return "ok"

    ex = Executor({"echo": make_tool(fn=flaky)}, {},
                  three_step_planner(), never_escalate, GRANTED)
    result = ex.run()
    assert result.final_state is State.DONE
    retries = [t for t in result.transitions
               if t.reason == "transient error, retrying"]
    assert len(retries) == 2


def test_permanent_error_replans_instead_of_retrying():
    calls = {"n": 0}

    def broken_then_fine(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermanentToolError("schema mismatch")
        return "ok"

    ex = Executor({"echo": make_tool(fn=broken_then_fine)}, {},
                  three_step_planner(), never_escalate, GRANTED)
    result = ex.run()
    assert result.final_state is State.DONE
    replans = [t for t in result.transitions
               if t.reason == "permanent tool error, re-planning"]
    assert len(replans) == 1


def test_verification_failure_escalates_after_budget():
    seen = []

    def escalate(esc):
        seen.append(esc)
        return Resolution.ABORT

    ex = Executor({"echo": make_tool(verifier="never")},
                  {"never": lambda r: False},
                  three_step_planner(), escalate, GRANTED)
    result = ex.run()
    assert result.final_state is State.FAILED
    assert seen and seen[0].reason == "verification failed"
    # retried exactly max_retries times before escalating
    retries = [t for t in result.transitions
               if t.reason == "verification failed, retrying"]
    assert len(retries) == 2


def test_permission_denied_escalates_without_retry():
    seen = []

    def escalate(esc):
        seen.append(esc)
        return Resolution.ABORT

    ex = Executor({"echo": make_tool(scopes=("repo:write",))}, {},
                  three_step_planner(), escalate,
                  granted_scopes=frozenset({"run:basic"}))  # write NOT granted
    result = ex.run()
    assert result.final_state is State.FAILED
    assert "requires scopes" in seen[0].reason
    assert not [t for t in result.transitions if "retry" in t.reason]


def test_human_resume_grants_replan():
    granted_once = {"done": False}

    def escalate(esc):
        if not granted_once["done"]:
            granted_once["done"] = True
            return Resolution.RESUME
        return Resolution.ABORT

    calls = {"n": 0}

    def eventually_valid(**kw):
        calls["n"] += 1
        return calls["n"]

    # verifier rejects until the 4th call: first step escalates once,
    # human resumes, later attempts pass
    ex = Executor({"echo": make_tool(verifier="gate", fn=eventually_valid)},
                  {"gate": lambda r: r >= 4},
                  three_step_planner(), escalate, GRANTED)
    result = ex.run()
    assert result.final_state is State.DONE
    resumes = [t for t in result.transitions if t.reason == "human: resume"]
    assert len(resumes) == 1


def test_trace_is_replayable_data():
    ex = Executor({"echo": make_tool(verifier="ok")},
                  {"ok": lambda r: True},
                  three_step_planner(), never_escalate, GRANTED)
    result = ex.run()
    trace = result.trace
    assert all(set(r) == {"seq", "step", "src", "dst", "reason", "detail", "t"}
               for r in trace)
    seqs = [r["seq"] for r in trace]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


def test_step_budget_escalates():
    def endless(checkpoints):
        return Step("echo", {})

    seen = []

    def escalate(esc):
        seen.append(esc)
        return Resolution.ABORT

    ex = Executor({"echo": make_tool(max_calls=1000)}, {},
                  endless, escalate, GRANTED, max_steps=5)
    result = ex.run()
    assert result.final_state is State.FAILED
    assert "step budget exhausted" in seen[0].reason
    assert len(result.checkpoints) == 5
