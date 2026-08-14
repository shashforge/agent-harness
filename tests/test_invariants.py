"""Seeded invariant sweep: the executor under randomized abuse.

Every other test file pins one known behavior. This one generates
behaviors: planners, tools, and verifiers whose conduct is drawn from
a seeded random.Random, then asserts the invariants that must survive
whatever the drawing produced.

Deliberately stdlib-only. A property-testing framework would shrink
counterexamples for me; a seed is already a perfect reproduction:
if seed 217 fails, run seed 217. No new dependency, no change to CI.

The one behavior randomness cannot be allowed to decide is
termination, so every generated human has finite patience: a few
resumes, then abort. A human who resumes forever earns a run that
runs forever — that is the design, not a defect of it.
"""
from __future__ import annotations

import random

import pytest

from agent_harness.contract import Tool, ToolContract
from agent_harness.errors import (
    PermanentToolError,
    TransientToolError,
    VerificationFailure,
)
from agent_harness.executor import Executor, Resolution, Step
from agent_harness.persistence import JsonlStore, replay
from agent_harness.states import State

SEEDS = range(250)
GRANTED = frozenset({"fuzz:granted"})
PLANNER_CALL_TRIPWIRE = 10_000     # liveness: no run may plan this often


def build_world(rng: random.Random):
    """A random toolbox. Most tools are permitted; some demand a scope
    this run was never granted. Bodies misbehave with fixed odds."""
    tools = {}
    for i in range(rng.randint(1, 4)):
        name = f"tool{i}"
        scopes = ("fuzz:granted",) if rng.random() < 0.8 else ("fuzz:forbidden",)
        contract = ToolContract(
            name=name,
            scopes=scopes,
            max_calls_per_run=rng.randint(1, 6),
            max_retries=rng.randint(0, 2),
            verifier=rng.choice([None, "coin"]),
        )

        def fn(_rng=rng, _name=name, **kwargs):
            roll = _rng.random()
            if roll < 0.15:
                raise TransientToolError(f"{_name} flaked")
            if roll < 0.25:
                raise PermanentToolError(f"{_name} cannot do this")
            if roll < 0.30:
                raise VerificationFailure(f"{_name} rejects its own work")
            return {"tool": _name, "ok": True}

        tools[name] = Tool(contract, fn)
    return tools


def build_run(seed: int, store):
    rng = random.Random(seed)
    tools = build_world(rng)
    names = list(tools)
    goal = rng.randint(0, 4)
    patience = rng.randint(0, 3)
    plans = {"n": 0}
    moods = {"resumes": 0}

    def planner(checkpoints):
        plans["n"] += 1
        assert plans["n"] < PLANNER_CALL_TRIPWIRE, f"seed {seed}: run never ends"
        if len(checkpoints) >= goal:
            return None
        return Step(rng.choice(names), {})

    def finite_patience(escalation):
        if moods["resumes"] < patience:
            moods["resumes"] += 1
            return Resolution.RESUME
        return Resolution.ABORT

    ex = Executor(tools, {"coin": lambda r: rng.random() < 0.7},
                  planner, finite_patience, GRANTED,
                  max_steps=rng.randint(1, 6), store=store)
    return ex, tools


def trace_shape(result):
    return [(t.seq, t.step_index, t.src, t.dst, t.reason)
            for t in result.transitions]


@pytest.mark.parametrize("seed", SEEDS)
def test_every_random_run_upholds_the_invariants(seed, tmp_path):
    store = JsonlStore(tmp_path / "run.jsonl")
    ex, tools = build_run(seed, store)
    result = ex.run()

    # terminal state, always
    assert result.final_state in (State.DONE, State.FAILED)

    # the trace is append-only and gapless
    assert [t.seq for t in result.transitions] == \
        list(range(1, len(result.transitions) + 1))

    # a checkpoint exists exactly when a verification was recorded
    verified = [t for t in result.transitions
                if t.src is State.VERIFY and t.dst is State.CHECKPOINT]
    assert len(result.checkpoints) == len(verified)

    # no tool was ever called past its budget
    for tool in tools.values():
        assert tool.calls_made <= tool.contract.max_calls_per_run

    # the stored trace supports its own story, tool-free
    summary = replay(store)
    assert summary.final_state == result.final_state
    assert summary.steps_checkpointed == len(result.checkpoints)
    assert summary.transition_count == len(result.transitions)


def test_same_seed_same_trace(tmp_path):
    """Reproducibility is the whole bargain of seeded randomness."""
    a = build_run(99, JsonlStore(tmp_path / "a.jsonl"))[0].run()
    b = build_run(99, JsonlStore(tmp_path / "b.jsonl"))[0].run()
    assert trace_shape(a) == trace_shape(b)
    assert a.final_state == b.final_state


def test_a_forbidden_tool_spends_no_budget():
    """The permission gate sits before the meter. Denied calls are
    not billed."""
    contract = ToolContract(name="locked", scopes=("fuzz:forbidden",),
                            max_calls_per_run=3)
    tool = Tool(contract, lambda **kw: {"ok": True})

    def planner(checkpoints):
        return Step("locked", {})

    ex = Executor({"locked": tool}, {}, planner,
                  lambda esc: Resolution.ABORT, GRANTED)
    result = ex.run()
    assert result.final_state is State.FAILED
    assert tool.calls_made == 0


def test_a_tireless_planner_is_stopped_by_the_step_budget():
    """A planner that never says done meets max_steps, and the human
    decides from there."""
    contract = ToolContract(name="yes", scopes=("fuzz:granted",),
                            max_calls_per_run=50)

    def planner(checkpoints):
        return Step("yes", {})

    ex = Executor({"yes": Tool(contract, lambda **kw: {"ok": True})}, {},
                  planner, lambda esc: Resolution.ABORT, GRANTED,
                  max_steps=4)
    result = ex.run()
    assert result.final_state is State.FAILED
    assert len(result.checkpoints) == 4
    assert result.transitions[-1].reason == "aborted after step budget"
