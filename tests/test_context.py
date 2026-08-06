"""Context compaction: a lens over the planner's view, never over
the record. Every test here checks both sides of that rule.
"""
from __future__ import annotations

from agent_harness.context import ContextBudget, FoldingCompactor
from agent_harness.executor import Executor, Resolution, Step
from agent_harness.persistence import JsonlStore, replay
from agent_harness.states import State

from test_executor import GRANTED, make_tool, never_escalate


def want(n, payload="x" * 40):
    """Planner wanting n checkpoints, each with a chunky result."""
    def plan(view):
        done = sum(len(c["result"].get("steps", [])) if c["tool"] == "compactor"
                   else 1 for c in view)
        if done >= n:
            return None
        return Step("echo", {"n": done, "pad": payload})
    return plan


def test_under_watermark_nothing_happens():
    ex = Executor({"echo": make_tool()}, {}, want(3), never_escalate,
                  GRANTED, context_budget=ContextBudget(100_000),
                  compactor=FoldingCompactor())
    result = ex.run()
    assert result.final_state is State.DONE
    assert not [t for t in result.transitions
                if t.reason == "context compacted"]


def test_watermark_triggers_compaction_but_record_keeps_everything(tmp_path):
    store = JsonlStore(tmp_path / "run.jsonl")
    ex = Executor({"echo": make_tool()}, {}, want(6), never_escalate,
                  GRANTED, store=store,
                  context_budget=ContextBudget(130),
                  compactor=FoldingCompactor(keep_last=1, result_chars=8))
    result = ex.run()
    assert result.final_state is State.DONE

    compactions = [t for t in result.transitions
                   if t.reason == "context compacted"]
    assert compactions, "watermark never triggered"
    for t in compactions:
        assert t.detail["units_after"] < t.detail["units_before"]

    # the lens shrank the view; the record shrank nothing
    assert len(result.checkpoints) == 6
    assert all(c["tool"] == "echo" for c in result.checkpoints)
    _, stored = store.load()
    assert len(stored) == 6
    assert replay(store).steps_checkpointed == 6


def test_planner_sees_digest_not_history():
    seen = []

    def spy_planner(view):
        seen.append(view)
        return want(6)(view)

    ex = Executor({"echo": make_tool()}, {}, spy_planner, never_escalate,
                  GRANTED, context_budget=ContextBudget(130),
                  compactor=FoldingCompactor(keep_last=1, result_chars=8))
    ex.run()
    compacted_views = [v for v in seen if v and v[0].get("tool") == "compactor"]
    assert compacted_views
    digest = compacted_views[-1][0]["result"]
    assert digest["compacted"] is True
    assert digest["steps"] == sorted(digest["steps"])
    assert len(compacted_views[-1]) <= 3          # digest + keep_last


def test_no_compactor_escalates_instead_of_guessing():
    seen = []

    def escalate(esc):
        seen.append(esc)
        return Resolution.ABORT

    ex = Executor({"echo": make_tool()}, {}, want(6), escalate,
                  GRANTED, context_budget=ContextBudget(60))
    result = ex.run()
    assert result.final_state is State.FAILED
    assert "watermark" in seen[0].reason


def test_useless_compactor_escalates_after_one_try():
    seen = []

    def escalate(esc):
        seen.append(esc)
        return Resolution.ABORT

    ex = Executor({"echo": make_tool()}, {}, want(6), escalate,
                  GRANTED, context_budget=ContextBudget(60),
                  compactor=lambda view: list(view))     # shrinks nothing
    result = ex.run()
    assert result.final_state is State.FAILED
    assert seen and seen[0].attempts == 1
    stills = [t for t in result.transitions
              if t.reason == "still over watermark after compaction"]
    assert len(stills) == 1


def test_folding_keeps_last_n_verbatim():
    checkpoints = [{"step": i, "tool": "echo", "result": {"n": i}}
                   for i in range(7)]
    out = FoldingCompactor(keep_last=3)(checkpoints)
    assert len(out) == 4
    assert out[1:] == checkpoints[-3:]            # untouched, not summarized
    assert out[0]["result"]["steps"] == [0, 1, 2, 3]
