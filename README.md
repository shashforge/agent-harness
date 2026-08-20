# agent-harness

[![tests](https://github.com/shashforge/agent-harness/actions/workflows/tests.yml/badge.svg)](https://github.com/shashforge/agent-harness/actions/workflows/tests.yml)

A reference implementation of an agent execution harness: an explicit
state machine instead of a `while` loop with vibes.

**Status: working core.** The executor, tool contracts, failure
taxonomy, transition trace, checkpoint persistence, crash-resume,
replay, golden-trace regression tests, context compaction, and an
LLM-backed planner all work and are tested. The planner is a plain
callable, which is the point: the harness owns control flow; the
model is a component.

Design writing behind this code:

- [The model was never the product](https://shashforge.dev/log/the-model-was-never-the-product/) — why the harness is the product
- [The harness gets a spine](https://shashforge.dev/log/the-harness-gets-a-spine/) — the state machine, failure taxonomy, and tool contract this repo implements
- [Replay is the feature](https://shashforge.dev/log/replay-is-the-feature/) — persistence, crash-resume, and golden traces
- [The model shows up](https://shashforge.dev/log/the-model-shows-up/) — an LLM behind the planner callable
- [The lens, not the eraser](https://shashforge.dev/log/the-lens-not-the-eraser/) — context compaction that never touches the record

## The rules the code enforces

- `PLAN → ACT → VERIFY → CHECKPOINT`, looping until the planner declares
  the goal met. Nothing is persisted as progress until it survives
  verification.
- Every transition is data: an append-only, replayable trace.
- The failure taxonomy decides responses, not ad-hoc `except` blocks:
  transient errors retry the same step; permanent errors re-plan;
  verification failures retry within budget, then escalate; permission
  denials escalate and never retry.
- The human is a state, not an exception handler. Escalation carries the
  full trace and returns a resolution: resume with guidance, or abort.
- Permissions are declared on the tool contract and enforced by the
  runtime before every call.
- The trace survives the process: transitions and checkpoints stream to
  an append-only JSONL store as they happen. A crashed run resumes from
  its last verified checkpoint with one continuous sequence.
- Replay reconstructs a run from its file without executing a single
  tool, and rejects traces that violate the invariants (unverified
  checkpoints, rewritten history).
- A golden trace committed to this repo pins the executor's canonical
  behavior; changing the control flow fails a test that names the exact
  transition that moved.
- The output contract applies to models too: the LLM planner answers
  with one JSON step or `{"done": true}`. Anything else is a
  `PlannerProtocolError`, not an improvised recovery.
- Compaction is a lens, not an eraser. Past the context watermark the
  compactor shrinks what the planner is *shown*; the checkpoint list,
  the store, and the trace never lose a byte. A compactor that can't
  get back under the watermark escalates after one try.
- The invariants hold under behavior nobody chose: a seeded sweep runs
  hundreds of randomized planners, tools, and verifiers through the
  executor and replays every trace they leave. Same seed, same trace.

## Run the tests

```bash
pip install pytest
python -m pytest tests/ -v
```

CI runs the same suite on every push, across CPython 3.10 through
3.13 — the versions `pyproject.toml` claims. The badge above is that
run, not a decoration.

## Running against a live model

```python
from agent_harness.llm_planner import LLMPlanner

planner = LLMPlanner(
    goal="...",
    tool_catalog={"search": "...", "write_file": "..."},
)  # uses ANTHROPIC_API_KEY; the tests use a scripted transport instead
```

The planner tests script the transport seam — they verify everything on
this side of the API. Live behavior needs a real key and a real run;
there is deliberately no offline fallback pretending otherwise.

## Layout

```
agent_harness/
  states.py       # State enum + Transition (the replayable trace record)
  errors.py       # the failure taxonomy, as exceptions
  contract.py     # ToolContract: scopes, budgets, idempotency, verifier
  executor.py     # the state machine
  persistence.py  # append-only JSONL store, crash-resume, replay
  golden.py       # golden traces as regression tests
  context.py      # context watermark + compaction (the lens)
  llm_planner.py  # an LLM behind the planner callable
examples/
  live_run.py     # a real model driving the harness on this repo
docs/adr/         # architecture decision records, numbered, never rewritten
tests/
  golden/happy_path.json  # the executor's canonical behavior, pinned
  test_invariants.py      # 250 seeded worlds; the invariants must hold in all
  test_*.py               # each other test pins one rule from the design
```

## Roadmap

- ~~Checkpoint persistence + replay from trace~~ done
- ~~An LLM-backed planner behind the same callable interface~~ done
- ~~Eval harness: golden traces as regression tests~~ done
- ~~Context compaction at the token watermark~~ done
- First live-model trace, published as a post — the run kit is
  `examples/live_run.py`; it needs a key and a human
- ~~ADRs in the repo~~ done: [docs/adr/](docs/adr/) holds 0001 (Python
  reference, C++ where it counts) and 0002 (seeded sweep over a
  property-testing framework); new decisions get a new number

MIT licensed. By [Shashi Shankar](https://shashforge.dev).
