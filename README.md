# agent-harness

A reference implementation of an agent execution harness: an explicit
state machine instead of a `while` loop with vibes.

**Status: skeleton.** The executor, tool contracts, failure taxonomy, and
transition trace work and are tested. No LLM planner is wired in yet —
the planner is a plain callable, which is the point: the harness owns
control flow; the model is a component.

Design writing behind this code:

- [The model was never the product](https://shashforge.dev/log/the-model-was-never-the-product/) — why the harness is the product
- [The harness gets a spine](https://shashforge.dev/log/the-harness-gets-a-spine/) — the state machine, failure taxonomy, and tool contract this repo implements

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

## Run the tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Layout

```
agent_harness/
  states.py     # State enum + Transition (the replayable trace record)
  errors.py     # the failure taxonomy, as exceptions
  contract.py   # ToolContract: scopes, budgets, idempotency, verifier
  executor.py   # the state machine
tests/
  test_executor.py  # each test pins one rule from the design
```

## Roadmap

- Checkpoint persistence + replay from trace
- An LLM-backed planner behind the same callable interface
- Eval harness: golden traces as regression tests
- ADRs, starting with [ADR-001: Python for the reference, C++ where it counts](https://shashforge.dev/log/adr-001-python-vs-cpp/)

MIT licensed. By [Shashi Shankar](https://shashforge.dev).
