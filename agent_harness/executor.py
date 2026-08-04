"""The executor: the state machine from the design, as code.

PLAN -> ACT -> VERIFY -> CHECKPOINT -> (loop | DONE)
VERIFY failure: retry within budget, then ESCALATE.
ESCALATE: a human is a state, not an exception handler.

Nothing is persisted as progress until it survives verification.
A checkpoint of unverified state is just a saved bug.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from .contract import Tool
from .errors import (
    BudgetExhausted,
    HarnessError,
    PermanentToolError,
    PermissionDenied,
    TransientToolError,
    VerificationFailure,
)
from .states import State, Transition


@dataclass(frozen=True)
class Step:
    """One planned unit of work: exactly one tool call."""

    tool: str
    args: dict


class Resolution(Enum):
    RESUME = "resume"    # human gave guidance; re-plan and continue
    ABORT = "abort"      # human stopped the run


@dataclass
class Escalation:
    step_index: int
    reason: str
    attempts: int
    trace: list[Transition]


@dataclass
class RunResult:
    final_state: State
    transitions: list[Transition]
    checkpoints: list[dict]

    @property
    def trace(self) -> list[dict]:
        return [t.as_record() for t in self.transitions]


class Executor:
    """Drives one run. Planner proposes steps; the harness owns control flow.

    planner(checkpoints) -> Step | None   (None means: goal met)
    verifiers: name -> fn(result) -> bool
    on_escalate(Escalation) -> Resolution
    """

    def __init__(
        self,
        tools: dict[str, Tool],
        verifiers: dict[str, Callable[[object], bool]],
        planner: Callable[[list[dict]], Optional[Step]],
        on_escalate: Callable[[Escalation], Resolution],
        granted_scopes: frozenset[str] = frozenset(),
        max_steps: int = 25,
    ) -> None:
        self.tools = tools
        self.verifiers = verifiers
        self.planner = planner
        self.on_escalate = on_escalate
        self.granted_scopes = granted_scopes
        self.max_steps = max_steps
        self._transitions: list[Transition] = []
        self._checkpoints: list[dict] = []
        self._seq = 0

    # -- trace ------------------------------------------------------------

    def _record(self, step: int, src: State, dst: State, reason: str,
                **detail: object) -> None:
        self._seq += 1
        self._transitions.append(
            Transition(self._seq, step, src, dst, reason, dict(detail))
        )

    # -- the loop ---------------------------------------------------------

    def run(self) -> RunResult:
        step_index = 0
        while True:
            if step_index >= self.max_steps:
                res = self._escalate(step_index, "step budget exhausted", 0)
                if res is Resolution.ABORT:
                    return self._finish(step_index, State.FAILED,
                                        "aborted after step budget")
                step_index = 0  # human granted a fresh budget

            # PLAN
            step = self.planner(list(self._checkpoints))
            if step is None:
                return self._finish(step_index, State.DONE, "goal met")
            self._record(step_index, State.PLAN, State.ACT,
                         "step planned", tool=step.tool)

            # ACT + VERIFY, with the taxonomy deciding what happens next
            outcome = self._act_and_verify(step_index, step)
            if outcome is _Outcome.CHECKPOINTED:
                step_index += 1
                continue
            if outcome is _Outcome.REPLAN:
                continue  # same step_index: a different plan, not progress
            if outcome is _Outcome.ABORTED:
                return self._finish(step_index, State.FAILED,
                                    "aborted by human")

    def _act_and_verify(self, step_index: int, step: Step) -> "_Outcome":
        tool = self.tools[step.tool]
        contract = tool.contract
        transient_left = contract.max_retries
        verify_left = contract.max_retries

        while True:
            # permission gate: checked before every call, never retried
            try:
                contract.check_scopes(self.granted_scopes)
            except PermissionDenied as e:
                self._record(step_index, State.ACT, State.ESCALATE,
                             "permission denied", error=str(e))
                res = self._escalate(step_index, str(e), 0)
                return (_Outcome.REPLAN if res is Resolution.RESUME
                        else _Outcome.ABORTED)

            if not tool.spend_call():
                res = self._escalate(step_index,
                                     f"call budget spent for {contract.name}", 0)
                return (_Outcome.REPLAN if res is Resolution.RESUME
                        else _Outcome.ABORTED)

            # ACT
            try:
                result = tool.fn(**step.args)
            except TransientToolError as e:
                if transient_left > 0:
                    transient_left -= 1
                    self._record(step_index, State.ACT, State.ACT,
                                 "transient error, retrying", error=str(e))
                    continue
                self._record(step_index, State.ACT, State.ESCALATE,
                             "transient retries exhausted", error=str(e))
                res = self._escalate(step_index, str(e), contract.max_retries)
                return (_Outcome.REPLAN if res is Resolution.RESUME
                        else _Outcome.ABORTED)
            except PermanentToolError as e:
                self._record(step_index, State.ACT, State.PLAN,
                             "permanent tool error, re-planning", error=str(e))
                return _Outcome.REPLAN

            # VERIFY
            self._record(step_index, State.ACT, State.VERIFY, "result produced")
            if contract.verifier is not None:
                ok = self.verifiers[contract.verifier](result)
                if not ok:
                    if verify_left > 0:
                        verify_left -= 1
                        self._record(step_index, State.VERIFY, State.ACT,
                                     "verification failed, retrying")
                        continue
                    self._record(step_index, State.VERIFY, State.ESCALATE,
                                 "verification budget spent")
                    res = self._escalate(step_index, "verification failed",
                                         contract.max_retries)
                    return (_Outcome.REPLAN if res is Resolution.RESUME
                            else _Outcome.ABORTED)

            # CHECKPOINT: only verified state is progress
            self._record(step_index, State.VERIFY, State.CHECKPOINT,
                         "verified")
            self._checkpoints.append(
                {"step": step_index, "tool": step.tool, "result": result}
            )
            return _Outcome.CHECKPOINTED

    # -- terminal & escalation --------------------------------------------

    def _escalate(self, step_index: int, reason: str,
                  attempts: int) -> Resolution:
        esc = Escalation(step_index, reason, attempts,
                         list(self._transitions))
        res = self.on_escalate(esc)
        self._record(step_index, State.ESCALATE,
                     State.PLAN if res is Resolution.RESUME else State.FAILED,
                     f"human: {res.value}")
        return res

    def _finish(self, step_index: int, final: State, reason: str) -> RunResult:
        src = State.PLAN if final is State.DONE else State.ESCALATE
        self._record(step_index, src, final, reason)
        return RunResult(final, list(self._transitions),
                         list(self._checkpoints))


class _Outcome(Enum):
    CHECKPOINTED = "checkpointed"
    REPLAN = "replan"
    ABORTED = "aborted"
