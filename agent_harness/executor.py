"""The executor: the state machine from the design, as code.

PLAN -> ACT -> VERIFY -> CHECKPOINT -> (loop | DONE)
VERIFY failure: retry within budget, then ESCALATE.
ESCALATE: a human is a state, not an exception handler.

Nothing is persisted as progress until it survives verification.
A checkpoint of unverified state is just a saved bug.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from .contract import Tool
from .errors import (
    ContextOverflow,
    PermanentToolError,
    PermissionDenied,
    TransientToolError,
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

    store: anything with append_transition / append_checkpoint / load /
    last_seq (see persistence.JsonlStore). With resume=True, verified
    checkpoints from the store are loaded before the first plan, and the
    sequence counter continues where the previous run stopped — one
    continuous trace across process deaths.

    context_budget / compactor (see context.ContextBudget and
    context.FoldingCompactor): when the budget's watermark is crossed,
    the compactor shrinks the *view* handed to the planner. The
    checkpoint list, the store, and the trace are never compacted.
    """

    def __init__(
        self,
        tools: dict[str, Tool],
        verifiers: dict[str, Callable[[object], bool]],
        planner: Callable[[list[dict]], Optional[Step]],
        on_escalate: Callable[[Escalation], Resolution],
        granted_scopes: frozenset[str] = frozenset(),
        max_steps: int = 25,
        store: object = None,
        resume: bool = False,
        context_budget: object = None,
        compactor: Optional[Callable[[list[dict]], list[dict]]] = None,
    ) -> None:
        self.tools = tools
        self.verifiers = verifiers
        self.planner = planner
        self.on_escalate = on_escalate
        self.granted_scopes = granted_scopes
        self.max_steps = max_steps
        self.store = store
        self.context_budget = context_budget
        self.compactor = compactor
        self._transitions: list[Transition] = []
        self._checkpoints: list[dict] = []
        self._seq = 0
        if resume:
            if store is None:
                raise ValueError("resume=True requires a store")
            _, checkpoints = store.load()
            self._checkpoints = list(checkpoints)
            self._seq = store.last_seq()

    # -- trace ------------------------------------------------------------

    def _record(self, step: int, src: State, dst: State, reason: str,
                **detail: object) -> None:
        self._seq += 1
        t = Transition(self._seq, step, src, dst, reason, dict(detail))
        self._transitions.append(t)
        if self.store is not None:
            self.store.append_transition(t)

    # -- the loop ---------------------------------------------------------

    def run(self) -> RunResult:
        step_index = len(self._checkpoints)   # 0 fresh; further on resume
        while True:
            if step_index >= self.max_steps:
                res = self._escalate(step_index, "step budget exhausted", 0)
                if res is Resolution.ABORT:
                    return self._finish(step_index, State.FAILED,
                                        "aborted after step budget")
                step_index = 0  # human granted a fresh budget

            # PLAN — through the context lens if a budget is set
            view = self._planner_view(step_index)
            if view is None:                      # human aborted at the lens
                return self._finish(step_index, State.FAILED,
                                    "aborted at context watermark")
            step = self.planner(view)
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

    def _planner_view(self, step_index: int) -> Optional[list[dict]]:
        """The checkpoints the planner is shown. Compaction happens
        here and only here — the record itself is never touched.

        Returns None only when the human aborts at an escalation.
        """
        view = list(self._checkpoints)
        if self.context_budget is None:
            return view
        try:
            self.context_budget.check(view)
            return view
        except ContextOverflow as e:
            if self.compactor is None:
                self._record(step_index, State.PLAN, State.ESCALATE,
                             "context watermark crossed, no compactor",
                             error=str(e))
                res = self._escalate(step_index, str(e), 0)
                return view if res is Resolution.RESUME else None

        before = self.context_budget.units(view)
        view = self.compactor(view)
        after = self.context_budget.units(view)
        self._record(step_index, State.PLAN, State.PLAN,
                     "context compacted", units_before=before,
                     units_after=after)
        try:
            self.context_budget.check(view)
            return view
        except ContextOverflow as e:
            self._record(step_index, State.PLAN, State.ESCALATE,
                         "still over watermark after compaction",
                         error=str(e))
            res = self._escalate(step_index, str(e), 1)
            return view if res is Resolution.RESUME else None

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
            checkpoint = {"step": step_index, "tool": step.tool,
                          "result": result}
            self._checkpoints.append(checkpoint)
            if self.store is not None:
                self.store.append_checkpoint(checkpoint)
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
