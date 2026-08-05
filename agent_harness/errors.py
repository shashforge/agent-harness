"""The failure taxonomy, as exceptions.

The rule: retries are for the world being flaky; escalation is for the
plan being wrong. Each class carries its allowed response with it, so the
executor never has to guess.
"""
from __future__ import annotations


class HarnessError(Exception):
    """Base class for every failure the executor understands."""


class TransientToolError(HarnessError):
    """Timeout, 5xx, flaky network. Allowed response: retry same step."""


class PermanentToolError(HarnessError):
    """4xx, schema mismatch. Allowed response: re-plan with another tool."""


class VerificationFailure(HarnessError):
    """Verifier rejected the step result. Retry within budget, then escalate."""


class BudgetExhausted(HarnessError):
    """Step, call, or retry budget spent. Allowed response: escalate."""


class PermissionDenied(HarnessError):
    """Scope check failed before the call. Escalate. Never retry."""


class ContextOverflow(HarnessError):
    """Token watermark crossed. Compact, then re-plan."""


class PlannerProtocolError(HarnessError):
    """The model broke the planner output contract. Never guess a fix."""
