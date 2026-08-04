"""Tool contracts.

Permissions live on the contract, not in the runtime. The runtime
enforces; the contract declares. A scope change is a reviewable diff.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .errors import PermissionDenied


@dataclass(frozen=True)
class ToolContract:
    name: str
    scopes: tuple[str, ...]              # e.g. ("repo:read",)
    idempotent: bool = False
    timeout_s: float = 8.0
    max_calls_per_run: int = 20
    max_retries: int = 2                 # for transient failures only
    verifier: Optional[str] = None       # name of the verifier to run

    def check_scopes(self, granted: frozenset[str]) -> None:
        missing = [s for s in self.scopes if s not in granted]
        if missing:
            raise PermissionDenied(
                f"tool '{self.name}' requires scopes {missing} "
                f"not granted to this run"
            )


@dataclass
class Tool:
    contract: ToolContract
    fn: Callable[..., object]
    calls_made: int = field(default=0)

    def spend_call(self) -> bool:
        """Returns False when the per-run call budget is exhausted."""
        if self.calls_made >= self.contract.max_calls_per_run:
            return False
        self.calls_made += 1
        return True
