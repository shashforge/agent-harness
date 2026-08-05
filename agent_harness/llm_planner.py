"""An LLM as the planner. The executor never finds out.

The design rule this module exists to prove: the planner is a plain
callable. This one happens to phone a model; the executor still owns
control flow, budgets, verification, and escalation. Swap it back for
a scripted function and nothing else in the harness changes.

The output contract is deliberately rigid. The model answers with one
JSON object per turn:

    {"tool": "<name>", "args": {...}}      one step
    {"done": true}                          goal met

Anything else raises PlannerProtocolError. No guessing, no repair by
imagination — the taxonomy applies to models the way it applies to
tools.

Transport is injectable: it is a function from request payload to
response body. Tests script it; production uses the Anthropic Messages
API with a key from ANTHROPIC_API_KEY. There is no third mode.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Callable, Optional

from .errors import PlannerProtocolError
from .executor import Step

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-5"

_SYSTEM = """You are the planner inside an agent execution harness.
The harness — not you — runs tools, verifies results, and checkpoints
progress. Your only job: given the goal and the verified checkpoints so
far, decide the single next step.

Respond with exactly one JSON object and nothing else:
{"tool": "<name>", "args": {...}} to run a tool, or {"done": true} when
the checkpoints show the goal is met. Only use tools from the catalog."""


def anthropic_transport(payload: dict) -> dict:
    """POST to the Messages API. Requires ANTHROPIC_API_KEY."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. LLMPlanner makes real model "
            "calls; there is no offline fallback by design."
        )
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": API_VERSION,
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


class LLMPlanner:
    """planner(checkpoints) -> Step | None, decided by a model.

    goal: what the run is trying to accomplish, in plain language
    tool_catalog: name -> one-line description shown to the model
    transport: request-payload dict -> response-body dict
    """

    def __init__(
        self,
        goal: str,
        tool_catalog: dict[str, str],
        model: str = DEFAULT_MODEL,
        transport: Callable[[dict], dict] = anthropic_transport,
        max_tokens: int = 1024,
    ) -> None:
        self.goal = goal
        self.tool_catalog = dict(tool_catalog)
        self.model = model
        self.transport = transport
        self.max_tokens = max_tokens

    def __call__(self, checkpoints: list[dict]) -> Optional[Step]:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": _SYSTEM,
            "messages": [{"role": "user", "content": self._prompt(checkpoints)}],
        }
        reply = self._text_of(self.transport(payload))
        return self._parse(reply)

    # -- prompt -----------------------------------------------------------

    def _prompt(self, checkpoints: list[dict]) -> str:
        catalog = "\n".join(f"- {name}: {desc}"
                            for name, desc in self.tool_catalog.items())
        return (
            f"Goal:\n{self.goal}\n\n"
            f"Tool catalog:\n{catalog}\n\n"
            f"Verified checkpoints so far:\n"
            f"{json.dumps(checkpoints, indent=2)}\n\n"
            f"Next step?"
        )

    # -- parsing ----------------------------------------------------------

    @staticmethod
    def _text_of(response: dict) -> str:
        try:
            blocks = response["content"]
            return "".join(b["text"] for b in blocks
                           if b.get("type") == "text")
        except (KeyError, TypeError) as e:
            raise PlannerProtocolError(
                f"unreadable API response: {e}") from e

    def _parse(self, reply: str) -> Optional[Step]:
        text = reply.strip()
        if text.startswith("```"):            # tolerate a markdown fence,
            text = text.strip("`")            # nothing more
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            decision = json.loads(text)
        except json.JSONDecodeError as e:
            raise PlannerProtocolError(
                f"planner reply is not JSON: {reply[:200]!r}") from e

        if decision.get("done") is True:
            return None
        tool = decision.get("tool")
        args = decision.get("args", {})
        if not isinstance(tool, str) or not isinstance(args, dict):
            raise PlannerProtocolError(
                f"planner reply fits no contract: {decision!r}")
        if tool not in self.tool_catalog:
            raise PlannerProtocolError(
                f"planner chose {tool!r}, not in the catalog")
        return Step(tool, args)
