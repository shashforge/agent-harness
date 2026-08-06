"""The first live run: a real model driving the harness.

Goal: the model reads this very repository — lists the Python files,
counts their lines, reads the README's title — and writes a short
report to out/report.md. Small enough to audit by hand, real enough
to mean something.

Run it from the repo root with your own key:

    ANTHROPIC_API_KEY=sk-... python examples/live_run.py

Everything the run does lands in out/run.jsonl as it happens. That
file is the deliverable: the unedited trace of a model behind the
planner callable, with every rule of the harness enforced on it.
If the model breaks the reply contract, the run stops and says so.
The human is a state: if the run escalates, you'll be asked to
resume or abort at the terminal.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_harness.contract import Tool, ToolContract          # noqa: E402
from agent_harness.executor import Executor, Resolution        # noqa: E402
from agent_harness.llm_planner import LLMPlanner               # noqa: E402
from agent_harness.persistence import JsonlStore               # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "out"

GOAL = """Produce a short report about the agent-harness repository and
save it. Do it in this order:
1. List the repository's Python files.
2. Count the lines of each file in agent_harness/ (the package, not tests).
3. Read the first line of README.md.
4. Write out/report.md: the README title, a table of package files with
   line counts, and a one-sentence conclusion. Then declare done."""


def _inside_repo(path: str) -> Path:
    p = (REPO / path).resolve()
    if not p.is_relative_to(REPO):
        raise ValueError(f"path escapes the repository: {path}")
    return p


def list_python_files() -> list[str]:
    return sorted(str(p.relative_to(REPO))
                  for p in REPO.rglob("*.py")
                  if "out" not in p.parts and "__pycache__" not in p.parts)


def count_lines(path: str) -> int:
    return len(_inside_repo(path).read_text().splitlines())


def read_first_line(path: str) -> str:
    return _inside_repo(path).read_text().splitlines()[0]


def write_report(content: str) -> str:
    OUT.mkdir(exist_ok=True)
    target = OUT / "report.md"
    target.write_text(content)
    return f"wrote {len(content)} chars to out/report.md"


def report_written(result: object) -> bool:
    target = OUT / "report.md"
    return target.exists() and len(target.read_text()) > 100


def ask_human(escalation) -> Resolution:
    print(f"\nESCALATION at step {escalation.step_index}: "
          f"{escalation.reason}")
    answer = input("resume or abort? [abort] ").strip().lower()
    return Resolution.RESUME if answer == "resume" else Resolution.ABORT


def tool(name, fn, scopes, verifier=None):
    return Tool(ToolContract(name=name, scopes=scopes, verifier=verifier,
                             max_calls_per_run=30), fn)


def main() -> None:
    trace_path = OUT / "run.jsonl"
    if trace_path.exists():
        sys.exit("out/run.jsonl already exists — move it aside first. "
                 "The trace is append-only and this script won't stack "
                 "two runs into one file.")

    tools = {
        "list_python_files": tool("list_python_files",
                                  lambda **kw: list_python_files(),
                                  ("fs:read",)),
        "count_lines": tool("count_lines",
                            lambda path, **kw: count_lines(path),
                            ("fs:read",)),
        "read_first_line": tool("read_first_line",
                                lambda path, **kw: read_first_line(path),
                                ("fs:read",)),
        "write_report": tool("write_report",
                             lambda content, **kw: write_report(content),
                             ("fs:write",), verifier="report_written"),
    }
    catalog = {
        "list_python_files": "no args; returns every .py path in the repo",
        "count_lines": "args: {path}; returns the file's line count",
        "read_first_line": "args: {path}; returns the file's first line",
        "write_report": "args: {content}; saves out/report.md",
    }

    OUT.mkdir(exist_ok=True)
    ex = Executor(
        tools=tools,
        verifiers={"report_written": report_written},
        planner=LLMPlanner(GOAL, catalog),
        on_escalate=ask_human,
        granted_scopes=frozenset({"fs:read", "fs:write"}),
        max_steps=15,
        store=JsonlStore(trace_path),
    )
    result = ex.run()

    print(f"\nfinal state:  {result.final_state.value}")
    print(f"checkpoints:  {len(result.checkpoints)}")
    print(f"transitions:  {len(result.transitions)}")
    print(f"trace:        {trace_path.relative_to(REPO)}")
    if (OUT / "report.md").exists():
        print(f"report:       out/report.md")


if __name__ == "__main__":
    main()
