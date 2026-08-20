# ADR-0001: Python for the reference, C++ where it counts

**Status:** accepted. **Date:** 2026-08-04. **Decider:** Shashi Shankar.
Written up in public first at
<https://shashforge.dev/log/adr-001-python-vs-cpp/>; this file backfills
the record into the repo that lives by it.

## Context

The harness needs a reference implementation. I have written C++ for a
living for most of my career; every LLM SDK, eval framework, and serving
integration worth touching is Python-first. The goal of this repo is a
reference architecture people can read in an afternoon, and evidence I
can ship regularly, not a performance record.

## Options

**A. Python end-to-end.** Fast to write, everyone can read it, plugs
straight into the model SDKs. Slower per operation, and it spends none
of my C++ depth.

**B. C++ core with Python bindings from day one.** Home turf, fast.
Also: binding maintenance, slower iteration, and a smaller audience for
a repo whose purpose is to be read.

**C. Python reference now; port hot paths to C++ only when a benchmark
says so.**

| Dimension | A: Python | B: C++ core | C: Hybrid |
|---|---|---|---|
| Iteration speed | high | low | high now |
| Readability as reference | high | medium | high |
| Ecosystem fit (SDKs, evals) | native | bindings | native |
| Runtime performance | fine | best | fine, path to best |
| Uses my C++ depth | no | yes | yes, where measured |

## Decision

Option C. The harness is control flow, not compute. A run spends
seconds waiting on model calls and tool I/O; a state transition costs
microseconds. Writing the executor in C++ optimizes the one part of the
system that is never the bottleneck, which is premature optimization
committed at the architecture level.

C++ gets spent where measurements will point: the Edge AI lane, where
on-device inference runtimes (ONNX Runtime, LiteRT, ExecuTorch) are C++
under the hood.

## Consequences

Easier: shipping the skeleton, wiring in a real LLM planner, letting
others read it. Harder: nothing yet. Revisit when: trace storage or
replay shows up hot in a profile. That module becomes the first C++
port, with the benchmark published before the port.

## Record of what followed

As of 2026-08-17: 868 lines of Python, 288 collected tests, no profile
has yet pointed anywhere. The decision stands untested in the way it
was meant to be tested, which is the correct state for it to be in.
