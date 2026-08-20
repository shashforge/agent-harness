# ADR-0002: Seeded sweep over a property-testing framework

**Status:** accepted. **Date:** 2026-08-17. **Decider:** Shashi Shankar.
The code this records landed on 2026-08-14 in `tests/test_invariants.py`;
the write-up is at <https://shashforge.dev/log/adr-002-seeds-over-hypothesis/>.

## Context

The executor's invariants (terminal state always; gapless, strictly
increasing trace; checkpoint one for one with verification; no tool
called past its budget; every trace replays) needed testing against
behavior I did not hand-pick. Two ways to generate that behavior:
a property-testing framework, or a seeded `random.Random` and a
parametrized test.

The repo has one runtime dependency (none) and one test dependency
(pytest). CI is a 29-line workflow that installs pytest and runs.
The repo's stated purpose is to be read as a reference in an afternoon.

## Options

**A. Hypothesis.** Strategies for planners, tools, verifiers; `@given`
over the executor; shrinking on failure. Battle-tested, expressive,
and the shrinker is real engineering I would not want to rewrite.

**B. Seeded stdlib random.** `random.Random(seed)` builds each world;
`@pytest.mark.parametrize("seed", range(N))` runs them; a failing seed
number is the reproduction.

**C. Both.** Hypothesis for the deep sweep, seeds for the CI-fast
version. Two mechanisms for one job.

| Dimension | A: Hypothesis | B: Seeds | C: Both |
|---|---|---|---|
| New dependency | yes | no | yes |
| Failure minimization | shrinker | none; seed only | shrinker |
| Reproduction | example database + seed | seed number alone | mixed |
| CI change | pin version, cache database | none | yes |
| Reader can follow the file cold | with framework fluency | yes | no |
| Runtime for 250 worlds | seconds, tunable | 0.7 s | more |

## Decision

Option B. The argument that settled it: **the failure mode I most need
to defend against is a reader not trusting the test, and every line of
framework between the reader and the executor costs trust.** A seeded
sweep is 169 lines a stranger can read top to bottom and see exactly
what is generated and exactly what is asserted. The reproduction story
is one integer.

Shrinking is the real thing given up. In this codebase a failing seed
already produces a trace of at most a few dozen transitions with every
reason string attached, which is a counterexample small enough to read
without minimization. If a future invariant fails in a way that needs
shrinking to understand, that is the signal to revisit.

Termination is not left to randomness: generated humans have finite
patience (0 to 3 resumes, then abort), and a tripwire fails any seed
whose planner is consulted 10,000 times.

## Consequences

Easier: adding a seed range is one number; CI unchanged; the test file
doubles as documentation of the invariants. Harder: no automatic
minimization; the distributions are hand-written and only as hostile as
I made them; coverage of the input space is what 250 draws give, not
what a strategy-aware engine would find.

Revisit when: a seed fails and the trace is too large to read; or the
generated worlds need structure (nested plans, stateful tools) that
hand-rolled generators make ugly. Either would justify Hypothesis as an
additional dev dependency, kept out of the default CI job.
