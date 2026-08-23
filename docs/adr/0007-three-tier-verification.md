# ADR 0007: Three-tier verification for expensive three-dimensional benchmarks

**Status:** Accepted

## Context

`DESIGN.md` Section 22.1 originally divided verification into two tiers: a fast
`not slow` PR subset and a `slow` full suite. The full suite had a five-minute budget,
each slow test had a roughly 90-second budget, and agents were required to run every
slow test touching their change before submission.

That contract worked for the two-dimensional and reduced milestones. The final
milestone-5.5 implementation measured 56.23 seconds for 289 fast tests and 193.30
seconds for all 309 tests. A literal recomputation of that milestone's fine table took
477.30 seconds and contained individual 105--168 second cases before solve sharing and
accepted restart states brought it back under the two-tier limits.

Phase 6 adds three-dimensional resolution, polynomial-order, anisotropy, and control
ladders. Section 8.6 deliberately requires those ladders to report cost, memory,
pollution, layer resolution, and pressure flattening. Making every row small enough for
the existing slow tier would either make the development loop unacceptably long or
create pressure to reduce scientifically meaningful coverage. Merely raising the
existing `slow` limits would also require agents to run multi-hour ladders locally
before each submission, preserving the same problem under a larger number.

The numerical acceptance criteria, convergence-rate expectations, pollution limits,
`min_layer_cells`, and mutation controls are not at issue and must not be relaxed.

## Options

1. **Raise the existing slow/full-suite limits.** Keep two tiers but allow slow tests
   and `make test-full` to run much longer.
2. **Add a remote exhaustive tier.** Preserve the fast and developer-slow limits, put
   full large parameter ladders in a separately marked remote tier, and require live
   fast/developer sentinels plus successful remote execution before milestone
   completion.
3. **Check in expensive tables without rerunning them.** Regenerate large tables only
   on explicit human request and let ordinary CI validate their schema and pinned
   values.

## Tradeoffs

Option 1 is mechanically simple, but it slows every agent that follows the touched-test
rule and encourages fewer local verification runs. It also repeats heavy numerical
physics across compatibility environments even though Python-version coverage and the
large parameter study answer different questions.

Option 2 introduces another marker, Make target, CI job, and completion rule. In
return, regressions are caught at three useful timescales: fast tests during editing,
bounded subsystem tests before submission, and full numerical evidence remotely. A
failure in the remote tier may arrive later than a local failure, so every exhaustive
family needs a small mutation-sensitive live sentinel. Remote compute can be sharded
without making an individual developer wait for the sum of all rows.

Option 3 minimizes compute but permits implementation drift between table
regenerations. Schema and provenance checks cannot establish that current solver code
still produces the recorded physics.

## Decision

Choose Option 2. Verification has these tiers:

1. **Fast:** tests marked neither `slow` nor `exhaustive`; `make test`; less than two
   minutes for the suite and roughly 20 seconds for any individual test on the
   reference laptop. This is the PR gate used by `make check`.
2. **Developer-slow:** all non-exhaustive tests, including `slow`; `make test-full`;
   less than five minutes for the suite and roughly 90 seconds for any individual slow
   test. Agents run the slow tests that touch their change before submission.
3. **Remote exhaustive:** all tests, including `exhaustive`; `make test-exhaustive`.
   There is no normative laptop wall-clock or per-test cap. The scheduled/manual
   workflow has an infrastructure timeout, not an accuracy-driven test budget.

An exhaustive test is legitimate only when:

- the parameter range or resolution is necessary for a documented numerical claim;
- the test declares its family with `@pytest.mark.sentinel("<family>")`, and that family
  has a fast live sentinel — and, where a useful intermediate form exists, a
  developer-slow sentinel — using the same production path and scientific gates;
- at least one live sentinel is mutation-sensitive to the physics the exhaustive rows
  claim to verify;
- independent rows are separate pytest nodes or modules so CI can shard them when the
  suite grows; and
- any checked-in table is produced only by its committed regeneration script.

The structural half of the sentinel rule is enforced mechanically, not by review alone.
`tests/unit/test_verification_tiers.py` runs in the fast tier and fails if an
`exhaustive` test declares no family, or if a declared family has no test outside both
slower tiers. A `slow` sentinel deliberately does not satisfy that requirement: `slow`
is not run on every change, which is the property the sentinel exists to supply. The
scientific half — same production path, same gates, genuine mutation sensitivity —
cannot be checked statically and remains a blocking reviewer obligation
(`.claude/commands/review-milestone.md` item 6).

Agents do not routinely run exhaustive tests locally. If a milestone adds an exhaustive
test or changes its solver path, inputs, controls, regeneration code, or asserted
artifact, the agent pushes the branch, manually dispatches
`.github/workflows/exhaustive.yml` for that branch, and records the successful run in
the PR body. That run must be a complete one: the workflow accepts a `pytest_args`
input for iterating on a single family, and writes to the job summary whether the run
was complete or filtered, so a partial run cannot be presented as the required
evidence. The milestone is not complete until a complete run passes. Intermediate
commits do not wait for the workflow.

The two tiers live in separate workflow files. `nightly.yml` runs the bounded developer
suite on the oldest and newest supported Python versions; `exhaustive.yml` runs the
numerical ladders once on a canonical Python/NGSolve environment. This separates
interpreter compatibility from numerical parameter coverage, lets the two cadences and
timeouts move independently, and means a branch dispatch of the exhaustive tier does not
also re-run the compatibility matrix. Both jobs install the optional extras, because a
tier whose tests `importorskip` away is not covering what it claims. The exhaustive job
should be split across independent CI matrix shards when one serial job becomes the
bottleneck.

The exhaustive tier is memory-bound rather than CPU-bound: the Section 21 direct solver
holds a full three-dimensional factorization per pytest-xdist worker, so the fixed
`-n 3` that suits the bounded tiers can exhaust a runner. The Make targets therefore
expose `PYTEST_WORKERS` and `EXHAUSTIVE_WORKERS`, and the exhaustive tier defaults to
fewer workers. Worker count is a resourcing knob only; it may never be used to reach a
budget that should have been met by a cheaper formulation.

Meshes, analytic topology data, and restart states may be versioned inputs or initial
guesses when their configuration and schema metadata are checked. Cached or checked-in
final solver output must never supply the diagnostic or value being asserted by a live
test.

The user approved this policy by requesting implementation on 2026-08-22.

**DECISION: Option 2 approved.**
