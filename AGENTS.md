# AGENTS.md — operational contract for remec

This file covers **how to work**. `docs/DESIGN.md` covers **what to build** and is
authoritative for architecture; `docs/20260815-01_Regularized_3D_MHD_equilibrium.tex`
("the note") is authoritative for the mathematics.

Read `docs/DESIGN.md` §26 before your first code change in a session. Do not restate it
here; it is binding.

## Routing — which files to read

Read these, in order, and stop:

1. `docs/STATUS.md` — the milestone ledger. It tells you what is done and what is next.
2. `docs/DESIGN.md` §25 for the milestone definition and acceptance criteria.
3. The `docs/DESIGN.md` section named in the STATUS.md row for that milestone.
4. The note section named in that row.
5. `docs/dev_notes.md` — accumulated NGSolve API reality.

Do **not** read the whole design document top to bottom every session. Do **not** read
the whole `.tex` note; read the cited section.

## Environment

```bash
python3 -m venv .venv  # Python 3.10 or newer
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

- NGSolve ships as a binary wheel. Never build it from source, never add a compiler
  requirement, never add MPI or PETSc to base dependencies (`docs/DESIGN.md` §26).
- Verify with `python -c "import ngsolve; print(ngsolve.__version__)"` before assuming an
  environment problem is a code problem.

## Commands

```bash
make test          # not-slow subset, the PR-CI gate; prints the 15 slowest tests
make test-full     # includes slow/nightly-marked tests
make lint          # ruff format --check && ruff check && mypy src/remec
make check         # lint + test; this is the gate
```

If a command does not exist yet, you are in Phase 0 — create it.

## Definition of done for a milestone

A milestone is complete only when **all** of the following hold. Do not open a PR before
they do. In that same PR, change the milestone's `docs/STATUS.md` marker to `[x]`; `[x]`
means complete in the submitted PR (or merged), not already merged. Do not leave a
review-ready or submitted milestone as `[~]`.

1. `make check` passes from a clean environment.
2. The specific acceptance criterion for this milestone in `docs/DESIGN.md` §25 is
   demonstrated by an automated test, not by a manual run or a claim in the PR body.
3. Every nontrivial weak form or operator carries a docstring with the equation label
   from the note (M1)–(M4b) and the formula it implements.
4. Convergence claims are backed by a measured rate table checked into
   `tests/manufactured/` and referenced in `docs/verification.md`. A residual that got
   small is not a convergence result (`docs/DESIGN.md` §26).
5. `docs/STATUS.md` is updated in the PR: mark the row `[x]`, record the measured
   numbers, and note anything the next milestone should know. Use `[~]` only while the
   work is genuinely incomplete and not ready for review.
6. Any NGSolve API surprise is appended to `docs/dev_notes.md`.
7. The test-time budgets below still hold with the milestone's new tests in place.

## Test-first, and tests that can fail

Write the test before the implementation. The design document specifies manufactured
solutions and expected rates precisely enough that this is possible.

A test that passes for an implementation you know to be wrong is worse than no test.
Where `docs/DESIGN.md` §22 names a term whose omission must be conspicuous — e.g. the
`D_u ∇⊥u·∇p` term in M3 — write the test so that deleting the term makes it fail, and
say in the PR body which mutation you verified it catches.

## Test speed — a budget, not an aspiration

A slow suite is a suite that stops being run. The budgets are normative
(`docs/DESIGN.md` §22.1); these are the working rules.

**The budgets**, wall-clock on the reference laptop (Apple-silicon macOS, the `-n 3`
xdist configuration in `pyproject.toml`):

| What | Budget |
|---|---|
| `make test` (`-m "not slow"`) | **< 2 min** |
| any single not-slow test | **< ~20 s** |
| `make test-full` | **< 5 min** |
| any single `slow` test | **< ~90 s** |

`make test` prints the 15 slowest tests on every run. Read that list; it is the only
early warning you get. Nothing fails purely on wall-clock — wall-clock assertions are
flaky across machines — so the budget is your responsibility, not CI's.

**Marking `slow`.** A test that cannot be brought under the not-slow caps gets
`@pytest.mark.slow` and runs only in `.github/workflows/nightly.yml` (`make test-full`).
Prefer marking the *longest* tests, and prefer keeping a cheap version of the same check
in the not-slow suite: a two-point rate check at low resolution in PR CI, the full
resolution/anisotropy scan nightly.

**While working a milestone.** You need not run the whole slow suite. Run `make test`,
plus every `slow` test that touches the code you changed (`pytest -m slow <path or -k>`),
and say in the PR body which slow tests you ran. If `make test` is over budget after
adding your tests, you may mark unrelated slow-but-passing tests `slow` to fit — but
**never** mark a test `slow` to get a failure out of your way. Marking a failing test
`slow` is the same offence as `xfail`-ing it (see STOP conditions).

**How to make a test fast**, in the order to try:

1. Lowest resolution that still demonstrates the claim. A convergence rate needs enough
   points to fit a slope, not a pretty table.
2. Do not recompute. Hoist mesh construction, assembly, and solves into module- or
   session-scoped fixtures and let several assertions share one solve. Note that
   `--dist=loadscope` keeps a module's tests on one worker, so module-scoped fixtures
   pay off.
3. `CoefficientFunction.Compile()` — see the `_compiled()` helpers and the timing entry
   in `docs/dev_notes.md`. It cut assembly ~24x for the M3 forms and is bitwise
   identical, so no recorded rate table moves.
4. Cheaper diagnostics: `diagnostic_detail="core"`, lower integration order, fewer
   sampled points, looser *algebraic* solver tolerance (never a looser *accuracy*
   tolerance in an assertion).

**Deleting tests.** If a test has been made irrelevant by a code change, an ADR, or a
change to the development plan, delete it and say so in the PR body. Dead tests cost
time and mislead reviewers. If it is still meaningful but expensive, mark it `slow`
instead.

**Retrofitting old tests.** If your milestone's own tests are lean and the suite is
still over budget, speed up the slowest existing tests, worst first. When you reduce an
existing test's resolution, minimize the loss of meaningful coverage — keep the same
mutation-detection and the same asymptotic rate, and record the before/after numbers in
`docs/STATUS.md` and any affected table in `docs/verification.md`. Rate tables checked
into `tests/manufactured/` must be regenerated, not hand-edited. Reducing an expected
convergence rate or loosening an accuracy tolerance to save time is a STOP condition,
not an optimization.

## Scope discipline

One milestone per branch, per PR. If you find work that belongs to a different
milestone, write it into `docs/STATUS.md` under that milestone's row and leave it alone.
Do not opportunistically refactor.

## STOP conditions — surface to the human, do not decide

Stop, write the artifact named below, and end your turn. Do not pick an option and
continue.

| Situation | What to write |
|---|---|
| The note and `docs/DESIGN.md` appear to conflict | Draft ADR in `docs/adr/`, quoting both passages |
| A `docs/DESIGN.md` decision looks wrong or infeasible as written | Draft ADR proposing the change with evidence |
| The mathematics is ambiguous and the choice affects results | Draft ADR with the candidate readings |
| An acceptance criterion cannot be met and you want to relax it | Draft ADR. **Never** loosen a tolerance, mark a test `xfail`, or reduce a convergence-rate expectation to make CI green |
| A new base dependency seems necessary | Draft ADR; see the MUST NOT list in §26 |
| The milestone needs a decision the design document does not cover | Draft ADR |

An ADR is a numbered file in `docs/adr/` with: context, the options, the tradeoffs, your
recommendation, and an explicit "DECISION: pending human sign-off" line. Commit it, push
it, and say in the PR body that the branch is blocked on it.

Purely mechanical discrepancies — a renamed NGSolve argument, a missing solver option —
are not stop conditions. Fix them and record them in `docs/dev_notes.md`.

## Git

- Branch name: `milestone/<number>-<slug>`, e.g. `milestone/1.3-pollution-benchmark`.
- Commit in logical groups, not one giant commit. Reference the milestone number.
- Open the PR with `gh pr create`. The PR body must contain: the milestone number, the
  equations and invariants touched, the acceptance criterion and how it is demonstrated,
  the measured numbers, mutations the tests were verified to catch, and any open ADR.
- Never merge your own PR. Never force-push to `main`.

## Reviewing (when acting as reviewer rather than implementer)

See `.claude/commands/review-milestone.md`. The question is not "is this clean Python"
but "would this pass if the physics were wrong".
