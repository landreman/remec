---
description: Adversarial review of a milestone PR against DESIGN.md and the note
argument-hint: "[PR number, or blank for the current branch]"
allowed-tools: Bash(git diff:*), Bash(git log:*), Bash(gh pr:*), Bash(make:*), Bash(pytest:*), Read, Grep, Glob
---

Review $ARGUMENTS as an adversarial reviewer. Assume the implementation is plausible,
well-formatted, and passing CI, and that if it is wrong the tests are wrong too. Your job
is to find that, not to comment on style — ruff already did that.

Read the diff, then `docs/DESIGN.md` §25 for this milestone's acceptance criteria,
§5 for the invariants, §26 for the MUST NOT list, and the note sections the PR body
names. Read the note yourself; do not take the PR body's characterization of it.

The milestone's `docs/STATUS.md` row MUST be changed to `[x]` in its implementation PR
once the PR satisfies its definition of done. Here `[x]` means complete in the submitted
PR (or merged), not previously merged. Do not report that marker as a finding merely
because the PR is unmerged; instead, report a finding only if the PR fails the documented
definition of done or if an unrelated milestone's state is changed.

Answer these, in order, each with a verdict and the evidence you checked:

**1. Does the test actually constrain the physics?**
For each new test: what implementation would pass it that is nevertheless wrong? If you
can construct one, that is a finding. Specifically check that the test would fail if a
term were dropped, if the anisotropic tensor were replaced by its isotropic part, if a
sign were flipped, or if a boundary condition were imposed on the wrong space. The PR
body claims certain mutations were verified — spot-check one of them yourself by
applying it and running the test.

**2. Is a convergence claim really a convergence claim?**
Rates must be measured across a refinement sweep in the norms `docs/DESIGN.md` §22
names, not inferred from a shrinking residual. Check that the sweep is wide enough for
the rate to be meaningful and that the reported rate matches the checked-in table.

**3. Do the equations in the code match the note?**
Term by term, for every weak form in the diff. Check signs, factors, which quantity each
test function is paired against, and whether any term present in the note is absent from
the code. This is the highest-value part of the review; spend the most time here.

**4. Any §26 MUST NOT violated?**
Silent clipping of pressure or field; isotropic approximation not behind an explicit
experimental flag; field-aligned coordinates assumed; histogram volume map in a Newton
path; NGSolve internals leaking into the public API; a new base dependency; a dense
global Jacobian; direct 3D factorization as the planned production path.

**5. Was an acceptance criterion quietly weakened?**
Diff the tolerances, expected rates, `xfail`/`skip` markers, and CI matrix against
`main`. Any loosening without an accepted ADR is a blocking finding regardless of how
reasonable the justification sounds.

**6. Undocumented decisions.**
Anywhere the implementer resolved an ambiguity in the note or the design document
without an ADR. §26 requires an ADR or an issue, not a silent choice.

Output as a table of findings: severity (blocking / should-fix / note), location, what
is wrong, and what to do about it. Then one line: **merge / fix first / needs ADR**.

If you find nothing blocking, say so plainly. Do not invent findings to seem thorough,
and do not soften a blocking finding into a suggestion.
