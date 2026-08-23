---
name: milestone
description: Implement one numbered milestone from docs/DESIGN.md §25 end to end — branch, test-first implementation, verification, STATUS.md update, and PR. Use whenever the user asks to work on a milestone, implement the next milestone, continue the plan, or names a milestone number like 1.3 or 4.2.
---

# Implement one milestone

The user will either name a milestone (`1.3`) or say "next". If they say next, take the
first unchecked row in `docs/STATUS.md` whose prerequisites are all checked.

Work on exactly one milestone. Follow `AGENTS.md` for environment, commands, definition
of done, and STOP conditions.

## 1. Orient

Read `docs/STATUS.md`, then the milestone's entry in `docs/DESIGN.md` §25, then the
design sections and note sections named in the STATUS.md row.

Before writing code, state in two or three sentences:

- which equations from the note this milestone touches (by label),
- which invariants from `docs/DESIGN.md` §5 it must preserve,
- the acceptance criterion you will demonstrate and the number it must hit.

If the previous phase's acceptance criteria are not all green in CI, stop and say so —
`docs/DESIGN.md` §25 forbids starting a phase before that holds.

## 2. Branch

```bash
git checkout main && git pull
git checkout -b milestone/<number>-<slug>
```

## 3. Write the tests first

Derive them from `docs/DESIGN.md` §22 and the note, not from the implementation you are
about to write. Include the manufactured solution, the mesh/order sweep, and the
expected rate.

Then confirm the tests fail for the right reason. Run them against the unimplemented
stub and check that the failure is the physics you are about to add, not an import
error.

## 4. Implement

The minimum that satisfies the milestone. NGSolve stays behind `src/remec/fem/` and
`src/remec/solvers/`. Docstrings carry equation labels and formulas.

## 5. Verify

```bash
make check
```

Then verify the tests can fail: pick the one or two mutations that matter for this
milestone — dropping a term, replacing the tensor by its isotropic part, halving the
stabilization parameter — apply each, confirm the suite goes red, revert. Record which
mutations you checked; this goes in the PR body.

Re-run `make check` from a clean venv if you touched packaging.

You do not need to run all of `make test-full` here. Run `make test`, plus the `slow`
tests that touch what you changed:

```bash
python -m pytest -m slow tests/path/touched_by_this_milestone.py
```

Then check the budget (`DESIGN.md` §22.1, `AGENTS.md` "Test speed"): `make test` under
2 minutes, no single fast test over ~20 s in the durations report it prints. If your
new tests blow it, make them cheaper — lower resolution, shared fixtures, `_compiled()`
coefficient expressions — before you consider marking anything `slow`. Copy the durations
tail into the PR body.

Do not run `make test-exhaustive` locally as part of the ordinary loop. If the milestone
adds an `exhaustive` test or changes its solver path, input grid, controls, regeneration
script, or asserted table, push the branch and manually dispatch `exhaustive.yml` against
that branch:

```bash
gh workflow run exhaustive.yml --ref "$(git branch --show-current)"
```

Continue with documentation and other non-dependent work while it runs; check back with
`gh run list --workflow exhaustive.yml --branch "$(git branch --show-current)"`. Do not
mark the milestone complete until a complete exhaustive run is green. The workflow's
optional `pytest_args` input is for iterating on one family while developing; a filtered
run is labelled as partial in the job summary and does not satisfy the requirement.

Any new `exhaustive` test also needs `@pytest.mark.sentinel("<family>")` and a fast
sentinel in the same family, or `tests/unit/test_verification_tiers.py` fails in
`make test` — by design.

## 6. Run GitHub Actions CI

Push the branch to github and let the CI run. Check to make sure it passes.
If anything fails, fix the issue, push again, and iterate until the CI is green.

## 7. Record

Update `docs/STATUS.md`: mark the row, paste the measured numbers (convergence rates,
pollution values, residuals — whatever the acceptance criterion names), and add a line
for anything the next milestone needs to know.

Append any NGSolve API discoveries to `docs/dev_notes.md`. Update
`docs/verification.md` with the new rate table.

## 8. Open the PR

```bash
gh pr create --fill --draft
```

Open it as a **draft**. `claude-code-review.yml` triggers only on `ready_for_review`
(and `reopened`) — not on `opened` or on every push — so a draft PR, and any commits you
push while it stays draft, does not consume a review. Mark it ready only when you want a
Claude Code review pass (step 9):

```bash
gh pr ready <number>
```

PR body must contain, in this order:

- Milestone number and one-line summary
- Equations (note labels) and invariants (§5) affected
- The acceptance criterion, and the test that demonstrates it
- Measured numbers
- Mutations verified to turn the suite red
- `make test` wall-clock and its slowest-test list; which `slow` tests you ran; any test
  you newly marked `slow` or `exhaustive`, sped up, or deleted, and why
- The successful exhaustive branch-workflow URL when the change affects exhaustive
  verification, or "not applicable" with the reason
- Open ADRs blocking merge, or "none"
- Anything you were unsure about and want the reviewer to look at hardest

## 9. Fix serious issues raised by Claude review

The review only runs when the PR transitions to ready-for-review, so trigger it
explicitly once the PR is in the state you want reviewed:

```bash
gh pr ready <number>
```

Periodically check for the `claude-review` workflow run. For everything that it flags as
`blocking` or `should-fix`, fix it. Before pushing more commits that you don't want
reviewed immediately (e.g. you're still iterating on the same round of fixes), convert
the PR back to draft so the pushes don't get seen as "ready" by anyone watching the PR
state:

```bash
gh pr ready <number> --undo
```

Push your fixes, then mark it ready again to trigger the next review pass:

```bash
gh pr ready <number>
```

Iterate until the review is satisfied. For items flagged as `note`, it is up to your
judgement whether to address them or not. If you disagree with a finding, write an ADR
and mark it in the PR body.

Once the `claude-review` workflow produces no `blocking` or `should-fix` findings, then
stop, leaving the PR marked ready for review. Do not merge. Do not start the next
milestone.

## If you hit a STOP condition

Write the ADR in `docs/adr/`, commit it, push the branch, open the PR marked as draft
with the ADR named in the body, and end your turn. Do not choose an option and proceed.
Relaxing a tolerance or marking a test `xfail` to get to green is never the answer.
