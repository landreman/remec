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

## 6. Record

Update `docs/STATUS.md`: mark the row, paste the measured numbers (convergence rates,
pollution values, residuals — whatever the acceptance criterion names), and add a line
for anything the next milestone needs to know.

Append any NGSolve API discoveries to `docs/dev_notes.md`. Update
`docs/verification.md` with the new rate table.

## 7. Open the PR

```bash
gh pr create --fill
```

PR body must contain, in this order:

- Milestone number and one-line summary
- Equations (note labels) and invariants (§5) affected
- The acceptance criterion, and the test that demonstrates it
- Measured numbers
- Mutations verified to turn the suite red
- Open ADRs blocking merge, or "none"
- Anything you were unsure about and want the reviewer to look at hardest

Then stop. Do not merge. Do not start the next milestone.

## If you hit a STOP condition

Write the ADR in `docs/adr/`, commit it, push the branch, open the PR marked as draft
with the ADR named in the body, and end your turn. Do not choose an option and proceed.
Relaxing a tolerance or marking a test `xfail` to get to green is never the answer.
