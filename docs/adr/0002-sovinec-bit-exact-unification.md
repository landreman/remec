# ADR 0002 — Sovinec bit-exact preservation versus common M4a assembly

## Context

Milestone 1.5 must route the constant, smoothly floored spatial, and rank-one
Sovinec M4a paths through one interface without changing their recorded results.
The prior Claude review found that routing the rank-one Sovinec path through the
spatial tensor assembly changed 8 of 9 recorded central amplitudes relative to
`origin/main`, by up to approximately `3.9e-8` relative. The existing CSV test
tolerance (`1e-5`) remains green, but the milestone acceptance criterion says
the results are bit-for-bit unchanged.

The common assembly is mathematically the intended rank-one M4a form when the
already normalized tangent field is passed through without an additional safe
normalization. The initial implementation did not convey that precondition to
the common kernel, causing it to normalize an already-unit direction again.

## Options

1. Keep the original dedicated Sovinec assembly for the exact regression table,
   while exposing it through `AnisotropicDiffusionSolver` as a separate strategy
   operation.
2. Regenerate `tests/verification/sovinec_pollution.csv` using the common tensor
   path, record the measured deviation, and change the 1.5 acceptance statement
   from bit-exact preservation to a stated numerical tolerance.
3. Continue investigating until the common path reproduces the original table
   bit-for-bit, without changing the table or acceptance criterion.

## Tradeoffs

Option 1 preserves the established verification baseline but retains a distinct
assembly implementation. Option 2 completes the common assembly extraction but
changes a stated acceptance criterion and the permanent baseline. Option 3 best
satisfies both objectives once the redundant normalization is removed.

## Decision history

The human initially selected Option 2 while the source of the difference was
unidentified. The subsequent `direction_is_normalized=True` correction removes
that redundant normalization. At the corrected head, all nine Sovinec rows are
bit-for-bit identical to `origin/main`; the regenerated table is therefore
discarded.

## DECISION: Option 3 (supersedes the provisional Option 2 outcome)
