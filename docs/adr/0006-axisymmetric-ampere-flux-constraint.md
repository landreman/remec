# ADR 0006: Axisymmetric Ampère gauge and compatible-current closure

## Context

Milestone 5.5 reduces the fixed-boundary magnetic update to scalar solves for
`psi` and `I = R B_phi`. The present implementation puts both scalars in an H1 space
with a constant Dirichlet trace. It then adds shell-local response columns to the
Grad--Shafranov right-hand side so the strong discrete curl preserves every prescribed
`I_0(s)` moment.

The note makes the global constraint explicit:

> "Fixed-boundary conditions: B dot n = 0 and prescribed toroidal flux Psi_t for
> (M1)--(M2)" (note, lines 1179--1181).

For the axisymmetric reduction it is more specific:

> "Equation (Igrad) determines I up to a constant, fixed by the prescribed toroidal
> flux Psi_t = integral_Omega (I/R) dR dZ" (note, lines 2240--2243).

The current constant-boundary-trace solve is stronger than this condition and does not
measure toroidal-flux error. Replacing it with a free trace plus one flux constraint
changes the nonlinear state and therefore the benchmark results; it is not a mechanical
API correction.

The compatible-current correction is monitored. At fixed final regularization its
relative norm on 100, 158, 238, 358, and 564 elements is respectively 0.23812,
0.17367, 0.20691, 0.13285, and 0.07555. The two finest effective-h rates are 2.170 and
2.483, but the finest correction is still 7.56%, and the non-ideal/analytic field error
has a non-monotone approximately 0.24 floor. The design's escalation rule is:

> "Discrete M2/M3/M3b incompatibility: projection with monitored correction and
> preserved shell moments; escalate to a mixed u--J formulation if either divergence
> or I_tor(s) correction does not converge" (`docs/DESIGN.md` §27.4, lines 1374--1376).

The h study demonstrates an asymptotic decrease, but it does not decide whether the
remaining correction and field-error floor are acceptable for the milestone or require
the mixed formulation now.

## Options

### Option 1: Free-I trace with one prescribed-flux constraint; retain projected psi

Put `I` in an unconstrained H1 space, solve (Igrad) modulo constants, and impose one
bordered constraint for the prescribed `Psi_t`. Report the toroidal-flux residual in
every continuation row. Retain the monitored shell-moment correction in the psi solve
and its checked-in h-convergence table.

Tradeoffs: this is the smallest note-compliant change and preserves the scalar
axisymmetric verification architecture. It does not remove the finite-mesh current
correction or by itself explain the approximately 0.24 field-error floor.

### Option 2: Implement a mixed u--J axisymmetric closure now

Promote the reconstructed current to a mixed unknown coupled to Ampère, M3, and M3b,
with prescribed toroidal flux and shell moments imposed directly. Use the scalar
projection only as a preconditioner or diagnostic.

Tradeoffs: this follows the conservative reading of §27.4 and removes the fictitious
shell source from the accepted magnetic equation. It is a materially larger milestone,
requires new spaces and inf-sup/solver verification, and will regenerate every 5.5
benchmark number.

### Option 3: Retain the current reduced benchmark contract

Keep the constant boundary trace and compatible-current projection, document the
refinement behavior, and treat the run only as a reduced algorithmic benchmark rather
than completion of the note's prescribed-flux fixed-boundary problem.

Tradeoffs: no further implementation work is required, but this weakens the milestone
5.5 acceptance claim and conflicts with the note's explicit toroidal-flux condition.
It would require revising `docs/DESIGN.md` §25 and must not be selected implicitly.

## Recommendation

Choose Option 1 for milestone 5.5, with toroidal-flux residual as a hard invariant and
the existing h-convergence table retained. If the free-trace solve leaves a non-convergent
current correction or the field-error floor unchanged under a longer fixed-pressure
regularization ladder, follow the existing §27.4 trigger and schedule Option 2 before
the Phase-6 gate. Do not choose Option 3 without explicitly revising the milestone.

DECISION: pending human sign-off
