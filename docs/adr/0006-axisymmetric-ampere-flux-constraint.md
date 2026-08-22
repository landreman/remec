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

The note-§12 shell-resolution check is also unresolved on the milestone acceptance
mesh: doubling the four-shell partition to eight shells (`edges=9`) at `maxh=0.18`
does not converge within the 40-step nonlinear limit. Three of the original four shells
already span less than one sampled radial cell on that mesh. This is recorded evidence
for the deferred mixed-u--J contingency, not evidence for weakening the four-shell
Option-1 gate.

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

## Escalation criteria (binding)

The §27.4 escalation clause above is quantified here so that the Option-1 → Option-2
decision is mechanical, not a fresh judgment call. Baselines are the constant-trace
measurements recorded in `docs/STATUS.md` (milestone 5.5) and
`tests/verification/axisymmetric_nonideal_refinement.csv`: finest-mesh (564-element)
relative compatible-current correction 0.07555 with fine effective-h rates 2.170 and
2.483; final-stage non-ideal/analytic L² errors 0.24109 and 0.23377; fixed-pressure
regularization-only ladder 0.32151 → 0.31796 → 0.30704 over D_u 0.060 → 0.015.

After the Option-1 free-trace solve lands, regenerate the five-mesh refinement study
and extend the fixed-pressure regularization ladder by at least two further halvings of
both D_u and epsilon_kappa (to D_u ≤ 0.00375, epsilon_kappa ≤ 0.0075), holding mesh,
pressure amplitude, and I₀ targets fixed. **Schedule Option 2 before the Phase-6 gate
if any of the following holds:**

1. **Current correction fails to converge.** The least-squares effective-h rate over
   the three finest meshes of the regenerated study falls below 1.0, or the finest-mesh
   relative correction exceeds 0.10 (i.e. is no better than ~1.3x the constant-trace
   baseline of 0.07555).
2. **Field-error floor is unchanged.** Over the extended fixed-pressure ladder, the
   non-ideal/analytic L² error at the final stage is not at least 20% below its
   first-stage value, or the error is non-monotone across the extended ladder by more
   than 2% of its running minimum.
3. **Flux constraint is not algebraically tight.** The relative toroidal-flux residual
   |Ψ_t,discrete − Ψ_t,prescribed| / |Ψ_t,prescribed| exceeds 1e-10 in any accepted
   continuation row. This is a hard invariant of the bordered solve: a violation is a
   solver defect to fix, and if it cannot be fixed within the scalar formulation it is
   also an escalation trigger.

If none of the three triggers fires, milestone 5.5 closes on the scalar Option-1
formulation and the mixed u--J closure remains a deferred §27.4 contingency. Relaxing
any threshold above requires amending this ADR with a new human sign-off, per the STOP
conditions in `CLAUDE.md`.

DECISION: option 1 approved, with the escalation criteria above binding.
