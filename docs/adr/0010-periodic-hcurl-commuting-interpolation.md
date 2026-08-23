# ADR 0010: Periodic curved-HCurl commuting interpolation contract

**Status:** Proposed — human decision required

## Context

Milestone 6.2 constructs the Reiman--Greenside field for note equation (M1) from the
closed-form vector potential

`A = Psi_t grad(Theta) - Psi_p grad(Phi)`

and requires, on the curved periodic cylinder selected by ADR 0009, both
`B_h = curl(A_h)` and `div(B_h) = 0` at roundoff.  The current implementation puts
`A` in periodic HCurl with `GridFunction.Set`, then mass-projects `curl(A_h)` into the
paired periodic HDiv space.  This gives the required discrete de Rham identities at
roundoff, but NGSolve documents the default `Set` operation as a local L2 projection,
not a canonical commuting interpolation.

An h-refinement test exposed the consequence.  For the reference field
(`t0=0.29`, `t1=0.38`, `epsilon_1=1e-3`, `epsilon_2=0`), the relative L2 error in
`B_h` converges at only about 0.6 for the order-1 complex on both the affine and
geometry-order-4 cylinder.  The error in `A_h` converges faster, while its curl does
not, which isolates the noncommuting projection rather than the analytic field or
wall geometry.

The obvious NGSolve alternatives do not mechanically resolve this:

| construction, macOS / NGSolve 6.2.2606 | measured coarse-to-refined B rate |
|---|---:|
| default `Set`, HCurl orders 1, 2, 3 | about 0.85, 1.25, 1.97 |
| `Set(..., dual=True, bonus_intorder=8)`, HCurl orders 1, 2, 3 | 0.952, 1.350, 2.170 |
| canonical HDiv interpolation, HDiv order 1, finest pair | 1.192 |

The order-1 default construction settles near 0.6 on three levels on Linux (measured
rates 0.607 and 0.615).  The `dual=True` results remain below the nominal HCurl curl
rates, so increasing quadrature is not a fix.  Direct canonical HDiv interpolation
also produced relative divergence `3.0e-3` on the coarse curved mesh before reaching
roundoff after refinement, and it no longer proves the binding ADR-0009 identity
`B_h = curl(A_h)`.  `ng.Interpolate(A, HCurl)` returns an elementwise coefficient
function for which NGSolve does not expose the `curl` operator.

This is a STOP condition: restoring the expected reference-field convergence appears
to require a new projection or mixed solve, while accepting the present p-scan plus
an unrelated periodic-scalar rate would change what counts as the milestone's
manufactured (M1) convergence evidence.  The rate must not be lowered or the failing
physics test deleted merely to make CI green.

## Options

1. **Implement a global commuting projection for periodic curved HCurl.** Assemble the
   canonical HCurl degrees of freedom explicitly (including the periodic equivalence
   classes), obtain `A_h`, and use its exact mapped curl in the paired HDiv space.
   Require a three-level reference-field h scan with the nominal order-1 curl rate and
   retain the existing p-scan and roundoff de Rham gates.
2. **Use a constrained mixed reconstruction.** Canonically interpolate or project the
   analytic `B` into periodic HDiv, then solve for a periodic HCurl potential whose curl
   equals that divergence-free field, with the gauge and harmonic compatibility made
   explicit.  Gate h convergence, `curl(A_h)=B_h`, divergence, and toroidal flux.
3. **Amend the milestone evidence contract.** Keep the exact discrete construction
   (`B_h=curl(A_h)`, `div(B_h)=0`), its four-order reference-field p-scan, and the
   independent periodic H1 manufactured h-rate test, but explicitly state that no
   algebraic h-rate is claimed for NGSolve's noncommuting `GridFunction.Set` path.
4. **Fall back to ADR 0009 Option 3.** Use affine geometry.  Existing probes show that
   geometry order 1 versus 4 does not remove the approximately 0.6 order-1 rate, so
   this is not expected to solve the projection problem; it is included only because
   ADR 0009 names it as the curved-complex fallback.

## Tradeoffs

Option 1 preserves the mathematical construction and provides the strongest evidence,
but it adds delicate finite-element interpolation code whose periodic orientation and
curved Piola behavior must be independently verified.  It may duplicate backend
functionality that is not exposed through NGSolve's Python API.

Option 2 can reuse the repository's gauge/null-space experience and makes the target
field approximation explicit.  It is a materially larger solver than milestone 6.2
currently calls for, introduces a new mixed-system conditioning question, and must
handle the solid-torus flux class without silently projecting it away.

Option 3 is the smallest implementation and retains every acceptance criterion stated
in DESIGN section 25 and ADR 0009, but it weakens the milestone skill's expectation
that the manufactured physics path itself carry an h-rate.  The scalar H1 rate does
not constrain the Reiman--Greenside implementation, so it cannot be described as a
substitute for an (M1) rate.

Option 4 gives up the accepted curved circular-wall contract, adds percent-level wall
geometry error on coarse meshes, and does not address the measured cause.  It is not
recommended on the available evidence.

## Recommendation

Choose Option 1 if milestone 6.2 must own a reference-field h-rate; it preserves the
accepted equations and tests the exact production path.  Choose Option 3 only if the
human explicitly decides that the design's required four-order reference-field scan,
roundoff de Rham identities, independent analytic oracles, and scalar periodic-space
rate are sufficient for this geometry milestone.  Do not choose Option 4 as a remedy
for this issue.

## Decision

DECISION: pending human sign-off
