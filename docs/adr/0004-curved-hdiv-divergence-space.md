# ADR 0004 — Curved-mesh terminal space for the compatible magnetic complex

## Context

Milestone 4.1 establishes the affine-tetrahedral polynomial sequence

\[
H^1_{p+1}\xrightarrow{\nabla}H(\mathrm{curl})_p
\xrightarrow{\nabla\times}H(\mathrm{div})_{p-1}
\xrightarrow{\nabla\cdot}L^2_{p-2}.
\]

`DESIGN.md` §7.1 requires both curved geometry and the full tested sequence:

> A in `HCurl`; projected **B** and **J** stored in `HDiv` when a stored field is
> required; `L2` for divergence constraints/diagnostics; scalar H¹ gauge multiplier
> chosen to form a stable mixed pair. Mesh geometry order comparable to the FE order
> (curved elements).

and

> the exact space/order pairing that yields the commuting discrete sequence MUST be
> established by small de Rham-sequence tests (∇, ∇×, ∇· mapping between the chosen
> spaces) **before** the 3D solver is built.

`DESIGN.md` §10 then specifies the curved-production current projection with an
ordinary L² multiplier:

> solve the constrained projection: (J_h, v) + (λ_h, ∇·v) = (J_raw, v),
> (∇·J_h, q) = 0, with J_h ∈ H(div)

and calls the result divergence-free.  The note is stricter about why this matters:

> a field with ∇·J ≠ 0 is not the curl of any B, so exactness is precisely what makes
> Ampère's law in (M1) integrable

(note §5.2, immediately after equation `u_equation`).

On affine tetrahedra, the milestone-4.1 sequence satisfies all three inclusions at
roundoff.  On an order-3 curved OCC tetrahedral ball with NGSolve 6.2.2606:

- the H¹→HCurl and HCurl→HDiv inclusions and both successive-derivative identities
  remain below `6.95e-13`;
- a random `HDiv(2)` divergence projected into ordinary scalar `L2(1)` has relative
  defect `0.23`–`0.32` (independently reproduced in review as `2.71e-1`);
- `dual_mapping`, `piola`, and `covariant` keyword flags are not documented scalar-L²
  flags and do not change that result in this wheel.

This behavior follows the mappings: an HDiv field uses the contravariant Piola map,
so its divergence carries `1/det(J)`, whereas ordinary scalar NGSolve L² uses the
ordinary pullback.  Therefore `div(HDiv)` is not strongly contained in the selected
scalar space on a curved element.  A direct NGSolve call `ng.div(ng.curl(a_h))` also
raises because `ng.curl(a_h)` is a plain coefficient function; the working diagnostic
is the symbolic trace

```python
b_h = ng.curl(a_h)
div_b_h = b_h.Diff(ng.x)[0] + b_h.Diff(ng.y)[1] + b_h.Diff(ng.z)[2]
```

which evaluates to zero on the curved test.  That verifies magnetic `div(curl)=0`, but
does not supply the missing terminal density space needed to interpret the §10 current
projection as strongly divergence-free.

The design therefore appears infeasible as written on the selected backend: ordinary
scalar NGSolve L² cannot simultaneously be the stated curved-mesh terminal space and
retain the affine strong-exactness interpretation.  This affects §5 invariant 2, the
§7.3 gauge-multiplier diagnostic, §10 current projection, and Ampère compatibility in
(M1).  It is not a tolerance issue.

## Options

1. **Add an algebraic density-mapped terminal space.** Keep NGSolve HCurl/HDiv, but
   represent the terminal 3-form by reference-element discontinuous coefficients with
   the `1/det(J)` physical evaluation.  Assemble the discrete divergence incidence
   operator into those coefficients, constrain it exactly, and provide separate
   physical L² norms for diagnostics.  This may be implementable with NGSolve matrices
   and Python-side space metadata; if it requires a compiled custom FESpace, the
   no-compiler/base-wheel constraint needs separate review.
2. **Project into the discrete curl image.** Parameterize the compatible part of the
   current as the curl of an HCurl field (plus explicitly represented harmonic current
   components), then impose the M3b shell moments in that representation.  Divergence
   is algebraic by construction, but the projection, topology, boundary-normal
   condition, and shell-moment solvability all change and require new inf-sup and
   approximation tests.
3. **Restrict compatible current projection to affine geometry.** Perform the §10
   projection on a piecewise-affine tetrahedral geometry and transfer its coefficients
   to the curved geometry used by the magnetic solve.  This preserves the proven
   affine complex but introduces a geometry-transfer error and may not preserve either
   normal fluxes or shell moments without another constrained projection.
4. **Amend the invariant to weak ordinary-L² divergence on curved meshes.** Retain the
   current §10 form, call the field weakly divergence-free in the chosen multiplier
   space, and require the unresolved strong divergence and gauge multiplier to converge
   under h/geometry refinement.  This is the smallest implementation change, but it
   gives up the exact integrability rationale quoted from the note and changes the
   meaning of DESIGN §5 invariant 2.

## Tradeoffs

Option 1 most directly restores the finite-element exterior-calculus interpretation
and keeps the current projection architecture, but it adds a backend-specific terminal
representation and may collide with the project's binary-wheel/no-compiler policy.
Option 2 makes (M1) compatibility explicit and avoids relying on scalar-L² mapping, but
it substantially changes milestone 4.4 and must handle nontrivial toroidal cohomology
and prescribed shell currents.  Option 3 contains the issue to geometry transfer, but
the transfer can reintroduce precisely the divergence and current-moment defects the
projection is meant to remove.  Option 4 uses only current NGSolve primitives, but it
weakens an exact mathematical invariant into a convergence claim and therefore needs
explicit physics/design approval plus a revised Ampère-compatibility criterion.

## Recommendation

Prototype Option 1 on the same curved OCC ball before milestone 4.2 or 4.4 chooses a
production representation.  Require the prototype to demonstrate strong mapped
`div(HDiv)` membership, an algebraic divergence operator with the correct one-dimensional
constant kernel/cokernel structure, and no compiler or new base dependency.  If that is
not feasible with the binary NGSolve wheel, compare Option 2 against Option 4 with a
manufactured curved-current projection and gauge-multiplier measurement before changing
the design.  Do not silently treat ordinary scalar L² weak orthogonality as the exact
invariant in the meantime.

## DECISION: pending human sign-off
