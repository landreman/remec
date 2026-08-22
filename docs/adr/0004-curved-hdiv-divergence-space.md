# ADR 0004 — Curved-mesh terminal space for the compatible magnetic complex

> **Superseded by ADR 0005, Option 1.** The containment measurements and NGSolve API
> observations below remain valid, but the inference that the paired weak constraint
> cannot enforce pointwise current continuity on curved meshes does not.

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

- the H¹→HCurl and HCurl→HDiv inclusions and projected `curl(grad)` identity remain
  below `6.95e-13`;
- the final automated regression independently projects `curl(A_h)` into HDiv with
  relative defect `7.12e-16`, measures its relative divergence with `ng.div` as
  `7.42e-15`, and measures `3.28` for a random-HDiv negative control;
- a random `HDiv(2)` divergence projected into ordinary scalar `L2(1)` has relative
  defect `0.23`–`0.32` (independently reproduced in review as `2.71e-1`);
- `dual_mapping`, `piola`, and `covariant` keyword flags are not documented scalar-L²
  flags and do not change that result in this wheel.

The adversarial Milestone-4.1 review then tested whether mesh refinement or simply
raising the ordinary scalar-L² order repairs the mismatch. For a random `HDiv(1)`
field on the same `Curve(3)` ball it measured:

| `maxh` | relative defect in `L2(0)` | relative defect in `L2(4)` |
| ---: | ---: | ---: |
| 0.50 | 3.14e-1 | 1.14e-1 |
| 0.35 | 2.85e-2 | 6.24e-6 |
| 0.25 | 2.47e-2 | 1.67e-6 |

The paired terminal order for `HDiv(1)` is `L2(0)`, whose defect stalls at about
2.5e-2 over the two finer meshes rather than becoming a roundoff identity. Raising the
ordinary-L² order approximates the non-polynomial `1/det(J)` factor increasingly well,
but does not restore strong inclusion or a finite exact sequence. The coarse result is
also quadrature-sensitive (2.93e-1 at integration order 6 and 3.59e-1 at order 26), so
the ADR relies on the refinement stall and mapping argument, not the original coarse
0.23--0.32 magnitude.

This behavior follows the mappings: an HDiv field uses the contravariant Piola map,
so its divergence carries `1/det(J)`, whereas ordinary scalar NGSolve L² uses the
ordinary pullback.  Therefore `div(HDiv)` is not strongly contained in the selected
scalar space on a curved element.  A direct NGSolve call `ng.div(ng.curl(a_h))` also
raises because `ng.curl(a_h)` is a plain coefficient function. The working diagnostic
first verifies its mass projection into the paired HDiv space, then applies `ng.div` to
that HDiv GridFunction:

```python
b_projection = mass_project(ng.curl(a_h), hdiv_space)
assert b_projection.relative_defect < roundoff_gate
div_b_h = ng.div(b_projection.field)
```

The projection defect and divergence are both at roundoff on the curved test. A random
HDiv field produces O(1) relative divergence under the same diagnostic. Do not use
`b_h.Diff(ng.x)`: for a GridFunction-backed coefficient function it is coefficient
differentiation and returns zero even for a divergent field. This verifies magnetic
`div(curl)=0`, but does not supply the missing terminal density space needed to
interpret the §10 current projection as strongly divergence-free.

The design therefore appears infeasible as written on the selected backend: ordinary
scalar NGSolve L² cannot simultaneously be the stated curved-mesh terminal space and
retain the affine strong-exactness interpretation.  This affects §5 invariant 2, the
§7.3 gauge-multiplier diagnostic, §10 current projection, and Ampère compatibility in
(M1).  It is not a tolerance issue.

### Scope: the defect does not reach ∇·B

The broken arrow is the terminal one, `div: HDiv → L2`.  It bites only where a
divergence must be *interpreted as a member of a named discrete space*, which is what
the §10 constrained projection for the current does.  The magnetic field is not
obtained that way: `B_h = ∇×A_h` with `A_h ∈ HCurl`, and `div(curl) = 0` is a pointwise
algebraic identity of the basis functions that survives the covariant/contravariant
Piola maps, since the curl of an HCurl function is an HDiv function by construction.
The curved-ball measurements above confirm this directly — the HCurl→HDiv inclusion
holds at roundoff and `ng.div` of the projected curl has relative norm `7.42e-15`, while
the random-HDiv control is O(1).
So ∇·B = 0 at roundoff on curved meshes is already established and is not what is at
stake here; only ∇·J is.

That asymmetry matches how the two fields are consumed.  Downstream users of the code
output may rely on ∇·B = 0 as a hard property — field-line tracing in particular
integrates B directly, and a nonzero divergence there produces trajectories that are
not confined to flux surfaces and accumulate error in a way the user cannot diagnose
from the output alone.  Users are much less likely to depend on ∇·J = 0 as a hard
property; the current is primarily an internal quantity of the equilibrium solve, and a
small, measured, converging divergence defect is defensible there in a way it is not
for B.

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
4. **Split the invariant: keep ∇·B exact, weaken only ∇·J on curved meshes.** Retain
   the current §10 form for the current projection and call `J_h` weakly divergence-free
   in the chosen multiplier space, requiring the unresolved strong divergence and the
   gauge multiplier to converge under h/geometry refinement.  Leave the magnetic
   invariant untouched: `B_h = ∇×A_h` is strongly divergence-free at roundoff on curved
   meshes by the `div(curl)=0` identity already measured above, and nothing in this ADR
   disturbs it.  This is the smallest implementation change, but it gives up the exact
   integrability rationale quoted from the note for the current, and splits DESIGN §5
   invariant 2 into an exact half and a convergence half.
5. **Raise the ordinary scalar-L² order and/or refine the mesh.** This reduces the
   projection defect, as the table above demonstrates for `L2(4)`, but it does not
   change the scalar pullback or make `div(HDiv)` a strong member of that space on a
   curved element. It is therefore useful for approximation diagnostics but cannot
   restore the exact terminal arrow or justify calling the constrained current strongly
   divergence-free.

## Tradeoffs

Option 1 most directly restores the finite-element exterior-calculus interpretation
and keeps the current projection architecture, but it adds a backend-specific terminal
representation and may collide with the project's binary-wheel/no-compiler policy.
Option 2 makes (M1) compatibility explicit and avoids relying on scalar-L² mapping, but
it substantially changes milestone 4.4 and must handle nontrivial toroidal cohomology
and prescribed shell currents.  Option 3 contains the issue to geometry transfer, but
the transfer can reintroduce precisely the divergence and current-moment defects the
projection is meant to remove.  Option 4 uses only current NGSolve primitives and
concedes exactness only for the current, where the exactness is least load-bearing for
users, but it does weaken half of an exact mathematical invariant into a convergence
claim and therefore needs explicit physics/design approval plus a revised
Ampère-compatibility criterion. Option 5 can lower a measured defect but cannot repair
the incompatible mapping, so it is not a substitute for choosing the invariant's
meaning.

## Recommendation

Adopt **Option 4**, with the invariant split explicitly rather than amended wholesale.

Rationale.  Option 1 is the option that most directly restores the exterior-calculus
interpretation, but it is the option that most likely requires a compiled custom
FESpace, which the project's binary-wheel/no-compiler policy (`DESIGN.md` §26) rules
out.  Option 2 does deliver algebraic ∇·J = 0, but it buys that exactness for the
quantity that needs it least, at the cost of explicit harmonic-current representation,
topology-dependent cohomology handling on the torus, restated shell-moment solvability,
and new inf-sup and approximation tests — a large redesign of milestone 4.4.  Option 3
reintroduces the same error through the transfer step in a form that is harder to
measure.  Option 4's weakness falls precisely where the tolerance for weakness is,
per the scope discussion above.

Concretely, replace DESIGN §5 invariant 2 with two invariants:

- **Magnetic (exact, unchanged).** `∇·B_h = 0` at roundoff on affine *and* curved
  meshes, justified by the `div(curl)=0` basis identity rather than by the terminal
  L² arrow, and tested to roundoff on the curved OCC ball.  This is not weakened and
  requires no new machinery.
- **Current (convergence).** The §10 projection yields `J_h` orthogonal to the chosen
  scalar `L2` multiplier space.  On affine meshes this remains strong divergence-freedom
  at roundoff.  On curved meshes, `‖∇·J_h‖` must be measured and must satisfy the
  two-part acceptance criterion below — a convergence rate under `h`- and
  geometry-refinement, plus a magnitude backstop — with the measured rate table checked
  into `tests/verification/` and the measured relative divergence recorded in
  `docs/verification.md`.

The acceptance test must also measure the gauge multiplier `λ_h` and confirm it
converges to zero under the same refinements.  This is the criterion that decides
whether Option 4 was the right call: with ∇·J only small, Ampère's law in (M1) is only
approximately integrable, and the residual inconsistency has to go somewhere — the
expected place is `λ_h`.  If `λ_h` stalls at a nonzero value under refinement, the
inconsistency is being parked rather than resolved, and this ADR should be revisited.
Record that fallback trigger explicitly in the milestone-4.4 row of `docs/STATUS.md`.

### Acceptance criterion for `‖∇·J_h‖` on curved geometry

The criterion has two parts, and the first is the one that actually detects a broken
projection.

**Primary gate — rate.** The measured `‖∇·J_h‖` must decrease at the predicted rate
under both `h`-refinement and geometry-order refinement, per the rate table checked into
`tests/verification/`.  A single small number at one resolution is not a convergence
result (`DESIGN.md` §26).

**Backstop — magnitude.** At production resolution, the dimensionless relative
divergence

\[
\delta \;\equiv\;
\frac{L_{\mathrm{ref}}\,\lVert \nabla\cdot J_h \rVert_{L^2(\Omega)}}
     {\lVert J_h \rVert_{L^2(\Omega)}}
\;<\; 0.03 ,
\]

where `L_ref` is the device/domain scale (e.g. the minor radius), **not** the local mesh
size `h`.  This catches the case where the rate is correct but the constant is
unacceptably large.

The relative form is preferred over an absolute one.  It is dimensionless — `∇·J` has
units of A/m³ while a current density is A/m², so any absolute ceiling written directly
against a reference current density is dimensionally inconsistent and needs a compensating
factor of `L_ref`.  It is also domain-size independent: the `√|Ω|` implicit in each
`L²(Ω)` norm cancels between numerator and denominator, so the same threshold means the
same thing on a small test ball and a full device.  And it avoids introducing `B_ref` and
`μ₀`, whose per-problem conventions would otherwise propagate into a pass/fail gate.  For
reference, the equivalent absolute statement via Ampère's law is
`‖∇·J_h‖ ≲ 0.03 · B_ref / (μ₀ L_ref²)`, with both norms taken as volume-averaged (RMS).

The value `0.03` is a deliberately loose backstop chosen before any curved measurement
exists.  A converged curved projection should land one to two orders of magnitude below
it, so a run that merely clears `0.03` deserves suspicion rather than confidence.
`docs/verification.md` must record the *measured* `δ`, not just pass/fail, and the ceiling
should be tightened to reflect the observed magnitude once milestone 4.4 produces real
numbers.

Do not silently treat ordinary scalar L² weak orthogonality as the exact invariant in
the meantime; where the current is described as divergence-free in the code, docs, or
output metadata, say weakly divergence-free on curved geometry and cite the measured
bound.

## DECISION: Option 4 was approved, then superseded by ADR 0005 Option 1.
