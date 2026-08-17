# NGSolve API notes

> Entries for milestones 3.3–3.4 mention `PrescribedCurrentProfile` and the former
> `u=F(p)+ũ` shift. After the 2026-08-15 model revision those entries remain accurate
> descriptions of NGSolve expression behavior, but not of the production current-profile
> closure. Follow `DESIGN.md` §9.2 and `STATUS.md` milestones 3.5–3.6.

- Milestone 4.4 (NGSolve 6.2.2606): the Section-10 current saddle uses one scalar
  `NumberSpace(mesh)` per shell moment; `NumberSpace(mesh, dim=N)` has one vector-valued
  DOF and cannot be compounded with the scalar HDiv/L2 spaces in this wheel. A constant
  NumberSpace test function integrates to the domain volume, so a scalar moment target
  on its RHS must be divided by that same-volume quadrature. For compact mollified
  shell weights on curved tetrahedra, use an explicit `dx(intrules={ET.TET: rule})`
  for both moment couplings and their RHS normalization; an unrelated diagnostic rule
  otherwise measures cut-layer quadrature mismatch instead of the algebraic constraint.
  The fixed-in-normalized-volume-width weight constructor introduced for the analytic
  torus is deliberately verification-only. Production callers must construct weights
  with milestone 3.6's shared gradient-scaled volume map and resolution guards, then
  pass those general vector weights through `CurrentMomentConstraint`.
  Compound bilinear forms assemble as one monolithic sparse matrix rather than exposing
  `.blocks`; the HDiv-to-L2 constraint block is recoverable from `mat.COO()` using the
  component DOF offsets. Restricting the discontinuous L2 rows to one curved element
  gives a cheap numerical-rank proof of oversized-row redundancy on the large torus.
  A generic analytic vector `CoefficientFunction` has no `derivname`, so `ng.div(raw)`
  is unavailable even when its components are differentiable; accept an independently
  transcribed raw divergence for the mandatory before/after diagnostic.

- Milestone 4.3 (Netgen/NGSolve 6.2.2606): `netgen.csg.Torus` generated 45,205
  tetrahedra for the R=2, a=0.6 verification torus regardless of requested `maxh` in
  the tested 0.65--2.0 range. Revolving an OCC `WorkPlane` circular face around the
  z-axis produced 1,414 tetrahedra on macOS and 1,389 on Linux, and remained stable
  under `mesh.Curve(order)` for orders 1--4. The verification CSV therefore records
  strict `sys.platform`-specific reference rows for NGSolve 6.2.2606 rather than
  relaxing numerical tolerances; a missing platform row reports the complete measured
  CSV needed for regeneration. Revolving the same disk through 180 degrees exposes
  OCC faces in wall/start-cut/end-cut order. Naming those faces and applying
  `mesh.Curve(6)` gives an explicit cut on which `ng.Integrate(field*normal, BND)`
  evaluates the actual NGSolve coefficient. The start face has outward normal `-e_y`,
  so the positive toroidal flux uses the negative boundary integral. For the normalized
  circular field this measures 0.9999999916288673; negating the field reverses the sign.
  Use this explicit-cut construction for flux regressions instead of a separate NumPy
  formula that can become disconnected from the field under test. When diagnosing
  tangency on this half-torus, restrict `ng.BND` with
  `definedon=mesh.Boundaries("wall")`: the artificial cut faces intentionally carry
  nonzero normal flux and are not part of the physical-wall tangency invariant.

- Milestone 4.2 (NGSolve 6.2.2606): the symmetric Coulomb-gauge saddle form on
  `FESpace([HCurl, H1])` is invertible with UMFPACK when both spaces carry the full
  fixed-boundary trace (`dirichlet=".*"`). For a compatible analytic current,
  NGSolve's default coefficient-function quadrature can leave a spurious gauge
  multiplier as large as 1e-4 on the coarsest trigonometric case; assembling both the
  operator and load with `dx(bonus_intorder=10)` reduces it below 4e-14. This is
  quadrature compatibility, not a gauge defect. Project `curl(A_h)` into the paired
  HDiv space with an independently assembled mass solve before applying `ng.div`, as
  in milestone 4.1. The HDiv normal trace can be evaluated directly with
  `magnetic_field * ng.specialcf.normal(3)` on `ng.BND`; it is below 7e-15 relative
  when the HCurl tangential trace is essential on the full cube boundary.

- Milestone 4.1 (NGSolve 6.2.2606): on affine tetrahedra, the exact order pairing is
  `H1(p+1) -> HCurl(p) -> HDiv(max(p-1, 0)) -> L2(max(p-2, 0))`; equal integer
  `order` arguments do not give the exact global complex.  The offsets are element-
  family specific: in particular, a hexahedron uses a different convention, so the
  tetrahedral factory rejects every non-tetrahedral volume element. On a third-order,
  107-tetrahedron curved OCC ball, projecting `curl(A_h)` into the paired HDiv space
  had relative defect 7.12e-16 and applying `ng.div` to that HDiv GridFunction gave
  relative divergence 7.42e-15; a random-HDiv negative control measured 3.28. A general
  `div(HDiv)` field did not lie in ordinary scalar `L2` there (relative projection
  defects 0.23--0.32), because the divergence carries the contravariant Piola
  `1/det(J)` density while scalar `L2` uses the ordinary pullback. Test the curved-mesh
  (M1) invariant by independently mass-projecting `ng.curl(A_h)` into HDiv, asserting
  the projection defect, and evaluating `ng.div(B_h)` on that HDiv GridFunction. A
  direct `ng.div(ng.curl(A_h))` raises because the intermediate coefficient function
  has no `derivname`. Do not use `B_h.Diff(ng.x)`: for a GridFunction-backed coefficient
  function it is coefficient differentiation and returns zero even for a divergent
  field. ADR 0005 Option 1 supersedes ADR 0004's relaxation: although general curved
  `div(HDiv)` is not strongly contained in ordinary `L2`, the paired weak constraint is
  exact because the Piola `1/det(J)` cancels the volume `det(J)` and the reference
  divergence spans the paired reference L2 space. Actual curved §10 saddle solves gave
  relative divergence below 7.0e-16; an undersized terminal order left O(1) divergence,
  while an oversized order produced a constraint block of rank 428 with 1070 rows (642
  redundant). Test that rank fact rather than a solver-specific singular-factorization
  exception. On the `Curve(3)` ball with `HDiv(2)`, the paired smallest-to-largest ratio
  was 2.02e-2 and the oversized first-discarded-to-largest ratio was 1.05e-15; paired
  `HDiv(3)`/`HDiv(4)` ratios were 4.22e-3/1.52e-3, all cleanly classified by relative
  rank tolerance 1e-10. Its λ is a continuity multiplier with a legitimate nonzero
  limit, not the
  magnetic gauge multiplier; normalize its mean if a non-natural trace leaves constants
  in the multiplier kernel. Milestone 4.4 must automate those positive and negative
  controls. An
  exploratory
  `MakeStructured3DMesh(secondorder=True, mapping=<nonlinear>)` construction segfaulted
  in this local wheel; `netgen.occ` mesh generation followed by `mesh.Curve(order)` was
  stable and should be the verification path for curved tetrahedra.
  On the same `Curve(3)` ball across base orders 0--5, `div(curl)` grows with mapped-
  basis conditioning from 2.53e-15 to 2.79e-12 while a random-HDiv control remains
  2.71--4.33. Use the measured curved gate `128*eps*(p+2)^3`; the affine constant 32 is
  insufficient at p=5 even though the identity remains exact.

- Milestone 3.6 (NGSolve 6.2.2606): the open linear spline
  `BSpline(2, [s0, *nodes, sN], nodal_values)(s)` gives the piecewise-linear G basis
  required by the bordered M3–M3b system. For a GridFunction-backed level set, do not
  rely on coordinate `.Diff` through that composition: transcribe the monotone PCHIP
  for `s=V_chi/V_omega` interval by interval and multiply its analytic `ds/dchi` by
  the supplied native `grad(chi)`. `MapToAllElements` retains the same element/rule
  ordering used by `extract_ngsolve_quadrature`; the resulting coefficient samples
  matched the array-backed volume-map coordinate within 2e-12. Compiling every G
  column repeats the nested PCHIP expression, so compile time grows visibly with the
  number of volume levels; keep that resolution explicit in configuration and scan it
  rather than burying an oversized table in each coefficient tree. One
  `bilinear_form.mat.Inverse(free_dofs, inverse="umfpack")` object can be reused for
  the base right-hand side and all A-inverse-P response vectors before forming the
  small dense Schur complement.
  `mesh(x_array, y_array)` returns a batched set of mesh points accepted by a
  GridFunction, so fields solved on two separately built but geometrically identical
  meshes can be evaluated on one common `MapToAllElements` quadrature rule. Use that
  route for an actual cross-mesh L² difference instead of substituting a stable point
  sample for an N-shell convergence norm.

- Milestone 3.4 (NGSolve 6.2.2606):
  `ngsolve.meshes.MakeStructured2DMesh(periodic_y=True)` registers a top/bottom mesh
  identification, but a plain `ngsolve.H1` space does not consume it: wrap the base
  space in `ngsolve.Periodic(...)` or the solved values remain discontinuous across
  the seam. The mesh still reports `bottom` and `top` from `mesh.GetBoundaries()`, so a
  periodic M3 harmonic exposes only the remaining physical Dirichlet regions
  (`left|right`). Explicit `(nx, ny)` structured counts provide a layer-aligned mesh
  without over-resolving the smooth periodic direction.

- Milestone 3.3 (NGSolve 6.2.2606): the coordinate derivative of a component of
  `ngsolve.grad(GridFunction)` is the zero coefficient, so the transformed (M3) source
  must not form `div(F'(p) grad_r(p))` by calling `.Diff` on a GridFunction-backed
  pressure gradient. `PrescribedCurrentProfile` therefore carries explicit
  perpendicular- and full-gradient divergence coefficients, and the runtime variant
  selects between them for the note-literal SUPG strong source. The symmetric Galerkin
  source is moved directly as `-D_u grad_r(v).grad_r(F)` so it remains exactly paired
  with the direct-u block. Analytic manufactured coefficient functions may construct
  both strong divergences with coordinate `.Diff` before entering the solver.

- Milestone 3.2 (NGSolve 6.2.2606): strong element-interior H¹ second derivatives are
  exposed as `ProxyFunction.Operator("hesse")` during assembly and
  `GridFunction.Operator("hesse")` for diagnostics. Calling `ngsolve.grad()` on an
  already differentiated H¹ proxy does not produce a Hessian, while differentiating a
  proxy gradient component with `.Diff(ngsolve.x)` returns zero. Coordinate `.Diff`
  also silently returns zero for a `GridFunction`; a varying GridFunction-backed B
  must pass its native `ngsolve.grad(B)` to the M3 coefficient record. Expand
  variable-tensor divergences explicitly as
  `P_ij hesse(u)_ij + (∂_i P_ij) ∂_j u` and construct `∂_i P_ij` from that supplied
  field gradient rather than differentiating the GridFunction expression tree.

- Milestone 3.1 (NGSolve 6.2.2606): the nonsymmetric direct-u M3 matrix can use
  `bilinear_form.mat.Inverse(free_dofs, inverse="umfpack")`; solve a nonzero
  Dirichlet lift by applying that inverse to `linear_form.vec - mat * field.vec`.
  The resulting frozen order-2 verification systems have free-DOF relative residuals
  below 4e-17 on macOS.

- Milestone 0.2 (NGSolve 6.2.2606): `ngsolve.SetNumThreads(int)` is available for
  process-global worker configuration; no `ngsolve.GetNumThreads` API is exposed.
  The direct API verification therefore runs in a subprocess so it cannot leak a
  process-global thread count into later tests.

- Milestone 1.1 (NGSolve 6.2.2606): `ngsolve.grad()` applies to a GridFunction but
  not to an arbitrary composed `CoefficientFunction` such as `sin(pi*x)*sin(pi*y)`.
  For manufactured energy norms, provide the analytic gradient explicitly as a vector
  `CoefficientFunction`.

- Milestone 1.1 (NGSolve 6.2.2606): a direct H¹ solve leaves nonzero residual entries
  on constrained Dirichlet rows. Compute the algebraic residual norm on free DOFs with
  `ngsolve.Projector(fes.FreeDofs(), True) * residual`, not on the full vector.

- Milestone 1.1 (NGSolve 6.2.2606): `ngsolve.meshes.MakeStructured2DMesh` produces a
  repeatable triangular slab mesh with the boundary order `bottom`, `right`, `top`,
  `left`. Prefer it to Netgen's unconstrained mesher for cross-platform manufactured
  error-constant regression tests.

- Milestone 1.3 (NGSolve 6.2.2606): a `GridFunction` can be evaluated at a physical
  point with `field(mesh(x, y))`, and `mesh.ne` provides the element count used in
  machine-readable scan tables. In the κ⊥ = 0 Sovinec solve, the free-DOF residual
  degrades as the measured pollution approaches roundoff (9.31e-8 for p=3 on 512
  elements here), despite a successful sparse Cholesky solve; record that residual
  separately from the pollution metric.

- Milestone 1.3 review follow-up (NGSolve 6.2.2606): analytic
  `CoefficientFunction` expressions support `.Diff(ngsolve.x)` and
  `.Diff(ngsolve.y)`, including repeated differentiation. Use these derivatives
  when a diagnostic must certify tangency to the exact coefficient function used
  in a form; a separately hand-written gradient can reproduce its own typo.

- Milestone 1.4 (NGSolve 6.2.2606): vector `CoefficientFunction` objects support
  division by a scalar `CoefficientFunction`, differentiation through the resulting
  `sqrt(B·B + B_floor²)` normalization, and finite vector evaluation at an exact
  analytic O- or X-point. This permits one source expression to be differentiated
  from the same smoothly floored tensor assembled in the manufactured island form.

- Milestone 2.2 (NGSolve 6.2.2606): `mesh.GetTrafo(element)(integration_point)`
  yields a mapped point with its local `measure`; multiply that by
  `integration_point.weight` for physical quadrature weights. A `CoefficientFunction`
  accepts the `mesh.MapToAllElements({element_type: rule}, ngsolve.VOL)` result as a
  batched array, so extract weights/geometric scales per element but evaluate values
  and analytic gradient magnitudes in one vectorized NGSolve pass without exposing an
  NGSolve object in the array-backed volume-map API.
  `BSpline(2, [x0, *knots, xN], values)` is the open linear spline representation
  that interpolates the supplied values; guard its right endpoint explicitly because
  the native spline's final knot is half-open.

- Milestone 2.4 (xfem 2.1.2606 / NGSolve 6.2.2606): the optional ngsxfem wheel is
  distributed as the `xfem` package. Its high-order static cut integration helper is
  `xfem.lsetcurv.LevelSetMeshAdaptation`, not a top-level `xfem` name: call
  `CalcDeformation(level_set - level)` and then `Integrate(POS, 1, order=...)` for
  \(\{\chi>\hat\chi\}\). The direct `xfem.Integrate` route only uses the piecewise-linear
  cut approximation unless this deformation is supplied.

- PR-CI timing (NGSolve 6.2.2606): `CoefficientFunction.Compile()` (default
  `realcompile=False`, so no C++ compiler is pulled in — `DESIGN.md` §26) evaluates an
  expression tree through a cached-node graph instead of by repeated recursive descent.
  NGSolve performs no common-subexpression elimination without it, so the regularized M3
  operators re-evaluated `B_safe`, `b_safe`, and `grad(b_safe)` many times per
  integration point. Compiling the M3 form integrands and the order-20 diagnostic
  integrands cut `BilinearForm.Assemble` by ~24x for the perpendicular SUPG variant
  (`grad_perp` p=1, maxh=1/32: 10.8 s → 0.58 s) and the `-m "not slow"` suite from 236 s
  to 35 s locally. Assembled matrix entries and every solver diagnostic are **bitwise**
  unchanged — `Compile()` reorders nothing, it only caches — so no recorded rate table
  moves. Compile any integrand whose coefficient tree repeats a normalization or a
  projector; the win grows with `bonus_intorder` and with expression depth.

- PR-CI timing (NGSolve 6.2.2606): the constrained M3–M3b verification rows that
  repeatedly check h/p/N and profile behavior use `diagnostic_detail="core"`. This
  still evaluates residuals, independent (M2) shell currents, G couplings, shell means,
  and shell-resolution gates, but intentionally omits supplemental SUPG, floor,
  sampled-field, and parallel-current L2 diagnostics. The default remains `"full"`;
  the gradient-comparison cost and active-floor test exercises it. Cross-mesh p≤2 L2
  comparisons converge to better than 1e-5 relative between mapped orders 8 and 20, so
  the tests retain order 8. `pytest-xdist` runs module scopes on three processes by
  default; every M3 test uses one NGSolve worker per process, avoiding thread
  oversubscription.

- Test-suite baseline, 2026-08-16 (reference laptop, macOS/CPython 3.12, NGSolve
  6.2.2606, `-n 3 --dist=loadscope`): 173 tests, `make test-full` 182 s, `make test`
  171 s. The budgets are now normative — `DESIGN.md` §22.1 — and `make test` is over
  its 2-minute limit; the offenders and the plan are recorded in `STATUS.md` under
  "Test-time budget". Two consequences worth knowing before optimizing: `--dist=loadscope`
  keeps a module's tests on one worker, so wall-clock is set by the *slowest module*, not
  by the total — moving one 50 s test out of the heaviest module buys more than trimming
  several small ones elsewhere; and setup time shows up in `--durations` separately
  (`test_m3_gradient_comparison` spends 15 s in a module-scoped fixture), so read the
  `setup` rows, not just the `call` rows.
