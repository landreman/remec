# NGSolve API notes

> Entries for milestones 3.3–3.4 mention `PrescribedCurrentProfile` and the former
> `u=F(p)+ũ` shift. After the 2026-08-15 model revision those entries remain accurate
> descriptions of NGSolve expression behavior, but not of the production current-profile
> closure. Follow `DESIGN.md` §9.2 and `STATUS.md` milestones 3.5–3.6.

- Milestone 4.1 (NGSolve 6.2.2606): on affine tetrahedra, the exact order pairing is
  `H1(p+1) -> HCurl(p) -> HDiv(max(p-1, 0)) -> L2(max(p-2, 0))`; equal integer
  `order` arguments do not give the exact global complex.  The offsets are element-
  family specific: in particular, a hexahedron uses a different convention, so the
  tetrahedral factory rejects every non-tetrahedral volume element.  On a third-order
  curved OCC tetrahedral ball, the Piola-mapped gradient/curl inclusions and both
  `curl(grad)` and `div(curl)` identities remained below 6.95e-13.  A general
  `div(HDiv)` field did not lie in ordinary scalar `L2` there (relative projection
  defects 0.23--0.32), because the divergence carries the contravariant Piola
  `1/det(J)` density while scalar `L2` uses the ordinary pullback.  Test the curved-
  mesh (M1) invariant by evaluating `div(curl(A_h))` directly and use `L2` as the weak
  divergence-constraint/diagnostic space; do not assert strong curved `div(HDiv)`
  membership in ordinary `L2`.  An exploratory
  `MakeStructured3DMesh(secondorder=True, mapping=<nonlinear>)` construction segfaulted
  in this local wheel; `netgen.occ` mesh generation followed by `mesh.Curve(order)` was
  stable and should be the verification path for curved tetrahedra.

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
