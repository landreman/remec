# NGSolve API notes

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
