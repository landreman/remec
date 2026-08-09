# NGSolve API notes

- Milestone 0.2 (NGSolve 6.2.2606): `ngsolve.SetNumThreads(int)` is available for
  process-global worker configuration; no `ngsolve.GetNumThreads` API is exposed.
  The direct API verification therefore runs in a subprocess so it cannot leak a
  process-global thread count into later tests.
