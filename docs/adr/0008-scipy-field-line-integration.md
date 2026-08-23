# ADR 0008: SciPy dependency for field-line integration

**Status:** Proposed

## Context

Milestone 6.1 requires a field-line tracer implemented with a SciPy ODE solver. The
acceptance criterion names both prescribed analytic magnetic fields and finite-element
H(div) fields, versioned trace persistence, recovered rotational transform, magnetic
O- and X-point locations, and island-width measurements. `DESIGN.md` Sections 8.6,
19, and 25 specifically call for a "SciPy ODE" tracer.

SciPy is not currently declared in `pyproject.toml`, and it is not installed in the
project's existing development environment. NGSolve and NumPy therefore do not provide
an undeclared transitive SciPy installation that the tracer can safely assume. Adding
SciPy to the base package is a new base-dependency decision under `AGENTS.md` Section
"STOP conditions". The decision matters for wheel size and installation support on
every platform, even though SciPy publishes binary wheels for the project's supported
CPython versions and supplies the mature adaptive integrators required here.

The numerical acceptance criteria and tolerances for milestone 6.1 are not at issue.

## Options

1. **Declare SciPy as a base dependency.** Add a bounded SciPy requirement compatible
   with Python 3.10--3.14 and implement the production tracer with
   `scipy.integrate.solve_ivp` (initially the explicit high-order `DOP853` method).
2. **Make tracing an optional dependency.** Add a `poincare` extra containing SciPy;
   the diagnostics API remains importable without the extra and raises an actionable
   error when tracing is requested.
3. **Implement and maintain a project-local adaptive ODE integrator.** Keep the base
   dependency set unchanged, but depart from the explicit SciPy-ODE design requirement
   and assume responsibility for dense output, event handling, and tolerance control.

## Tradeoffs

Option 1 makes the milestone-6.1 capability available in a normal remec installation
and follows the design literally. SciPy is familiar in the target scientific-Python
community and its tested adaptive integration avoids a substantial numerical-method
maintenance burden. It increases the base installation size and adds another binary
wheel compatibility constraint; the version bounds must retain Python 3.10 and 3.14
coverage in CI.

Option 2 keeps users who never trace field lines from paying that installation cost.
It complicates the public diagnostics contract and permits a normal installation to
lack a capability required by the Phase-6 plan. CI, examples, and downstream users
would have to install the extra explicitly, and a missing extra could turn scientific
coverage into a skip unless tests guard against that.

Option 3 minimizes third-party dependencies but duplicates mature SciPy functionality.
It creates extra work to demonstrate local-error control and event accuracy before the
physics acceptance tests are meaningful, and it conflicts with the stated "SciPy ODE"
implementation requirement unless `DESIGN.md` is changed at the same time.

## Recommendation

Choose Option 1. SciPy is a community-familiar scientific dependency consistent with
ADR 0001, and the tracer is a standard Phase-6 diagnostic rather than an optional file
format or accelerator. Pin a bounded range only after checking the available releases'
Python support in the Linux/macOS CI matrix; do not vendor or silently fall back to a
different integrator.

DECISION: pending human sign-off
