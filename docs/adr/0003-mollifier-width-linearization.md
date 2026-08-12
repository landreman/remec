# ADR 0003 — Linearization of gradient-scaled mollifier widths

## Context

Milestone 2.3 implements the nonlocal derivative of the interpretive M4b pressure
transplant.  The note defines the mollified volume functional in `(mollified_V)` with
the local width

\[
\varepsilon(\mathbf r)=c h\lvert\nabla\chi\rvert,
\qquad
V_\chi^{(\varepsilon)}(\hat\chi)
=\int_\Omega H_\varepsilon(\chi-\hat\chi)\,d^3r.
\]

It subsequently gives `(V_derivatives)`,

\[
\frac{\partial V_\chi^{(\varepsilon)}(\hat\chi)}{\partial\chi_i}
=H'_{\varepsilon}(\chi_i-\hat\chi)w_i.
\]

The displayed derivative contains the explicit variation of the Heaviside argument,
but does not state whether the derivative must also propagate through the
gradient-dependent width. `DESIGN.md` §12.2 requires that width scaling, while §12.6
requires a JVP built from the `H'_epsilon w_i` data. These statements permit two
readings with materially different Newton actions.

The Claude review for PR #14 measured the distinction on a variable-gradient
manufactured level set: the current frozen-width action agrees with its matching
finite difference to `2.8e-9` relative, while it differs from a rebuild-the-width
finite difference by `3.75e-5` relative. The latter discrepancy scales as O(epsilon)
in the reviewer’s refinement experiment. This is not a tolerance issue: it determines
which mathematical functional the Newton derivative represents.

## Options

1. **Frozen-width (quasi-Newton) action.** At each linearization point, form
   `epsilon_i = c h_i max(|grad chi_i|, floor)` once and use exactly the displayed
   `(V_derivatives)` action `sum_i H'_epsilon_i w_i delta_chi_i`. Rebuild widths only
   after accepting the nonlinear iterate.
2. **Full live-width derivative.** Differentiate through `|grad chi|` (and through its
   smooth critical-level safeguard) so the JVP is the derivative of the complete
   rebuild-the-map functional. This requires a precise discrete-gradient and
   critical-floor derivative specification, including the non-smooth `max` currently
   used in the width construction.
3. **Redefine the regularization.** Replace the present width rule with a smooth
   state-independent or explicitly differentiable width model, then derive and verify
   its full Jacobian. This changes the current §12.2 implementation and its existing
   verification baseline.

## Tradeoffs

Option 1 is the literal finite-dimensional formula displayed in `(V_derivatives)` and
keeps the nonlocal action to one level-set averaging pass, but it is a quasi-Newton
linearization rather than the derivative of a map that recomputes widths. Option 2
matches that rebuilt functional but may add local gradient-coupled terms, needs a
decision at the critical-level floor, and must be verified through the FEM gradient
operator rather than only quadrature values. Option 3 makes differentiability explicit
but risks changing the spatially uniform smoothing required by §12.2 and would require
new consistency and convergence verification.

## Recommendation

Do not select an implementation until a human chooses whether Newton is permitted to
use the frozen-width quasi-Newton reading, or must linearize the live width. If Option
1 is selected, label it as a quasi-Newton action and retain a live-width discrepancy
regression. If Option 2 or 3 is selected, specify the critical-floor derivative before
implementation.

## DECISION: Option 1

For now we choose the simplest option so we can proceed to get a minimal version
of the code working. The issue is noted here so if poor convergence is observed
later when nonlinear solves are performed, we can reconsider option 2.
