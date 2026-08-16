# ADR 0005 — Coercion of the paired curved HDiv/L2 current constraint

## Context

ADR 0004 Option 4 was approved on the basis of a correct NGSolve mapping observation:
on a curved element, a general physical `div(HDiv)` field contains the contravariant
Piola factor `1/det(J)` and is not strongly contained in ordinary scalar `L2`. It then
made the stronger inference

> The design therefore appears infeasible as written on the selected backend

and changed `DESIGN.md` §5 and §10 so that curved projected-current divergence need
only converge, with a dimensionless 3% backstop. The note gives the reason the original
exact invariant matters:

> a field with ∇·J ≠ 0 is not the curl of any B, so exactness is precisely what makes
> Ampère's law in (M1) integrable

(note §5.2, immediately after equation `u_equation`).

The fourth adversarial Milestone-4.1 review found that ADR 0004 measured **containment**
of a general divergence in ordinary L2, while `DESIGN.md` §10 uses the different
property of **coercion** by the constraint

\[
(\nabla\cdot J_h,q)_\Omega=0 \qquad \forall q\in L^2_{p-2}.
\]

For the paired spaces established by Milestone 4.1, the contravariant Piola map gives

\[
\nabla_x\cdot v_h=\frac{1}{\det F'}\,\widehat\nabla\cdot\widehat v_h,
\qquad dx=\det F'\,d\widehat x.
\]

The determinant therefore cancels in the weak constraint:

\[
(\nabla_x\cdot v_h,q_h)_K
=\int_{\widehat K}
  (\widehat\nabla\cdot\widehat v_h)\,\widehat q_h\,d\widehat x.
\]

Because `div_ref(HDiv(p-1))` is exactly the reference span of `L2(p-2)`, orthogonality
to the paired L2 space forces the reference divergence, and hence the physical
divergence, to vanish pointwise. That mechanism does not require the general physical
divergence to be an ordinary-L2 member.

The review measured the actual §10 saddle-point solve on NGSolve 6.2.2606:

| curved mesh and pairing | general-divergence L2 containment defect | constrained `||div J_h||/||J_h||` | pointwise relative maximum |
| --- | ---: | ---: | ---: |
| ball, `Curve(3)`, 107 tets, `HDiv(1)→L2(0)` | 3.38e-1 | 5.78e-16 | 1.52e-15 |
| ball, `Curve(3)`, 107 tets, `HDiv(2)→L2(1)` | 2.40e-1 | 6.97e-16 | 2.63e-15 |
| ball, `Curve(4)`, 107 tets, `HDiv(2)→L2(1)` | 2.28e-1 | 6.62e-16 | 1.76e-15 |
| torus, `Curve(3)`, 1356 tets, `HDiv(2)→L2(1)` | — | 6.26e-16 | — |
| torus, `Curve(4)`, 1356 tets, `HDiv(3)→L2(2)` | — | 5.98e-16 | — |

Natural and essential normal traces gave the same roundoff result. The physical
divergence theorem closed to 12 digits, excluding a reference-coordinate artifact.
Varying only the terminal order on the same curved ball made the pairing decisive:

| `HDiv(2)` terminal space | constrained `||div J_h||/||J_h||` |
| --- | ---: |
| `L2(0)` (undersized) | 6.03e+0 |
| `L2(1)` (Milestone-4.1 pairing) | 6.97e-16 |
| `L2(2)` (oversized) | singular saddle point |

The same review also found a separate terminology and acceptance error in ADR 0004.
The `λ_h` in §10 is the **continuity multiplier**, not the H1 Coulomb-gauge multiplier
from §7.3. For deliberately non-solenoidal `J_raw`, `||λ_h||/||J_h||` approached a
nonzero continuous limit (approximately 0.175–0.181 under h-refinement and 0.175 across
geometry orders 2–5). It must not be required to converge to zero. Conversely,
`||div J_h||` is already at roundoff for the paired spaces, so requiring a convergence
rate for it is not meaningful and makes the current Milestone-4.4 criterion
unsatisfiable.

This evidence conflicts with the approved conclusion of ADR 0004 and with the design
changes made from it. Under the repository STOP rule, an agent must not decide whether
to supersede a signed ADR.

## Options

1. **Supersede ADR 0004 and restore exact current continuity.** Keep ordinary scalar
   L2 with the exact affine/curved order pairing. Restore the §5/§10 invariant that the
   constrained current is strongly divergence-free to roundoff on affine and curved
   meshes. Rename `λ_h` the continuity multiplier and do not require it to vanish.
   Milestone 4.4 must automate the actual mixed projection on curved ball and torus
   meshes, include undersized/oversized terminal-space negative controls, and retain
   the existing projection-correction, shell-moment, and Ampère-compatibility gates.
2. **Amend ADR 0004 in place to the same exact-coercion outcome.** Preserve ADR 0004 as
   the active record, replace its Option-4 conclusion with the determinant-cancellation
   argument and measurements above, and restore the exact design invariant. This has
   the same implementation outcome as Option 1 but rewrites an already approved record
   instead of preserving its mistaken conclusion as history.
3. **Retain ADR 0004 Option 4 despite the coercion evidence.** Continue to call the
   curved current only weakly divergence-free and keep a nonzero magnitude allowance.
   The current rate and `λ_h→0` requirements would still have to be removed because the
   measured strong divergence is resolution-independent roundoff and the continuity
   multiplier has a legitimate nonzero limit.

## Tradeoffs

Option 1 preserves a clear audit trail: ADR 0004 records the containment fact and the
decision it originally motivated, while ADR 0005 records why that inference was
superseded. It restores the note's exact Ampère-integrability invariant using the
existing NGSolve wheel and makes the terminal-order convention physically falsifiable.
Its cost is new Milestone-4.4 mixed-projection tests on curved ball and torus meshes.

Option 2 yields the same mathematics with one fewer active ADR, but obscures that the
signed Option-4 decision was reversed by new evidence. Option 3 preserves the user's
earlier choice in name, but knowingly weakens a property the paired formulation already
delivers and leaves the 3% backstop disconnected from observed backend behavior.

## Recommendation

Choose **Option 1**: supersede ADR 0004, restore exact curved current continuity, and
make the determinant-cancellation mechanism plus wrong-terminal-order controls part of
Milestone 4.4 acceptance. Keep the two valid NGSolve API facts from ADR 0004 in
`docs/dev_notes.md`: `ng.div(ng.curl(A_h))` is not a supported nested expression, and
the ordinary-L2 projection of a *general* curved `div(HDiv)` field is not a roundoff
identity. Neither fact prevents the paired weak constraint from being strongly
coercive.

## DECISION: Option 1 approved.
