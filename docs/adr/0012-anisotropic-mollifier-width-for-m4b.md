# ADR 0012: Anisotropic mollifier width for M4b

**Status:** Proposed

**Date:** 2026-08-23

## Context

Milestone 6.3 must evaluate the mollified level-set map in (M4b),

\[
V_\chi(\hat\chi)=\int_\Omega H(\chi-\hat\chi)\,d^3x,
\qquad p=p_0(V_\chi(\chi)/V_\Omega),
\]

on the radially graded, tangentially stretched tetrahedral mesh accepted by ADR 0011.
`MollifiedVolumeMap` currently converts the spatial smoothing width to a level-set
width with

\[
\delta\chi_q=c_h |\det J_q|^{1/3}|\nabla\chi_q|.
\]

That isotropic element-size surrogate was verified on the earlier shape-regular meshes,
but neither DESIGN section 12.3 nor ADR 0011 defines its meaning on anisotropic cells.
On the accepted mesh, `|det J|^(1/3)` mixes the deliberately fine radial spacing with
the coarse angular and axial spacings. It can therefore span many radial layers even
though the resolution gate counts at least six radial widths across `w_c`.

This is material, not cosmetic. On the current p=2, 24-angular-cell, four-axial-cell
probe at `epsilon_kappa=1e-6`, the live (M4a) solution has a minimum O-point-ray
`|dchi/dr|` ratio of **0.55894** relative to the same-mesh integrable control. Thus the
anisotropic PDE has produced the required island response. With the existing
determinant-based M4b mollifier, however, the reported pressure-flattening width is
**0.0**. At `epsilon_kappa=1e-4`, the corresponding raw-gradient minimum is **0.94989**,
which also confirms that the reference threshold row is an onset rather than a fully
saturated island. Choosing a different anisotropic width changes the reported M4b
flattening, plateau, and co-area diagnostics, so the operational contract requires a
human decision before implementation continues.

## Options

### Option 1: Level-set-normal metric width (recommended)

At each quadrature point with unit normal
`n = grad(chi)/|grad(chi)|`, define the physical smoothing length from the mapped-element
metric,

\[
h_n = \frac{1}{\lVert J^{-1}n\rVert},
\qquad \delta\chi_q=c_h h_n |\nabla\chi_q|,
\]

with a documented reference-element normalization calibrated so it reproduces the
existing size on isotropically scaled reference tetrahedra. Use a guarded fallback at
critical points where `|grad(chi)|` is floored. Add affine, curved, isotropic-scaling,
rotation, and anisotropic manufactured tests, then re-run all Section 12.3 consistency
and milestone-6.3 flattening/refinement gates.

Tradeoffs: this measures the mesh resolution in the direction in which the smoothed
Heaviside transitions, is invariant under physical rotations, and preserves tangential
elongation. It is more implementation work and requires careful reference-cell
normalization and critical-point handling.

### Option 2: Minimum-singular-value width

Use `h_min = sigma_min(J)` (with the same reference-element normalization) at every
quadrature point.

Tradeoffs: simple and conservative, and it cannot let a long tangential direction
over-smear a thin layer. It under-smooths whenever the shortest cell direction is
tangential to the level set, so it can add quadrature noise and makes results depend on
an irrelevant direction.

### Option 3: Minimum physical edge length

Use the minimum curved or vertex-edge length of the containing tetrahedron.

Tradeoffs: easiest to explain and compute, but it is only an indirect metric measure,
is constant over an element, and has the same tangential under-smoothing problem as
Option 2. Curved high-order elements also need a sampling convention.

### Option 4: Keep `|det J|^(1/3)` and refine angularly/axially

Retain the current mollifier and make the stretched cells shape-regular enough that the
determinant surrogate approaches the radial width.

Tradeoffs: no level-set API change, but it gives back the cost reduction accepted in
ADR 0011 and makes an unrelated tangential resolution determine whether a radially
resolved island is visible. It is likely to return the benchmark to the multi-million
element regime that ADR 0011 rejected.

### Option 5: Tune `spatial_width_cells` for milestone 6.3

Choose a smaller scalar `c_h` on the stretched mesh.

Tradeoffs: minimal code change, but it is mesh-aspect dependent, is not a physical local
width, and turns the M4b topology signature into a tunable benchmark output. This is not
a defensible production rule.

## Recommendation

Accept Option 1. The mollified Heaviside varies in the level-set-normal direction, so
the normal metric width is the relevant spatial resolution. Require the implementation
to reproduce the existing isotropic-width behavior under uniform scaling, remain
rotation invariant, expose the chosen width and critical-point fallback in diagnostics,
and regenerate every affected verification table through committed scripts. Reject
Options 4 and 5 because they respectively undo ADR 0011 and tune the asserted result.

DECISION: Option 1 approved by the user on 2026-08-23.
