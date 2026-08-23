# ADR 0011 — Radially graded mesh for the 3D island resolution gate

Status: proposed

DECISION: pending human sign-off

## Context

Milestone 6.3 must solve (M4a)–(M4b) on the exact-CAD periodic cylinder and satisfy
three resolution gates simultaneously. `docs/STATUS.md` requires that “`w_c` is spanned
by at least `min_layer_cells` element widths” with the default `min_layer_cells=6`, and
explicitly says that inability to reach this with the direct default is never grounds
for reducing that number. `docs/DESIGN.md` §21 also says that sparse direct
factorization is the verification default, while “the planned production 3D algorithm
is iterative” and a production-scale design must not depend on direct factorization of
multi-million-DOF blocks.

The test-first milestone-6.3 prototype assembles the note-literal tensor

`K = epsilon_kappa I + (1-epsilon_kappa) b_safe b_safe^T`

on `PeriodicCylinder3D`, solves only (M4a), constructs the mollified normalized volume
map, and applies (M4b). It also runs the integrable, isotropic, and rank-one pollution
controls. The following local macOS measurements use NGSolve 6.2.2606:

| mesh / order | elements | epsilon_kappa | local widths across w_c | pollution ratio |
|---|---:|---:|---:|---:|
| maxh 0.45, p=1 | 800 | 1e-2 | 0.591 | 1.351 |
| maxh 0.30, p=2 | 2,504 | 1e-4 | 0.263 | 0.0499 |
| maxh 0.30 + one refinement, p=2 | 20,032 | 1e-4 | 0.504 | 0.00295 |

The p=2 rows clear the pollution gate, but polynomial order does not increase the number
of *element widths* across the layer. The refined 20,032-element row still reports no
resolved island flattening, as expected from the 0.504-cell measurement. Uniform
tetrahedral refinement multiplies the element count by approximately eight for each
factor two in local width; four more passes from that row would be about 82 million
elements. A prototype annular marking pass on the coarser `maxh=0.45` OCC mesh produced
6,387, 49,372, and 325,651 tetrahedra after one, two, and three passes. Because coarse
tetrahedra span much of the annulus, this isotropic local refinement still approaches a
multi-million-element problem before six radial widths are reached. It cannot support
the full epsilon_kappa ladder with the required live controls under the current direct
verification policy.

This is not grounds to reinterpret `min_layer_cells`, claim flattening from an
under-resolved row, lower the anisotropy target, or substitute a reduced-dimensional
calculation for the required 3D solve. The design does not currently select the mesh
technology needed to make the radial resolution criterion affordable.

## Options

### Option 1 — Add a radially graded, field-elongated periodic-cylinder mesh

Construct a conforming cylindrical mesh with explicit radial layers concentrated around
the resonant annulus, coarse angular/axial spacing along the field, periodic axial trace
pairing, and the same exact circular OCC wall/curved-geometry error contract as ADR 0009.
Split prisms into a compatible tetrahedral complex only if the split preserves the
periodic and de Rham contracts; otherwise add and verify the prism-specific space/order
pairing before using it. Measure the *radial projection* of each intersecting cell as the
layer width. Use the direct solver below its configured threshold and a native
CG/preconditioner path above it, recording both in the cost table.

Tradeoffs: this directly targets the physical resolution requirement and keeps the 3D
benchmark affordable, but it expands milestone 6.3 geometry work and requires new
periodic/high-aspect-ratio element-quality and solver tests. Prism support would also
need a new compatible-space contract because the existing tetrahedral de Rham factory
correctly rejects non-tetrahedral elements.

### Option 2 — Keep OCC isotropic tetrahedra and add only an iterative solver

Retain the existing mesh and replace sparse Cholesky with native preconditioned CG once
the DOF threshold is crossed.

Tradeoffs: this follows the planned Section-8.5 production direction and removes the
factorization-memory ceiling, but it does not remove the multi-million-element geometry,
quadrature, volume-map, and output costs. It is unlikely to make the required full ladder
practical on one node and does not address radial inefficiency.

### Option 3 — Count high-order nodes as layer cells

Define the reported width as `h/p` and let polynomial order satisfy the six-cell gate.

Tradeoffs: this is cheap and often useful as a spectral-resolution diagnostic, but it
changes the explicit acceptance criterion from element widths to interpolation
subscales. It could hide a layer inside a single element and is therefore a relaxation,
not an implementation of the current milestone.

### Option 4 — Use a reduced 2D island calculation for the threshold table

Retain one coarse 3D sentinel and obtain the resolved threshold from a poloidal model.

Tradeoffs: this is inexpensive, but `DESIGN.md` §8.6 says that the deliverable exists
specifically to establish the cost and pressure response of the three-dimensional solve.
It would not satisfy the phase gate.

## Recommendation

Adopt Option 1. A radially graded, field-elongated mesh is the only option that preserves
the six-element-width criterion, the exact 3D periodic-cylinder problem, and a plausible
one-node cost. The implementation should retain Option 2 as the solver path above the
direct threshold, but iterative algebra alone should not be treated as the resolution
solution. Reject Options 3 and 4 because they relax the acceptance criterion.

Until this decision is signed, milestone 6.3 remains blocked and no coupled 3D milestone
may start.
