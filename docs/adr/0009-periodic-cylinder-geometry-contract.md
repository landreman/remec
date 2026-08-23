# ADR 0009: Periodic-cylinder geometry contract

**Status:** Proposed

## Context

Milestone 6.2 requires `PeriodicCylinder3D` as the exact-geometry verification domain
for the Reiman--Greenside field. `DESIGN.md` Section 8.6 defines the domain as
"a periodic straight cylinder Ω = {r < a}" and says that "its mesh is affine and its
geometry exact -- no geometry-approximation error is entangled with the anisotropy
measurement." Section 16.2 likewise requires the cylinder to be "meshed with affine
tetrahedra" and says that it is "exactly representable -- its geometry-approximation
error is zero." The milestone row in `docs/STATUS.md` repeats both the circular domain
and affine-tetrahedra requirements.

Those requirements cannot all hold for a finite mesh. Every boundary face of an affine
tetrahedron is planar, so the boundary of a finite affine tetrahedral mesh is a finite
union of planar triangles. A nondegenerate circular cylindrical wall has nonzero
curvature in its cross-section and cannot equal that piecewise-planar boundary. An OCC
cylinder retains the analytic CAD surface as meshing input, but an order-one NGSolve
volume mesh still uses planar boundary facets; curving the mesh replaces affine element
maps with finite-order isoparametric maps and therefore violates the affine requirement.

This is a design-level conflict, not an NGSolve API discrepancy. Choosing a polygonal
cylinder, accepting a geometric approximation, or using non-affine elements changes the
milestone's stated scientific isolation and its periodic-space verification contract.
The `AGENTS.md` STOP conditions therefore prohibit implementing one choice without
human sign-off.

## Options

1. **Keep affine tetrahedra and define a polygonal straight cylinder.** Replace
   `{r<a}` by an explicitly specified regular-polygon cross-section extruded over the
   periodic length. The computational domain is then exactly represented by affine
   tetrahedra, and the axial periodic identification is exact. Record the polygon and
   its apothem/circumradius in every benchmark artifact.
2. **Keep the circular cylinder and allow curved tetrahedral geometry.** Generate the
   periodic circular cylinder from exact CAD and curve the mesh to an order comparable
   to the FE order. Measure circular-wall geometry error and include a geometry-order
   scan in milestone 6.2 before the anisotropy benchmark uses the domain.
3. **Keep an affine approximation to the circular cylinder and make its error
   explicit.** Use planar tetrahedra whose boundary vertices lie on the circle, remove
   the zero-geometry-error claim, and quantify area, volume, wall-radius, and relevant
   flux/balance errors under circumferential refinement. Pin the cross-section
   resolution independently of the volume `maxh` so later cost tables remain
   interpretable.

## Tradeoffs

Option 1 preserves the exact affine de Rham setting and cleanly isolates periodic-space
and solver-wrapper behavior. It changes the physical wall from `r=a`, removes exact
rotational symmetry, complicates radial power-balance references, and makes the
Reiman--Greenside radial coordinate no longer a wall-fitted coordinate. Results would
need a polygon-specific interpretation and would not literally satisfy Sections 8.6
and 16.2.

Option 2 preserves the intended circular domain and the radial analytic benchmark. It
introduces mapped-geometry conditioning and a nonzero finite-order geometry error, so
the claim that anisotropy results cannot be entangled with geometry error must be
replaced by a measured bound. It also broadens milestone 6.2's periodic H1, H(curl), and
H(div) compatibility tests from affine to curved mapped elements; the exact
HCurl-to-HDiv composition remains available, but high-order periodic wrapper support
must be demonstrated on the chosen curved mesh.

Option 3 keeps the simplest and cheapest periodic tetrahedral systems and makes the
existing affine wording honest. It preserves an asymptotically circular domain but not
an exact one; coarse 3D cost rows may be contaminated by wall-facet error unless the
geometry scan establishes a bound well below each asserted physics tolerance. It also
requires benchmark tables to distinguish circumferential geometry resolution from
interior h-refinement.

## Recommendation

Choose Option 2. The circular radial coordinate, resonance locations, analytic vector
potential, and later radial power-balance diagnostics are central to the Section 8.6
benchmark, so changing the wall to a polygon creates more physics ambiguity than it
removes. Amend Sections 8.6, 16.2, and 25 plus the milestone ledger to say that axial
periodicity and the straight-cylinder centerline are exact while the circular wall is
represented by a measured curved geometry approximation. Require the geometry error to
remain below a stated fraction of every milestone-6.3 accuracy gate; if the measured
cost is unacceptable, Option 3 is the appropriate fallback with an explicit geometry
error budget, not a claim of zero error.

DECISION: pending human sign-off
