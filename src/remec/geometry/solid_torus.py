"""Analytic solid-torus geometry for compatible magnetic verification."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True, slots=True)
class _TorusMeshBundle:
    """Internal carrier that keeps the NGSolve mesh out of the public API."""

    _mesh: Any
    boundary_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AnalyticSolidTorus:
    """Simple circular solid torus linking the cylindrical axis.

    The domain is ``(sqrt(x^2+y^2)-R)^2 + z^2 < a^2``.  Restricting
    ``geometry_order`` to the measured 1--4 range keeps the curved-mesh behavior
    covered by the milestone-4.3 verification table.
    """

    major_radius: float
    minor_radius: float
    max_element_size: float = 1.2
    geometry_order: int = 3

    def __post_init__(self) -> None:
        """Reject horn/spindle tori and unverified meshing parameters."""
        if (
            not isfinite(self.major_radius)
            or not isfinite(self.minor_radius)
            or self.minor_radius <= 0.0
            or self.major_radius <= self.minor_radius
        ):
            raise ValueError("radii must be finite and satisfy major_radius > minor_radius > 0")
        if not isfinite(self.max_element_size) or self.max_element_size <= 0.0:
            raise ValueError("max_element_size must be finite and positive")
        if isinstance(self.geometry_order, bool) or not isinstance(self.geometry_order, int):
            raise TypeError("geometry_order must be an integer")
        if not 1 <= self.geometry_order <= 4:
            raise ValueError("geometry_order must be in the verified range 1 through 4")

    def _revolved_solid(self, angle: float) -> Any:
        """Return the OCC solid swept through ``angle`` degrees."""
        from netgen.occ import (  # type: ignore[import-untyped]
            Axes,
            Axis,
            Revolve,
            WorkPlane,
            X,
            Y,
            Z,
        )

        meridional_disk = (
            WorkPlane(Axes((0.0, 0.0, 0.0), n=Y, h=X))
            .MoveTo(self.major_radius, 0.0)
            .Circle(self.minor_radius)
            .Face()
        )
        return Revolve(meridional_disk, Axis((0.0, 0.0, 0.0), Z), angle)

    def build_mesh(self) -> _TorusMeshBundle:
        """Create the curved tetrahedral verification mesh with a named wall."""
        import ngsolve as ng  # type: ignore[import-untyped]
        from netgen.occ import OCCGeometry

        solid = self._revolved_solid(360.0)
        solid.faces.name = "wall"
        mesh = ng.Mesh(OCCGeometry(solid).GenerateMesh(maxh=self.max_element_size))
        mesh.Curve(self.geometry_order)
        return _TorusMeshBundle(_mesh=mesh, boundary_names=("wall",))

    def _build_poloidal_cut_mesh(self, *, geometry_order: int = 6) -> _TorusMeshBundle:
        """Build a half torus whose start face is an explicit poloidal cut."""
        import ngsolve as ng
        from netgen.occ import OCCGeometry

        solid = self._revolved_solid(180.0)
        for face, name in zip(solid.faces, ("wall", "cut_start", "cut_end"), strict=True):
            face.name = name
        mesh = ng.Mesh(OCCGeometry(solid).GenerateMesh(maxh=self.max_element_size))
        mesh.Curve(geometry_order)
        return _TorusMeshBundle(
            _mesh=mesh,
            boundary_names=("wall", "cut_start", "cut_end"),
        )

    def boundary_regions(self) -> dict[str, str]:
        """Return the named physical wall region."""
        return {"wall": "wall"}

    def characteristic_length(self) -> float:
        """Return the largest diameter used to nondimensionalize the torus."""
        return 2.0 * (self.major_radius + self.minor_radius)

    def harmonic_basis(self, mesh_bundle: _TorusMeshBundle) -> list[object]:
        """Return the one normalized harmonic field of this solid torus."""
        from remec.fem._harmonic_flux import build_analytic_torus_harmonic_field

        solution = build_analytic_torus_harmonic_field(mesh_bundle._mesh, self)
        return [solution.field]

    def metadata(self) -> dict[str, object]:
        """Return reproducible geometry metadata without backend objects."""
        return {
            "geometry": "AnalyticSolidTorus",
            "major_radius": self.major_radius,
            "minor_radius": self.minor_radius,
            "max_element_size": self.max_element_size,
            "geometry_order": self.geometry_order,
            "boundary_regions": self.boundary_regions(),
        }
