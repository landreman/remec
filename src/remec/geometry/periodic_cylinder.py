"""Periodic circular-cylinder geometry selected by ADR 0009."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
from typing import Any


@dataclass(frozen=True, slots=True)
class PeriodicCylinderGeometryMetrics:
    """Measured ADR-0009 geometry errors for one mesh configuration."""

    geometry_order: int
    refinements: int
    elements: int
    wall_radius_rms_relative_error: float
    cross_section_area_relative_error: float
    volume_relative_error: float
    boundary_flux_relative_error: float
    minimum_mapped_jacobian_determinant: float
    maximum_mapped_jacobian_determinant: float
    minimum_mapped_jacobian_ratio: float

    @property
    def maximum_relative_error(self) -> float:
        """Return the conservative scalar budget consumed by milestone 6.3."""
        return max(
            self.wall_radius_rms_relative_error,
            self.cross_section_area_relative_error,
            self.volume_relative_error,
            self.boundary_flux_relative_error,
        )


@dataclass(frozen=True, slots=True)
class _PeriodicCylinderMeshBundle:
    """Internal carrier keeping the NGSolve mesh out of the public API."""

    _mesh: Any
    boundary_names: tuple[str, ...]
    periodic_identification_count: int
    _geometry_owner: Any = None


@dataclass(frozen=True, slots=True)
class PeriodicCylinder3D:
    """Circular cylinder with its axial end faces periodically identified."""

    radius: float = 1.0
    major_radius: float = 1.0
    # Netgen's coarser maxh families can contain nearly singular tetrahedra after
    # Curve(4), despite small integral geometry errors. 0.45 is the coarsest
    # cross-platform family validated by the mapped-Jacobian gate below.
    max_element_size: float = 0.45
    geometry_order: int = 3
    refinements: int = 0

    def __post_init__(self) -> None:
        for name, value in (("radius", self.radius), ("major_radius", self.major_radius)):
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not isfinite(self.max_element_size) or self.max_element_size <= 0.0:
            raise ValueError("max_element_size must be finite and positive")
        if isinstance(self.geometry_order, bool) or not isinstance(self.geometry_order, int):
            raise TypeError("geometry_order must be an integer")
        if not 1 <= self.geometry_order <= 4:
            raise ValueError("geometry_order must be in the verified range 1 through 4")
        if isinstance(self.refinements, bool) or not isinstance(self.refinements, int):
            raise TypeError("refinements must be an integer")
        if self.refinements < 0:
            raise ValueError("refinements must be non-negative")

    @property
    def periodic_length(self) -> float:
        """Return the identified axial length ``2*pi*R0``."""
        return 2.0 * pi * self.major_radius

    def build_mesh(
        self,
        *,
        local_refinement_radius: float | None = None,
        local_refinement_half_width: float | None = None,
    ) -> _PeriodicCylinderMeshBundle:
        """Build the curved periodic tetrahedral mesh selected by ADR 0009.

        When a radius and half-width are supplied, ``refinements`` marks only elements
        intersecting that annulus before each refinement pass.  Milestone 6.3 uses this
        shell grading to resolve ``w_c`` without uniformly refining the long cylinder.
        """
        if (local_refinement_radius is None) != (local_refinement_half_width is None):
            raise ValueError("local refinement requires both radius and half-width")
        if local_refinement_radius is not None and (
            not isfinite(local_refinement_radius)
            or local_refinement_radius < 0.0
            or local_refinement_radius > self.radius
        ):
            raise ValueError("local refinement radius must lie in the cylinder")
        if local_refinement_half_width is not None and (
            not isfinite(local_refinement_half_width) or local_refinement_half_width <= 0.0
        ):
            raise ValueError("local refinement half-width must be finite and positive")
        import ngsolve as ng  # type: ignore[import-untyped]
        from netgen.occ import (  # type: ignore[import-untyped]
            Cylinder,
            IdentificationType,
            OCCGeometry,
            Pnt,
            Translation,
            Z,
        )

        solid = Cylinder(Pnt(0.0, 0.0, 0.0), Z, self.radius, self.periodic_length)
        lower = None
        upper = None
        wall = None
        coordinate_tolerance = 64.0 * 2.220446049250313e-16 * self.periodic_length
        for face in solid.faces:
            axial_coordinate = float(face.center.z)
            if abs(axial_coordinate) <= coordinate_tolerance:
                lower = face
            elif abs(axial_coordinate - self.periodic_length) <= coordinate_tolerance:
                upper = face
            else:
                wall = face
        if lower is None or upper is None or wall is None:
            raise RuntimeError("OCC cylinder did not expose one wall and two axial end faces")
        wall.name = "wall"
        lower.name = "periodic_lower"
        upper.name = "periodic_upper"
        lower.Identify(
            upper,
            "periodic_z",
            IdentificationType.PERIODIC,
            Translation((0.0, 0.0, self.periodic_length)),
        )
        geometry = OCCGeometry(solid)
        mesh = ng.Mesh(geometry.GenerateMesh(maxh=self.max_element_size))
        # Refine the straight mesh before curving: curving first leaves newly refined
        # children with order-one geometry and destroys the ADR-0009 refinement scan.
        for _ in range(self.refinements):
            if local_refinement_radius is not None:
                assert local_refinement_half_width is not None
                lower = max(0.0, local_refinement_radius - local_refinement_half_width)
                upper = min(self.radius, local_refinement_radius + local_refinement_half_width)
                for element in mesh.Elements(ng.VOL):
                    radii = [
                        (mesh[vertex].point[0] ** 2 + mesh[vertex].point[1] ** 2) ** 0.5
                        for vertex in element.vertices
                    ]
                    intersects_shell = min(radii) <= upper and max(radii) >= lower
                    mesh.SetRefinementFlag(ng.ElementId(ng.VOL, element.nr), intersects_shell)
            mesh.Refine()
        mesh.Curve(self.geometry_order)
        actual_boundaries = set(mesh.GetBoundaries())
        expected_boundaries = {"wall", "periodic_lower", "periodic_upper"}
        if actual_boundaries != expected_boundaries:
            raise RuntimeError(
                f"periodic cylinder boundaries {actual_boundaries} do not match {expected_boundaries}"
            )
        identification_count = int(mesh.ngmesh.GetNrIdentifications())
        if identification_count != 1:
            raise RuntimeError(
                "periodic cylinder must contain exactly one Netgen identification, "
                f"found {identification_count}"
            )
        return _PeriodicCylinderMeshBundle(
            _mesh=mesh,
            boundary_names=("wall", "periodic_lower", "periodic_upper"),
            periodic_identification_count=identification_count,
            _geometry_owner=(solid, geometry),
        )

    def boundary_regions(self) -> dict[str, str]:
        """Return the physical wall and named periodic end faces."""
        return {
            "wall": "wall",
            "periodic_lower": "periodic_lower",
            "periodic_upper": "periodic_upper",
        }

    def characteristic_length(self) -> float:
        """Return the larger of the cylinder diameter and periodic length."""
        return max(2.0 * self.radius, self.periodic_length)

    def harmonic_basis(self, mesh_bundle: _PeriodicCylinderMeshBundle) -> list[object]:
        """Return the normalized axial harmonic field required by (M1)."""
        if not isinstance(mesh_bundle, _PeriodicCylinderMeshBundle):
            raise TypeError("mesh_bundle must be built by PeriodicCylinder3D")
        from remec.fem._harmonic_flux import build_periodic_cylinder_harmonic_field

        solution = build_periodic_cylinder_harmonic_field(mesh_bundle._mesh, self)
        return [solution.field]

    def measure_geometry(
        self, mesh_bundle: _PeriodicCylinderMeshBundle
    ) -> PeriodicCylinderGeometryMetrics:
        """Measure the ADR-0009 circular-wall geometry-error budget."""
        if not isinstance(mesh_bundle, _PeriodicCylinderMeshBundle):
            raise TypeError("mesh_bundle must be built by PeriodicCylinder3D")
        import ngsolve as ng
        import numpy as np

        mesh = mesh_bundle._mesh
        quadrature_order = 2 * self.geometry_order + 10
        wall = mesh.Boundaries("wall")
        upper = mesh.Boundaries("periodic_upper")
        normal = ng.specialcf.normal(3)
        radius = ng.sqrt(ng.x**2 + ng.y**2)
        tangent = ng.CoefficientFunction((-ng.y, ng.x, 0.0))
        wall_area = float(ng.Integrate(1.0, mesh, ng.BND, definedon=wall, order=quadrature_order))
        radius_rms_error = float(
            ng.sqrt(
                ng.Integrate(
                    (radius - self.radius) ** 2,
                    mesh,
                    ng.BND,
                    definedon=wall,
                    order=quadrature_order,
                )
                / wall_area
            )
            / self.radius
        )
        measured_area = float(
            ng.Integrate(1.0, mesh, ng.BND, definedon=upper, order=quadrature_order)
        )
        exact_area = pi * self.radius**2
        measured_volume = float(ng.Integrate(1.0, mesh, order=quadrature_order))
        exact_volume = exact_area * self.periodic_length
        boundary_flux_error = float(
            ng.sqrt(
                ng.Integrate(
                    (tangent * normal) ** 2,
                    mesh,
                    ng.BND,
                    definedon=wall,
                    order=quadrature_order,
                )
                / ng.Integrate(
                    tangent * tangent,
                    mesh,
                    ng.BND,
                    definedon=wall,
                    order=quadrature_order,
                )
            )
        )
        element_types = {element.type for element in mesh.Elements(ng.VOL)}
        integration_rules = {
            element_type: ng.IntegrationRule(element_type, quadrature_order)
            for element_type in element_types
        }
        mapped_points = mesh.MapToAllElements(integration_rules, ng.VOL)
        mapped_jacobian_determinants = np.asarray(
            ng.Det(ng.specialcf.JacobianMatrix(3))(mapped_points),
            dtype=float,
        ).reshape(-1)
        minimum_mapped_jacobian_determinant = float(np.min(mapped_jacobian_determinants))
        maximum_mapped_jacobian_determinant = float(np.max(mapped_jacobian_determinants))
        return PeriodicCylinderGeometryMetrics(
            geometry_order=self.geometry_order,
            refinements=self.refinements,
            elements=mesh.ne,
            wall_radius_rms_relative_error=radius_rms_error,
            cross_section_area_relative_error=abs(measured_area - exact_area) / exact_area,
            volume_relative_error=abs(measured_volume - exact_volume) / exact_volume,
            boundary_flux_relative_error=boundary_flux_error,
            minimum_mapped_jacobian_determinant=minimum_mapped_jacobian_determinant,
            maximum_mapped_jacobian_determinant=maximum_mapped_jacobian_determinant,
            minimum_mapped_jacobian_ratio=(
                minimum_mapped_jacobian_determinant / maximum_mapped_jacobian_determinant
            ),
        )

    def metadata(self) -> dict[str, object]:
        """Return deterministic geometry metadata without backend objects."""
        return {
            "geometry": "PeriodicCylinder3D",
            "radius": self.radius,
            "major_radius": self.major_radius,
            "periodic_length": self.periodic_length,
            "max_element_size": self.max_element_size,
            "geometry_order": self.geometry_order,
            "refinements": self.refinements,
            "boundary_regions": self.boundary_regions(),
        }
