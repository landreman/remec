"""Periodic circular-cylinder geometry selected by ADR 0009."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
from typing import Any


@dataclass(frozen=True, slots=True)
class GradedAnnulus:
    """One unperturbed cylindrical annulus targeted by ADR 0011 radial packing."""

    radius: float
    half_width: float
    maximum_radial_width: float

    def __post_init__(self) -> None:
        if not isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("annulus radius must be finite and positive")
        if not isfinite(self.half_width) or self.half_width <= 0.0:
            raise ValueError("annulus half_width must be finite and positive")
        if not isfinite(self.maximum_radial_width) or self.maximum_radial_width <= 0.0:
            raise ValueError("annulus maximum_radial_width must be finite and positive")


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
    radial_coordinates: tuple[float, ...] = ()
    element_types: tuple[str, ...] = ()
    maximum_aspect_ratio: float = 1.0
    maximum_target_aspect_ratio: float = 1.0


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

    def build_graded_mesh(
        self,
        target_annuli: tuple[GradedAnnulus, ...],
        *,
        angular_cells: int,
        axial_cells: int,
        background_radial_width: float = 0.125,
    ) -> _PeriodicCylinderMeshBundle:
        """Build the accepted ADR-0011 extrude-then-split graded tetrahedral mesh.

        A parameterized polar disk is packed around every unperturbed target annulus,
        extruded through the periodic length, and each triangular prism is split into
        three consistently oriented tetrahedra. The exact OCC cylinder remains attached
        to the manual mesh so ``Curve(geometry_order)`` projects the wall to the same
        analytic circle used by ADR 0009.
        """
        if not target_annuli:
            raise ValueError("target_annuli must not be empty")
        if not all(isinstance(target, GradedAnnulus) for target in target_annuli):
            raise TypeError("target_annuli must contain GradedAnnulus values")
        for target in target_annuli:
            if target.radius + target.half_width >= self.radius:
                raise ValueError("target annulus must lie strictly inside the cylinder")
        for name, value, minimum in (
            ("angular_cells", angular_cells, 8),
            ("axial_cells", axial_cells, 2),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        if angular_cells % 2:
            raise ValueError("angular_cells must be even for the periodic tetrahedral split")
        if not isfinite(background_radial_width) or background_radial_width <= 0.0:
            raise ValueError("background_radial_width must be finite and positive")

        from math import ceil, cos, dist, sin, tau

        import ngsolve as ng
        from netgen.csg import Pnt as MeshPointCoordinate  # type: ignore[import-untyped]
        from netgen.meshing import (  # type: ignore[import-untyped]
            Element2D,
            Element3D,
            FaceDescriptor,
            IdentificationType,
            Mesh,
            MeshPoint,
        )
        from netgen.occ import Cylinder, OCCGeometry, Pnt, Z

        background_points = {
            self.radius * index / ceil(self.radius / background_radial_width)
            for index in range(ceil(self.radius / background_radial_width) + 1)
        }
        radial_points = {
            point
            for point in background_points
            if not any(
                target.radius - target.half_width < point < target.radius + target.half_width
                for target in target_annuli
            )
        }
        for target in target_annuli:
            lower = target.radius - target.half_width
            upper = target.radius + target.half_width
            intervals = ceil((upper - lower) / target.maximum_radial_width)
            radial_points.update(
                lower + (upper - lower) * index / intervals for index in range(intervals + 1)
            )
        radial_coordinates = tuple(sorted(radial_points))

        solid = Cylinder(Pnt(0.0, 0.0, 0.0), Z, self.radius, self.periodic_length)
        geometry = OCCGeometry(solid)
        wall_surface = 0
        upper_surface = 0
        lower_surface = 0
        tolerance = 64.0 * 2.220446049250313e-16 * self.periodic_length
        for surface_number, face in enumerate(solid.faces, start=1):
            axial_coordinate = float(face.center.z)
            if abs(axial_coordinate) <= tolerance:
                lower_surface = surface_number
            elif abs(axial_coordinate - self.periodic_length) <= tolerance:
                upper_surface = surface_number
            else:
                wall_surface = surface_number
        if not wall_surface or not lower_surface or not upper_surface:
            raise RuntimeError("OCC cylinder did not expose the expected three surfaces")

        netgen_mesh = Mesh(dim=3)
        netgen_mesh.SetGeometry(geometry)
        netgen_mesh.SetMaterial(1, "domain")
        wall_descriptor = netgen_mesh.Add(
            FaceDescriptor(surfnr=wall_surface, domin=1, domout=0, bc=1)
        )
        upper_descriptor = netgen_mesh.Add(
            FaceDescriptor(surfnr=upper_surface, domin=1, domout=0, bc=2)
        )
        lower_descriptor = netgen_mesh.Add(
            FaceDescriptor(surfnr=lower_surface, domin=1, domout=0, bc=3)
        )
        for index, name in enumerate(("wall", "periodic_upper", "periodic_lower")):
            netgen_mesh.SetBCName(index, name)

        point_ids: dict[tuple[int, int, int], Any] = {}
        coordinates: dict[tuple[int, int, int], tuple[float, float, float]] = {}
        for axial_index in range(axial_cells + 1):
            axial_coordinate = self.periodic_length * axial_index / axial_cells
            key = (axial_index, 0, 0)
            coordinates[key] = (0.0, 0.0, axial_coordinate)
            point_ids[key] = netgen_mesh.Add(MeshPoint(MeshPointCoordinate(*coordinates[key])))
            for radial_index, radius in enumerate(radial_coordinates[1:], start=1):
                for angular_index in range(angular_cells):
                    angle = tau * angular_index / angular_cells
                    key = (axial_index, radial_index, angular_index)
                    coordinates[key] = (
                        radius * cos(angle),
                        radius * sin(angle),
                        axial_coordinate,
                    )
                    point_ids[key] = netgen_mesh.Add(
                        MeshPoint(MeshPointCoordinate(*coordinates[key]))
                    )

        disk_triangles: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
        for angular_index in range(angular_cells):
            disk_triangles.append(
                ((0, 0), (1, angular_index), (1, (angular_index + 1) % angular_cells))
            )
        for radial_index in range(1, len(radial_coordinates) - 1):
            for angular_index in range(angular_cells):
                next_angle = (angular_index + 1) % angular_cells
                disk_triangles.extend(
                    (
                        (
                            (radial_index, angular_index),
                            (radial_index + 1, angular_index),
                            (radial_index + 1, next_angle),
                        ),
                        (
                            (radial_index, angular_index),
                            (radial_index + 1, next_angle),
                            (radial_index, next_angle),
                        ),
                    )
                )

        maximum_aspect_ratio = 1.0
        maximum_target_aspect_ratio = 1.0

        def disk_vertex_rank(vertex: tuple[int, int]) -> int:
            radial_index, angular_index = vertex
            return (
                0 if radial_index == 0 else 1 + (radial_index - 1) * angular_cells + angular_index
            )

        for axial_index in range(axial_cells):
            for triangle in disk_triangles:
                ordered_triangle = tuple(sorted(triangle, key=disk_vertex_rank))
                bottom = [point_ids[(axial_index,) + vertex] for vertex in ordered_triangle]
                top = [point_ids[(axial_index + 1,) + vertex] for vertex in ordered_triangle]
                tetrahedra = (
                    (bottom[0], bottom[1], bottom[2], top[2]),
                    (bottom[0], bottom[1], top[1], top[2]),
                    (bottom[0], top[0], top[1], top[2]),
                )
                coordinate_keys = (
                    (
                        (axial_index, *ordered_triangle[0]),
                        (axial_index, *ordered_triangle[1]),
                        (axial_index, *ordered_triangle[2]),
                        (axial_index + 1, *ordered_triangle[2]),
                    ),
                    (
                        (axial_index, *ordered_triangle[0]),
                        (axial_index, *ordered_triangle[1]),
                        (axial_index + 1, *ordered_triangle[1]),
                        (axial_index + 1, *ordered_triangle[2]),
                    ),
                    (
                        (axial_index, *ordered_triangle[0]),
                        (axial_index + 1, *ordered_triangle[0]),
                        (axial_index + 1, *ordered_triangle[1]),
                        (axial_index + 1, *ordered_triangle[2]),
                    ),
                )
                for tetrahedron, keys in zip(tetrahedra, coordinate_keys, strict=True):
                    oriented_tetrahedron = list(tetrahedron)
                    oriented_keys = list(keys)
                    origin, first, second, third = (coordinates[key] for key in oriented_keys)
                    first_vector = tuple(first[index] - origin[index] for index in range(3))
                    second_vector = tuple(second[index] - origin[index] for index in range(3))
                    third_vector = tuple(third[index] - origin[index] for index in range(3))
                    cross_product = (
                        second_vector[1] * third_vector[2] - second_vector[2] * third_vector[1],
                        second_vector[2] * third_vector[0] - second_vector[0] * third_vector[2],
                        second_vector[0] * third_vector[1] - second_vector[1] * third_vector[0],
                    )
                    signed_jacobian = sum(
                        first_vector[index] * cross_product[index] for index in range(3)
                    )
                    # Netgen's tetrahedral reference orientation has the opposite sign
                    # of the Cartesian determinant above.
                    if signed_jacobian > 0.0:
                        oriented_tetrahedron[0], oriented_tetrahedron[1] = (
                            oriented_tetrahedron[1],
                            oriented_tetrahedron[0],
                        )
                        oriented_keys[0], oriented_keys[1] = oriented_keys[1], oriented_keys[0]
                    netgen_mesh.Add(Element3D(1, oriented_tetrahedron))
                    edge_lengths = [
                        dist(
                            coordinates[oriented_keys[left]],
                            coordinates[oriented_keys[right]],
                        )
                        for left in range(4)
                        for right in range(left + 1, 4)
                    ]
                    maximum_aspect_ratio = max(
                        maximum_aspect_ratio, max(edge_lengths) / min(edge_lengths)
                    )
                    centroid_radius = (
                        sum(coordinates[key][0] for key in oriented_keys) / 4.0
                    ) ** 2 + (sum(coordinates[key][1] for key in oriented_keys) / 4.0) ** 2
                    centroid_radius = centroid_radius**0.5
                    if any(
                        abs(centroid_radius - target.radius) <= target.half_width
                        for target in target_annuli
                    ):
                        maximum_target_aspect_ratio = max(
                            maximum_target_aspect_ratio,
                            max(edge_lengths) / min(edge_lengths),
                        )

        for triangle in disk_triangles:
            bottom = [point_ids[(0,) + vertex] for vertex in triangle]
            top = [point_ids[(axial_cells,) + vertex] for vertex in triangle]
            bottom_uv = [coordinates[(0,) + vertex][:2] for vertex in triangle]
            top_uv = [coordinates[(axial_cells,) + vertex][:2] for vertex in triangle]
            netgen_mesh.Add(
                Element2D(
                    lower_descriptor,
                    (bottom[0], bottom[2], bottom[1]),
                    uv=(bottom_uv[0], bottom_uv[2], bottom_uv[1]),
                )
            )
            netgen_mesh.Add(Element2D(upper_descriptor, top, uv=top_uv))

        wall_radial_index = len(radial_coordinates) - 1
        for axial_index in range(axial_cells):
            lower_z = self.periodic_length * axial_index / axial_cells
            upper_z = self.periodic_length * (axial_index + 1) / axial_cells
            for angular_index in range(angular_cells):
                next_angle = (angular_index + 1) % angular_cells
                lower_angle = tau * angular_index / angular_cells
                upper_angle = tau * (angular_index + 1) / angular_cells
                lower_left = point_ids[axial_index, wall_radial_index, angular_index]
                lower_right = point_ids[axial_index, wall_radial_index, next_angle]
                upper_left = point_ids[axial_index + 1, wall_radial_index, angular_index]
                upper_right = point_ids[axial_index + 1, wall_radial_index, next_angle]
                if angular_index < next_angle:
                    wall_triangles = (
                        (
                            (lower_left, lower_right, upper_right),
                            (
                                (lower_angle, lower_z),
                                (upper_angle, lower_z),
                                (upper_angle, upper_z),
                            ),
                        ),
                        (
                            (lower_left, upper_right, upper_left),
                            (
                                (lower_angle, lower_z),
                                (upper_angle, upper_z),
                                (lower_angle, upper_z),
                            ),
                        ),
                    )
                else:
                    wall_triangles = (
                        (
                            (lower_left, lower_right, upper_left),
                            (
                                (lower_angle, lower_z),
                                (upper_angle, lower_z),
                                (lower_angle, upper_z),
                            ),
                        ),
                        (
                            (lower_right, upper_right, upper_left),
                            (
                                (upper_angle, lower_z),
                                (upper_angle, upper_z),
                                (lower_angle, upper_z),
                            ),
                        ),
                    )
                for vertices, uv in wall_triangles:
                    netgen_mesh.Add(Element2D(wall_descriptor, vertices, uv=uv))

        for radial_index in range(len(radial_coordinates)):
            angular_indices = (0,) if radial_index == 0 else range(angular_cells)
            for angular_index in angular_indices:
                netgen_mesh.AddPointIdentification(
                    point_ids[0, radial_index, angular_index],
                    point_ids[axial_cells, radial_index, angular_index],
                    1,
                    IdentificationType.PERIODIC,
                )
        netgen_mesh.Compress()
        mesh = ng.Mesh(netgen_mesh)
        mesh.Curve(self.geometry_order)
        boundaries = tuple(mesh.GetBoundaries())
        expected = {"wall", "periodic_lower", "periodic_upper"}
        if set(boundaries) != expected:
            raise RuntimeError(
                f"graded cylinder boundaries {set(boundaries)} do not match {expected}"
            )
        identification_count = int(mesh.ngmesh.GetNrIdentifications())
        if identification_count != 1:
            raise RuntimeError("graded cylinder must contain exactly one periodic identification")
        element_types = tuple(sorted({element.type.name for element in mesh.Elements(ng.VOL)}))
        if element_types != ("TET",):
            raise RuntimeError(f"graded cylinder split produced {element_types}, expected TET")
        return _PeriodicCylinderMeshBundle(
            _mesh=mesh,
            boundary_names=("wall", "periodic_lower", "periodic_upper"),
            periodic_identification_count=identification_count,
            _geometry_owner=(solid, geometry, netgen_mesh),
            radial_coordinates=radial_coordinates,
            element_types=element_types,
            maximum_aspect_ratio=maximum_aspect_ratio,
            maximum_target_aspect_ratio=maximum_target_aspect_ratio,
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
        mapped_jacobian_ratio = (
            minimum_mapped_jacobian_determinant / maximum_mapped_jacobian_determinant
        )
        if mesh_bundle.radial_coordinates:
            local_ratios: list[float] = []
            for element in mesh.Elements(ng.VOL):
                transformation = mesh.GetTrafo(element)
                measures = [
                    float(transformation(point).measure)
                    for point in integration_rules[element.type]
                ]
                local_ratios.append(min(measures) / max(measures))
            mapped_jacobian_ratio = min(local_ratios)
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
            minimum_mapped_jacobian_ratio=mapped_jacobian_ratio,
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
