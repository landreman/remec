"""Analytic solid-torus geometry for compatible magnetic verification."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


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

    def mesh(self) -> Any:
        """Create the curved tetrahedral verification mesh."""
        import ngsolve as ng  # type: ignore[import-untyped]
        from netgen.occ import (  # type: ignore[import-untyped]
            Axes,
            Axis,
            OCCGeometry,
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
        solid = Revolve(meridional_disk, Axis((0.0, 0.0, 0.0), Z), 360.0)
        mesh = ng.Mesh(OCCGeometry(solid).GenerateMesh(maxh=self.max_element_size))
        mesh.Curve(self.geometry_order)
        return mesh
