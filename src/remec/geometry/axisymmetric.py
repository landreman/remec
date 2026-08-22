"""True two-dimensional axisymmetric R-Z geometry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import ceil, isfinite
from typing import Any

import numpy as np
from numpy.typing import NDArray

from remec.analytic_equilibria import FluxContour
from remec.geometry.slab import _MeshBundle


@dataclass(frozen=True, slots=True)
class AxisymmetricFluxContourDomain:
    """A shaped R-Z domain whose single wall is an analytic constant-flux contour."""

    contour: FluxContour
    maxh: float
    geometry_order: int = 2

    def __post_init__(self) -> None:
        if not isfinite(self.maxh) or self.maxh <= 0.0:
            raise ValueError("maxh must be finite and positive")
        if self.geometry_order < 1:
            raise ValueError("geometry_order must be at least one")

    def build_mesh(self) -> _MeshBundle:
        """Build and curve a triangular mesh whose only boundary is ``wall``."""
        import ngsolve as ng  # type: ignore[import-untyped]
        from netgen.geom2d import SplineGeometry  # type: ignore[import-untyped]

        geometry = SplineGeometry()
        if self.contour.parameterizations:
            curves = self.contour.parameterizations
        else:
            curves = tuple(
                _piecewise_linear_curve(radius, height)
                for radius, height in self.contour.curve_segments()
            )
        _append_quadratic_spline_chains(geometry, curves, maxh=self.maxh)
        mesh = ng.Mesh(geometry.GenerateMesh(maxh=self.maxh))
        if self.geometry_order > 1:
            mesh.Curve(self.geometry_order)
        return _MeshBundle(mesh, ("wall",), geometry)

    def boundary_regions(self) -> dict[str, str]:
        return {"flux_surface": "wall"}

    def metadata(self) -> dict[str, object]:
        return {
            "geometry": "AxisymmetricFluxContourDomain",
            "maxh": self.maxh,
            "geometry_order": self.geometry_order,
            "boundary_flux": 0.0,
            "contour_samples": int(self.contour.radius.size),
            "contour_corners": len(self.contour.corner_indices),
            "boundary_regions": self.boundary_regions(),
            "toroidal_discretization": None,
        }


def _piecewise_linear_curve(
    radius: NDArray[np.float64], height: NDArray[np.float64]
) -> Callable[[float], tuple[float, float]]:
    """Return a Netgen ``[0,1]`` parameterization through sampled contour points."""
    count = radius.size

    def parameterization(parameter: float) -> tuple[float, float]:
        position = min(max(float(parameter), 0.0), 1.0) * (count - 1)
        lower = min(int(position), count - 2)
        fraction = position - lower
        return (
            float((1.0 - fraction) * radius[lower] + fraction * radius[lower + 1]),
            float((1.0 - fraction) * height[lower] + fraction * height[lower + 1]),
        )

    return parameterization


def _append_quadratic_spline_chains(
    geometry: Any,
    curves: tuple[Callable[[float], tuple[float, float]], ...],
    *,
    maxh: float,
) -> None:
    """Append explicit quadratic splines without retaining Python geometry callbacks."""
    point_indices: dict[tuple[float, float], int] = {}

    def append_point(point: tuple[float, float]) -> int:
        key = (round(point[0], 14), round(point[1], 14))
        if key not in point_indices:
            point_indices[key] = geometry.AppendPoint(*point, maxh=maxh)
        return point_indices[key]

    target_segments = max(len(curves), ceil(10.0 / maxh))
    for curve_index, curve in enumerate(curves):
        first = (curve_index * target_segments) // len(curves)
        last = ((curve_index + 1) * target_segments) // len(curves)
        subdivisions = last - first
        for index in range(subdivisions):
            lower = index / subdivisions
            upper = (index + 1) / subdivisions
            midpoint = 0.5 * (lower + upper)
            start = curve(lower)
            interpolated_midpoint = curve(midpoint)
            stop = curve(upper)
            control = (
                2.0 * interpolated_midpoint[0] - 0.5 * (start[0] + stop[0]),
                2.0 * interpolated_midpoint[1] - 0.5 * (start[1] + stop[1]),
            )
            start_index = append_point(start)
            midpoint_index = append_point(control)
            stop_index = append_point(stop)
            geometry.Append(
                ["spline3", start_index, midpoint_index, stop_index],
                leftdomain=1,
                rightdomain=0,
                bc="wall",
                maxh=maxh,
            )


@dataclass(frozen=True, slots=True)
class AxisymmetricRZDomain:
    """A rectangular poloidal domain bounded away from the cylindrical axis."""

    radial_bounds: tuple[float, float]
    vertical_bounds: tuple[float, float]
    maxh: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (*self.radial_bounds, *self.vertical_bounds)):
            raise ValueError("axisymmetric bounds must be finite")
        if self.radial_bounds[0] <= 0.0 or self.radial_bounds[0] >= self.radial_bounds[1]:
            raise ValueError("radial bounds must be strictly ordered and have R_min > 0")
        if self.vertical_bounds[0] >= self.vertical_bounds[1]:
            raise ValueError("vertical bounds must be strictly ordered")
        if not isfinite(self.maxh) or self.maxh <= 0.0:
            raise ValueError("maxh must be finite and positive")

    def build_mesh(self) -> _MeshBundle:
        """Build the deterministic triangular R-Z verification mesh."""
        from ngsolve.meshes import MakeStructured2DMesh  # type: ignore[import-untyped]

        radial_width = self.radial_bounds[1] - self.radial_bounds[0]
        vertical_width = self.vertical_bounds[1] - self.vertical_bounds[0]
        nx = max(1, ceil(radial_width / self.maxh))
        ny = max(1, ceil(vertical_width / self.maxh))
        mesh = MakeStructured2DMesh(
            quads=False,
            nx=nx,
            ny=ny,
            mapping=lambda x, y: (
                self.radial_bounds[0] + radial_width * x,
                self.vertical_bounds[0] + vertical_width * y,
            ),
        )
        return _MeshBundle(mesh, ("bottom", "right", "top", "left"))

    def boundary_regions(self) -> dict[str, str]:
        """Map physical R-Z side names to backend boundary regions."""
        return {"z_min": "bottom", "r_max": "right", "z_max": "top", "r_min": "left"}

    def metadata(self) -> dict[str, object]:
        """Return the backend-independent reduced-geometry record."""
        return {
            "geometry": "AxisymmetricRZDomain",
            "radial_bounds": self.radial_bounds,
            "vertical_bounds": self.vertical_bounds,
            "maxh": self.maxh,
            "boundary_regions": self.boundary_regions(),
            "toroidal_discretization": None,
        }
