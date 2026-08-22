"""Two-dimensional slab geometry."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Any


@dataclass(frozen=True, slots=True)
class _MeshBundle:
    """Internal mesh carrier returned by a geometry construction.

    The NGSolve mesh remains private so future public geometry consumers depend on
    named regions and metadata rather than backend-specific mesh operations.
    """

    _mesh: Any
    boundary_names: tuple[str, ...]
    _geometry_owner: Any = None


@dataclass(frozen=True, slots=True)
class Slab2D:
    """A rectangular structured domain with named physical boundary regions.

    ``periodic_y`` records and builds the top/bottom mesh identification used by
    finite-element consumers to construct a periodic space. ``subdivisions``
    optionally fixes the two axis counts, which permits a layer-aligned verification
    mesh to resolve the layer-normal direction without over-resolving its smooth
    periodic harmonic.
    """

    maxh: float
    lower: tuple[float, float] = (0.0, 0.0)
    upper: tuple[float, float] = (1.0, 1.0)
    subdivisions: tuple[int, int] | None = None
    periodic_y: bool = False

    def __post_init__(self) -> None:
        if not isfinite(self.maxh) or self.maxh <= 0.0:
            raise ValueError("maxh must be finite and positive")
        if not all(isfinite(value) for value in (*self.lower, *self.upper)):
            raise ValueError("slab bounds must be finite")
        if self.lower[0] >= self.upper[0] or self.lower[1] >= self.upper[1]:
            raise ValueError("slab lower bounds must be strictly below upper bounds")
        if self.subdivisions is not None and (
            len(self.subdivisions) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in self.subdivisions
            )
        ):
            raise ValueError("subdivisions must contain two positive integers")
        if not isinstance(self.periodic_y, bool):
            raise TypeError("periodic_y must be a boolean")

    @classmethod
    def unit_square(cls, *, maxh: float) -> Slab2D:
        """Return the unit-square slab used by early kernel verification."""
        return cls(maxh=maxh)

    def build_mesh(self) -> _MeshBundle:
        """Build a deterministic structured slab mesh with named boundaries."""
        from ngsolve.meshes import MakeStructured2DMesh  # type: ignore[import-untyped]

        width = self.upper[0] - self.lower[0]
        height = self.upper[1] - self.lower[1]
        if self.subdivisions is None:
            nx = max(1, ceil(width / self.maxh))
            ny = max(1, ceil(height / self.maxh))
        else:
            nx, ny = self.subdivisions
        boundary_names = (
            ("right", "left")
            if self.periodic_y
            else (
                "bottom",
                "right",
                "top",
                "left",
            )
        )
        mesh = MakeStructured2DMesh(
            quads=False,
            nx=nx,
            ny=ny,
            periodic_y=self.periodic_y,
            mapping=lambda x, y: (self.lower[0] + width * x, self.lower[1] + height * y),
        )
        return _MeshBundle(
            _mesh=mesh,
            boundary_names=boundary_names,
        )

    def boundary_regions(self) -> dict[str, str]:
        """Return named regions for the slab's non-periodic exterior sides."""
        names = ("right", "left") if self.periodic_y else ("bottom", "right", "top", "left")
        return {name: name for name in names}

    def characteristic_length(self) -> float:
        """Return the largest side length used to nondimensionalize the slab."""
        return max(self.upper[0] - self.lower[0], self.upper[1] - self.lower[1])

    def harmonic_basis(self, mesh_bundle: _MeshBundle) -> list[object]:
        """Return no 3D magnetic harmonic-flux fields for this verification slab."""
        del mesh_bundle
        return []

    def metadata(self) -> dict[str, object]:
        """Return reproducible geometry metadata without exposing backend objects."""
        return {
            "geometry": "Slab2D",
            "lower": self.lower,
            "upper": self.upper,
            "maxh": self.maxh,
            "subdivisions": self.subdivisions,
            "periodic_y": self.periodic_y,
            "boundary_regions": self.boundary_regions(),
        }
