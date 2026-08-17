"""True two-dimensional axisymmetric R-Z geometry."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite

from remec.geometry.slab import _MeshBundle


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
