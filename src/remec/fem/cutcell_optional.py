"""Optional high-order sharp level-set volume reference using ngsxfem."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import ngsolve as ng  # type: ignore[import-untyped]


class CutCellUnavailableError(ImportError):
    """The optional ``xfem`` extra is required for a sharp volume reference."""


def _load_cutcell_dependencies() -> tuple[Any, Any, Any]:
    """Load ngsxfem only when the optional reference is constructed."""
    try:
        import ngsolve as ng
        from xfem import POS  # type: ignore[import-untyped]
        from xfem.lsetcurv import LevelSetMeshAdaptation  # type: ignore[import-untyped]
    except ModuleNotFoundError as error:
        raise CutCellUnavailableError(
            "CutCellVolumeReference requires the optional remec[cutcell] extra"
        ) from error
    return ng, POS, LevelSetMeshAdaptation


class CutCellVolumeReference:
    """Sharp note `(M4b)` / §8.1 reference for ``V_chi(level) = int H(chi-level) dOmega``.

    ``ngsxfem`` maps its piecewise-linear cut geometry to the supplied high-order
    level set before integrating the positive domain ``{chi > level}``.  This is
    a verification reference for `(mollified_V)`, not its differentiable solver
    replacement: it deliberately has no JVP.
    """

    def __init__(
        self,
        mesh: ng.Mesh,
        level_set: ng.CoefficientFunction,
        *,
        geometry_order: int = 3,
        integration_order: int | None = None,
    ) -> None:
        if geometry_order < 1:
            raise ValueError("geometry_order must be at least one")
        if integration_order is not None and integration_order < 1:
            raise ValueError("integration_order must be at least one")
        ng, positive_domain, level_set_adaptation = _load_cutcell_dependencies()

        self._mesh = mesh
        self._level_set = level_set
        self.geometry_order = geometry_order
        self._integration_order = (
            2 * geometry_order + 2 if integration_order is None else integration_order
        )
        self._positive_domain = positive_domain
        self._level_set_adaptation = level_set_adaptation
        self.total_volume = float(ng.Integrate(1.0, mesh))

    def volume(self, level: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
        """Integrate ``{chi > level}`` using high-order implicit-domain quadrature."""
        points = np.asarray(level, dtype=float)
        if not np.all(np.isfinite(points)):
            raise ValueError("cut-cell reference levels must be finite")
        result = np.empty(points.size, dtype=float)
        for index, value in enumerate(points.reshape(-1)):
            adaptation = self._level_set_adaptation(self._mesh, order=self.geometry_order)
            adaptation.CalcDeformation(self._level_set - float(value))
            result[index] = float(
                adaptation.Integrate(
                    self._positive_domain,
                    1.0,
                    order=self._integration_order,
                )
            )
        volumes = result.reshape(points.shape)
        return float(volumes) if points.ndim == 0 else volumes
