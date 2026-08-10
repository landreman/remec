"""Public strategy interface for the frozen-field M4a diffusion solve."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from remec.fem._anisotropic_diffusion import (
    DirectionalConductivity,
    _EnergyDiagnostics,
    solve_anisotropic_diffusion,
    solve_frozen_field_anisotropic_diffusion,
)
from remec.geometry.slab import Slab2D
from remec.options import RuntimeOptions


@dataclass(frozen=True, slots=True)
class SpatialAnisotropicConductivity:
    """Smoothly floored coefficients for the spatial M4a tensor.

    The tensor is ``K = κ_perp I + (κ_parallel - κ_perp)b_safe⊗b_safe`` with
    ``b_safe = B/sqrt(B·B + B_floor²)``.  This is the smooth small-field
    protection prescribed by DESIGN §6; it preserves symmetry and positive
    eigenvalues while reporting floor activity.
    """

    parallel: float
    perpendicular: float
    field_floor: float

    def __post_init__(self) -> None:
        if not all(
            isfinite(value) and value > 0.0
            for value in (self.parallel, self.perpendicular, self.field_floor)
        ):
            raise ValueError("conductivities and field_floor must be finite and positive")
        if self.parallel < self.perpendicular:
            raise ValueError("parallel conductivity must not be below perpendicular conductivity")

    def tensor(self, field: tuple[float, float]) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the symmetric 2-D M4a tensor at a plain field vector."""
        denominator = field[0] ** 2 + field[1] ** 2 + self.field_floor**2
        bx = field[0] / denominator**0.5
        by = field[1] / denominator**0.5
        contrast = self.parallel - self.perpendicular
        return (
            (self.perpendicular + contrast * bx * bx, contrast * bx * by),
            (contrast * bx * by, self.perpendicular + contrast * by * by),
        )

    def floor_activity(self, field: tuple[float, float]) -> float:
        """Return ``1-|b_safe|² = B_floor²/(|B|²+B_floor²)``."""
        field_squared = field[0] ** 2 + field[1] ** 2
        return self.field_floor**2 / (field_squared + self.field_floor**2)


@dataclass(frozen=True, slots=True)
class AnisotropicDiffusionResult:
    """Public frozen-field M4a result and diagnostics."""

    mesh: Any
    field: Any
    polynomial_order: int
    free_dof_residual_norm: float
    free_dof_relative_residual_norm: float
    energy_diagnostics: _EnergyDiagnostics
    diagnostics: dict[str, float]


class AnisotropicDiffusionSolver:
    """StandardCG strategy for note equation (M4a).

    ``solve`` keeps NGSolve behind the FEM boundary and supports both the
    constant-direction verification kernel and the smoothly floored spatial
    field kernel.  The assembled operator is retained for ``apply`` and the
    preconditioner hook used by later Newton strategies.
    """

    def __init__(
        self,
        *,
        polynomial_order: int = 3,
        runtime: RuntimeOptions | None = None,
    ) -> None:
        if polynomial_order < 1:
            raise ValueError("polynomial_order must be at least one")
        self.polynomial_order = polynomial_order
        self.runtime = runtime
        self._operator: Any = None

    def solve(
        self,
        field: Slab2D | Any,
        coefficients: DirectionalConductivity | SpatialAnisotropicConductivity,
        source: Any,
        boundary: str = "bottom|right|top|left",
        initial: Any | None = None,
    ) -> AnisotropicDiffusionResult:
        """Solve ``∫ grad(v)·K grad(χ) = ∫ v S_ref`` for homogeneous Dirichlet data.

        ``field`` is a :class:`Slab2D` for constant directions.  For a spatial
        field, pass ``(slab, raw_field)``; ``boundary`` is currently the named
        unit-square boundary set used by the verification meshes.
        """
        del boundary, initial  # Reserved for the strategy protocol.
        if isinstance(coefficients, DirectionalConductivity):
            if not isinstance(field, Slab2D):
                raise TypeError("constant conductivity requires a Slab2D field")
            result: Any = solve_anisotropic_diffusion(
                field,
                polynomial_order=self.polynomial_order,
                source=source,
                conductivity=coefficients,
                runtime=self.runtime,
            )
            floor_activity = 0.0
        elif isinstance(coefficients, SpatialAnisotropicConductivity):
            if not isinstance(field, tuple) or len(field) != 2 or not isinstance(field[0], Slab2D):
                raise TypeError("spatial conductivity requires (Slab2D, raw_field)")
            result = solve_frozen_field_anisotropic_diffusion(
                field[0],
                polynomial_order=self.polynomial_order,
                source=source,
                raw_field=field[1],
                parallel_conductivity=coefficients.parallel,
                perpendicular_conductivity=coefficients.perpendicular,
                field_floor=coefficients.field_floor,
                runtime=self.runtime,
            )
            floor_activity = result.field_direction_diagnostics.floor_activity_l2_squared
        else:
            raise TypeError("unsupported anisotropic conductivity strategy")
        self._operator = result.operator

        diagnostics = {
            "free_dof_residual_norm": result.free_dof_residual_norm,
            "free_dof_relative_residual_norm": result.free_dof_relative_residual_norm,
            "parallel_energy": result.energy_diagnostics.parallel,
            "perpendicular_energy": result.energy_diagnostics.perpendicular,
            "total_energy": result.energy_diagnostics.total,
            "floor_activity_l2_squared": floor_activity,
        }
        return AnisotropicDiffusionResult(
            mesh=result._mesh,
            field=result._field,
            polynomial_order=result.polynomial_order,
            free_dof_residual_norm=result.free_dof_residual_norm,
            free_dof_relative_residual_norm=result.free_dof_relative_residual_norm,
            energy_diagnostics=result.energy_diagnostics,
            diagnostics=diagnostics,
        )

    def apply(self, x: Any) -> Any:
        """Apply the last assembled M4a operator to a vector."""
        if self._operator is None:
            raise RuntimeError("solve must be called before apply")
        return self._operator * x

    def build_preconditioner(self) -> Any:
        """Return the last operator's sparse factorization as a preconditioner hook."""
        if self._operator is None:
            raise RuntimeError("solve must be called before build_preconditioner")
        return self._operator
