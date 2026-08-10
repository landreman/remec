"""Public StandardCG strategy for the anisotropic reference-potential equation."""

from __future__ import annotations

import warnings
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


class AnisotropyPollutionError(RuntimeError):
    """Raised when numerical cross-field transport exceeds the strict safety gate."""


class AnisotropyPollutionWarning(RuntimeWarning):
    """Numerical cross-field transport is too large for the requested physics."""


@dataclass(frozen=True, slots=True)
class PollutionSafetyDiagnostic:
    """§8.3 comparison of measured ``κ_perp,num`` to physical ``κ_perp``."""

    numerical_perpendicular_diffusivity: float
    physical_perpendicular_diffusivity: float
    safety_factor: float
    ratio_to_physical: float
    is_safe: bool


@dataclass(frozen=True, slots=True)
class FloorSensitivityDiagnostic:
    """§6 observable sensitivity to the smooth ``B_safe`` regularization."""

    relative_change: float
    tolerance: float
    is_acceptable: bool


@dataclass(frozen=True, slots=True)
class SpatialAnisotropicConductivity:
    """Smooth M4a tensor ``K=κ⊥I+(κ∥-κ⊥)b_safe⊗b_safe``.

    Here ``b_safe=B/sqrt(B·B+B_floor²)``.  This retains the note's (M4a)
    tensor form at a field null, rather than silently replacing it with an
    isotropic approximation.
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
        """Return the symmetric 2-D M4a tensor for a plain frozen field vector."""
        norm_squared = field[0] ** 2 + field[1] ** 2 + self.field_floor**2
        bx, by = field[0] / norm_squared**0.5, field[1] / norm_squared**0.5
        contrast = self.parallel - self.perpendicular
        return (
            (self.perpendicular + contrast * bx * bx, contrast * bx * by),
            (contrast * bx * by, self.perpendicular + contrast * by * by),
        )

    def floor_activity(self, field: tuple[float, float]) -> float:
        """Return ``1-|b_safe|²=B_floor²/(B·B+B_floor²)``."""
        return self.field_floor**2 / (field[0] ** 2 + field[1] ** 2 + self.field_floor**2)


@dataclass(frozen=True, slots=True)
class AnisotropicDiffusionResult:
    """M4a solution, separate energies, and the retained assembled operator."""

    mesh: Any
    field: Any
    polynomial_order: int
    free_dof_residual_norm: float
    free_dof_relative_residual_norm: float
    energy_diagnostics: _EnergyDiagnostics
    diagnostics: dict[str, float]
    operator: Any

    def apply(self, x: Any) -> Any:
        """Apply the assembled M4a operator to ``x`` for Newton/preconditioners."""
        return self.operator * x


class AnisotropicDiffusionSolver:
    """``StandardCG`` strategy for the note equation (M4a).

    ``solve`` selects the established constant-direction or smoothly floored
    spatial-field M4a assembly behind the FEM boundary.  Both paths retain their
    historical quadrature rules and formulas, so the Phase-1 regression tables
    remain bit-for-bit stable while later strategies share this public API.
    """

    def __init__(
        self,
        *,
        polynomial_order: int = 3,
        runtime: RuntimeOptions | None = None,
        pollution_safety_factor: float = 0.1,
        floor_sensitivity_tolerance: float = 0.01,
    ) -> None:
        if polynomial_order < 1:
            raise ValueError("polynomial_order must be at least one")
        if not isfinite(pollution_safety_factor) or not 0.0 < pollution_safety_factor <= 1.0:
            raise ValueError("pollution_safety_factor must be finite and in (0, 1]")
        if not isfinite(floor_sensitivity_tolerance) or floor_sensitivity_tolerance < 0.0:
            raise ValueError("floor_sensitivity_tolerance must be finite and non-negative")
        self.polynomial_order = polynomial_order
        self.runtime = runtime
        self.pollution_safety_factor = pollution_safety_factor
        self.floor_sensitivity_tolerance = floor_sensitivity_tolerance
        self._last_result: AnisotropicDiffusionResult | None = None

    def solve(
        self,
        field: Slab2D | tuple[Slab2D, Any],
        coefficients: DirectionalConductivity | SpatialAnisotropicConductivity,
        source: Any,
        boundary: str = "bottom|right|top|left",
        initial: Any | None = None,
    ) -> AnisotropicDiffusionResult:
        """Solve ``∫∇v·K∇χ=∫vS_ref`` with the M4a conductivity tensor.

        ``field`` is a ``Slab2D`` for a constant unit direction, or
        ``(Slab2D, raw_field)`` for ``SpatialAnisotropicConductivity``.  The
        current verification mesh has homogeneous named Dirichlet boundaries;
        ``initial`` is reserved for iterative future strategies.
        """
        if boundary != "bottom|right|top|left":
            raise ValueError("only the unit-square named Dirichlet boundary is supported")
        if initial is not None:
            raise ValueError("StandardCG does not accept an initial iterate")
        internal: Any
        if isinstance(coefficients, DirectionalConductivity):
            if not isinstance(field, Slab2D):
                raise TypeError("constant conductivity requires a Slab2D")
            internal = solve_anisotropic_diffusion(
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
            internal = solve_frozen_field_anisotropic_diffusion(
                field[0],
                polynomial_order=self.polynomial_order,
                source=source,
                raw_field=field[1],
                parallel_conductivity=coefficients.parallel,
                perpendicular_conductivity=coefficients.perpendicular,
                field_floor=coefficients.field_floor,
                runtime=self.runtime,
            )
            floor_activity = internal.field_direction_diagnostics.floor_activity_l2_squared
        else:
            raise TypeError("unsupported anisotropic conductivity")
        result = AnisotropicDiffusionResult(
            mesh=internal._mesh,
            field=internal._field,
            polynomial_order=internal.polynomial_order,
            free_dof_residual_norm=internal.free_dof_residual_norm,
            free_dof_relative_residual_norm=internal.free_dof_relative_residual_norm,
            energy_diagnostics=internal.energy_diagnostics,
            operator=internal.operator,
            diagnostics={
                "free_dof_residual_norm": internal.free_dof_residual_norm,
                "free_dof_relative_residual_norm": internal.free_dof_relative_residual_norm,
                "parallel_energy": internal.energy_diagnostics.parallel,
                "perpendicular_energy": internal.energy_diagnostics.perpendicular,
                "total_energy": internal.energy_diagnostics.total,
                "floor_activity_l2_squared": floor_activity,
            },
        )
        self._last_result = result
        return result

    def apply(self, x: Any) -> Any:
        """Apply the last M4a operator, as required by the strategy interface."""
        if self._last_result is None:
            raise RuntimeError("solve must be called before apply")
        return self._last_result.apply(x)

    def build_preconditioner(self) -> Any:
        """Return the last assembled SPD operator as the StandardCG hook."""
        if self._last_result is None:
            raise RuntimeError("solve must be called before build_preconditioner")
        return self._last_result.operator

    def diagnostics(self) -> dict[str, float]:
        """Return scalar diagnostics from the most recent M4a solve."""
        if self._last_result is None:
            raise RuntimeError("solve must be called before diagnostics")
        return dict(self._last_result.diagnostics)

    def assess_pollution(
        self,
        *,
        numerical_perpendicular_diffusivity: float,
        physical_perpendicular_diffusivity: float,
        strict: bool = False,
    ) -> PollutionSafetyDiagnostic:
        """Apply §8.3 ``κ_perp,num < safety_factor κ_perp`` safety gate."""
        if not all(
            isfinite(value) and value > 0.0
            for value in (
                numerical_perpendicular_diffusivity,
                physical_perpendicular_diffusivity,
            )
        ):
            raise ValueError("perpendicular diffusivities must be finite and positive")
        ratio = numerical_perpendicular_diffusivity / physical_perpendicular_diffusivity
        diagnostic = PollutionSafetyDiagnostic(
            numerical_perpendicular_diffusivity,
            physical_perpendicular_diffusivity,
            self.pollution_safety_factor,
            ratio,
            ratio < self.pollution_safety_factor,
        )
        if not diagnostic.is_safe:
            message = (
                "numerical perpendicular diffusion is too large: "
                f"κ_perp,num/κ_perp={ratio:.3e}, required below "
                f"{self.pollution_safety_factor:.3e}"
            )
            if strict:
                raise AnisotropyPollutionError(message)
            warnings.warn(message, AnisotropyPollutionWarning, stacklevel=2)
        return diagnostic

    def assess_floor_sensitivity(
        self,
        *,
        observable_with_floor: float,
        observable_with_smaller_floor: float,
        strict: bool = False,
    ) -> FloorSensitivityDiagnostic:
        """Compare observables at two smooth floors as required by DESIGN §6."""
        if not all(
            isfinite(value) for value in (observable_with_floor, observable_with_smaller_floor)
        ):
            raise ValueError("floor-sensitivity observables must be finite")
        relative_change = abs(observable_with_floor - observable_with_smaller_floor) / max(
            1.0, abs(observable_with_smaller_floor)
        )
        diagnostic = FloorSensitivityDiagnostic(
            relative_change,
            self.floor_sensitivity_tolerance,
            relative_change <= self.floor_sensitivity_tolerance,
        )
        if not diagnostic.is_acceptable:
            message = (
                "B floor materially affects the observable: relative change "
                f"{relative_change:.3e} exceeds {self.floor_sensitivity_tolerance:.3e}"
            )
            if strict:
                raise RuntimeError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        return diagnostic
