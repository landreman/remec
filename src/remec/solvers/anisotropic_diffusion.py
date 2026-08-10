"""Public StandardCG strategy for the anisotropic reference-potential equation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import isfinite
from typing import Any

from remec.fem._anisotropic_diffusion import (
    DirectionalConductivity,
    PollutionDiagnostic,
    measure_sovinec_pollution,
    preconditioner_identity_defect,
    solve_anisotropic_diffusion,
    solve_frozen_field_anisotropic_diffusion,
)
from remec.geometry.slab import Slab2D
from remec.options import RuntimeOptions


class AnisotropyPollutionError(RuntimeError):
    """Raised when §8.3 numerical cross-field transport is unsafe."""


class AnisotropyPollutionWarning(RuntimeWarning):
    """Numerical cross-field transport is too large for requested physics."""


class FloorSensitivityError(RuntimeError):
    """Raised when the smooth §6 B-floor materially changes an observable."""


class FloorSensitivityWarning(RuntimeWarning):
    """The smooth §6 B-floor materially changes an observable."""


@dataclass(frozen=True, slots=True)
class EnergyDiagnostics:
    """Separate non-negative M4a energies and their total."""

    parallel: float
    perpendicular: float
    total: float


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
    """§6 relative observable sensitivity to the smooth ``B_safe`` floor."""

    relative_change: float
    tolerance: float
    is_acceptable: bool


@dataclass(frozen=True, slots=True)
class SpatialAnisotropicConductivity:
    """Frozen M4a coefficients ``K=κ⊥I+(κ∥-κ⊥)b_safe⊗b_safe``.

    ``raw_field`` is evaluated only inside ``remec.fem``. The public solve
    remains uniformly elliptic; rank-one Sovinec diagnostics use their dedicated
    :meth:`AnisotropicDiffusionSolver.measure_sovinec_pollution` entry point.
    """

    parallel: float
    perpendicular: float
    field_floor: float
    raw_field: Any

    def __post_init__(self) -> None:
        if not isfinite(self.parallel) or self.parallel <= 0.0:
            raise ValueError("parallel conductivity must be finite and positive")
        if not isfinite(self.perpendicular) or self.perpendicular <= 0.0:
            raise ValueError("perpendicular conductivity must be finite and positive")
        if self.parallel < self.perpendicular:
            raise ValueError("parallel conductivity must not be below perpendicular conductivity")
        if not isfinite(self.field_floor) or self.field_floor <= 0.0:
            raise ValueError("field_floor must be finite and positive")


@dataclass(frozen=True, slots=True)
class AnisotropicDiffusionResult:
    """Public scalar M4a result; NGSolve objects remain in ``remec.fem``."""

    polynomial_order: int
    free_dof_residual_norm: float
    free_dof_relative_residual_norm: float
    energy_diagnostics: EnergyDiagnostics
    diagnostics: dict[str, float]


class AnisotropicDiffusionSolver:
    """``StandardCG`` strategy for note equation (M4a).

    The public interface routes constant directions, smoothly floored frozen
    fields, and the rank-one Sovinec diagnostic while retaining NGSolve handles
    privately.  ``apply`` and ``build_preconditioner`` expose distinct operator
    actions required by the §8.4 strategy protocol.
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
        self._operator: Any = None
        self._preconditioner: Any = None
        self._internal_solution: Any = None
        self._last_diagnostics: dict[str, float] | None = None

    def solve(
        self,
        field: Slab2D,
        coefficients: DirectionalConductivity | SpatialAnisotropicConductivity,
        source: Any,
        boundary: str = "bottom|right|top|left",
        initial: Any | None = None,
    ) -> AnisotropicDiffusionResult:
        """Solve M4a; direct StandardCG ignores an optional initial iterate."""
        if boundary != "bottom|right|top|left":
            raise ValueError("only the unit-square named Dirichlet boundary is supported")
        if not isinstance(field, Slab2D):
            raise TypeError("StandardCG requires a Slab2D mesh")
        del initial
        internal: Any
        if isinstance(coefficients, DirectionalConductivity):
            internal = solve_anisotropic_diffusion(
                field,
                polynomial_order=self.polynomial_order,
                source=source,
                conductivity=coefficients,
                runtime=self.runtime,
            )
        elif isinstance(coefficients, SpatialAnisotropicConductivity):
            internal = solve_frozen_field_anisotropic_diffusion(
                field,
                polynomial_order=self.polynomial_order,
                source=source,
                raw_field=coefficients.raw_field,
                parallel_conductivity=coefficients.parallel,
                perpendicular_conductivity=coefficients.perpendicular,
                field_floor=coefficients.field_floor,
                runtime=self.runtime,
            )
        else:
            raise TypeError("unsupported anisotropic conductivity")
        energy = EnergyDiagnostics(
            internal.energy_diagnostics.parallel,
            internal.energy_diagnostics.perpendicular,
            internal.energy_diagnostics.total,
        )
        diagnostics = {
            "free_dof_residual_norm": internal.free_dof_residual_norm,
            "free_dof_relative_residual_norm": internal.free_dof_relative_residual_norm,
            "parallel_energy": energy.parallel,
            "perpendicular_energy": energy.perpendicular,
            "total_energy": energy.total,
            "floor_activity_l2_squared": internal.field_direction_diagnostics.floor_activity_l2_squared,
            "central_amplitude": float(internal._field(internal._mesh(0.5, 0.5))),
        }
        self._operator = internal.operator
        self._preconditioner = internal.preconditioner
        self._internal_solution = internal
        self._last_diagnostics = diagnostics
        return AnisotropicDiffusionResult(
            self.polynomial_order,
            internal.free_dof_residual_norm,
            internal.free_dof_relative_residual_norm,
            energy,
            dict(diagnostics),
        )

    def measure_sovinec_pollution(
        self, field: Slab2D, *, strict: bool = False
    ) -> PollutionDiagnostic:
        """Route the rank-one M4a Sovinec solve and apply its §8.3 gate."""
        diagnostic = measure_sovinec_pollution(
            field, polynomial_order=self.polynomial_order, runtime=self.runtime
        )
        self.assess_pollution_diagnostic(diagnostic, strict=strict)
        return diagnostic

    def apply(self, x: Any) -> Any:
        """Apply the most recently assembled M4a operator to ``x``."""
        if self._operator is None:
            raise RuntimeError("solve must be called before apply")
        return self._operator * x

    def build_preconditioner(self) -> Any:
        """Return the reused sparse-Cholesky inverse action for StandardCG."""
        if self._preconditioner is None:
            raise RuntimeError("solve must be called before build_preconditioner")
        return self._preconditioner

    def diagnostics(self) -> dict[str, float]:
        """Return scalar diagnostics from the most recent M4a solve."""
        if self._last_diagnostics is None:
            raise RuntimeError("solve must be called before diagnostics")
        return dict(self._last_diagnostics)

    def preconditioner_identity_defect(self) -> float:
        """Verify the public ``P(Ax)≈x`` preconditioner contract for M4a."""
        if self._internal_solution is None:
            raise RuntimeError("solve must be called before preconditioner verification")
        return preconditioner_identity_defect(
            self._internal_solution, self.apply, self.build_preconditioner()
        )

    def assess_pollution_diagnostic(
        self, diagnostic: PollutionDiagnostic, *, strict: bool = False
    ) -> PollutionSafetyDiagnostic:
        """Apply §8.3 directly to a measured frozen-field pollution diagnostic."""
        return self.assess_pollution(
            numerical_perpendicular_diffusivity=diagnostic.numerical_perpendicular_diffusivity,
            physical_perpendicular_diffusivity=diagnostic.physical_perpendicular_conductivity,
            strict=strict,
        )

    def assess_pollution(
        self,
        *,
        numerical_perpendicular_diffusivity: float,
        physical_perpendicular_diffusivity: float,
        strict: bool = False,
    ) -> PollutionSafetyDiagnostic:
        """Apply §8.3 ``κ_perp,num < safety_factor κ_perp`` safety gate."""
        if (
            not isfinite(numerical_perpendicular_diffusivity)
            or numerical_perpendicular_diffusivity < 0.0
        ):
            raise ValueError("numerical perpendicular diffusivity must be finite and non-negative")
        if (
            not isfinite(physical_perpendicular_diffusivity)
            or physical_perpendicular_diffusivity < 0.0
        ):
            raise ValueError("physical perpendicular diffusivity must be finite and non-negative")
        ratio = (
            0.0
            if numerical_perpendicular_diffusivity == 0.0
            and physical_perpendicular_diffusivity == 0.0
            else float("inf")
            if physical_perpendicular_diffusivity == 0.0
            else numerical_perpendicular_diffusivity / physical_perpendicular_diffusivity
        )
        result = PollutionSafetyDiagnostic(
            numerical_perpendicular_diffusivity,
            physical_perpendicular_diffusivity,
            self.pollution_safety_factor,
            ratio,
            ratio < self.pollution_safety_factor,
        )
        if not result.is_safe:
            message = (
                "numerical perpendicular diffusion is too large: "
                f"κ_perp,num/κ_perp={ratio:.3e}, required below "
                f"{self.pollution_safety_factor:.3e}"
            )
            if strict:
                raise AnisotropyPollutionError(message)
            warnings.warn(message, AnisotropyPollutionWarning, stacklevel=2)
        return result

    def assess_floor_sensitivity(
        self,
        *,
        observable_with_floor: float,
        observable_with_smaller_floor: float,
        strict: bool = False,
    ) -> FloorSensitivityDiagnostic:
        """Compare §6 observables at two smooth field floors on their own scale."""
        if not all(
            isfinite(value) for value in (observable_with_floor, observable_with_smaller_floor)
        ):
            raise ValueError("floor-sensitivity observables must be finite")
        scale = max(abs(observable_with_floor), abs(observable_with_smaller_floor), 1.0e-300)
        relative_change = abs(observable_with_floor - observable_with_smaller_floor) / scale
        result = FloorSensitivityDiagnostic(
            relative_change,
            self.floor_sensitivity_tolerance,
            relative_change <= self.floor_sensitivity_tolerance,
        )
        if not result.is_acceptable:
            message = (
                "B floor materially affects the observable: relative change "
                f"{relative_change:.3e} exceeds {self.floor_sensitivity_tolerance:.3e}"
            )
            if strict:
                raise FloorSensitivityError(message)
            warnings.warn(message, FloorSensitivityWarning, stacklevel=2)
        return result
