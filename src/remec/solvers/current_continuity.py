"""Public interface for the regularized current-continuity equation (M3)."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from remec.common.logging import JsonEventLogger
from remec.common.serialization import configuration_digest
from remec.fem._current_continuity import solve_frozen_current_continuity
from remec.geometry.slab import Slab2D
from remec.options import RegularizationGradient, RuntimeOptions


@dataclass(frozen=True, slots=True)
class FrozenCurrentContinuityCoefficients:
    """Frozen coefficients required by the direct-u form of note equation (M3).

    ``magnetic_magnitude_gradient`` is normally ``grad(|B|)`` derived from
    ``magnetic_field``. It is explicit so verification can inject a prescribed
    strong-form M3 drive; production callers are responsible for their consistency.
    """

    magnetic_field: Any
    pressure_gradient: Any
    magnetic_magnitude_gradient: Any
    current_diffusivity: float
    magnetic_floor: float = 1.0e-12
    vacuum_permeability: float = 1.0

    def __post_init__(self) -> None:
        if not isfinite(self.current_diffusivity) or self.current_diffusivity <= 0.0:
            raise ValueError("current_diffusivity must be finite and positive")
        if not isfinite(self.magnetic_floor) or self.magnetic_floor <= 0.0:
            raise ValueError("magnetic_floor must be finite and positive")
        if not isfinite(self.vacuum_permeability) or self.vacuum_permeability <= 0.0:
            raise ValueError("vacuum_permeability must be finite and positive")


@dataclass(frozen=True, slots=True)
class CurrentContinuityResult:
    """Scalar public result for a frozen-field direct-u solve of (M3)."""

    polynomial_order: int
    regularization_gradient: RegularizationGradient
    configuration_digest: str
    free_dof_residual_norm: float
    free_dof_relative_residual_norm: float
    diagnostics: dict[str, float]


class CurrentContinuitySolver:
    r"""Direct-u solver for note equations (M2)--(M3).

    The selected ``grad_r`` is used consistently in
    ``J = uB + B x grad(p)/B_safe^2 - D_u grad_r(u)`` and in the M3 weak form
    documented by :func:`remec.fem._current_continuity.solve_frozen_current_continuity`.
    """

    def __init__(
        self,
        *,
        polynomial_order: int = 3,
        runtime: RuntimeOptions | None = None,
        logger: JsonEventLogger | None = None,
    ) -> None:
        if polynomial_order < 1:
            raise ValueError("polynomial_order must be at least one")
        self.polynomial_order = polynomial_order
        self.runtime = RuntimeOptions() if runtime is None else runtime
        self.logger = logger
        self._internal_solution: Any = None

    def solve(
        self,
        field: Slab2D,
        coefficients: FrozenCurrentContinuityCoefficients,
        *,
        boundary: str = "bottom|right|top|left",
        boundary_value: Any = 0.0,
    ) -> CurrentContinuityResult:
        """Solve the frozen-field direct-u weak form of note equation (M3)."""
        if not isinstance(field, Slab2D):
            raise TypeError("the direct-u M3 kernel requires a Slab2D mesh")
        if not isinstance(coefficients, FrozenCurrentContinuityCoefficients):
            raise TypeError("coefficients must be FrozenCurrentContinuityCoefficients")
        configuration = {
            "polynomial_order": self.polynomial_order,
            "runtime": self.runtime,
            "current_diffusivity": coefficients.current_diffusivity,
            "magnetic_floor": coefficients.magnetic_floor,
            "vacuum_permeability": coefficients.vacuum_permeability,
        }
        digest = configuration_digest(configuration)
        log_fields = {
            "configuration_digest": digest,
            "polynomial_order": self.polynomial_order,
            "regularization_gradient": self.runtime.regularization_gradient,
        }
        if self.logger is not None:
            self.logger.info("m3_solve_started", **log_fields)

        internal = solve_frozen_current_continuity(
            field,
            polynomial_order=self.polynomial_order,
            coefficients=coefficients,
            runtime=self.runtime,
            boundary=boundary,
            boundary_value=boundary_value,
        )
        self._internal_solution = internal
        diagnostics = {
            **internal.diagnostics,
            "free_dof_residual_norm": internal.free_dof_residual_norm,
            "free_dof_relative_residual_norm": internal.free_dof_relative_residual_norm,
        }
        if self.logger is not None:
            self.logger.info(
                "m3_solve_completed",
                **log_fields,
                free_dof_relative_residual_norm=internal.free_dof_relative_residual_norm,
            )
        return CurrentContinuityResult(
            polynomial_order=self.polynomial_order,
            regularization_gradient=self.runtime.regularization_gradient,
            configuration_digest=digest,
            free_dof_residual_norm=internal.free_dof_residual_norm,
            free_dof_relative_residual_norm=internal.free_dof_relative_residual_norm,
            diagnostics=diagnostics,
        )

    def _solution(self) -> Any:
        if self._internal_solution is None:
            raise RuntimeError("solve must be called before evaluating the M3 result")
        return self._internal_solution

    def solution_at(self, x_coordinate: float, y_coordinate: float) -> float:
        """Return u at a physical point after a solve."""
        return float(self._solution().solution_at(x_coordinate, y_coordinate))

    def solution_gradient_at(self, x_coordinate: float, y_coordinate: float) -> tuple[float, float]:
        """Return the in-plane gradient of u at a physical point after a solve."""
        value: tuple[float, float] = self._solution().solution_gradient_at(
            x_coordinate, y_coordinate
        )
        return value

    def current_at(self, x_coordinate: float, y_coordinate: float) -> tuple[float, float, float]:
        """Return the reconstructed note-(M2) current at a physical point."""
        value: tuple[float, float, float] = self._solution().current_at(x_coordinate, y_coordinate)
        return value

    def parallel_current_over_field_at(self, x_coordinate: float, y_coordinate: float) -> float:
        r"""Return J_parallel/B; full grad uses ``u-(D_u/B)b.grad(u)``."""
        return float(self._solution().parallel_current_over_field_at(x_coordinate, y_coordinate))

    def diagnostics(self) -> dict[str, float]:
        """Return scalar diagnostics from the latest direct-u M3 solve."""
        solution = self._solution()
        diagnostics: dict[str, float] = {
            **solution.diagnostics,
            "free_dof_residual_norm": solution.free_dof_residual_norm,
            "free_dof_relative_residual_norm": solution.free_dof_relative_residual_norm,
        }
        return diagnostics
