"""Public interface for the regularized current-continuity equation (M3)."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

from remec.common.logging import JsonEventLogger
from remec.common.serialization import configuration_digest
from remec.fem._current_continuity import solve_frozen_current_continuity
from remec.geometry.slab import Slab2D
from remec.options import (
    CurrentContinuityStabilization,
    RegularizationGradient,
    RuntimeOptions,
)

CurrentContinuityFormulation = Literal["direct-u", "utilde"]


@dataclass(frozen=True, slots=True)
class FrozenCurrentContinuityCoefficients:
    """Frozen coefficients required by the direct-u form of note equation (M3).

    ``magnetic_magnitude_gradient`` is normally ``grad(|B|)`` derived from
    ``magnetic_field``. It is explicit so verification can inject a prescribed
    strong-form M3 drive; production callers are responsible for their consistency.
    ``magnetic_field_gradient``, when supplied, is either NGSolve's native 2-by-3
    vector gradient or the transposed 3-by-2 matrix ``(∂_x B_i, ∂_y B_i)`` required by
    the perpendicular SUPG strong divergence. It is mandatory for a varying
    GridFunction-backed magnetic field because NGSolve coordinate differentiation of
    such a field silently returns zero.
    """

    magnetic_field: Any
    pressure_gradient: Any
    magnetic_magnitude_gradient: Any
    current_diffusivity: float
    magnetic_field_gradient: Any | None = None
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
class PrescribedCurrentProfile:
    r"""Frozen ``F(p)`` data for the note's ``u = F(p) + utilde`` split.

    ``value`` is ``F(p)`` and ``pressure_derivative`` is ``F'(p)``. Together with
    ``FrozenCurrentContinuityCoefficients.pressure_gradient`` they define
    ``grad(F(p)) = F'(p) grad(p)``. The two explicit divergence coefficients are
    ``div(F'(p) grad_perp(p))`` and ``div(F'(p) grad(p))``, where
    ``grad_perp = (I - b_safe b_safe^T) grad`` is the note-literal single projection;
    the solver selects the runtime variant for the SUPG strong source. The Galerkin
    load instead moves its symmetric ``grad_r(v).grad_r(F)`` term directly, so it
    remains algebraically identical to direct-u when the B floor is active. The
    divergence coefficients are explicit because NGSolve coordinate differentiation
    can silently return zero for a GridFunction-backed pressure gradient.
    """

    value: Any
    pressure_derivative: Any
    perpendicular_gradient_divergence: Any
    full_gradient_divergence: Any

    def __post_init__(self) -> None:
        for name in (
            "value",
            "pressure_derivative",
            "perpendicular_gradient_divergence",
            "full_gradient_divergence",
        ):
            if getattr(getattr(self, name), "dim", 1) != 1:
                raise ValueError(f"{name} must be scalar")


@dataclass(frozen=True, slots=True)
class CurrentContinuityResult:
    """Scalar public result for a frozen-field direct-u or utilde solve of (M3)."""

    formulation: CurrentContinuityFormulation
    polynomial_order: int
    regularization_gradient: RegularizationGradient
    stabilization: CurrentContinuityStabilization
    configuration_digest: str
    free_dof_residual_norm: float
    free_dof_relative_residual_norm: float
    diagnostics: dict[str, float]


class CurrentContinuitySolver:
    r"""Direct-u and preferred utilde solver for note equations (M2)--(M3).

    The selected ``grad_r`` is used consistently in
    ``J = uB + B x grad(p)/B_safe^2 - D_u grad_r(u)`` and in the M3 weak form
    documented by :func:`remec.fem._current_continuity.solve_frozen_current_continuity`.
    The DESIGN §9.1 baseline is ``stabilization="supg"``; ``"none"`` retains the
    unstabilized Galerkin form for cross-verification.
    """

    def __init__(
        self,
        *,
        polynomial_order: int = 3,
        runtime: RuntimeOptions | None = None,
        stabilization: CurrentContinuityStabilization = "supg",
        logger: JsonEventLogger | None = None,
    ) -> None:
        if polynomial_order < 1:
            raise ValueError("polynomial_order must be at least one")
        if stabilization not in ("none", "supg"):
            raise ValueError("stabilization must be 'none' or 'supg'")
        self.polynomial_order = polynomial_order
        self.runtime = RuntimeOptions() if runtime is None else runtime
        self.stabilization = stabilization
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
        return self._solve(
            field,
            coefficients,
            formulation="direct-u",
            profile=None,
            boundary=boundary,
            boundary_value=boundary_value,
        )

    def _solve(
        self,
        field: Slab2D,
        coefficients: FrozenCurrentContinuityCoefficients,
        *,
        formulation: CurrentContinuityFormulation,
        profile: PrescribedCurrentProfile | None,
        boundary: str,
        boundary_value: Any,
    ) -> CurrentContinuityResult:
        """Run one validated direct-u or transformed frozen-field solve."""
        if not isinstance(field, Slab2D):
            raise TypeError("the M3 kernel requires a Slab2D mesh")
        if not isinstance(coefficients, FrozenCurrentContinuityCoefficients):
            raise TypeError("coefficients must be FrozenCurrentContinuityCoefficients")
        if formulation == "utilde" and not isinstance(profile, PrescribedCurrentProfile):
            raise TypeError("profile must be PrescribedCurrentProfile")
        configuration = {
            "formulation": formulation,
            "polynomial_order": self.polynomial_order,
            "runtime": self.runtime,
            "current_diffusivity": coefficients.current_diffusivity,
            "magnetic_floor": coefficients.magnetic_floor,
            "vacuum_permeability": coefficients.vacuum_permeability,
            "stabilization": self.stabilization,
        }
        digest = configuration_digest(configuration)
        log_fields = {
            "configuration_digest": digest,
            "formulation": formulation,
            "polynomial_order": self.polynomial_order,
            "regularization_gradient": self.runtime.regularization_gradient,
            "stabilization": self.stabilization,
        }
        if self.logger is not None:
            self.logger.info("m3_solve_started", **log_fields)

        internal = solve_frozen_current_continuity(
            field,
            polynomial_order=self.polynomial_order,
            coefficients=coefficients,
            runtime=self.runtime,
            prescribed_current_profile=profile,
            boundary=boundary,
            boundary_value=boundary_value,
            stabilization=self.stabilization,
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
                m3_supg_stabilization_norm=internal.diagnostics["m3_supg_stabilization_norm"],
            )
        return CurrentContinuityResult(
            formulation=formulation,
            polynomial_order=self.polynomial_order,
            regularization_gradient=self.runtime.regularization_gradient,
            stabilization=self.stabilization,
            configuration_digest=digest,
            free_dof_residual_norm=internal.free_dof_residual_norm,
            free_dof_relative_residual_norm=internal.free_dof_relative_residual_norm,
            diagnostics=diagnostics,
        )

    def solve_utilde(
        self,
        field: Slab2D,
        coefficients: FrozenCurrentContinuityCoefficients,
        profile: PrescribedCurrentProfile,
        *,
        boundary: str = "bottom|right|top|left",
    ) -> CurrentContinuityResult:
        r"""Solve note Eq. ``utilde_equation`` with homogeneous ``utilde`` data.

        This is the preferred (M3) formulation. It solves for ``utilde`` on the same
        Galerkin/SUPG operator as direct ``u``, then reconstructs
        ``u = F(p) + utilde`` for note-(M2) current and physical diagnostics.
        """
        return self._solve(
            field,
            coefficients,
            formulation="utilde",
            profile=profile,
            boundary=boundary,
            boundary_value=0.0,
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

    def utilde_at(self, x_coordinate: float, y_coordinate: float) -> float:
        """Return the homogeneous utilde unknown after :meth:`solve_utilde`."""
        return float(self._solution().utilde_at(x_coordinate, y_coordinate))

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
