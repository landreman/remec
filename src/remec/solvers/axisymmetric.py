"""Axisymmetric profile closure for the reduced Grad-Shafranov model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import isfinite, pi

import numpy as np
from numpy.typing import ArrayLike, NDArray

from remec.fem._axisymmetric import (
    AxisymmetricGradShafranovCoefficients,
    solve_axisymmetric_grad_shafranov,
)
from remec.geometry.axisymmetric import AxisymmetricRZDomain
from remec.options import RuntimeOptions
from remec.profiles import PressureProfile, ToroidalCurrentProfile

ScalarOrArray = float | NDArray[np.float64]

__all__ = [
    "AxisymmetricGradShafranovCoefficients",
    "AxisymmetricGradShafranovResult",
    "AxisymmetricGradShafranovSolver",
    "AxisymmetricProfileClosure",
    "AxisymmetricProfileClosureEvaluation",
]


@dataclass(frozen=True, slots=True)
class AxisymmetricProfileClosureEvaluation:
    """Evaluated note-``(M4b)`` profiles and ``(M2)``-``(M3b)`` GS sources."""

    normalized_volume: ScalarOrArray
    pressure: ScalarOrArray
    target_enclosed_current: ScalarOrArray
    pressure_flux_derivative: ScalarOrArray
    toroidal_field_drive: ScalarOrArray


@dataclass(frozen=True, slots=True)
class AxisymmetricProfileClosure:
    r"""Normalized ``p_0(s)``/``I_0(s)`` closure from note ``(M2)``-``(M4b)``.

    With ``s=V(psi)/V_omega``, this evaluates

    ``p'(psi) = p_0'(s) ds/dpsi`` and
    ``I I' = mu0/<R^-2> [2*pi*I_0'(s)/V_omega - p'(psi)]``.

    The second identity is ``I_ODE``, the axisymmetric form of (M3b).  It makes the
    toroidal current reconstructed independently from ``GS_recovered`` satisfy
    ``I_tor(s)=I_0(s)`` while retaining the same normalized coordinate as M4b.
    """

    pressure_profile: PressureProfile
    toroidal_current_profile: ToroidalCurrentProfile
    total_volume: float
    mu0: float = 1.0

    def __post_init__(self) -> None:
        if not isfinite(self.total_volume) or self.total_volume <= 0.0:
            raise ValueError("total_volume must be finite and positive")
        if not isfinite(self.mu0) or self.mu0 <= 0.0:
            raise ValueError("mu0 must be finite and positive")
        self.pressure_profile.validate()
        self.toroidal_current_profile.validate()

    def evaluate(
        self,
        normalized_volume: float | ArrayLike,
        *,
        d_normalized_volume_d_flux: float | ArrayLike,
        mean_inverse_radius_squared: float | ArrayLike,
    ) -> AxisymmetricProfileClosureEvaluation:
        """Evaluate ``GS_recovered`` sources from normalized M4b/M3b profiles."""
        s, ds_dflux, mean_inverse_r2 = np.broadcast_arrays(
            np.asarray(normalized_volume, dtype=float),
            np.asarray(d_normalized_volume_d_flux, dtype=float),
            np.asarray(mean_inverse_radius_squared, dtype=float),
        )
        if not np.all(np.isfinite(ds_dflux)) or np.any(ds_dflux == 0.0):
            raise ValueError("d_normalized_volume_d_flux must be finite and nonzero")
        if not np.all(np.isfinite(mean_inverse_r2)) or np.any(mean_inverse_r2 <= 0.0):
            raise ValueError("mean_inverse_radius_squared must be finite and positive")
        pressure = np.asarray(self.pressure_profile.value(s), dtype=float)
        target_current = np.asarray(self.toroidal_current_profile.enclosed_current(s), dtype=float)
        pressure_s_derivative = np.asarray(self.pressure_profile.derivative(s), dtype=float)
        current_s_derivative = np.asarray(self.toroidal_current_profile.derivative(s), dtype=float)
        pressure_flux_derivative = pressure_s_derivative * ds_dflux
        toroidal_field_drive = (
            self.mu0
            / mean_inverse_r2
            * (2.0 * pi * current_s_derivative / self.total_volume - pressure_flux_derivative)
        )
        scalar = s.ndim == 0
        return AxisymmetricProfileClosureEvaluation(
            _scalar_or_array(s, scalar),
            _scalar_or_array(pressure, scalar),
            _scalar_or_array(target_current, scalar),
            _scalar_or_array(pressure_flux_derivative, scalar),
            _scalar_or_array(toroidal_field_drive, scalar),
        )


def _scalar_or_array(values: NDArray[np.float64], scalar: bool) -> ScalarOrArray:
    """Preserve scalar profile calls while retaining vectorized array evaluation."""
    return float(values) if scalar else values


@dataclass(frozen=True, slots=True)
class AxisymmetricGradShafranovResult:
    """Backend-independent summary of a reduced note-``(M1)`` GS solve."""

    polynomial_order: int
    elements: int
    free_dof_residual_norm: float
    free_dof_relative_residual_norm: float
    weighted_magnetic_energy: float
    _flux_evaluator: Callable[[float, float], float] = field(repr=False, compare=False)

    def flux_at(self, radius: float, vertical_coordinate: float) -> float:
        """Evaluate this immutable solve's reduced note-``(M1)`` flux."""
        return self._flux_evaluator(radius, vertical_coordinate)


class AxisymmetricGradShafranovSolver:
    """Sparse-direct verification solver for note ``(M1)`` in true R-Z form."""

    def __init__(
        self,
        *,
        polynomial_order: int = 2,
        runtime: RuntimeOptions | None = None,
    ) -> None:
        if polynomial_order < 1:
            raise ValueError("polynomial_order must be at least one")
        self.polynomial_order = polynomial_order
        self.runtime = runtime

    def solve(
        self,
        domain: AxisymmetricRZDomain,
        coefficients: AxisymmetricGradShafranovCoefficients,
    ) -> AxisymmetricGradShafranovResult:
        """Solve ``GS_recovered`` with frozen profile-derived source coefficients."""
        internal = solve_axisymmetric_grad_shafranov(
            domain,
            polynomial_order=self.polynomial_order,
            coefficients=coefficients,
            runtime=self.runtime,
        )

        def flux_at(radius: float, vertical_coordinate: float) -> float:
            return float(internal._flux(internal._mesh(radius, vertical_coordinate)))

        return AxisymmetricGradShafranovResult(
            polynomial_order=internal.polynomial_order,
            elements=internal.elements,
            free_dof_residual_norm=internal.free_dof_residual_norm,
            free_dof_relative_residual_norm=internal.free_dof_relative_residual_norm,
            weighted_magnetic_energy=internal.weighted_magnetic_energy,
            _flux_evaluator=flux_at,
        )
