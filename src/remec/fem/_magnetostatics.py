"""Gauge-fixed compatible finite-element kernel for note equation (M1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GaugeFixedCurlCurlSolution:
    """Internal fields and diagnostics from the mixed Coulomb-gauge (M1) solve."""

    vector_potential: Any
    gauge_multiplier: Any
    magnetic_field: Any
    vector_potential_order: int
    gauge_order: int
    magnetic_field_order: int
    free_dof_relative_residual: float
    gauge_constraint_relative_residual: float
    magnetic_divergence_relative_norm: float
    gauge_multiplier_l2_norm: float
    magnetic_energy: float


def solve_gauge_fixed_curl_curl(
    mesh: Any,
    current_density: Any,
    *,
    base_order: int,
    vacuum_permeability: float = 1.0,
    boundary: str = ".*",
    bonus_integration_order: int = 8,
) -> GaugeFixedCurlCurlSolution:
    r"""Solve the mixed Coulomb-gauge form of note equation ``(M1)``.

    Implements

    ``(curl A, curl v)/mu0 + (grad lambda, v) = (J, v)`` and
    ``(A, grad q) = 0``

    with essential ``n x A = 0`` and ``lambda = 0`` on ``boundary``.  The
    resulting magnetic field is ``B = curl A``.
    """
    raise NotImplementedError("milestone 4.2 gauge-fixed curl-curl solve")
