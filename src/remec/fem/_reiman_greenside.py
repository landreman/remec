"""NGSolve realization of the Reiman--Greenside field for note equation (M1)."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from remec.fem._harmonic_flux import _quadrature_extrema
from remec.fem._magnetostatics import reconstruct_periodic_magnetic_potential
from remec.reiman_greenside import ReimanGreensideField


@dataclass(frozen=True, slots=True)
class ReimanGreensideDiscreteField:
    r"""Periodic discrete ``B_h = curl(A_h)`` and note-(M1) diagnostics."""

    vector_potential: Any
    target_magnetic_field: Any
    magnetic_field: Any
    analytic_vector_potential: Any
    analytic_magnetic_field: Any
    curl_projection_relative_defect: float
    divergence_relative_norm: float
    analytic_field_relative_error: float
    bz_l2_error: float
    sampled_magnetic_magnitude_minimum: float
    sampled_magnetic_magnitude_maximum: float
    b_floor_relative_activity: float
    requested_axial_flux: float
    target_axial_flux: float
    reconstructed_axial_flux: float
    gauge_constraint_relative_residual: float
    harmonic_constraint_relative_residual: float


def reiman_greenside_coefficient_functions(model: ReimanGreensideField) -> tuple[Any, Any]:
    r"""Return the exact Section-8.6 ``(A, B=curl(A))`` coefficient functions.

    This is the single NGSolve transcription of
    ``A=Psi_t grad(Theta)-Psi_p grad(Phi)`` and
    ``B=grad(Psi_t) x grad(Theta)+grad(Phi) x grad(Psi_p)``.  Both the milestone-6.2
    compatible reconstruction and the milestone-6.3 frozen (M4a) operator consume it.
    """
    if not isinstance(model, ReimanGreensideField):
        raise TypeError("model must be a ReimanGreensideField")
    import ngsolve as ng  # type: ignore[import-untyped]

    phi = ng.z / model.major_radius
    cosine = ng.cos(phi)
    sine = ng.sin(phi)
    radius_squared = ng.x**2 + ng.y**2
    harmonic_2 = (ng.x**2 - ng.y**2) * cosine + 2.0 * ng.x * ng.y * sine
    harmonic_3 = (ng.x**3 - 3.0 * ng.x * ng.y**2) * cosine + (3.0 * ng.x**2 * ng.y - ng.y**3) * sine
    psi_p = (
        0.5 * model.t0 * radius_squared
        + 0.25 * model.t1 * radius_squared**2
        - model.epsilon_1 * harmonic_2
        - model.epsilon_2 * harmonic_3
    )
    psi_x = (
        model.t0 * ng.x
        + model.t1 * radius_squared * ng.x
        - model.epsilon_1 * (2.0 * ng.x * cosine + 2.0 * ng.y * sine)
        - model.epsilon_2 * (3.0 * (ng.x**2 - ng.y**2) * cosine + 6.0 * ng.x * ng.y * sine)
    )
    psi_y = (
        model.t0 * ng.y
        + model.t1 * radius_squared * ng.y
        - model.epsilon_1 * (-2.0 * ng.y * cosine + 2.0 * ng.x * sine)
        - model.epsilon_2 * (-6.0 * ng.x * ng.y * cosine + 3.0 * (ng.x**2 - ng.y**2) * sine)
    )
    vector_potential = ng.CoefficientFunction(
        (-0.5 * ng.y, 0.5 * ng.x, -psi_p / model.major_radius)
    )
    magnetic_field = ng.CoefficientFunction(
        (-psi_y / model.major_radius, psi_x / model.major_radius, 1.0)
    )
    return vector_potential, magnetic_field


def build_reiman_greenside_discrete_field(
    mesh: Any,
    model: ReimanGreensideField,
    *,
    order: int,
    harmonic_field: Any,
    axial_flux: float,
    b_floor: float = 1.0e-8,
) -> ReimanGreensideDiscreteField:
    r"""Reconstruct ``A_h``, form ``B_h=curl(A_h)``, and diagnose note equation (M1).

    Implements the Section-8.6 formulas

    ``A = Psi_t grad(Theta) - Psi_p grad(Phi)`` and
    ``B = curl(A) = grad(Psi_t) x grad(Theta) + grad(Phi) x grad(Psi_p)``.

    Per accepted ADR 0010 Option 2, the analytic ``B`` is projected into periodic
    H(div) subject to the paired divergence constraint and requested axial flux.
    The Section-7.3 periodic gauge-fixed curl solve, augmented by the normalized
    harmonic constraint from milestone 4.3, then reconstructs ``A_h``. Its curl is
    represented in the paired periodic H(div) space as ``B_h``. The returned curl
    and strong-divergence defects demonstrate
    ``div(B_h)=div(curl(A_h))=0`` required by note equation (M1).
    """
    if getattr(mesh, "dim", None) != 3:
        raise ValueError("mesh must be three-dimensional")
    if not isinstance(model, ReimanGreensideField):
        raise TypeError("model must be a ReimanGreensideField")
    if not isfinite(b_floor) or b_floor <= 0.0:
        raise ValueError("b_floor must be finite and positive")
    import ngsolve as ng

    analytic_vector_potential, analytic_magnetic_field = reiman_greenside_coefficient_functions(
        model
    )

    reconstruction = reconstruct_periodic_magnetic_potential(
        mesh,
        analytic_magnetic_field,
        harmonic_field,
        base_order=order,
        axial_flux=axial_flux,
    )
    vector_potential = reconstruction.vector_potential
    magnetic_field = reconstruction.magnetic_field

    integration_order = 2 * order + 12
    curl_norm = float(
        ng.sqrt(
            ng.Integrate(
                ng.InnerProduct(ng.curl(vector_potential), ng.curl(vector_potential)),
                mesh,
                order=integration_order,
            )
        )
    )
    magnetic_norm = float(
        ng.sqrt(
            ng.Integrate(
                ng.InnerProduct(magnetic_field, magnetic_field),
                mesh,
                order=integration_order,
            )
        )
    )
    projection_defect = float(
        ng.sqrt(
            ng.Integrate(
                ng.InnerProduct(
                    ng.curl(vector_potential) - magnetic_field,
                    ng.curl(vector_potential) - magnetic_field,
                ),
                mesh,
                order=integration_order,
            )
        )
        / max(curl_norm, np.finfo(float).tiny)
    )
    divergence_relative_norm = float(
        ng.sqrt(ng.Integrate(ng.div(magnetic_field) ** 2, mesh, order=integration_order))
        / max(magnetic_norm, np.finfo(float).tiny)
    )
    analytic_norm = float(
        ng.sqrt(
            ng.Integrate(
                ng.InnerProduct(analytic_magnetic_field, analytic_magnetic_field),
                mesh,
                order=integration_order,
            )
        )
    )
    analytic_error = float(
        ng.sqrt(
            ng.Integrate(
                ng.InnerProduct(
                    magnetic_field - analytic_magnetic_field,
                    magnetic_field - analytic_magnetic_field,
                ),
                mesh,
                order=integration_order,
            )
        )
        / max(analytic_norm, np.finfo(float).tiny)
    )
    bz_l2_error = float(
        ng.sqrt(ng.Integrate((magnetic_field[2] - 1.0) ** 2, mesh, order=integration_order))
    )
    magnetic_magnitude = ng.sqrt(ng.InnerProduct(magnetic_field, magnetic_field))
    sampled_minimum, sampled_maximum = _quadrature_extrema(
        mesh,
        magnetic_magnitude,
        integration_order=integration_order,
    )
    safe_magnitude = ng.sqrt(magnetic_magnitude**2 + b_floor**2)
    _, floor_activity = _quadrature_extrema(
        mesh,
        (safe_magnitude - magnetic_magnitude) / magnetic_magnitude,
        integration_order=integration_order,
    )
    return ReimanGreensideDiscreteField(
        vector_potential=vector_potential,
        target_magnetic_field=reconstruction.target_magnetic_field,
        magnetic_field=magnetic_field,
        analytic_vector_potential=analytic_vector_potential,
        analytic_magnetic_field=analytic_magnetic_field,
        curl_projection_relative_defect=projection_defect,
        divergence_relative_norm=divergence_relative_norm,
        analytic_field_relative_error=analytic_error,
        bz_l2_error=bz_l2_error,
        sampled_magnetic_magnitude_minimum=sampled_minimum,
        sampled_magnetic_magnitude_maximum=sampled_maximum,
        b_floor_relative_activity=floor_activity,
        requested_axial_flux=reconstruction.requested_axial_flux,
        target_axial_flux=reconstruction.target_axial_flux,
        reconstructed_axial_flux=reconstruction.reconstructed_axial_flux,
        gauge_constraint_relative_residual=reconstruction.gauge_constraint_relative_residual,
        harmonic_constraint_relative_residual=reconstruction.harmonic_constraint_relative_residual,
    )
