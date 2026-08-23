"""NGSolve realization of the Reiman--Greenside field for note equation (M1)."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from remec.fem._harmonic_flux import _quadrature_extrema
from remec.fem.spaces import make_periodic_tetrahedral_de_rham_sequence
from remec.reiman_greenside import ReimanGreensideField


@dataclass(frozen=True, slots=True)
class ReimanGreensideDiscreteField:
    r"""Periodic discrete ``B_h = curl(A_h)`` and note-(M1) diagnostics."""

    vector_potential: Any
    magnetic_field: Any
    analytic_vector_potential: Any
    analytic_magnetic_field: Any
    curl_projection_relative_defect: float
    divergence_relative_norm: float
    analytic_field_relative_error: float
    sampled_magnetic_magnitude_minimum: float
    sampled_magnetic_magnitude_maximum: float
    b_floor_relative_activity: float


def build_reiman_greenside_discrete_field(
    mesh: Any,
    model: ReimanGreensideField,
    *,
    order: int,
    b_floor: float = 1.0e-8,
) -> ReimanGreensideDiscreteField:
    r"""Interpolate ``A``, form ``B_h=curl(A_h)``, and diagnose note equation (M1).

    Implements the Section-8.6 formulas

    ``A = Psi_t grad(Theta) - Psi_p grad(Phi)`` and
    ``B = curl(A) = grad(Psi_t) x grad(Theta) + grad(Phi) x grad(Psi_p)``.

    ``A_h`` is interpolated into periodic HCurl and its curl is mass-projected into
    the paired periodic HDiv space. The returned projection and strong-divergence
    defects independently demonstrate the mapped de Rham identity
    ``div(B_h)=div(curl(A_h))=0`` required by note equation (M1).
    """
    if getattr(mesh, "dim", None) != 3:
        raise ValueError("mesh must be three-dimensional")
    if not isinstance(model, ReimanGreensideField):
        raise TypeError("model must be a ReimanGreensideField")
    if not isfinite(b_floor) or b_floor <= 0.0:
        raise ValueError("b_floor must be finite and positive")
    import ngsolve as ng  # type: ignore[import-untyped]

    sequence = make_periodic_tetrahedral_de_rham_sequence(mesh, order=order)
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
    analytic_vector_potential = ng.CoefficientFunction(
        (-0.5 * ng.y, 0.5 * ng.x, -psi_p / model.major_radius)
    )
    analytic_magnetic_field = ng.CoefficientFunction(
        (-psi_y / model.major_radius, psi_x / model.major_radius, 1.0)
    )

    vector_potential = ng.GridFunction(sequence.hcurl)
    vector_potential.Set(analytic_vector_potential)
    trial, test = sequence.hdiv.TnT()
    mass = ng.BilinearForm(sequence.hdiv)
    mass += ng.InnerProduct(trial, test) * ng.dx
    load = ng.LinearForm(sequence.hdiv)
    load += ng.InnerProduct(ng.curl(vector_potential), test) * ng.dx
    mass.Assemble()
    load.Assemble()
    magnetic_field = ng.GridFunction(sequence.hdiv)
    magnetic_field.vec.data = (
        mass.mat.Inverse(sequence.hdiv.FreeDofs(), inverse="sparsecholesky") * load.vec
    )

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
    analytic_magnitude = ng.sqrt(ng.InnerProduct(analytic_magnetic_field, analytic_magnetic_field))
    sampled_minimum, sampled_maximum = _quadrature_extrema(
        mesh,
        analytic_magnitude,
        integration_order=integration_order,
    )
    safe_magnitude = ng.sqrt(analytic_magnitude**2 + b_floor**2)
    _, floor_activity = _quadrature_extrema(
        mesh,
        (safe_magnitude - analytic_magnitude) / analytic_magnitude,
        integration_order=integration_order,
    )
    return ReimanGreensideDiscreteField(
        vector_potential=vector_potential,
        magnetic_field=magnetic_field,
        analytic_vector_potential=analytic_vector_potential,
        analytic_magnetic_field=analytic_magnetic_field,
        curl_projection_relative_defect=projection_defect,
        divergence_relative_norm=divergence_relative_norm,
        analytic_field_relative_error=analytic_error,
        sampled_magnetic_magnitude_minimum=sampled_minimum,
        sampled_magnetic_magnitude_maximum=sampled_maximum,
        b_floor_relative_activity=floor_activity,
    )
