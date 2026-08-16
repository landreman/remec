"""Gauge-fixed compatible finite-element kernel for note equation (M1)."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from remec.fem.spaces import make_tetrahedral_de_rham_sequence


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
    curl_projection_relative_defect: float
    magnetic_divergence_relative_norm: float
    boundary_normal_relative_norm: float
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
    if not isfinite(vacuum_permeability) or vacuum_permeability <= 0.0:
        raise ValueError("vacuum_permeability must be finite and positive")
    if isinstance(bonus_integration_order, bool) or not isinstance(bonus_integration_order, int):
        raise TypeError("bonus_integration_order must be an integer")
    if bonus_integration_order < 0:
        raise ValueError("bonus_integration_order must be non-negative")
    if boundary != ".*":
        raise ValueError("the fixed-boundary magnetic kernel requires the full boundary '.*'")
    if getattr(current_density, "dim", None) != 3:
        raise ValueError("current_density must be a three-component coefficient function")

    import ngsolve as ng  # type: ignore[import-untyped]

    sequence = make_tetrahedral_de_rham_sequence(mesh, order=base_order)
    vector_space = ng.HCurl(mesh, order=sequence.hcurl_order, dirichlet=boundary)
    gauge_space = ng.H1(mesh, order=sequence.h1_order, dirichlet=boundary)
    mixed_space = ng.FESpace([vector_space, gauge_space])
    (vector_trial, gauge_trial), (vector_test, gauge_test) = mixed_space.TnT()
    dx = ng.dx(bonus_intorder=bonus_integration_order)

    operator = ng.BilinearForm(mixed_space, symmetric=True)
    operator += (
        ng.InnerProduct(ng.curl(vector_trial), ng.curl(vector_test)) / vacuum_permeability
        + ng.InnerProduct(ng.grad(gauge_trial), vector_test)
        + ng.InnerProduct(vector_trial, ng.grad(gauge_test))
    ) * dx
    right_hand_side = ng.LinearForm(mixed_space)
    right_hand_side += ng.InnerProduct(current_density, vector_test) * dx
    operator.Assemble()
    right_hand_side.Assemble()

    mixed_field = ng.GridFunction(mixed_space)
    mixed_field.vec.data = (
        operator.mat.Inverse(mixed_space.FreeDofs(), inverse="umfpack") * right_hand_side.vec
    )
    vector_potential, gauge_multiplier = mixed_field.components

    free_projector = ng.Projector(mixed_space.FreeDofs(), True)
    algebraic_residual = free_projector * (operator.mat * mixed_field.vec - right_hand_side.vec)
    free_rhs = free_projector * right_hand_side.vec
    residual_norm = float(ng.Norm(algebraic_residual))
    free_rhs_norm = float(ng.Norm(free_rhs))
    free_dof_relative_residual = residual_norm / max(
        free_rhs_norm,
        np.finfo(float).tiny,
    )

    gauge_residual = ng.LinearForm(gauge_space)
    gauge_probe = gauge_space.TestFunction()
    gauge_residual += ng.InnerProduct(vector_potential, ng.grad(gauge_probe)) * dx
    gauge_residual.Assemble()
    free_gauge_residual = ng.Projector(gauge_space.FreeDofs(), True) * gauge_residual.vec
    vector_coefficient_norm = float(ng.Norm(vector_potential.vec))
    gauge_constraint_relative_residual = float(ng.Norm(free_gauge_residual)) / max(
        vector_coefficient_norm,
        1.0,
    )

    magnetic_space = ng.HDiv(mesh, order=sequence.hdiv_order)
    magnetic_trial, magnetic_test = magnetic_space.TnT()
    magnetic_mass = ng.BilinearForm(magnetic_space)
    magnetic_mass += ng.InnerProduct(magnetic_trial, magnetic_test) * dx
    magnetic_rhs = ng.LinearForm(magnetic_space)
    magnetic_rhs += ng.InnerProduct(ng.curl(vector_potential), magnetic_test) * dx
    magnetic_mass.Assemble()
    magnetic_rhs.Assemble()
    magnetic_field = ng.GridFunction(magnetic_space)
    magnetic_field.vec.data = (
        magnetic_mass.mat.Inverse(magnetic_space.FreeDofs(), inverse="sparsecholesky")
        * magnetic_rhs.vec
    )

    integration_order = 2 * max(sequence.h1_order, sequence.hcurl_order) + 6
    curl_field = ng.curl(vector_potential)
    curl_norm = float(
        ng.sqrt(
            ng.Integrate(
                ng.InnerProduct(curl_field, curl_field),
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
    curl_projection_defect = float(
        ng.sqrt(
            ng.Integrate(
                ng.InnerProduct(
                    curl_field - magnetic_field,
                    curl_field - magnetic_field,
                ),
                mesh,
                order=integration_order,
            )
        )
    )
    curl_projection_relative_defect = curl_projection_defect / max(
        curl_norm,
        np.finfo(float).tiny,
    )
    magnetic_divergence_norm = float(
        ng.sqrt(ng.Integrate(ng.div(magnetic_field) ** 2, mesh, order=integration_order))
    )
    magnetic_divergence_relative_norm = magnetic_divergence_norm / max(
        magnetic_norm,
        np.finfo(float).tiny,
    )
    boundary_normal_norm = float(
        ng.sqrt(
            ng.Integrate(
                (magnetic_field * ng.specialcf.normal(3)) ** 2,
                mesh,
                ng.BND,
                order=integration_order,
            )
        )
    )
    boundary_normal_relative_norm = boundary_normal_norm / max(
        magnetic_norm,
        np.finfo(float).tiny,
    )
    gauge_multiplier_l2_norm = float(
        ng.sqrt(ng.Integrate(gauge_multiplier**2, mesh, order=integration_order))
    )
    magnetic_energy = 0.5 * magnetic_norm**2 / vacuum_permeability

    return GaugeFixedCurlCurlSolution(
        vector_potential=vector_potential,
        gauge_multiplier=gauge_multiplier,
        magnetic_field=magnetic_field,
        vector_potential_order=sequence.hcurl_order,
        gauge_order=sequence.h1_order,
        magnetic_field_order=sequence.hdiv_order,
        free_dof_relative_residual=free_dof_relative_residual,
        gauge_constraint_relative_residual=gauge_constraint_relative_residual,
        curl_projection_relative_defect=curl_projection_relative_defect,
        magnetic_divergence_relative_norm=magnetic_divergence_relative_norm,
        boundary_normal_relative_norm=boundary_normal_relative_norm,
        gauge_multiplier_l2_norm=gauge_multiplier_l2_norm,
        magnetic_energy=magnetic_energy,
    )
