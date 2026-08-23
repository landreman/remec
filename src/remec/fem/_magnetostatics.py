"""Gauge-fixed compatible finite-element kernel for note equation (M1)."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from remec.fem.spaces import (
    make_periodic_tetrahedral_de_rham_sequence,
    make_tetrahedral_de_rham_sequence,
)


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
    sampled_magnetic_magnitude_minimum: float
    sampled_magnetic_magnitude_maximum: float


@dataclass(frozen=True, slots=True)
class PeriodicCurlReconstruction:
    r"""Periodic gauge-fixed reconstruction diagnostics for note equation ``(M1)``."""

    vector_potential: Any
    gauge_multiplier: Any
    target_magnetic_field: Any
    magnetic_field: Any
    target_projection_relative_error: float
    target_divergence_relative_norm: float
    curl_target_relative_defect: float
    magnetic_divergence_relative_norm: float
    requested_axial_flux: float
    target_axial_flux: float
    reconstructed_axial_flux: float
    free_dof_relative_residual: float
    gauge_constraint_relative_residual: float
    harmonic_constraint_relative_residual: float


def _quadrature_extrema(
    mesh: Any, coefficient: Any, *, integration_order: int
) -> tuple[float, float]:
    """Return deterministic volume-quadrature extrema for an NGSolve coefficient."""
    import ngsolve as ng  # type: ignore[import-untyped]

    element_types = {element.type for element in mesh.Elements(ng.VOL)}
    rules = {
        element_type: ng.IntegrationRule(element_type, integration_order)
        for element_type in element_types
    }
    mapped_points = mesh.MapToAllElements(rules, ng.VOL)
    values = np.asarray(coefficient(mapped_points), dtype=float).reshape(-1)
    return float(np.min(values)), float(np.max(values))


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

    import ngsolve as ng

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
    magnetic_magnitude = ng.sqrt(ng.InnerProduct(magnetic_field, magnetic_field))
    sampled_magnetic_magnitude_minimum, sampled_magnetic_magnitude_maximum = _quadrature_extrema(
        mesh, magnetic_magnitude, integration_order=integration_order
    )

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
        sampled_magnetic_magnitude_minimum=sampled_magnetic_magnitude_minimum,
        sampled_magnetic_magnitude_maximum=sampled_magnetic_magnitude_maximum,
    )


def reconstruct_periodic_magnetic_potential(
    mesh: Any,
    analytic_magnetic_field: Any,
    harmonic_field: Any,
    *,
    base_order: int,
    axial_flux: float,
    flux_boundary: str = "periodic_upper",
    bonus_integration_order: int = 8,
) -> PeriodicCurlReconstruction:
    r"""Reconstruct ``A_h`` with ``curl(A_h)=B_h`` for note equation ``(M1)``.

    ADR 0010 selects the Section-7.3 gauge-fixed curl-constrained solve after a
    divergence-constrained, axial-flux-normalized periodic-HDiv projection of ``B``.
    """
    if getattr(mesh, "dim", None) != 3:
        raise ValueError("mesh must be three-dimensional")
    if getattr(analytic_magnetic_field, "dim", None) != 3:
        raise ValueError("analytic_magnetic_field must have three components")
    if getattr(harmonic_field, "dim", None) != 3:
        raise ValueError("harmonic_field must have three components")
    if not isfinite(axial_flux):
        raise ValueError("axial_flux must be finite")
    if not isinstance(flux_boundary, str) or not flux_boundary:
        raise ValueError("flux_boundary must be a nonempty string")
    if isinstance(bonus_integration_order, bool) or not isinstance(bonus_integration_order, int):
        raise TypeError("bonus_integration_order must be an integer")
    if bonus_integration_order < 0:
        raise ValueError("bonus_integration_order must be non-negative")

    import ngsolve as ng

    sequence = make_periodic_tetrahedral_de_rham_sequence(mesh, order=base_order)
    dx = ng.dx(bonus_intorder=bonus_integration_order)
    flux_region = mesh.Boundaries(flux_boundary)
    if not flux_region.Mask().NumSet():
        raise ValueError(f"mesh has no boundary named {flux_boundary!r}")
    normal = ng.specialcf.normal(3)
    number_space = ng.NumberSpace(mesh)

    # ADR 0010 stage 1: closest periodic HDiv field subject to the paired curved
    # divergence constraint, followed by a global normalization that preserves that
    # constraint while setting the requested axial-cut flux.
    target_space = ng.FESpace([sequence.hdiv, sequence.l2])
    (field_trial, divergence_trial), (field_test, divergence_test) = target_space.TnT()
    target_operator = ng.BilinearForm(target_space, symmetric=True)
    target_operator += (
        ng.InnerProduct(field_trial, field_test)
        + ng.div(field_trial) * divergence_test
        + divergence_trial * ng.div(field_test)
    ) * dx
    target_rhs = ng.LinearForm(target_space)
    target_rhs += ng.InnerProduct(analytic_magnetic_field, field_test) * dx
    target_operator.Assemble()
    target_rhs.Assemble()
    target_solution = ng.GridFunction(target_space)
    target_solution.vec.data = (
        target_operator.mat.Inverse(target_space.FreeDofs(), inverse="umfpack") * target_rhs.vec
    )
    target_magnetic_field = target_solution.components[0]
    unnormalized_flux = float(
        ng.Integrate(
            target_magnetic_field * normal,
            mesh,
            ng.BND,
            definedon=flux_region,
            order=2 * sequence.hcurl_order + 10,
        )
    )
    if abs(unnormalized_flux) <= np.finfo(float).tiny:
        raise RuntimeError("periodic HDiv target has zero axial flux and cannot be normalized")
    target_magnetic_field.vec.data *= axial_flux / unnormalized_flux

    # ADR 0010 stage 2: the milestone-4.2 Coulomb-gauge curl--curl block, now on
    # periodic spaces. Two scalar constraints remove the constant multiplier mode
    # and the milestone-4.3 axial harmonic from ker(curl), respectively.
    potential_space = ng.FESpace([sequence.hcurl, sequence.h1, number_space, ng.NumberSpace(mesh)])
    (
        (potential_trial, gauge_trial, harmonic_trial, mean_trial),
        (
            potential_test,
            gauge_test,
            harmonic_test,
            mean_test,
        ),
    ) = potential_space.TnT()
    potential_operator = ng.BilinearForm(potential_space, symmetric=True)
    potential_operator += (
        ng.InnerProduct(ng.curl(potential_trial), ng.curl(potential_test))
        + ng.InnerProduct(ng.grad(gauge_trial), potential_test)
        + ng.InnerProduct(potential_trial, ng.grad(gauge_test))
        + harmonic_trial * ng.InnerProduct(harmonic_field, potential_test)
        + harmonic_test * ng.InnerProduct(potential_trial, harmonic_field)
        + mean_trial * gauge_test
        + mean_test * gauge_trial
    ) * dx
    potential_rhs = ng.LinearForm(potential_space)
    potential_rhs += ng.InnerProduct(target_magnetic_field, ng.curl(potential_test)) * dx
    potential_operator.Assemble()
    potential_rhs.Assemble()
    potential_solution = ng.GridFunction(potential_space)
    potential_solution.vec.data = (
        potential_operator.mat.Inverse(potential_space.FreeDofs(), inverse="umfpack")
        * potential_rhs.vec
    )
    vector_potential, gauge_multiplier = potential_solution.components[:2]

    magnetic_trial, magnetic_test = sequence.hdiv.TnT()
    magnetic_mass = ng.BilinearForm(sequence.hdiv)
    magnetic_mass += ng.InnerProduct(magnetic_trial, magnetic_test) * dx
    magnetic_rhs = ng.LinearForm(sequence.hdiv)
    magnetic_rhs += ng.InnerProduct(ng.curl(vector_potential), magnetic_test) * dx
    magnetic_mass.Assemble()
    magnetic_rhs.Assemble()
    magnetic_field = ng.GridFunction(sequence.hdiv)
    magnetic_field.vec.data = (
        magnetic_mass.mat.Inverse(sequence.hdiv.FreeDofs(), inverse="sparsecholesky")
        * magnetic_rhs.vec
    )

    integration_order = 2 * max(sequence.h1_order, sequence.hcurl_order) + 10
    analytic_norm = float(
        ng.sqrt(
            ng.Integrate(
                ng.InnerProduct(analytic_magnetic_field, analytic_magnetic_field),
                mesh,
                order=integration_order,
            )
        )
    )
    target_norm = float(
        ng.sqrt(
            ng.Integrate(
                ng.InnerProduct(target_magnetic_field, target_magnetic_field),
                mesh,
                order=integration_order,
            )
        )
    )
    scale = max(target_norm, np.finfo(float).tiny)
    target_projection_relative_error = float(
        ng.sqrt(
            ng.Integrate(
                ng.InnerProduct(
                    target_magnetic_field - analytic_magnetic_field,
                    target_magnetic_field - analytic_magnetic_field,
                ),
                mesh,
                order=integration_order,
            )
        )
        / max(analytic_norm, np.finfo(float).tiny)
    )
    target_divergence_relative_norm = float(
        ng.sqrt(
            ng.Integrate(
                ng.div(target_magnetic_field) ** 2,
                mesh,
                order=integration_order,
            )
        )
        / scale
    )
    curl_target_relative_defect = float(
        ng.sqrt(
            ng.Integrate(
                ng.InnerProduct(
                    ng.curl(vector_potential) - target_magnetic_field,
                    ng.curl(vector_potential) - target_magnetic_field,
                ),
                mesh,
                order=integration_order,
            )
        )
        / scale
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
    magnetic_divergence_relative_norm = float(
        ng.sqrt(ng.Integrate(ng.div(magnetic_field) ** 2, mesh, order=integration_order))
        / max(magnetic_norm, np.finfo(float).tiny)
    )

    free_projector = ng.Projector(potential_space.FreeDofs(), True)
    algebraic_residual = potential_solution.vec.CreateVector()
    algebraic_residual.data = free_projector * (
        potential_operator.mat * potential_solution.vec - potential_rhs.vec
    )
    free_rhs = free_projector * potential_rhs.vec
    free_dof_relative_residual = float(ng.Norm(algebraic_residual)) / max(
        float(ng.Norm(free_rhs)), np.finfo(float).tiny
    )
    gauge_residual = ng.LinearForm(sequence.h1)
    gauge_probe = sequence.h1.TestFunction()
    gauge_residual += ng.InnerProduct(vector_potential, ng.grad(gauge_probe)) * dx
    gauge_residual.Assemble()
    gauge_constraint_relative_residual = float(ng.Norm(gauge_residual.vec)) / max(
        float(ng.Norm(vector_potential.vec)), 1.0
    )
    harmonic_dof = potential_space.Range(2).start
    harmonic_constraint_relative_residual = abs(float(algebraic_residual[harmonic_dof])) / max(
        float(ng.Norm(free_rhs)), 1.0
    )

    target_axial_flux = float(
        ng.Integrate(
            target_magnetic_field * normal,
            mesh,
            ng.BND,
            definedon=flux_region,
            order=integration_order,
        )
    )
    reconstructed_axial_flux = float(
        ng.Integrate(
            magnetic_field * normal,
            mesh,
            ng.BND,
            definedon=flux_region,
            order=integration_order,
        )
    )
    return PeriodicCurlReconstruction(
        vector_potential=vector_potential,
        gauge_multiplier=gauge_multiplier,
        target_magnetic_field=target_magnetic_field,
        magnetic_field=magnetic_field,
        target_projection_relative_error=target_projection_relative_error,
        target_divergence_relative_norm=target_divergence_relative_norm,
        curl_target_relative_defect=curl_target_relative_defect,
        magnetic_divergence_relative_norm=magnetic_divergence_relative_norm,
        requested_axial_flux=axial_flux,
        target_axial_flux=target_axial_flux,
        reconstructed_axial_flux=reconstructed_axial_flux,
        free_dof_relative_residual=free_dof_relative_residual,
        gauge_constraint_relative_residual=gauge_constraint_relative_residual,
        harmonic_constraint_relative_residual=harmonic_constraint_relative_residual,
    )
