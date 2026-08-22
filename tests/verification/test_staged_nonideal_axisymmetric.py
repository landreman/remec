"""Shaped non-ideal continuation against independent Zheng ideal equilibria."""

from __future__ import annotations

import numpy as np
import pytest

from remec.analytic_equilibria import ZhengShape, solve_zheng_equilibrium
from remec.profiles import TabulatedPressureProfile
from remec.solvers.axisymmetric_nonideal import (
    _ZhengContinuationContext,
    run_zheng_nonideal_continuation,
)
from remec.solvers.continuation import ContinuationStage

_STAGES = (
    ContinuationStage(0.6, 0.06, 0.12),
    ContinuationStage(0.8, 0.03, 0.06),
    ContinuationStage(1.0, 0.015, 0.03),
)


def test_shaped_nonideal_continuation_sentinel() -> None:
    """Cheap real-block sentinel realizes profiles and reports the independent error split."""
    result = run_zheng_nonideal_continuation(
        plasma_current=0.8e6,
        stages=(_STAGES[0],),
        maxh=0.32,
        polynomial_order=2,
    )
    row = result.stages[0]
    assert row.nonideal_to_analytic_relative_l2_error > row.ideal_fem_to_analytic_relative_l2_error
    assert row.ideal_fem_to_analytic_relative_l2_error < 0.01
    assert row.pressure_profile_error < 1.0e-10
    assert row.current_profile_error < 1.0e-10
    assert row.projected_current_profile_error < 1.0e-10
    assert row.toroidal_flux_relative_error < 1.0e-10
    assert row.minimum_magnetic_magnitude > 0.0
    assert row.magnetic_floor_activity_l2 < 1.0e-10


def test_toroidal_field_uses_a_free_trace_and_configurable_magnetic_floor() -> None:
    """The Igrad null mode is fixed only by prescribed toroidal flux, not wall data."""
    equilibrium = solve_zheng_equilibrium(
        shape=ZhengShape(0.70, 0.49, 1.7, 0.125),
        poloidal_beta=0.40,
        plasma_current=0.8e6,
    )
    context = _ZhengContinuationContext(
        equilibrium,
        maxh=0.32,
        polynomial_order=2,
        toroidal_flux=1.25,
        magnetic_floor=2.0e-9,
    )

    assert context.toroidal_flux == 1.25
    assert context.magnetic_floor == 2.0e-9
    assert sum(bool(value) for value in context.toroidal_space.FreeDofs()) == (
        context.toroidal_space.ndof
    )


def test_axisymmetric_m4a_anisotropy_is_live() -> None:
    """Replacing the in-plane M4a tensor by its isotropic part is conspicuous."""
    equilibrium = solve_zheng_equilibrium(
        shape=ZhengShape(0.70, 0.49, 1.7, 0.125),
        poloidal_beta=0.40,
        plasma_current=0.8e6,
    )
    context = _ZhengContinuationContext(equilibrium, maxh=0.32, polynomial_order=2)
    stage = _STAGES[0]
    psi, toroidal_field = context.fields_from_state(context.initial_state(stage))
    magnetic_field = context.magnetic_field(psi, toroidal_field)
    _, anisotropic_map, _, _ = context.solve_reference_potential(
        magnetic_field, stage.perpendicular_ratio
    )
    _, isotropic_map, _, _ = context.solve_reference_potential(magnetic_field, 1.0)
    anisotropic_s = anisotropic_map.quadrature_normalized_volume
    isotropic_s = isotropic_map.quadrature_normalized_volume
    relative_difference = float(
        np.linalg.norm(anisotropic_s - isotropic_s) / np.linalg.norm(isotropic_s)
    )
    assert relative_difference == pytest.approx(4.631311904911209e-4, rel=0.05)


def test_pressure_profile_realization_error_is_independently_measured() -> None:
    """Doubling the prescribed p0 shape fails the analytic Zheng pressure oracle."""
    equilibrium = solve_zheng_equilibrium(
        shape=ZhengShape(0.70, 0.49, 1.7, 0.125),
        poloidal_beta=0.40,
        plasma_current=0.8e6,
    )
    context = _ZhengContinuationContext(equilibrium, maxh=0.32, polynomial_order=2)
    stage = _STAGES[0]
    psi, toroidal_field = context.fields_from_state(context.initial_state(stage))
    magnetic_field = context.magnetic_field(psi, toroidal_field)
    chi, volume_map, mapped_points, _ = context.solve_reference_potential(
        magnetic_field, stage.perpendicular_ratio
    )
    context._last_chi = chi
    pressure, current = context.profiles(stage)
    edge_pressure = float(pressure.pressures[-1])
    doubled = TabulatedPressureProfile(
        pressure.normalized_volumes,
        tuple(edge_pressure + 2.0 * (value - edge_pressure) for value in pressure.pressures),
    )
    result = context.solve_current(
        magnetic_field,
        doubled,
        current,
        volume_map,
        mapped_points,
        stage=stage,
        current_diffusivity=stage.current_diffusivity,
    )

    assert result.pressure_profile_error > 0.9


def test_axisymmetric_m3_pressure_gradient_terms_are_live() -> None:
    """Deleting the M3 pressure-gradient drive and coupling changes the solved utilde."""
    equilibrium = solve_zheng_equilibrium(
        shape=ZhengShape(0.70, 0.49, 1.7, 0.125),
        poloidal_beta=0.40,
        plasma_current=0.8e6,
    )
    context = _ZhengContinuationContext(equilibrium, maxh=0.32, polynomial_order=2)
    stage = _STAGES[0]
    psi, toroidal_field = context.fields_from_state(context.initial_state(stage))
    magnetic_field = context.magnetic_field(psi, toroidal_field)
    chi, volume_map, mapped_points, _ = context.solve_reference_potential(
        magnetic_field, stage.perpendicular_ratio
    )
    context._last_chi = chi
    pressure, current = context.profiles(stage)
    constant_pressure = TabulatedPressureProfile(
        pressure.normalized_volumes,
        tuple(float(pressure.pressures[-1]) for _ in pressure.pressures),
    )
    with_pressure = context.solve_current(
        magnetic_field,
        pressure,
        current,
        volume_map,
        mapped_points,
        stage=stage,
        current_diffusivity=stage.current_diffusivity,
    )
    without_pressure_gradient = context.solve_current(
        magnetic_field,
        constant_pressure,
        current,
        volume_map,
        mapped_points,
        stage=stage,
        current_diffusivity=stage.current_diffusivity,
    )
    with_values = np.asarray(with_pressure.utilde.vec.FV().NumPy(), dtype=float)
    without_values = np.asarray(without_pressure_gradient.utilde.vec.FV().NumPy(), dtype=float)
    relative_difference = float(
        np.linalg.norm(with_values - without_values) / np.linalg.norm(with_values)
    )

    assert relative_difference == pytest.approx(0.3623776571562323, rel=0.05)


def test_axisymmetric_m2_current_has_an_independent_oracle() -> None:
    """A separately sampled ``(M2)`` oracle detects consistent component mutations."""
    import ngsolve as ng

    equilibrium = solve_zheng_equilibrium(
        shape=ZhengShape(0.70, 0.49, 1.7, 0.125),
        poloidal_beta=0.40,
        plasma_current=0.8e6,
    )
    context = _ZhengContinuationContext(equilibrium, maxh=0.32, polynomial_order=2)
    stage = _STAGES[0]
    psi, toroidal_field = context.fields_from_state(context.initial_state(stage))
    magnetic_field = context.magnetic_field(psi, toroidal_field)
    chi, volume_map, mapped_points, _ = context.solve_reference_potential(
        magnetic_field, stage.perpendicular_ratio
    )
    context._last_chi = chi
    pressure, current = context.profiles(stage)
    result = context.solve_current(
        magnetic_field,
        pressure,
        current,
        volume_map,
        mapped_points,
        stage=stage,
        current_diffusivity=stage.current_diffusivity,
    )

    difference = result.physical_current - result.independent_current
    separately_integrated_error = float(
        ng.Integrate(ng.x * ng.InnerProduct(difference, difference), context.mesh, order=8)
    )

    assert result.independent_m2_relative_error < 1.0e-12
    assert separately_integrated_error < 1.0e-24
