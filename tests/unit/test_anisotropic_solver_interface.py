"""Public StandardCG contracts for the anisotropic M4a solver."""

from __future__ import annotations

import warnings

import ngsolve as ng
import pytest

from remec.fem._anisotropic_diffusion import DirectionalConductivity
from remec.geometry.slab import Slab2D
from remec.solvers.anisotropic_diffusion import (
    AnisotropicDiffusionSolver,
    AnisotropyPollutionError,
    AnisotropyPollutionWarning,
    FloorSensitivityError,
    FloorSensitivityWarning,
    SpatialAnisotropicConductivity,
)


def test_standard_cg_public_path_preserves_constant_m4a_diagnostics() -> None:
    """(M4a) StandardCG reports separately positive constant-field energies."""
    source = ng.sin(ng.pi * ng.x) * ng.sin(ng.pi * ng.y)
    result = AnisotropicDiffusionSolver(polynomial_order=2).solve(
        Slab2D.unit_square(maxh=0.25),
        DirectionalConductivity(parallel=7.0, perpendicular=0.3, direction=(3.0, 4.0)),
        source,
    )

    assert result.free_dof_relative_residual_norm < 1.0e-11
    assert result.energy_diagnostics.parallel > 0.0
    assert result.energy_diagnostics.perpendicular > 0.0
    assert result.energy_diagnostics.total == pytest.approx(
        result.energy_diagnostics.parallel + result.energy_diagnostics.perpendicular
    )
    assert result.diagnostics["preconditioner_identity_defect"] < 1.0e-11
    assert not hasattr(result, "field")
    assert not hasattr(result, "operator")


def test_standard_cg_public_spatial_path_catches_missing_m4a_tensor_contrast() -> None:
    """(M4a) public island solve reproduces a non-tangent manufactured response."""
    flux = 0.5 * (ng.y - 0.5) ** 2 + ng.cos(2.0 * ng.pi * ng.x) / (2.0 * ng.pi) ** 2
    raw_field = ng.CoefficientFunction((flux.Diff(ng.y), -flux.Diff(ng.x)))
    exact = ng.sin(ng.pi * ng.x) * ng.sin(ng.pi * ng.y)
    field_floor = 0.05
    direction = raw_field / ng.sqrt(ng.InnerProduct(raw_field, raw_field) + field_floor**2)
    exact_gradient = ng.CoefficientFunction((exact.Diff(ng.x), exact.Diff(ng.y)))
    parallel_gradient = ng.InnerProduct(direction, exact_gradient)
    flux_vector = exact_gradient + 9.0 * direction * parallel_gradient
    source = -(flux_vector[0].Diff(ng.x) + flux_vector[1].Diff(ng.y))
    result = AnisotropicDiffusionSolver(polynomial_order=3).solve(
        Slab2D.unit_square(maxh=0.1),
        SpatialAnisotropicConductivity(
            parallel=10.0,
            perpendicular=1.0,
            field_floor=field_floor,
            raw_field=raw_field,
        ),
        source,
    )

    assert result.diagnostics["central_amplitude"] == pytest.approx(1.0, rel=3.0e-3)
    assert result.energy_diagnostics.parallel > 0.0
    assert result.energy_diagnostics.perpendicular > 0.0
    assert result.diagnostics["floor_activity_l2_squared"] > 0.0


def test_sovinec_measurement_is_routed_through_production_safety_gate() -> None:
    """§8.3 treats its physical κ_perp=0 rank-one measurement as unsafe."""
    solver = AnisotropicDiffusionSolver(polynomial_order=1)

    with pytest.raises(AnisotropyPollutionError, match="numerical perpendicular diffusion"):
        solver.measure_sovinec_pollution(Slab2D.unit_square(maxh=0.25), strict=True)

    with pytest.warns(AnisotropyPollutionWarning, match="numerical perpendicular diffusion"):
        diagnostic = solver.assess_pollution(
            numerical_perpendicular_diffusivity=2.0e-3,
            physical_perpendicular_diffusivity=1.0e-2,
        )
    assert not diagnostic.is_safe


def test_floor_sensitivity_is_relative_even_for_small_observables() -> None:
    """§6 detects a 100% floor-induced change at an O(1e-3) observable."""
    solver = AnisotropicDiffusionSolver(floor_sensitivity_tolerance=1.0e-2)

    with pytest.warns(FloorSensitivityWarning, match="B floor materially affects"):
        diagnostic = solver.assess_floor_sensitivity(
            observable_with_floor=2.0e-3,
            observable_with_smaller_floor=1.0e-3,
        )
    assert diagnostic.relative_change == pytest.approx(0.5)
    assert not diagnostic.is_acceptable
    with pytest.raises(FloorSensitivityError, match="B floor materially affects"):
        solver.assess_floor_sensitivity(
            observable_with_floor=2.0e-3,
            observable_with_smaller_floor=1.0e-3,
            strict=True,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert solver.assess_floor_sensitivity(
            observable_with_floor=1.001e-3,
            observable_with_smaller_floor=1.0e-3,
        ).is_acceptable
