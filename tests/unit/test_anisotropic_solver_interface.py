"""Public StandardCG contracts for the anisotropic M4a solver."""

from __future__ import annotations

import warnings

import ngsolve as ng
import pytest

from remec.fem._anisotropic_diffusion import (
    DirectionalConductivity,
    solve_anisotropic_diffusion,
    solve_frozen_field_anisotropic_diffusion,
)
from remec.geometry.slab import Slab2D
from remec.solvers.anisotropic_diffusion import (
    AnisotropicDiffusionSolver,
    AnisotropyPollutionError,
    AnisotropyPollutionWarning,
    SpatialAnisotropicConductivity,
)


def test_standard_cg_preserves_constant_kernel_bit_for_bit() -> None:
    """(M4a) the extracted path retains the established constant-tensor result."""
    slab = Slab2D.unit_square(maxh=0.25)
    source = ng.sin(ng.pi * ng.x) * ng.sin(ng.pi * ng.y)
    coefficients = DirectionalConductivity(parallel=7.0, perpendicular=0.3, direction=(3.0, 4.0))

    reference = solve_anisotropic_diffusion(
        slab, polynomial_order=2, source=source, conductivity=coefficients
    )
    result = AnisotropicDiffusionSolver(polynomial_order=2).solve(slab, coefficients, source)

    assert result.field.vec.FV().NumPy() == pytest.approx(
        reference._field.vec.FV().NumPy(), abs=0.0
    )
    assert result.diagnostics["parallel_energy"] == reference.energy_diagnostics.parallel
    assert result.diagnostics["perpendicular_energy"] == reference.energy_diagnostics.perpendicular
    applied = result.field.vec.CreateVector()
    applied.data = result.apply(result.field.vec)
    direct = result.field.vec.CreateVector()
    direct.data = result.operator * result.field.vec
    assert applied.FV().NumPy() == pytest.approx(direct.FV().NumPy(), abs=0.0)


def test_standard_cg_preserves_floored_spatial_kernel_bit_for_bit() -> None:
    """(M4a) the unified path retains the smoothly floored tensor result."""
    slab = Slab2D.unit_square(maxh=0.25)
    raw_field = ng.CoefficientFunction((ng.y - 0.5, -(ng.x - 0.5)))
    source = ng.sin(ng.pi * ng.x) * ng.sin(ng.pi * ng.y)
    coefficients = SpatialAnisotropicConductivity(parallel=3.0, perpendicular=0.2, field_floor=0.1)

    reference = solve_frozen_field_anisotropic_diffusion(
        slab,
        polynomial_order=1,
        source=source,
        raw_field=raw_field,
        parallel_conductivity=coefficients.parallel,
        perpendicular_conductivity=coefficients.perpendicular,
        field_floor=coefficients.field_floor,
    )
    result = AnisotropicDiffusionSolver(polynomial_order=1).solve(
        (slab, raw_field), coefficients, source
    )

    assert result.field.vec.FV().NumPy() == pytest.approx(
        reference._field.vec.FV().NumPy(), abs=0.0
    )
    assert result.diagnostics["floor_activity_l2_squared"] == (
        reference.field_direction_diagnostics.floor_activity_l2_squared
    )
    assert result.energy_diagnostics.total == pytest.approx(
        result.energy_diagnostics.parallel + result.energy_diagnostics.perpendicular,
        abs=1.0e-14,
    )


def test_spatial_tensor_is_symmetric_and_softens_only_parallel_eigenvalue() -> None:
    """The M4a safe-field tensor remains symmetric and bounded at a field null."""
    coefficients = SpatialAnisotropicConductivity(
        parallel=10.0, perpendicular=0.25, field_floor=0.5
    )

    tensor = coefficients.tensor((3.0, 4.0))
    assert tensor[0][1] == tensor[1][0]
    assert coefficients.tensor((0.0, 0.0)) == ((0.25, 0.0), (0.0, 0.25))
    assert coefficients.floor_activity((0.0, 0.0)) == pytest.approx(1.0)


def test_pollution_gate_warns_or_fails_before_physical_transport_is_overstated() -> None:
    """DESIGN §8.3 enforces κ_perp,num < safety_factor κ_perp in strict mode."""
    solver = AnisotropicDiffusionSolver(pollution_safety_factor=0.1)

    with pytest.warns(AnisotropyPollutionWarning, match="numerical perpendicular diffusion"):
        diagnostic = solver.assess_pollution(
            numerical_perpendicular_diffusivity=2.0e-3,
            physical_perpendicular_diffusivity=1.0e-2,
        )
    assert not diagnostic.is_safe
    assert diagnostic.ratio_to_physical == pytest.approx(0.2)

    with pytest.raises(AnisotropyPollutionError, match="numerical perpendicular diffusion"):
        solver.assess_pollution(
            numerical_perpendicular_diffusivity=2.0e-3,
            physical_perpendicular_diffusivity=1.0e-2,
            strict=True,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert solver.assess_pollution(
            numerical_perpendicular_diffusivity=5.0e-4,
            physical_perpendicular_diffusivity=1.0e-2,
        ).is_safe


def test_floor_sensitivity_gate_reports_material_observable_change() -> None:
    """DESIGN §6 makes a materially active B floor visible to production callers."""
    solver = AnisotropicDiffusionSolver(floor_sensitivity_tolerance=1.0e-2)

    with pytest.warns(RuntimeWarning, match="B floor materially affects"):
        diagnostic = solver.assess_floor_sensitivity(
            observable_with_floor=1.02,
            observable_with_smaller_floor=1.0,
        )
    assert not diagnostic.is_acceptable

    with pytest.raises(RuntimeError, match="B floor materially affects"):
        solver.assess_floor_sensitivity(
            observable_with_floor=1.02,
            observable_with_smaller_floor=1.0,
            strict=True,
        )
