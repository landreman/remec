"""Public anisotropic diffusion strategy contracts for note equation (M4a)."""

from __future__ import annotations

import ngsolve as ng
import pytest

from remec.fem._anisotropic_diffusion import DirectionalConductivity
from remec.geometry.slab import Slab2D
from remec.solvers.anisotropic_diffusion import (
    AnisotropicDiffusionSolver,
    SpatialAnisotropicConductivity,
)


def test_standard_solver_preserves_constant_kernel_result() -> None:
    """The extracted strategy preserves the established constant M4a result."""
    slab = Slab2D.unit_square(maxh=0.25)
    source = ng.sin(ng.pi * ng.x) * ng.sin(ng.pi * ng.y)
    coefficients = DirectionalConductivity(parallel=7.0, perpendicular=0.3, direction=(3.0, 4.0))
    solver = AnisotropicDiffusionSolver(polynomial_order=2)

    result = solver.solve(slab, coefficients, source)

    assert result.polynomial_order == 2
    assert result.mesh.ne == 32
    assert result.energy_diagnostics.total == pytest.approx(
        result.energy_diagnostics.parallel + result.energy_diagnostics.perpendicular,
        rel=0.0,
        abs=1.0e-15,
    )
    assert result.diagnostics["free_dof_relative_residual_norm"] < 1.0e-11
    assert result.diagnostics["parallel_energy"] == pytest.approx(
        result.energy_diagnostics.parallel, rel=0.0, abs=1.0e-15
    )


def test_spatial_tensor_is_symmetric_with_floored_eigenvalues() -> None:
    """The smooth M4a tensor has eigenvalues in the configured conductivity range."""
    coefficients = SpatialAnisotropicConductivity(
        parallel=10.0,
        perpendicular=0.25,
        field_floor=0.5,
    )
    tensor = coefficients.tensor((3.0, 4.0))

    expected = ((3.725247524752475, 4.633663366336634), (4.633663366336634, 6.428217821782178))
    for row, expected_row in zip(tensor, expected):
        assert row == pytest.approx(expected_row, rel=0.0, abs=1.0e-12)
    assert tensor[0][1] == tensor[1][0]
    trace = tensor[0][0] + tensor[1][1]
    determinant = tensor[0][0] * tensor[1][1] - tensor[0][1] ** 2
    discriminant = (trace**2 - 4.0 * determinant) ** 0.5
    eigenvalues = sorted(((trace - discriminant) / 2.0, (trace + discriminant) / 2.0))
    assert eigenvalues[0] == pytest.approx(0.25, rel=0.0, abs=1.0e-12)
    assert 0.25 < eigenvalues[1] < 10.0
    assert coefficients.tensor((0.0, 0.0)) == ((0.25, 0.0), (0.0, 0.25))
    assert coefficients.floor_activity((0.0, 0.0)) == pytest.approx(1.0)


def test_public_solver_routes_spatial_field_and_reports_floor_activity() -> None:
    """The same interface accepts a spatial frozen field and its M4a diagnostics."""
    slab = Slab2D.unit_square(maxh=0.25)
    raw_field = ng.CoefficientFunction((ng.y - 0.5, -(ng.x - 0.5)))
    source = ng.sin(ng.pi * ng.x) * ng.sin(ng.pi * ng.y)
    coefficients = SpatialAnisotropicConductivity(parallel=3.0, perpendicular=0.2, field_floor=0.1)

    result = AnisotropicDiffusionSolver(polynomial_order=1).solve(
        (slab, raw_field), coefficients, source
    )

    assert result.mesh.ne == 32
    assert result.diagnostics["floor_activity_l2_squared"] > 0.0
    assert result.diagnostics["free_dof_relative_residual_norm"] < 1.0e-11
