"""Public contracts for the true two-dimensional axisymmetric block."""

from __future__ import annotations

import pickle

import ngsolve as ng
import numpy as np
import pytest

from remec.common.serialization import canonical_json
from remec.geometry import AxisymmetricRZDomain
from remec.profiles import AnalyticPressureProfile, AnalyticToroidalCurrentProfile
from remec.solvers import (
    AxisymmetricGradShafranovCoefficients,
    AxisymmetricGradShafranovSolver,
    AxisymmetricProfileClosure,
)


def _profiles() -> tuple[AnalyticPressureProfile, AnalyticToroidalCurrentProfile]:
    """Return valid normalized profiles for closure guard tests."""
    return (
        AnalyticPressureProfile(lambda s: 1.0 - s, lambda s: -1.0 + 0.0 * s),
        AnalyticToroidalCurrentProfile(lambda s: s, lambda s: 1.0 + 0.0 * s),
    )


def test_axisymmetric_domain_is_a_named_two_dimensional_rz_mesh() -> None:
    """The public geometry carries no hidden toroidal wedge discretization."""
    domain = AxisymmetricRZDomain((1.0, 2.0), (-0.5, 0.5), maxh=0.5)
    mesh = domain.build_mesh()._mesh

    assert mesh.dim == 2
    assert mesh.ne == 8
    assert domain.boundary_regions() == {
        "z_min": "bottom",
        "r_max": "right",
        "z_max": "top",
        "r_min": "left",
    }
    assert domain.metadata()["toroidal_discretization"] is None
    assert ng.Integrate(ng.x, mesh, definedon=mesh.Boundaries("left")) == pytest.approx(1.0)
    assert ng.Integrate(ng.x, mesh, definedon=mesh.Boundaries("right")) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "arguments, message",
    [
        (((0.0, 2.0), (0.0, 1.0), 0.5), "R_min"),
        (((2.0, 1.0), (0.0, 1.0), 0.5), "R_min"),
        (((1.0, 2.0), (1.0, 1.0), 0.5), "vertical"),
        (((1.0, 2.0), (0.0, 1.0), 0.0), "maxh"),
        (((1.0, np.inf), (0.0, 1.0), 0.5), "finite"),
    ],
)
def test_axisymmetric_domain_rejects_invalid_bounds(
    arguments: tuple[tuple[float, float], tuple[float, float], float],
    message: str,
) -> None:
    """Invalid cylindrical domains fail before NGSolve mesh allocation."""
    with pytest.raises(ValueError, match=message):
        AxisymmetricRZDomain(*arguments)


def test_public_axisymmetric_solver_reports_the_weighted_energy() -> None:
    """The public (M1) wrapper exercises and exposes its nontrivial energy diagnostic."""
    solver = AxisymmetricGradShafranovSolver(polynomial_order=2)
    domain = AxisymmetricRZDomain((1.0, 2.0), (0.0, 1.0), maxh=0.5)
    coefficients = AxisymmetricGradShafranovCoefficients(
        pressure_flux_derivative=-0.4,
        toroidal_field_drive=3.0,
        mu0=2.3,
    )
    first_solution = solver.solve_with_flux(domain, coefficients)
    result = first_solution.result

    assert result.elements == 8
    assert result.free_dof_relative_residual_norm < 1.0e-12
    assert result.weighted_magnetic_energy > 0.0
    first_flux = first_solution.flux_at(1.5, 0.5)
    second_solution = solver.solve_with_flux(
        AxisymmetricRZDomain((1.0, 2.0), (0.0, 1.0), maxh=0.5),
        AxisymmetricGradShafranovCoefficients(
            pressure_flux_derivative=0.0,
            toroidal_field_drive=0.0,
            mu0=2.3,
        ),
    )
    assert first_flux != pytest.approx(0.0)
    assert second_solution.flux_at(1.5, 0.5) == pytest.approx(0.0, abs=1.0e-15)
    assert first_solution.flux_at(1.5, 0.5) == pytest.approx(first_flux)
    assert canonical_json(result)
    assert pickle.loads(pickle.dumps(result)) == result
    assert solver.solve(domain, coefficients) == result

    with pytest.raises(ValueError, match="polynomial_order"):
        AxisymmetricGradShafranovSolver(polynomial_order=0)
    with pytest.raises(ValueError, match="mu0"):
        AxisymmetricGradShafranovCoefficients(0.0, 0.0, mu0=0.0)


def test_axisymmetric_profile_closure_rejects_degenerate_geometry_inputs() -> None:
    """The (M2)–(M4b) closure rejects singular normalized-volume geometry."""
    pressure, current = _profiles()
    closure = AxisymmetricProfileClosure(pressure, current, total_volume=4.0, mu0=2.3)

    with pytest.raises(ValueError, match="d_normalized_volume_d_flux"):
        closure.evaluate(0.5, d_normalized_volume_d_flux=0.0, mean_inverse_radius_squared=1.0)
    with pytest.raises(ValueError, match="mean_inverse_radius_squared"):
        closure.evaluate(0.5, d_normalized_volume_d_flux=1.0, mean_inverse_radius_squared=0.0)
    with pytest.raises(ValueError, match="total_volume"):
        AxisymmetricProfileClosure(pressure, current, total_volume=0.0)
    with pytest.raises(ValueError, match="mu0"):
        AxisymmetricProfileClosure(pressure, current, total_volume=4.0, mu0=0.0)
