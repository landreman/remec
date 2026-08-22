"""Manufactured verification of the note's axisymmetric GS reduction."""

from __future__ import annotations

import csv
from math import log, pi, sqrt
from pathlib import Path

import ngsolve as ng
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from remec.fem._axisymmetric import (
    AxisymmetricGradShafranovCoefficients,
    solve_axisymmetric_grad_shafranov,
)
from remec.geometry.axisymmetric import AxisymmetricRZDomain
from remec.profiles import AnalyticPressureProfile, AnalyticToroidalCurrentProfile
from remec.solvers.axisymmetric import AxisymmetricProfileClosure

_MESH_SIZES = (1.0 / 6.0, 1.0 / 12.0, 1.0 / 24.0)
_MU0 = 2.3
_PRESSURE_FLUX_DERIVATIVE = -0.4
_RATE_TABLE = Path(__file__).with_name("axisymmetric_grad_shafranov_rates.csv")
_CURRENT_TABLE = Path(__file__).with_name("axisymmetric_enclosed_current.csv")


def _recorded_errors() -> dict[tuple[int, float], tuple[float, float]]:
    """Read the checked-in weighted GS manufactured-error table."""
    with _RATE_TABLE.open(newline="") as table_file:
        return {
            (int(row["polynomial_order"]), float(row["maxh"])): (
                float(row["l2_error"]),
                float(row["weighted_energy_error"]),
            )
            for row in csv.DictReader(table_file)
        }


def _recorded_current_errors() -> dict[str, tuple[float, float]]:
    """Read the two independently integrated ``I_0(s)`` measurements."""
    with _CURRENT_TABLE.open(newline="") as table_file:
        return {
            row["profile"]: (float(row["maximum_absolute_error"]), float(row["edge_error"]))
            for row in csv.DictReader(table_file)
        }


@pytest.mark.parametrize("polynomial_order", [1, 2, 3])
def test_axisymmetric_grad_shafranov_manufactured_convergence(
    polynomial_order: int,
) -> None:
    """``GS_recovered`` has L2 order p+1 and weighted-energy order p.

    On ``1 < R < 2, 0 < Z < 1``, the exact flux is
    ``psi=sin(pi(R-1)) sin(pi Z)`` and
    ``-Delta*psi=2 pi^2 psi + pi cos(pi(R-1)) sin(pi Z)/R``.  The source is
    split between nonzero ``p'(psi)`` at ``mu0 != 1`` and ``I I'``. The explicit
    ``1/R`` term and split make this reject a Cartesian form or either omitted drive.
    """
    exact = ng.sin(pi * (ng.x - 1.0)) * ng.sin(pi * ng.y)
    exact_gradient = ng.CoefficientFunction(
        (
            pi * ng.cos(pi * (ng.x - 1.0)) * ng.sin(pi * ng.y),
            pi * ng.sin(pi * (ng.x - 1.0)) * ng.cos(pi * ng.y),
        )
    )
    negative_delta_star = (
        2.0 * pi**2 * exact + pi * ng.cos(pi * (ng.x - 1.0)) * ng.sin(pi * ng.y) / ng.x
    )
    errors: list[tuple[float, float]] = []
    recorded_errors = _recorded_errors()

    for maxh in _MESH_SIZES:
        domain = AxisymmetricRZDomain((1.0, 2.0), (0.0, 1.0), maxh)
        solution = solve_axisymmetric_grad_shafranov(
            domain,
            polynomial_order=polynomial_order,
            coefficients=AxisymmetricGradShafranovCoefficients(
                pressure_flux_derivative=_PRESSURE_FLUX_DERIVATIVE,
                toroidal_field_drive=negative_delta_star
                - _MU0 * ng.x**2 * _PRESSURE_FLUX_DERIVATIVE,
                mu0=_MU0,
            ),
        )
        assert domain.metadata()["toroidal_discretization"] is None
        assert solution._mesh.dim == 2
        l2_error = ng.sqrt(ng.Integrate((solution._flux - exact) ** 2, solution._mesh, order=10))
        gradient_error = ng.grad(solution._flux) - exact_gradient
        energy_error = ng.sqrt(
            ng.Integrate(
                ng.InnerProduct(gradient_error, gradient_error) / ng.x,
                solution._mesh,
                order=10,
            )
        )
        expected = recorded_errors[polynomial_order, maxh]
        assert float(l2_error) == pytest.approx(expected[0], rel=0.05, abs=1.0e-12)
        assert float(energy_error) == pytest.approx(expected[1], rel=0.05, abs=1.0e-12)
        assert solution.free_dof_relative_residual_norm < 1.0e-11
        errors.append((float(l2_error), float(energy_error)))

    l2_rate = log(errors[-2][0] / errors[-1][0]) / log(2.0)
    energy_rate = log(errors[-2][1] / errors[-1][1]) / log(2.0)
    assert l2_rate >= polynomial_order + 0.8
    assert energy_rate >= polynomial_order - 0.2


def _integrated_toroidal_current(
    closure: AxisymmetricProfileClosure,
    normalized_volume: float,
    *,
    major_radius: float,
    minor_radius: float,
) -> float:
    """Independently integrate J_phi over a circular poloidal cross-section."""
    radial_nodes, radial_weights = leggauss(48)
    angle_nodes, angle_weights = leggauss(72)
    outer_radius = minor_radius * sqrt(normalized_volume)
    radius = 0.5 * outer_radius * (radial_nodes + 1.0)
    radial_weights = 0.5 * outer_radius * radial_weights
    angle = pi * (angle_nodes + 1.0)
    angle_weights = pi * angle_weights
    radius_grid = radius[:, None]
    cylindrical_radius = major_radius + radius_grid * np.cos(angle[None, :])
    s = np.broadcast_to((radius_grid / minor_radius) ** 2, cylindrical_radius.shape)
    ds_dflux = 0.6 + 0.5 * s
    mean_inverse_radius_squared = 1.0 / (major_radius * np.sqrt(major_radius**2 - radius_grid**2))
    closure_values = closure.evaluate(
        s,
        d_normalized_volume_d_flux=ds_dflux,
        mean_inverse_radius_squared=np.broadcast_to(
            mean_inverse_radius_squared, cylindrical_radius.shape
        ),
    )
    toroidal_current_density = (
        cylindrical_radius * closure_values.pressure_flux_derivative
        + closure_values.toroidal_field_drive / (closure.mu0 * cylindrical_radius)
    )
    weights = radial_weights[:, None] * angle_weights[None, :] * radius_grid
    return float(np.sum(toroidal_current_density * weights))


@pytest.mark.parametrize(
    "current_profile",
    [
        pytest.param(
            AnalyticToroidalCurrentProfile(lambda s: 0.7 * s, lambda s: 0.7 + 0.0 * s),
            id="linear",
        ),
        pytest.param(
            AnalyticToroidalCurrentProfile(
                lambda s: 0.4 * s * (2.0 - s), lambda s: 0.8 * (1.0 - s)
            ),
            id="centrally-peaked",
        ),
    ],
)
def test_enclosed_current_relation_recovers_two_normalized_profiles(
    current_profile: AnalyticToroidalCurrentProfile,
    request: pytest.FixtureRequest,
) -> None:
    """``I_ODE`` independently realizes two distinct cumulative ``I_0(s)`` inputs."""
    major_radius = 2.0
    minor_radius = 0.6
    total_volume = 2.0 * pi**2 * major_radius * minor_radius**2
    pressure_profile = AnalyticPressureProfile(
        lambda s: 2.1 - 2.0 * s,
        lambda s: -2.0 + 0.0 * s,
    )
    closure = AxisymmetricProfileClosure(
        pressure_profile,
        current_profile,
        total_volume,
        mu0=_MU0,
    )
    pressure_profile.validate(edge_value=0.1)
    current_profile.validate()
    shells = np.asarray((0.1, 0.3, 0.55, 0.8, 1.0))
    shell_radii = minor_radius * np.sqrt(shells)
    ds_dflux = 0.6 + 0.5 * shells
    closure_rows = closure.evaluate(
        shells,
        d_normalized_volume_d_flux=ds_dflux,
        mean_inverse_radius_squared=1.0
        / (major_radius * np.sqrt(major_radius**2 - shell_radii**2)),
    )
    assert closure_rows.pressure_flux_derivative == pytest.approx(
        np.asarray(pressure_profile.derivative(shells)) * ds_dflux
    )
    measured = np.asarray(
        [
            _integrated_toroidal_current(
                closure,
                float(shell),
                major_radius=major_radius,
                minor_radius=minor_radius,
            )
            for shell in shells
        ]
    )
    target = np.asarray(current_profile.enclosed_current(shells))
    maximum_error = float(np.max(np.abs(measured - target)))
    edge_error = float(measured[-1] - target[-1])
    profile_name = str(request.node.callspec.id)
    recorded_maximum, recorded_edge = _recorded_current_errors()[profile_name]
    assert maximum_error == pytest.approx(recorded_maximum, rel=0.1, abs=1.0e-14)
    assert edge_error == pytest.approx(recorded_edge, rel=0.1, abs=1.0e-14)
    assert maximum_error < 1.0e-8
    assert measured[-1] == pytest.approx(current_profile.enclosed_current(1.0), abs=1.0e-8)
