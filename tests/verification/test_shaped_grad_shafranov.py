"""Ideal analytic Grad-Shafranov benchmarks on shaped flux-contour domains."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from itertools import pairwise
from math import log, pi, sqrt
from pathlib import Path

import ngsolve as ng
import pytest

from remec.analytic_equilibria import (
    CerfonFreidbergBoundary,
    CerfonFreidbergShape,
    ZhengShape,
    recover_smooth_flux_observables,
    solve_cerfon_freidberg,
    solve_zheng_equilibrium,
)
from remec.fem._axisymmetric import (
    AxisymmetricGradShafranovCoefficients,
    solve_axisymmetric_grad_shafranov,
)
from remec.geometry.axisymmetric import AxisymmetricFluxContourDomain


@dataclass(frozen=True)
class _ErrorRow:
    maxh: float
    elements: int
    relative_l2_error: float
    relative_weighted_energy_error: float
    boundary_geometry_error: float
    axis_radius_error: float
    major_radius_error: float
    minor_radius_error: float
    elongation_error: float
    triangularity_error: float


@dataclass(frozen=True)
class _XPointRow:
    maxh: float
    elements: int
    relative_l2_error: float
    boundary_geometry_error: float
    free_dof_relative_residual_norm: float


_ZHENG_TABLE = Path(__file__).with_name("shaped_zheng_rates.csv")
_XPOINT_TABLE = Path(__file__).with_name("cerfon_freidberg_xpoint_rates.csv")


_ZHENG = solve_zheng_equilibrium(
    shape=ZhengShape(0.70, 0.49, 1.7, 0.125),
    poloidal_beta=0.40,
    plasma_current=1.0e6,
)


def _zheng_row(polynomial_order: int, maxh: float) -> _ErrorRow:
    """Solve note ``GS_recovered`` on Zheng's exact Psi=0 contour and measure it."""
    contour = _ZHENG.boundary_contour(samples=257)
    domain = AxisymmetricFluxContourDomain(
        contour,
        maxh=maxh,
        geometry_order=polynomial_order + 1,
    )
    solution = solve_axisymmetric_grad_shafranov(
        domain,
        polynomial_order=polynomial_order,
        coefficients=AxisymmetricGradShafranovCoefficients(
            pressure_flux_derivative=-_ZHENG.a1 / (4.0e-7 * pi),
            toroidal_field_drive=_ZHENG.a2,
            mu0=4.0e-7 * pi,
        ),
    )
    exact = _ZHENG.flux(ng.x, ng.y)
    exact_gradient = ng.CoefficientFunction(
        (
            _ZHENG.radial_derivative(ng.x, ng.y),
            _ZHENG.vertical_derivative(ng.x, ng.y),
        )
    )
    integration_order = 2 * polynomial_order + 8
    l2_error = float(
        ng.sqrt(
            ng.Integrate((solution._flux - exact) ** 2, solution._mesh, order=integration_order)
        )
    )
    l2_norm = float(ng.sqrt(ng.Integrate(exact**2, solution._mesh, order=integration_order)))
    gradient_error = ng.grad(solution._flux) - exact_gradient
    energy_error = float(
        ng.sqrt(
            ng.Integrate(
                ng.InnerProduct(gradient_error, gradient_error) / ng.x,
                solution._mesh,
                order=integration_order,
            )
        )
    )
    energy_norm = float(
        ng.sqrt(
            ng.Integrate(
                ng.InnerProduct(exact_gradient, exact_gradient) / ng.x,
                solution._mesh,
                order=integration_order,
            )
        )
    )
    boundary_length = float(ng.Integrate(1.0, solution._mesh, ng.BND, order=integration_order))
    boundary_geometry_error = float(
        ng.sqrt(
            ng.Integrate(exact**2, solution._mesh, ng.BND, order=integration_order)
            / boundary_length
        )
        / abs(_ZHENG.magnetic_axis()[1])
    )
    observables = recover_smooth_flux_observables(
        mesh=solution._mesh,
        flux=solution._flux,
        search_contour=contour,
    )
    exact_axis_radius, _ = _ZHENG.magnetic_axis()
    requested = _ZHENG.shape
    assert solution.free_dof_relative_residual_norm < 1.0e-11
    return _ErrorRow(
        maxh,
        solution.elements,
        l2_error / l2_norm,
        energy_error / energy_norm,
        boundary_geometry_error,
        abs(observables.axis_radius - exact_axis_radius),
        abs(observables.major_radius - requested.major_radius),
        abs(observables.minor_radius - requested.minor_radius),
        abs(observables.elongation - requested.elongation),
        abs(observables.triangularity - requested.triangularity),
    )


def _rate(coarse: float, fine: float, coarse_elements: int, fine_elements: int) -> float:
    """Return a 2-D unstructured-mesh rate using ``h_eff proportional sqrt(1/ne)``."""
    return log(coarse / fine) / log(sqrt(fine_elements / coarse_elements))


def _recorded_rows(path: Path) -> list[dict[str, str]]:
    """Load every row of one generated shaped-equilibrium measurement table."""
    with path.open(newline="") as table_file:
        return list(csv.DictReader(table_file))


def test_shaped_zheng_smooth_boundary_sentinel() -> None:
    """Cheap p=2 sentinel has near-optimal FEM rates on a smooth Psi=0 wall."""
    rows = [_zheng_row(2, maxh) for maxh in (0.24, 0.12)]
    assert (
        _rate(
            rows[0].relative_l2_error,
            rows[1].relative_l2_error,
            rows[0].elements,
            rows[1].elements,
        )
        >= 2.8
    )
    assert (
        _rate(
            rows[0].relative_weighted_energy_error,
            rows[1].relative_weighted_energy_error,
            rows[0].elements,
            rows[1].elements,
        )
        >= 1.8
    )
    assert rows[1].boundary_geometry_error < rows[0].boundary_geometry_error
    assert rows[1].axis_radius_error < 1.0e-3
    assert rows[1].major_radius_error < 2.0e-3
    assert rows[1].minor_radius_error < 2.0e-3
    assert rows[1].elongation_error < 1.0e-2
    assert rows[1].triangularity_error < 2.0e-2


@pytest.mark.slow
@pytest.mark.parametrize("polynomial_order", [1, 2, 3])
def test_shaped_zheng_full_order_scan(polynomial_order: int) -> None:
    """Full smooth-boundary scan reaches L2 order p+1 and weighted-energy order p."""
    rows = [_zheng_row(polynomial_order, maxh) for maxh in (0.20, 0.12, 0.07)]
    assert (
        _rate(
            rows[-2].relative_l2_error,
            rows[-1].relative_l2_error,
            rows[-2].elements,
            rows[-1].elements,
        )
        >= polynomial_order + 0.8
    )
    assert (
        _rate(
            rows[-2].relative_weighted_energy_error,
            rows[-1].relative_weighted_energy_error,
            rows[-2].elements,
            rows[-1].elements,
        )
        >= polynomial_order - 0.2
    )
    recorded = {
        (int(row["polynomial_order"]), float(row["maxh"])): row
        for row in _recorded_rows(_ZHENG_TABLE)
    }
    for row in rows:
        expected = recorded[polynomial_order, row.maxh]
        assert row.elements == int(expected["elements"])
        assert row.relative_l2_error == pytest.approx(
            float(expected["relative_l2_error"]), rel=0.05
        )
        assert row.relative_weighted_energy_error == pytest.approx(
            float(expected["relative_weighted_energy_error"]), rel=0.05
        )
        assert row.boundary_geometry_error == pytest.approx(
            float(expected["boundary_geometry_error"]), rel=0.1, abs=1.0e-12
        )


def _cerfon_xpoint_rows() -> list[_XPointRow]:
    """Measure ideal-analytic and geometry errors on the double-null separatrix."""
    equilibrium = solve_cerfon_freidberg(
        shape=CerfonFreidbergShape(0.78, 2.0, 0.35),
        source_parameter=0.0,
        boundary=CerfonFreidbergBoundary.DOUBLE_NULL,
    )
    rows: list[_XPointRow] = []
    for maxh in (0.38, 0.26, 0.18):
        domain = AxisymmetricFluxContourDomain(
            equilibrium.boundary_contour(samples=385),
            maxh=maxh,
            geometry_order=2,
        )
        solution = solve_axisymmetric_grad_shafranov(
            domain,
            polynomial_order=2,
            coefficients=AxisymmetricGradShafranovCoefficients(
                pressure_flux_derivative=-1.0,
                toroidal_field_drive=0.0,
                mu0=1.0,
            ),
        )
        exact = equilibrium.flux(ng.x, ng.y)
        l2_error = float(
            ng.sqrt(ng.Integrate((solution._flux - exact) ** 2, solution._mesh, order=12))
        )
        l2_norm = float(ng.sqrt(ng.Integrate(exact**2, solution._mesh, order=12)))
        length = float(ng.Integrate(1.0, solution._mesh, ng.BND, order=12))
        geometry_error = float(
            ng.sqrt(ng.Integrate(exact**2, solution._mesh, ng.BND, order=12) / length)
        )
        rows.append(
            _XPointRow(
                maxh,
                solution.elements,
                l2_error / l2_norm,
                geometry_error,
                solution.free_dof_relative_residual_norm,
            )
        )
    return rows


@pytest.mark.slow
def test_cerfon_freidberg_xpoint_boundary_converges() -> None:
    """The ideal FEM error and explicit geometry error fall on an X-point separatrix."""
    rows = _cerfon_xpoint_rows()
    assert all(row.free_dof_relative_residual_norm < 1.0e-11 for row in rows)
    assert all(fine.relative_l2_error < coarse.relative_l2_error for coarse, fine in pairwise(rows))
    assert all(
        fine.boundary_geometry_error < coarse.boundary_geometry_error
        for coarse, fine in pairwise(rows)
    )
    recorded = {float(row["maxh"]): row for row in _recorded_rows(_XPOINT_TABLE)}
    for row in rows:
        expected = recorded[row.maxh]
        assert row.elements == int(expected["elements"])
        assert row.relative_l2_error == pytest.approx(
            float(expected["relative_l2_error"]), rel=0.05
        )
        assert row.boundary_geometry_error == pytest.approx(
            float(expected["boundary_geometry_error"]), rel=0.1, abs=1.0e-12
        )
