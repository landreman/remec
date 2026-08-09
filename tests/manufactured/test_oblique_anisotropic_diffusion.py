"""Manufactured tests for the oblique anisotropic form of note equation (M4a)."""

from __future__ import annotations

import csv
from math import log
from pathlib import Path

import ngsolve as ng
import pytest

from remec.fem._isotropic_poisson import ObliqueConductivity, solve_isotropic_poisson
from remec.geometry.slab import Slab2D

_MESH_SIZES = (0.35, 0.175, 0.0875)
_RATE_TABLE = Path(__file__).with_name("oblique_anisotropic_rates.csv")
_CONDUCTIVITY = ObliqueConductivity(
    parallel=7.0,
    perpendicular=2.0,
    direction=(3.0 / 5.0, 4.0 / 5.0),
)


def _recorded_errors() -> dict[tuple[int, float], tuple[float, float]]:
    """Read the checked-in manufactured-error record keyed by order and maxh."""
    with _RATE_TABLE.open(newline="") as table_file:
        rows = csv.DictReader(table_file)
        return {
            (int(row["polynomial_order"]), float(row["maxh"])): (
                float(row["l2_error"]),
                float(row["energy_error"]),
            )
            for row in rows
        }


def _manufactured_data() -> tuple[
    ng.CoefficientFunction, ng.CoefficientFunction, ng.CoefficientFunction
]:
    """Return smooth χ and -div(K grad χ) for the constant oblique tensor K."""
    exact = ng.sin(ng.pi * ng.x) * ng.sin(ng.pi * ng.y)
    cosine_product = ng.cos(ng.pi * ng.x) * ng.cos(ng.pi * ng.y)
    k_xx, k_xy, k_yy = _CONDUCTIVITY.components
    source = ng.pi**2 * (k_xx + k_yy) * exact - 2.0 * ng.pi**2 * k_xy * cosine_product
    exact_gradient = ng.CoefficientFunction(
        (
            ng.pi * ng.cos(ng.pi * ng.x) * ng.sin(ng.pi * ng.y),
            ng.pi * ng.sin(ng.pi * ng.x) * ng.cos(ng.pi * ng.y),
        )
    )
    return exact, exact_gradient, source


def _diagnostic_data() -> tuple[ng.CoefficientFunction, ng.CoefficientFunction]:
    """Return a direction-sensitive solution and source for the M4a diagnostics."""
    exact = ng.sin(ng.pi * ng.x) * ng.sin(2.0 * ng.pi * ng.y)
    k_xx, k_xy, k_yy = _CONDUCTIVITY.components
    source = ng.pi**2 * (k_xx + 4.0 * k_yy) * exact - 4.0 * ng.pi**2 * k_xy * ng.cos(
        ng.pi * ng.x
    ) * ng.cos(2.0 * ng.pi * ng.y)
    return exact, source


def test_oblique_conductivity_has_parallel_and_perpendicular_eigenpairs() -> None:
    """K has eigenvalues κ∥ along b and κ⊥ in the transverse direction."""
    assert _CONDUCTIVITY.apply(_CONDUCTIVITY.direction) == pytest.approx(
        tuple(_CONDUCTIVITY.parallel * component for component in _CONDUCTIVITY.direction)
    )
    transverse = (-_CONDUCTIVITY.direction[1], _CONDUCTIVITY.direction[0])
    assert _CONDUCTIVITY.apply(transverse) == pytest.approx(
        tuple(_CONDUCTIVITY.perpendicular * component for component in transverse)
    )
    assert _CONDUCTIVITY.components == pytest.approx((3.8, 2.4, 5.2))


@pytest.mark.parametrize("polynomial_order", [1, 2, 3])
def test_oblique_anisotropic_manufactured_convergence(polynomial_order: int) -> None:
    """(M4a) keeps L² order p+1 and K-energy order p for constant oblique K."""
    exact, exact_gradient, source = _manufactured_data()
    errors: list[tuple[float, float]] = []
    recorded_errors = _recorded_errors()

    for maxh in _MESH_SIZES:
        solution = solve_isotropic_poisson(
            Slab2D.unit_square(maxh=maxh),
            polynomial_order=polynomial_order,
            source=source,
            conductivity=_CONDUCTIVITY,
        )
        l2_error = ng.sqrt(ng.Integrate((solution._field - exact) ** 2, solution._mesh, order=8))
        gradient_error = ng.grad(solution._field) - exact_gradient
        energy_error = ng.sqrt(
            ng.Integrate(_CONDUCTIVITY.quadratic_form(gradient_error), solution._mesh, order=8)
        )
        observed = (float(l2_error), float(energy_error))
        expected = recorded_errors[polynomial_order, maxh]
        assert observed[0] == pytest.approx(expected[0], rel=0.05, abs=1.0e-12)
        assert observed[1] == pytest.approx(expected[1], rel=0.05, abs=1.0e-12)
        assert solution.free_dof_residual_norm < 1.0e-10
        boundary_trace_norm_squared = sum(
            ng.Integrate(
                solution._field**2,
                solution._mesh,
                definedon=solution._mesh.Boundaries(boundary_name),
            )
            for boundary_name in Slab2D.unit_square(maxh=maxh).boundary_regions()
        )
        assert boundary_trace_norm_squared < 1.0e-24
        errors.append(observed)

    l2_rate = log(errors[-2][0] / errors[-1][0]) / log(_MESH_SIZES[-2] / _MESH_SIZES[-1])
    energy_rate = log(errors[-2][1] / errors[-1][1]) / log(_MESH_SIZES[-2] / _MESH_SIZES[-1])
    assert l2_rate >= polynomial_order + 0.8
    assert energy_rate >= polynomial_order - 0.2


def test_oblique_solution_reports_separate_parallel_and_perpendicular_energy() -> None:
    """The M4a diagnostics retain both positive contributions to the weak form."""
    exact, source = _diagnostic_data()
    solution = solve_isotropic_poisson(
        Slab2D.unit_square(maxh=0.0875),
        polynomial_order=4,
        source=source,
        conductivity=_CONDUCTIVITY,
    )

    diagnostics = solution.energy_diagnostics
    assert diagnostics.parallel > 0.0
    assert diagnostics.perpendicular > 0.0
    assert diagnostics.total == pytest.approx(diagnostics.parallel + diagnostics.perpendicular)
    bx, by = _CONDUCTIVITY.direction
    expected_parallel = _CONDUCTIVITY.parallel * ng.pi**2 * (bx**2 / 4.0 + by**2)
    expected_perpendicular = (
        _CONDUCTIVITY.perpendicular * ng.pi**2 * (5.0 / 4.0 - bx**2 / 4.0 - by**2)
    )
    assert diagnostics.parallel == pytest.approx(expected_parallel, rel=1.0e-5)
    assert diagnostics.perpendicular == pytest.approx(expected_perpendicular, rel=1.0e-5)
    assert ng.sqrt(ng.Integrate((solution._field - exact) ** 2, solution._mesh, order=8)) < 2.0e-5
