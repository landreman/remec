"""Manufactured convergence test for the isotropic reduction of (M4a)."""

from __future__ import annotations

import csv
from math import log
from pathlib import Path

import ngsolve as ng
import pytest

from remec.fem._isotropic_poisson import ObliqueConductivity, solve_isotropic_poisson
from remec.geometry.slab import Slab2D

_MESH_SIZES = (0.35, 0.175, 0.0875)
_RATE_TABLE = Path(__file__).with_name("isotropic_poisson_rates.csv")
_ISOTROPIC_CONDUCTIVITY = ObliqueConductivity(1.0, 1.0, (1.0, 0.0))


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


def test_unit_square_boundary_names_match_boundary_locations() -> None:
    """`Slab2D` preserves its named boundary contract for future BC variants."""
    mesh = Slab2D.unit_square(maxh=0.35).build_mesh()._mesh

    assert mesh.GetBoundaries() == ("bottom", "right", "top", "left")
    assert ng.Integrate(ng.y, mesh, definedon=mesh.Boundaries("bottom")) == pytest.approx(0.0)
    assert ng.Integrate(ng.y, mesh, definedon=mesh.Boundaries("top")) == pytest.approx(1.0)
    assert ng.Integrate(ng.x, mesh, definedon=mesh.Boundaries("left")) == pytest.approx(0.0)
    assert ng.Integrate(ng.x, mesh, definedon=mesh.Boundaries("right")) == pytest.approx(1.0)


@pytest.mark.parametrize("polynomial_order", [1, 2, 3])
def test_isotropic_poisson_manufactured_convergence(polynomial_order: int) -> None:
    """M4a has L² order p+1 and energy order p for a smooth slab solution.

    The manufactured solution is ``χ = sin(πx) sin(πy)`` on ``[0, 1]²`` with
    zero Dirichlet boundary data and ``S_ref = 2π²χ``.  For isotropic unit
    conductivity, (M4a) reduces to ``-Δχ = S_ref``.
    """
    exact = ng.sin(ng.pi * ng.x) * ng.sin(ng.pi * ng.y)
    exact_gradient = ng.CoefficientFunction(
        (
            ng.pi * ng.cos(ng.pi * ng.x) * ng.sin(ng.pi * ng.y),
            ng.pi * ng.sin(ng.pi * ng.x) * ng.cos(ng.pi * ng.y),
        )
    )
    source = 2.0 * ng.pi**2 * exact
    errors: list[tuple[float, float]] = []
    recorded_errors = _recorded_errors()

    for maxh in _MESH_SIZES:
        solution = solve_isotropic_poisson(
            Slab2D.unit_square(maxh=maxh),
            polynomial_order=polynomial_order,
            source=source,
            conductivity=_ISOTROPIC_CONDUCTIVITY,
        )
        l2_error = ng.sqrt(ng.Integrate((solution._field - exact) ** 2, solution._mesh, order=8))
        energy_error = ng.sqrt(
            ng.Integrate(
                ng.InnerProduct(
                    ng.grad(solution._field) - exact_gradient,
                    ng.grad(solution._field) - exact_gradient,
                ),
                solution._mesh,
                order=8,
            )
        )
        observed = (float(l2_error), float(energy_error))
        expected = recorded_errors[polynomial_order, maxh]
        assert observed[0] == pytest.approx(expected[0], rel=0.05, abs=1.0e-12)
        assert observed[1] == pytest.approx(expected[1], rel=0.05, abs=1.0e-12)
        boundary_trace_norm_squared = sum(
            ng.Integrate(
                solution._field**2,
                solution._mesh,
                definedon=solution._mesh.Boundaries(boundary_name),
            )
            for boundary_name in Slab2D.unit_square(maxh=maxh).boundary_regions()
        )
        assert boundary_trace_norm_squared < 1.0e-24
        assert solution.free_dof_residual_norm < 1.0e-10
        errors.append(observed)

    l2_rate = log(errors[-2][0] / errors[-1][0]) / log(_MESH_SIZES[-2] / _MESH_SIZES[-1])
    energy_rate = log(errors[-2][1] / errors[-1][1]) / log(_MESH_SIZES[-2] / _MESH_SIZES[-1])
    assert l2_rate >= polynomial_order + 0.8
    assert energy_rate >= polynomial_order - 0.2
