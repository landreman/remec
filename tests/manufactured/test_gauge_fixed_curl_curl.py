"""Manufactured magnetostatics tests for the gauge-fixed form of note (M1)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import log, pi
from pathlib import Path

import ngsolve as ng
import numpy as np
import pytest
from ngsolve.meshes import MakeStructured3DMesh

from remec.fem._magnetostatics import solve_gauge_fixed_curl_curl

_BASE_ORDERS = (1, 2, 3)
_SUBDIVISIONS = (2, 3, 4)
_TABLE_PATH = Path(__file__).with_name("gauge_fixed_curl_curl_rates.csv")


@dataclass(frozen=True, slots=True)
class _ManufacturedRow:
    base_order: int
    subdivisions: int
    vector_potential_l2_error: float
    magnetic_field_l2_error: float
    gauge_multiplier_l2_norm: float
    curl_projection_relative_defect: float
    magnetic_divergence_relative_norm: float
    boundary_normal_relative_norm: float
    free_dof_relative_residual: float
    gauge_constraint_relative_residual: float


def _manufactured_row(base_order: int, subdivisions: int) -> _ManufacturedRow:
    mesh = MakeStructured3DMesh(
        hexes=False,
        nx=subdivisions,
        ny=subdivisions,
        nz=subdivisions,
    )
    scalar = ng.sin(pi * ng.x) * ng.sin(pi * ng.y)
    exact_vector_potential = ng.CoefficientFunction((0.0, 0.0, scalar))
    exact_magnetic_field = ng.CoefficientFunction(
        (
            pi * ng.sin(pi * ng.x) * ng.cos(pi * ng.y),
            -pi * ng.cos(pi * ng.x) * ng.sin(pi * ng.y),
            0.0,
        )
    )
    current_density = ng.CoefficientFunction((0.0, 0.0, 2.0 * pi**2 * scalar))
    solution = solve_gauge_fixed_curl_curl(
        mesh,
        current_density,
        base_order=base_order,
        bonus_integration_order=10,
    )
    integration_order = 2 * base_order + 10
    vector_error = float(
        ng.sqrt(
            ng.Integrate(
                ng.InnerProduct(
                    solution.vector_potential - exact_vector_potential,
                    solution.vector_potential - exact_vector_potential,
                ),
                mesh,
                order=integration_order,
            )
        )
    )
    magnetic_error = float(
        ng.sqrt(
            ng.Integrate(
                ng.InnerProduct(
                    solution.magnetic_field - exact_magnetic_field,
                    solution.magnetic_field - exact_magnetic_field,
                ),
                mesh,
                order=integration_order,
            )
        )
    )
    return _ManufacturedRow(
        base_order=base_order,
        subdivisions=subdivisions,
        vector_potential_l2_error=vector_error,
        magnetic_field_l2_error=magnetic_error,
        gauge_multiplier_l2_norm=solution.gauge_multiplier_l2_norm,
        curl_projection_relative_defect=solution.curl_projection_relative_defect,
        magnetic_divergence_relative_norm=solution.magnetic_divergence_relative_norm,
        boundary_normal_relative_norm=solution.boundary_normal_relative_norm,
        free_dof_relative_residual=solution.free_dof_relative_residual,
        gauge_constraint_relative_residual=solution.gauge_constraint_relative_residual,
    )


def _recorded_rows() -> dict[tuple[int, int], dict[str, str]]:
    with _TABLE_PATH.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    indexed = {(int(row["base_order"]), int(row["subdivisions"])): row for row in rows}
    assert len(indexed) == len(rows), "verification table contains duplicate mesh/order rows"
    return indexed


@pytest.fixture(scope="module")
def manufactured_rows() -> dict[tuple[int, int], _ManufacturedRow]:
    """Share the nine mixed solves across convergence and invariant assertions."""
    return {
        (base_order, subdivisions): _manufactured_row(base_order, subdivisions)
        for base_order in _BASE_ORDERS
        for subdivisions in _SUBDIVISIONS
    }


@pytest.mark.parametrize("base_order", _BASE_ORDERS)
def test_manufactured_magnetostatics_converges_at_expected_orders(
    base_order: int,
    manufactured_rows: dict[tuple[int, int], _ManufacturedRow],
) -> None:
    r"""The mixed (M1) solve converges at HCurl L2 order p+1 and curl order p."""
    rows = [manufactured_rows[(base_order, subdivisions)] for subdivisions in _SUBDIVISIONS]
    vector_rate = log(
        rows[-2].vector_potential_l2_error / rows[-1].vector_potential_l2_error
    ) / log(_SUBDIVISIONS[-1] / _SUBDIVISIONS[-2])
    magnetic_rate = log(rows[-2].magnetic_field_l2_error / rows[-1].magnetic_field_l2_error) / log(
        _SUBDIVISIONS[-1] / _SUBDIVISIONS[-2]
    )
    assert vector_rate > base_order + 0.75
    assert magnetic_rate > base_order - 0.10


def test_manufactured_magnetostatics_preserves_m1_and_coulomb_gauge(
    manufactured_rows: dict[tuple[int, int], _ManufacturedRow],
) -> None:
    """Every manufactured row satisfies the algebraic, gauge, and div(B) invariants."""
    for row in manufactured_rows.values():
        roundoff_gate = 128.0 * np.finfo(float).eps * (row.base_order + 2) ** 3
        assert row.free_dof_relative_residual < 1.0e-11
        assert row.gauge_constraint_relative_residual < 1.0e-11
        assert row.gauge_multiplier_l2_norm < 1.0e-10
        assert row.curl_projection_relative_defect < roundoff_gate
        assert row.magnetic_divergence_relative_norm < roundoff_gate
        assert row.boundary_normal_relative_norm < roundoff_gate


def test_gauge_fixed_rate_table_matches_every_manufactured_row(
    manufactured_rows: dict[tuple[int, int], _ManufacturedRow],
) -> None:
    """The checked-in (M1) rate table covers and reproduces the full tested sweep."""
    recorded = _recorded_rows()
    assert set(recorded) == set(manufactured_rows)
    for key, row in manufactured_rows.items():
        table_row = recorded[key]
        assert int(table_row["elements"]) == 6 * row.subdivisions**3
        for column in (
            "vector_potential_l2_error",
            "magnetic_field_l2_error",
            "gauge_multiplier_l2_norm",
            "magnetic_divergence_relative_norm",
            "free_dof_relative_residual",
            "gauge_constraint_relative_residual",
        ):
            assert getattr(row, column) == pytest.approx(
                float(table_row[column]),
                rel=2.0e-9,
                abs=2.0e-15,
            )


def test_pure_gradient_current_is_removed_by_the_gauge_multiplier() -> None:
    r"""A discrete ``J=grad(phi)`` produces ``A=B=0`` and ``lambda=phi``.

    This is the gauge-null-space control: deleting either mixed coupling makes the
    system singular or leaves a conspicuous magnetic response.
    """
    mesh = MakeStructured3DMesh(hexes=False, nx=1, ny=1, nz=1)
    base_order = 3
    gauge_space = ng.H1(mesh, order=base_order + 1, dirichlet=".*")
    exact_multiplier = ng.GridFunction(gauge_space)
    rng = np.random.default_rng(4202)
    exact_multiplier.vec.FV().NumPy()[:] = rng.standard_normal(gauge_space.ndof)
    exact_multiplier.vec.data = ng.Projector(gauge_space.FreeDofs(), True) * exact_multiplier.vec

    solution = solve_gauge_fixed_curl_curl(
        mesh,
        ng.grad(exact_multiplier),
        base_order=base_order,
        bonus_integration_order=6,
    )
    multiplier_error = float(
        ng.sqrt(
            ng.Integrate(
                (solution.gauge_multiplier - exact_multiplier) ** 2,
                mesh,
                order=12,
            )
        )
    )
    magnetic_norm = float(
        ng.sqrt(
            ng.Integrate(
                ng.InnerProduct(solution.magnetic_field, solution.magnetic_field),
                mesh,
                order=12,
            )
        )
    )
    assert multiplier_error < 1.0e-11
    assert solution.gauge_multiplier_l2_norm > 1.0e-3
    assert magnetic_norm < 1.0e-11
    assert solution.free_dof_relative_residual < 1.0e-11
