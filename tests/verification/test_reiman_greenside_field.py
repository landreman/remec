"""Curved-mesh (M1) verification of the production Reiman--Greenside field."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import ngsolve as ng
import numpy as np
import pytest

from remec.diagnostics import make_hdiv_field_evaluator
from remec.diagnostics.poincare import recover_rotational_transform, trace_poincare
from remec.fem._reiman_greenside import (
    ReimanGreensideDiscreteField,
    build_reiman_greenside_discrete_field,
)
from remec.geometry import PeriodicCylinder3D
from remec.reiman_greenside import ReimanGreensideField

_TABLE_PATH = Path(__file__).with_name("reiman_greenside_m1.csv")


@dataclass(frozen=True, slots=True)
class _Row:
    order: int
    elements: int
    hcurl_dofs: int
    hdiv_dofs: int
    result: ReimanGreensideDiscreteField
    mesh: object
    periodic_length: float


@pytest.fixture(scope="module")
def m1_rows() -> dict[int, _Row]:
    model = ReimanGreensideField(epsilon_1=1.0e-3, epsilon_2=0.0)
    rows: dict[int, _Row] = {}
    for order in (1, 2, 3, 4):
        cylinder = PeriodicCylinder3D(geometry_order=min(order + 1, 4))
        bundle = cylinder.build_mesh()
        result = build_reiman_greenside_discrete_field(bundle._mesh, model, order=order)
        rows[order] = _Row(
            order=order,
            elements=bundle._mesh.ne,
            hcurl_dofs=result.vector_potential.space.ndof,
            hdiv_dofs=result.magnetic_field.space.ndof,
            result=result,
            mesh=bundle._mesh,
            periodic_length=cylinder.periodic_length,
        )
    return rows


def test_curved_periodic_de_rham_field_preserves_m1_at_roundoff(
    m1_rows: dict[int, _Row],
) -> None:
    r"""On every order, ``B_h=curl(A_h)`` has discrete ``div(B_h)=0`` for (M1)."""
    eps = np.finfo(float).eps
    for order, row in m1_rows.items():
        gate = 256.0 * eps * (order + 2) ** 3
        assert row.result.curl_projection_relative_defect < gate
        assert row.result.divergence_relative_norm < gate


def test_m1_analytic_field_error_decreases_systematically_with_order(
    m1_rows: dict[int, _Row],
) -> None:
    """The periodic HCurl/HDiv realization converges toward the independent analytic B."""
    errors = [m1_rows[order].result.analytic_field_relative_error for order in (1, 2, 3, 4)]
    for coarse, fine in pairwise(errors):
        assert coarse > 2.0 * fine
    assert errors[-1] < 1.0e-3


def test_bz_and_small_b_protection_are_inactive(m1_rows: dict[int, _Row]) -> None:
    """The exact field has ``B_z=1``, no null, and negligible smooth-floor activity."""
    for row in m1_rows.values():
        bz_error = float(
            ng.sqrt(ng.Integrate((row.result.analytic_magnetic_field[2] - 1.0) ** 2, row.mesh))
        )
        assert bz_error < 1.0e-14
        assert row.result.sampled_magnetic_magnitude_minimum >= 1.0
        assert row.result.b_floor_relative_activity < 2.0e-16


def test_production_analytic_and_interpolated_fields_run_through_tracer(
    m1_rows: dict[int, _Row],
) -> None:
    """Both public analytic and solver-facing H(div) fields use the milestone-6.1 tracer."""
    model = ReimanGreensideField()
    seed = np.array([[0.35, 0.2]])
    analytic = trace_poincare(model, seed, turns=3)
    expected = model.rotational_transform(seed[0, 0])
    assert recover_rotational_transform(analytic)[0] == pytest.approx(expected, abs=2.0e-10)

    island_model = ReimanGreensideField(epsilon_1=1.0e-3)
    island_analytic = trace_poincare(island_model, seed, turns=3)
    island_transform = recover_rotational_transform(island_analytic)[0]
    row = m1_rows[4]
    evaluator = make_hdiv_field_evaluator(
        row.mesh,
        row.result.magnetic_field,
        periodic_z_length=row.periodic_length,
    )
    with pytest.raises(ValueError, match="mesh axial extent"):
        make_hdiv_field_evaluator(
            row.mesh,
            row.result.magnetic_field,
            periodic_z_length=0.9 * row.periodic_length,
        )
    interpolated = trace_poincare(evaluator, seed, turns=3)
    assert recover_rotational_transform(interpolated)[0] == pytest.approx(
        island_transform, abs=8.0e-4
    )


def test_m1_order_scan_matches_checked_in_table(m1_rows: dict[int, _Row]) -> None:
    with _TABLE_PATH.open(newline="", encoding="utf-8") as stream:
        recorded = {
            int(row["order"]): row
            for row in csv.DictReader(stream)
            if row["platform"] == sys.platform
        }
    measured = {
        order: (
            actual.elements,
            actual.hcurl_dofs,
            actual.hdiv_dofs,
            actual.result.curl_projection_relative_defect,
            actual.result.divergence_relative_norm,
            actual.result.analytic_field_relative_error,
            actual.result.sampled_magnetic_magnitude_minimum,
            actual.result.sampled_magnetic_magnitude_maximum,
            actual.result.b_floor_relative_activity,
        )
        for order, actual in m1_rows.items()
    }
    assert set(recorded) == set(m1_rows), f"missing {sys.platform} M1 rows: {measured!r}"
    for order, actual in m1_rows.items():
        row = recorded[order]
        assert int(row["elements"]) == actual.elements
        assert int(row["hcurl_dofs"]) == actual.hcurl_dofs
        assert int(row["hdiv_dofs"]) == actual.hdiv_dofs
        for column in (
            "curl_projection_relative_defect",
            "divergence_relative_norm",
            "analytic_field_relative_error",
            "sampled_magnetic_magnitude_minimum",
            "sampled_magnetic_magnitude_maximum",
            "b_floor_relative_activity",
        ):
            assert getattr(actual.result, column) == pytest.approx(float(row[column]), rel=2.0e-8)
