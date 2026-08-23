"""Curved-mesh (M1) verification of the production Reiman--Greenside field."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from itertools import pairwise
from math import log, pi
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
_H_TABLE_PATH = Path(__file__).with_name("reiman_greenside_m1_h_rates.csv")


@dataclass(frozen=True, slots=True)
class _Row:
    order: int
    elements: int
    hcurl_dofs: int
    hdiv_dofs: int
    result: ReimanGreensideDiscreteField
    mesh: object
    periodic_length: float


def _build_row(order: int) -> _Row:
    model = ReimanGreensideField(epsilon_1=1.0e-3, epsilon_2=0.0)
    cylinder = PeriodicCylinder3D(geometry_order=min(order + 1, 4))
    bundle = cylinder.build_mesh()
    result = build_reiman_greenside_discrete_field(
        bundle._mesh,
        model,
        order=order,
        harmonic_field=cylinder.harmonic_basis(bundle)[0],
        axial_flux=pi * cylinder.radius**2,
    )
    return _Row(
        order=order,
        elements=bundle._mesh.ne,
        hcurl_dofs=result.vector_potential.space.ndof,
        hdiv_dofs=result.magnetic_field.space.ndof,
        result=result,
        mesh=bundle._mesh,
        periodic_length=cylinder.periodic_length,
    )


@pytest.fixture(scope="module")
def m1_row() -> _Row:
    """Keep one live ADR-0010 production reconstruction in every fast run."""
    return _build_row(1)


@pytest.fixture(scope="module")
def m1_rows(m1_row: _Row) -> dict[int, _Row]:
    """Share the wider p ladder across developer-slow verification tests."""
    rows = {1: m1_row}
    for order in (2, 3, 4):
        rows[order] = _build_row(order)
    return rows


def test_curved_periodic_de_rham_field_preserves_m1_at_roundoff(
    m1_row: _Row,
) -> None:
    r"""The fast reconstruction has ``B_h=curl(A_h)`` and ``div(B_h)=0`` for (M1)."""
    eps = np.finfo(float).eps
    gate = 128.0 * eps * (m1_row.order + 2) ** 3
    assert m1_row.result.curl_projection_relative_defect < gate
    assert m1_row.result.divergence_relative_norm < gate
    assert m1_row.result.gauge_constraint_relative_residual < 1.0e-10
    assert m1_row.result.harmonic_constraint_relative_residual < 1.0e-10
    assert m1_row.result.target_axial_flux == pytest.approx(pi, abs=2.0e-10)
    assert m1_row.result.reconstructed_axial_flux == pytest.approx(pi, abs=2.0e-10)


@pytest.mark.slow
def test_m1_analytic_field_error_decreases_systematically_with_order(
    m1_rows: dict[int, _Row],
) -> None:
    """The periodic HCurl/HDiv realization converges toward the independent analytic B."""
    errors = [m1_rows[order].result.analytic_field_relative_error for order in (1, 2, 3, 4)]
    for coarse, fine in pairwise(errors):
        assert coarse > 2.0 * fine
    assert errors[-1] < 1.0e-3
    finest = m1_rows[4]
    assert finest.result.bz_l2_error < 1.0e-3
    assert finest.result.sampled_magnetic_magnitude_minimum > 0.65
    assert finest.result.sampled_magnetic_magnitude_maximum < 2.0


def test_bz_and_small_b_protection_are_inactive(m1_row: _Row) -> None:
    """The live reconstructed field has non-null B and an inactive smooth floor."""
    assert m1_row.result.bz_l2_error < 0.5
    assert m1_row.result.sampled_magnetic_magnitude_minimum > 0.25
    assert m1_row.result.sampled_magnetic_magnitude_maximum < 100.0
    assert m1_row.result.b_floor_relative_activity < 1.0e-15


def test_wrong_axial_flux_control_is_detected() -> None:
    """The ADR-0010 flux constraint cannot silently discard or replace net flux."""
    cylinder = PeriodicCylinder3D(geometry_order=2)
    bundle = cylinder.build_mesh()
    result = build_reiman_greenside_discrete_field(
        bundle._mesh,
        ReimanGreensideField(epsilon_1=1.0e-3),
        order=1,
        harmonic_field=cylinder.harmonic_basis(bundle)[0],
        axial_flux=0.5 * pi * cylinder.radius**2,
    )
    assert result.reconstructed_axial_flux == pytest.approx(0.5 * pi, abs=2.0e-10)
    assert abs(result.reconstructed_axial_flux - pi) > 1.0


def test_reference_field_reconstruction_has_nominal_order_one_h_rate() -> None:
    """The ADR-0010 production reconstruction reaches the nominal (M1) curl rate."""
    model = ReimanGreensideField(epsilon_1=1.0e-3)
    measured: list[tuple[float, int, float, float]] = []
    for maxh in (0.6, 0.45, 0.35):
        cylinder = PeriodicCylinder3D(
            max_element_size=maxh,
            geometry_order=4,
        )
        bundle = cylinder.build_mesh()
        result = build_reiman_greenside_discrete_field(
            bundle._mesh,
            model,
            order=1,
            harmonic_field=cylinder.harmonic_basis(bundle)[0],
            axial_flux=pi * cylinder.radius**2,
        )
        volume = float(ng.Integrate(1.0, bundle._mesh, order=14))
        h_eff = (volume / bundle._mesh.ne) ** (1.0 / 3.0)
        measured.append((maxh, bundle._mesh.ne, h_eff, result.analytic_field_relative_error))
    rates = [
        log(coarse[3] / fine[3]) / log(coarse[2] / fine[2]) for coarse, fine in pairwise(measured)
    ]
    assert rates[-1] > 0.9, (measured, rates)

    with _H_TABLE_PATH.open(newline="", encoding="utf-8") as stream:
        recorded = [row for row in csv.DictReader(stream) if row["platform"] == sys.platform]
    assert len(recorded) == len(measured), (
        f"missing {sys.platform} reference-field h rows: {measured!r}, rates={rates!r}"
    )
    for index, (row, actual) in enumerate(zip(recorded, measured, strict=True)):
        maxh, elements, h_eff, error = actual
        assert float(row["max_element_size"]) == maxh
        assert int(row["elements"]) == elements
        assert float(row["h_eff"]) == pytest.approx(h_eff, rel=2.0e-8)
        assert float(row["relative_b_error"]) == pytest.approx(error, rel=2.0e-8)
        if index:
            assert float(row["pair_rate"]) == pytest.approx(rates[index - 1], rel=2.0e-8)


@pytest.mark.slow
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


@pytest.mark.slow
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
            actual.result.bz_l2_error,
            actual.result.sampled_magnetic_magnitude_minimum,
            actual.result.sampled_magnetic_magnitude_maximum,
            actual.result.b_floor_relative_activity,
            actual.result.target_axial_flux,
            actual.result.reconstructed_axial_flux,
            actual.result.gauge_constraint_relative_residual,
            actual.result.harmonic_constraint_relative_residual,
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
            "bz_l2_error",
            "sampled_magnetic_magnitude_minimum",
            "sampled_magnetic_magnitude_maximum",
            "b_floor_relative_activity",
            "target_axial_flux",
            "reconstructed_axial_flux",
            "gauge_constraint_relative_residual",
            "harmonic_constraint_relative_residual",
        ):
            assert getattr(actual.result, column) == pytest.approx(float(row[column]), rel=2.0e-8)
