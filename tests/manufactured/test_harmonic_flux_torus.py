"""Analytic-torus verification for the harmonic part of note equation (M1)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import ngsolve as ng
import numpy as np
import pytest

from remec.fem._harmonic_flux import (
    build_analytic_torus_harmonic_field,
    poloidal_cut_flux,
)
from remec.geometry import AnalyticSolidTorus

_GEOMETRY_ORDERS = (1, 2, 3, 4)
_TABLE_PATH = Path(__file__).with_name("harmonic_flux_torus.csv")


@dataclass(frozen=True, slots=True)
class _HarmonicRow:
    geometry_order: int
    elements: int
    curved_volume: float
    weak_curl_relative_residual: float
    weak_divergence_relative_residual: float
    boundary_normal_relative_norm: float
    toroidal_flux: float
    sampled_magnetic_magnitude_minimum: float
    sampled_magnetic_magnitude_maximum: float


def _measured_row(geometry_order: int) -> _HarmonicRow:
    torus = AnalyticSolidTorus(
        major_radius=2.0,
        minor_radius=0.6,
        max_element_size=1.2,
        geometry_order=geometry_order,
    )
    mesh = torus.mesh()
    solution = build_analytic_torus_harmonic_field(mesh, torus, test_order=2)
    return _HarmonicRow(
        geometry_order=geometry_order,
        elements=mesh.ne,
        curved_volume=float(ng.Integrate(1.0, mesh, order=12)),
        weak_curl_relative_residual=solution.weak_curl_relative_residual,
        weak_divergence_relative_residual=solution.weak_divergence_relative_residual,
        boundary_normal_relative_norm=solution.boundary_normal_relative_norm,
        toroidal_flux=poloidal_cut_flux(
            torus,
            solution.poloidal_normal_component,
            quadrature_order=24,
        ),
        sampled_magnetic_magnitude_minimum=solution.sampled_magnetic_magnitude_minimum,
        sampled_magnetic_magnitude_maximum=solution.sampled_magnetic_magnitude_maximum,
    )


def _recorded_rows() -> dict[int, dict[str, str]]:
    with _TABLE_PATH.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    indexed = {int(row["geometry_order"]): row for row in rows}
    assert len(indexed) == len(rows), "verification table contains duplicate geometry orders"
    return indexed


def _actual_csv(rows: dict[int, _HarmonicRow]) -> str:
    """Format all measured rows so a cross-platform mismatch is reproducible."""
    header = (
        "geometry_order,elements,curved_volume,weak_curl_relative_residual,"
        "weak_divergence_relative_residual,boundary_normal_relative_norm,toroidal_flux,"
        "sampled_magnetic_magnitude_minimum,sampled_magnetic_magnitude_maximum"
    )
    body = [
        ",".join(
            (
                str(row.geometry_order),
                str(row.elements),
                f"{row.curved_volume:.16e}",
                f"{row.weak_curl_relative_residual:.16e}",
                f"{row.weak_divergence_relative_residual:.16e}",
                f"{row.boundary_normal_relative_norm:.16e}",
                f"{row.toroidal_flux:.16e}",
                f"{row.sampled_magnetic_magnitude_minimum:.16e}",
                f"{row.sampled_magnetic_magnitude_maximum:.16e}",
            )
        )
        for row in rows.values()
    ]
    return "\n".join((header, *body))


@pytest.fixture(scope="module")
def harmonic_rows() -> dict[int, _HarmonicRow]:
    """Share the curved-torus constructions across all acceptance assertions."""
    return {order: _measured_row(order) for order in _GEOMETRY_ORDERS}


def test_harmonic_field_satisfies_m1_and_has_unit_toroidal_flux(
    harmonic_rows: dict[int, _HarmonicRow],
) -> None:
    r"""The harmonic ``B_h`` preserves (M1), tangency, and normalized flux."""
    for row in harmonic_rows.values():
        roundoff_gate = 128.0 * np.finfo(float).eps * (row.geometry_order + 2) ** 3
        assert row.weak_curl_relative_residual < roundoff_gate
        assert row.weak_divergence_relative_residual < roundoff_gate
        assert row.toroidal_flux == pytest.approx(1.0, abs=1.0e-10)
        assert 0.65 < row.sampled_magnetic_magnitude_minimum < 0.68
        assert 1.22 < row.sampled_magnetic_magnitude_maximum < 1.25

    assert harmonic_rows[4].boundary_normal_relative_norm < 1.2e-4
    assert all(
        harmonic_rows[coarse].boundary_normal_relative_norm
        > harmonic_rows[fine].boundary_normal_relative_norm
        for coarse, fine in pairwise(_GEOMETRY_ORDERS)
    )
    exact_volume = 2.0 * np.pi**2 * 2.0 * 0.6**2
    assert all(
        abs(harmonic_rows[coarse].curved_volume - exact_volume)
        > abs(harmonic_rows[fine].curved_volume - exact_volume)
        for coarse, fine in pairwise(_GEOMETRY_ORDERS)
    )
    assert harmonic_rows[4].curved_volume == pytest.approx(exact_volume, rel=2.0e-6)


def test_curl_field_carries_zero_toroidal_flux() -> None:
    r"""A curl with zero tangential boundary trace cannot alter the (M1) flux."""
    torus = AnalyticSolidTorus(
        major_radius=2.0,
        minor_radius=0.6,
        max_element_size=1.2,
        geometry_order=3,
    )
    mesh = torus.mesh()
    cylindrical_radius = ng.sqrt(ng.x**2 + ng.y**2)
    boundary_factor = torus.minor_radius**2 - (
        (cylindrical_radius - torus.major_radius) ** 2 + ng.z**2
    )
    scalar_potential = boundary_factor * (1.0 + 0.2 * ng.x + 0.1 * ng.z)
    # A=0 on the analytic torus boundary, hence its tangential trace is zero.
    vector_potential = ng.CoefficientFunction((0.0, 0.0, scalar_potential))
    boundary_trace_norm = float(
        ng.sqrt(
            ng.Integrate(
                ng.InnerProduct(vector_potential, vector_potential),
                mesh,
                ng.BND,
                order=14,
            )
        )
    )
    assert boundary_trace_norm < 2.0e-3
    vector_space = ng.HCurl(mesh, order=2, dirichlet=".*")
    discrete_potential = ng.GridFunction(vector_space)
    discrete_potential.Set(vector_potential)
    discrete_potential.vec.data = (
        ng.Projector(vector_space.FreeDofs(), True) * discrete_potential.vec
    )
    normal = ng.specialcf.normal(3)
    tangential_trace_norm = float(
        ng.sqrt(
            ng.Integrate(
                ng.InnerProduct(
                    ng.Cross(normal, discrete_potential),
                    ng.Cross(normal, discrete_potential),
                ),
                mesh,
                ng.BND,
                order=14,
            )
        )
    )
    assert tangential_trace_norm < 1.0e-14

    def curl_normal_component(x_coordinate: np.ndarray, z_coordinate: np.ndarray) -> np.ndarray:
        radial_offset = x_coordinate - torus.major_radius
        boundary = torus.minor_radius**2 - (radial_offset**2 + z_coordinate**2)
        multiplier = 1.0 + 0.2 * x_coordinate + 0.1 * z_coordinate
        # At y=0 and x>0, (curl A).e_y = -partial_x A_z.
        return 2.0 * radial_offset * multiplier - 0.2 * boundary

    assert poloidal_cut_flux(
        torus,
        curl_normal_component,
        quadrature_order=24,
    ) == pytest.approx(
        0.0,
        abs=1.0e-10,
    )


def test_harmonic_flux_table_matches_every_geometry_order(
    harmonic_rows: dict[int, _HarmonicRow],
) -> None:
    """The machine-readable table covers and reproduces the full curvature sweep."""
    recorded = _recorded_rows()
    assert set(recorded) == set(harmonic_rows)
    mismatch_report = "measured platform table:\n" + _actual_csv(harmonic_rows)
    for order, row in harmonic_rows.items():
        table_row = recorded[order]
        assert int(table_row["elements"]) == row.elements, mismatch_report
        assert row.curved_volume == pytest.approx(float(table_row["curved_volume"]), abs=2e-8)
        assert row.toroidal_flux == pytest.approx(float(table_row["toroidal_flux"]), abs=2e-11)
        for column in (
            "weak_curl_relative_residual",
            "weak_divergence_relative_residual",
            "boundary_normal_relative_norm",
        ):
            recorded_value = float(table_row[column])
            assert getattr(row, column) <= max(
                8.0 * recorded_value,
                64.0 * np.finfo(float).eps,
            )
        for column in (
            "sampled_magnetic_magnitude_minimum",
            "sampled_magnetic_magnitude_maximum",
        ):
            assert getattr(row, column) == pytest.approx(
                float(table_row[column]),
                rel=2.0e-6,
            )
