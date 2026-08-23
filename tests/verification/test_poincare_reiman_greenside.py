"""Poincare verification on the Section 8.6 Reiman--Greenside field."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import cos, pi, sin, sqrt
from pathlib import Path

import ngsolve as ng
import numpy as np
import pytest
from ngsolve.meshes import MakeStructured3DMesh

from remec.diagnostics.poincare import (
    find_periodic_point,
    radial_excursion,
    recover_rotational_transform,
    separatrix_width_from_invariant,
    trace_poincare,
)
from remec.fem._field_evaluation import make_hdiv_field_evaluator

_T0 = 0.29
_T1 = 0.38
_RESONANCE_RADIUS = sqrt((0.5 - _T0) / _T1)
_TABLE_PATH = Path(__file__).with_name("poincare_reiman_greenside.csv")


def _recorded_row(epsilon_1: float) -> dict[str, str]:
    with _TABLE_PATH.open(newline="", encoding="utf-8") as stream:
        return next(
            row
            for row in csv.DictReader(stream)
            if float(row["epsilon_1"]) == pytest.approx(epsilon_1)
        )


@dataclass(frozen=True, slots=True)
class _ReimanGreensideField:
    """Independent Cartesian transcription of the Section 8.6 analytic field."""

    epsilon_1: float
    t0: float = _T0
    t1: float = _T1
    major_radius: float = 1.0

    def __call__(self, x: float, y: float, z: float) -> np.ndarray:
        radius = sqrt(x * x + y * y)
        theta = np.arctan2(y, x)
        phi = z / self.major_radius
        zeta = 2.0 * theta - phi
        radial = -2.0 * self.epsilon_1 * radius * sin(zeta) / self.major_radius
        poloidal = (
            radius
            * (self.t0 + self.t1 * radius * radius - 2.0 * self.epsilon_1 * cos(zeta))
            / self.major_radius
        )
        return np.array(
            [
                radial * cos(theta) - poloidal * sin(theta),
                radial * sin(theta) + poloidal * cos(theta),
                1.0,
            ]
        )

    def invariant(self, radius: float, theta: float) -> float:
        """Return the exactly conserved resonant Hamiltonian K at Phi=0."""
        psi_t = 0.5 * radius * radius
        return (
            (2.0 * self.t0 - 1.0) * psi_t
            + 2.0 * self.t1 * psi_t * psi_t
            - 4.0 * self.epsilon_1 * psi_t * cos(2.0 * theta)
        )


def test_integrable_trace_recovers_iota_and_zero_apparent_island_width() -> None:
    seeds = np.array([[0.2, 0.17], [0.5, -0.31], [0.8, 0.42]])
    trace = trace_poincare(_ReimanGreensideField(0.0), seeds, turns=32)

    expected_iota = _T0 + _T1 * seeds[:, 0] ** 2
    np.testing.assert_allclose(recover_rotational_transform(trace), expected_iota, atol=2.0e-10)
    assert np.max(radial_excursion(trace)) < 2.0e-10


@pytest.mark.parametrize("major_radius", [1.0, 2.5])
@pytest.mark.parametrize("epsilon_1", [1.0e-4, sqrt(1.0e-7), 1.0e-3])
def test_tracer_locates_o_x_points_and_conserved_k_width(
    epsilon_1: float, major_radius: float
) -> None:
    field = _ReimanGreensideField(epsilon_1, major_radius=major_radius)
    recorded = _recorded_row(epsilon_1)
    o_point = find_periodic_point(
        field,
        (_RESONANCE_RADIUS, 0.0),
        period_turns=2,
        poloidal_winding=1,
        major_radius=major_radius,
    )
    x_point = find_periodic_point(
        field,
        (_RESONANCE_RADIUS, 0.5 * pi),
        period_turns=2,
        poloidal_winding=1,
        major_radius=major_radius,
    )

    a_coefficient = 1.0 - 2.0 * _T0
    exact_o_radius = sqrt((a_coefficient + 4.0 * epsilon_1) / (2.0 * _T1))
    exact_x_radius = sqrt((a_coefficient - 4.0 * epsilon_1) / (2.0 * _T1))
    assert o_point.kind == "O"
    assert x_point.kind == "X"
    assert o_point.radius == pytest.approx(float(recorded["o_radius"]), abs=2.0e-9)
    assert x_point.radius == pytest.approx(float(recorded["x_radius"]), abs=2.0e-9)
    assert o_point.radius == pytest.approx(exact_o_radius, abs=2.0e-9)
    assert x_point.radius == pytest.approx(exact_x_radius, abs=2.0e-9)
    assert abs(o_point.theta) < 2.0e-9
    assert x_point.theta == pytest.approx(0.5 * pi, abs=2.0e-9)
    assert max(o_point.residual_norm, x_point.residual_norm) < 2.0e-10

    invariant_width = separatrix_width_from_invariant(
        field.invariant,
        level=field.invariant(x_point.radius, x_point.theta),
        cut_theta=o_point.theta,
        inner_bracket=(0.1, _RESONANCE_RADIUS),
        outer_bracket=(_RESONANCE_RADIUS, 0.99),
    )
    exact_width = 4.0 * sqrt(epsilon_1 / (2.0 * _T1))
    assert invariant_width == pytest.approx(float(recorded["invariant_width"]), abs=2.0e-9)
    assert exact_width == pytest.approx(float(recorded["closed_form_width"]), abs=2.0e-12)
    assert invariant_width == pytest.approx(exact_width, abs=2.0e-9)

    inner_separatrix_radius = sqrt(
        2.0 * (sqrt(a_coefficient) - 2.0 * sqrt(epsilon_1)) ** 2 / (4.0 * _T1)
    )
    near_separatrix = trace_poincare(
        field,
        np.array([[inner_separatrix_radius + 0.01 * exact_width, o_point.theta]]),
        turns=80,
        major_radius=major_radius,
    )
    traced_width = radial_excursion(near_separatrix)[0]
    assert 0.97 * invariant_width < traced_width < invariant_width


@pytest.mark.parametrize(
    ("subdivisions", "axial_field", "major_radius"),
    [(1, 1.0, 1.0), (2, 0.5, 2.5)],
)
def test_same_tracer_runs_on_an_hdiv_magnetic_field(
    subdivisions: int, axial_field: float, major_radius: float
) -> None:
    rotation = 0.37
    mesh = MakeStructured3DMesh(
        hexes=False,
        nx=subdivisions,
        ny=subdivisions,
        nz=subdivisions,
        mapping=lambda x, y, z: (
            2.0 * x - 1.0,
            2.0 * y - 1.0,
            2.0 * pi * major_radius * z,
        ),
    )
    space = ng.HDiv(mesh, order=1)
    magnetic_field = ng.GridFunction(space)
    magnetic_field.Set(ng.CoefficientFunction((-rotation * ng.y, rotation * ng.x, axial_field)))
    evaluator = make_hdiv_field_evaluator(
        mesh,
        magnetic_field,
        periodic_z_length=2.0 * pi * major_radius,
    )

    trace = trace_poincare(
        evaluator,
        np.array([[0.3, 0.2]]),
        turns=3,
        major_radius=major_radius,
    )

    assert float(ng.Integrate(ng.div(magnetic_field) ** 2, mesh, order=5)) < 1.0e-24
    expected_iota = major_radius * rotation / axial_field
    assert recover_rotational_transform(trace)[0] == pytest.approx(expected_iota, abs=2.0e-10)
    assert radial_excursion(trace)[0] < 2.0e-10
