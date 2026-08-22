"""Manufactured acceptance tests for the compatible ``(M1)`` current projection."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import ngsolve as ng
import numpy as np
import pytest
from netgen.occ import OCCGeometry, Pnt, Sphere
from ngsolve.meshes import MakeStructured3DMesh

from remec.fem._current_projection import (
    CurrentMomentConstraint,
    CurrentProjectionSolution,
    _verification_compact_moment_matched_heaviside,
    analyze_divergence_constraint_rank,
    solve_constrained_current_projection,
    verification_mollified_shell_moment_weights,
)
from remec.fem._magnetostatics import solve_gauge_fixed_curl_curl
from remec.geometry import AnalyticSolidTorus
from remec.level_set import compact_moment_matched_heaviside

_TABLE_PATH = Path(__file__).with_name("current_projection_rates.csv")
_AFFINE_SWEEP = ((2, 1), (2, 2), (2, 4), (2, 8), (3, 1), (3, 2), (3, 4))


@dataclass(frozen=True, slots=True)
class _ProjectionRow:
    base_order: int
    subdivisions: int
    elements: int
    projection_correction_relative_norm: float
    post_projection_divergence_relative_norm: float
    ampere_compatibility_relative_residual: float
    continuity_multiplier_l2_norm: float


def _cube_current() -> Any:
    r"""Return a smooth tangent current satisfying ``div(J_raw)=0`` exactly."""
    potential = ng.sin(np.pi * ng.x) * ng.sin(np.pi * ng.y) * ng.sin(np.pi * ng.z)
    return ng.CoefficientFunction(
        (
            np.pi * ng.cos(np.pi * ng.y) * ng.sin(np.pi * ng.x) * ng.sin(np.pi * ng.z),
            -np.pi * ng.cos(np.pi * ng.x) * ng.sin(np.pi * ng.y) * ng.sin(np.pi * ng.z),
            0.0 * potential,
        )
    )


def _projection_row(base_order: int, subdivisions: int) -> _ProjectionRow:
    mesh = MakeStructured3DMesh(
        hexes=False,
        nx=subdivisions,
        ny=subdivisions,
        nz=subdivisions,
    )
    solution = solve_constrained_current_projection(
        mesh,
        _cube_current(),
        base_order=base_order,
        raw_divergence=ng.CoefficientFunction(0.0),
        bonus_integration_order=10,
    )
    return _ProjectionRow(
        base_order=base_order,
        subdivisions=subdivisions,
        elements=mesh.ne,
        projection_correction_relative_norm=solution.projection_correction_relative_norm,
        post_projection_divergence_relative_norm=solution.post_projection_divergence_relative_norm,
        ampere_compatibility_relative_residual=solution.ampere_compatibility_relative_residual,
        continuity_multiplier_l2_norm=solution.continuity_multiplier_l2_norm,
    )


@pytest.fixture(scope="module")
def affine_rows() -> dict[tuple[int, int], _ProjectionRow]:
    """Share the h/p projection sweep across the convergence and table assertions."""
    return {key: _projection_row(*key) for key in _AFFINE_SWEEP}


@pytest.fixture(scope="module")
def curved_ball_projection() -> tuple[Any, CurrentProjectionSolution]:
    """Project one non-solenoidal raw current on the established curved OCC ball."""
    mesh = ng.Mesh(OCCGeometry(Sphere(Pnt(0.0, 0.0, 0.0), 1.0)).GenerateMesh(maxh=0.9))
    mesh.Curve(3)
    solenoidal = ng.CoefficientFunction((-ng.y, ng.x, 0.0))
    divergent = ng.CoefficientFunction((ng.x, ng.y, ng.z))
    solution = solve_constrained_current_projection(
        mesh,
        solenoidal + 0.15 * divergent,
        base_order=3,
        raw_divergence=ng.CoefficientFunction(0.45),
        bonus_integration_order=10,
    )
    return mesh, solution


def _recorded_rows() -> dict[tuple[int, int], dict[str, str]]:
    with _TABLE_PATH.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    indexed = {(int(row["base_order"]), int(row["subdivisions"])): row for row in rows}
    assert len(indexed) == len(rows), "verification table contains duplicate p/h rows"
    return indexed


def _roundoff_gate(base_order: int, *, curved: bool) -> float:
    factor = 128.0 if curved else 32.0
    return float(factor * np.finfo(float).eps * (base_order + 2) ** 3)


def test_affine_projection_correction_converges_and_current_is_exactly_continuous(
    affine_rows: dict[tuple[int, int], _ProjectionRow],
) -> None:
    r"""The Section-10 correction converges while ``div(J_h)=0`` stays at roundoff."""
    for base_order, subdivisions in _AFFINE_SWEEP:
        row = affine_rows[(base_order, subdivisions)]
        assert row.post_projection_divergence_relative_norm < _roundoff_gate(
            base_order, curved=False
        )
        assert row.ampere_compatibility_relative_residual < 2.0e-11

    degree_two_errors = [
        affine_rows[(2, subdivisions)].projection_correction_relative_norm
        for subdivisions in (1, 2, 4, 8)
    ]
    degree_three_errors = [
        affine_rows[(3, subdivisions)].projection_correction_relative_norm
        for subdivisions in (1, 2, 4)
    ]
    degree_two_rates = [
        np.log(coarse / fine) / np.log(2.0) for coarse, fine in pairwise(degree_two_errors)
    ]
    degree_three_rates = [
        np.log(coarse / fine) / np.log(2.0) for coarse, fine in pairwise(degree_three_errors)
    ]
    assert degree_two_rates[-1] > 1.8
    assert degree_three_rates[-1] > 2.7
    assert (
        affine_rows[(3, 2)].projection_correction_relative_norm
        < affine_rows[(2, 2)].projection_correction_relative_norm
    )


def test_affine_projection_table_reproduces_every_h_p_row(
    affine_rows: dict[tuple[int, int], _ProjectionRow],
) -> None:
    """The machine-readable table pins every measured convergence row."""
    recorded = _recorded_rows()
    assert set(recorded) == set(affine_rows)
    for key, row in affine_rows.items():
        expected = recorded[key]
        assert row.elements == int(expected["elements"])
        for column in (
            "projection_correction_relative_norm",
            "post_projection_divergence_relative_norm",
            "ampere_compatibility_relative_residual",
            "continuity_multiplier_l2_norm",
        ):
            actual = getattr(row, column)
            reference = float(expected[column])
            assert actual <= max(8.0 * reference, 64.0 * np.finfo(float).eps)
            if column in (
                "projection_correction_relative_norm",
                "continuity_multiplier_l2_norm",
            ):
                assert actual == pytest.approx(reference, rel=2.0e-5, abs=1.0e-12)


def test_curved_ball_pairing_and_ampere_compatibility(
    curved_ball_projection: tuple[Any, CurrentProjectionSolution],
) -> None:
    r"""The paired curved projection makes ``(M1)`` integrable at roundoff."""
    mesh, solution = curved_ball_projection
    assert solution.pre_projection_divergence_relative_norm > 0.2
    assert solution.post_projection_divergence_relative_norm < _roundoff_gate(3, curved=True)
    assert solution.ampere_compatibility_relative_residual < 2.0e-11
    assert solution.continuity_multiplier_l2_norm > 1.0e-2
    assert solution.continuity_multiplier_relative_norm > 1.0e-2
    assert solution.free_dof_relative_residual < 1.0e-11

    magnetic = solve_gauge_fixed_curl_curl(
        mesh,
        solution.current_density,
        base_order=3,
        bonus_integration_order=10,
    )
    assert magnetic.gauge_multiplier_l2_norm < 1.0e-10
    assert magnetic.gauge_constraint_relative_residual < 1.0e-11
    assert magnetic.magnetic_divergence_relative_norm < _roundoff_gate(3, curved=True)


def test_curved_ball_wrong_terminal_orders_are_falsifiable(
    curved_ball_projection: tuple[Any, CurrentProjectionSolution],
) -> None:
    """Undersizing leaves divergence; oversizing adds numerically redundant rows."""
    mesh, paired_solution = curved_ball_projection
    raw_current = ng.CoefficientFunction((-ng.y + 0.15 * ng.x, ng.x + 0.15 * ng.y, 0.15 * ng.z))
    undersized = solve_constrained_current_projection(
        mesh,
        raw_current,
        base_order=3,
        raw_divergence=ng.CoefficientFunction(0.45),
        terminal_order=0,
        bonus_integration_order=10,
    )
    assert paired_solution.terminal_order == 1
    assert undersized.post_projection_divergence_relative_norm > 1.0e-2

    paired_rank = analyze_divergence_constraint_rank(
        mesh,
        hdiv_order=2,
        terminal_order=1,
    )
    oversized_rank = analyze_divergence_constraint_rank(
        mesh,
        hdiv_order=2,
        terminal_order=2,
    )
    assert paired_rank.rows == paired_rank.rank == 428
    assert paired_rank.retained_singular_value_ratio > 1.0e-10
    assert oversized_rank.rows == 1070
    assert oversized_rank.rank == 428
    assert oversized_rank.first_discarded_singular_value_ratio < 1.0e-10


def test_verification_mollifier_matches_the_shared_moment_matched_kernel() -> None:
    r"""The verification-only ``(M3b)`` copy matches shared ``(mollified_V)`` exactly."""
    mesh = MakeStructured3DMesh(hexes=False, nx=1, ny=1, nz=1)
    point = mesh(0.25, 0.25, 0.25)
    arguments = np.linspace(-1.5, 1.5, 13)
    actual = np.asarray(
        [
            float(
                _verification_compact_moment_matched_heaviside(
                    ng.CoefficientFunction(float(argument))
                )(point)
            )
            for argument in arguments
        ]
    )
    expected = compact_moment_matched_heaviside(arguments)
    assert actual == pytest.approx(expected, abs=4.0 * np.finfo(float).eps)


def test_curved_torus_projection_preserves_m3b_shell_moments_and_pairing() -> None:
    r"""The current passed to ``(M1)`` retains independent mollified ``(M3b)`` rows."""
    torus = AnalyticSolidTorus(
        major_radius=2.0,
        minor_radius=0.6,
        max_element_size=1.2,
        geometry_order=4,
    )
    mesh_bundle = torus.build_mesh()
    mesh = mesh_bundle._mesh
    cylindrical_radius = ng.sqrt(ng.x**2 + ng.y**2)
    normalized_volume = (
        (cylindrical_radius - torus.major_radius) ** 2 + ng.z**2
    ) / torus.minor_radius**2
    toroidal_angle_gradient = ng.CoefficientFunction(
        (-ng.y / cylindrical_radius**2, ng.x / cylindrical_radius**2, 0.0)
    )
    shell_edges = (0.0, 0.5, 1.0)
    weights = verification_mollified_shell_moment_weights(
        normalized_volume,
        toroidal_angle_gradient,
        shell_edges,
        mollifier_width=0.08,
    )
    harmonic = torus.harmonic_basis(mesh_bundle)[0]
    # The added poloidal gradient is deliberately divergent but has zero toroidal moment.
    poloidal_gradient = ng.CoefficientFunction(
        (
            2.0 * (cylindrical_radius - torus.major_radius) * ng.x / cylindrical_radius,
            2.0 * (cylindrical_radius - torus.major_radius) * ng.y / cylindrical_radius,
            2.0 * ng.z,
        )
    )
    raw_current = (1.0 + 0.01 * normalized_volume) * harmonic + 0.04 * poloidal_gradient
    targets = tuple(
        float(ng.Integrate(ng.InnerProduct(harmonic, weight), mesh, order=14)) for weight in weights
    )
    # Independent circular-torus anchors catch shell exchange and normalization errors.
    inner_minor_radius = torus.minor_radius * np.sqrt(shell_edges[1])
    inner_fraction = (
        torus.major_radius - np.sqrt(torus.major_radius**2 - inner_minor_radius**2)
    ) / (torus.major_radius - np.sqrt(torus.major_radius**2 - torus.minor_radius**2))
    assert targets == pytest.approx((inner_fraction, 1.0 - inner_fraction), abs=1.0e-3)
    assert sum(targets) == pytest.approx(1.0, abs=1.0e-3)
    constraints = tuple(
        CurrentMomentConstraint(weight=weight, target=target, name=f"shell-{index}")
        for index, (weight, target) in enumerate(zip(weights, targets, strict=True))
    )
    paired = solve_constrained_current_projection(
        mesh,
        raw_current,
        base_order=3,
        raw_divergence=0.04
        * ng.CoefficientFunction(4.0 + 2.0 * (cylindrical_radius - 2.0) / cylindrical_radius),
        moment_constraints=constraints,
        bonus_integration_order=10,
        moment_integration_order=14,
    )
    assert paired.post_projection_divergence_relative_norm < _roundoff_gate(3, curved=True)
    assert paired.ampere_compatibility_relative_residual < 2.0e-11
    assert max(paired.raw_moment_relative_residuals) > 1.0e-4
    assert max(paired.moment_relative_residuals) < 1.0e-10
    assert paired.projected_moments == pytest.approx(targets, rel=1.0e-10, abs=1.0e-11)
    assert paired.raw_cumulative_moments == pytest.approx(
        (0.0, *np.cumsum(paired.raw_moments)), abs=1.0e-13
    )
    assert paired.target_cumulative_moments == pytest.approx(
        (0.0, *np.cumsum(targets)), abs=1.0e-13
    )
    assert paired.projected_cumulative_moments == pytest.approx(
        paired.target_cumulative_moments, rel=1.0e-10, abs=1.0e-11
    )
    assert max(paired.raw_cumulative_moment_relative_residuals) > 1.0e-4
    assert max(paired.cumulative_moment_relative_residuals) < 1.0e-10

    # The cut-shell integral is not claimed below its quadrature sensitivity.
    quadrature_scan = tuple(
        float(ng.Integrate(ng.InnerProduct(paired.current_density, weights[0]), mesh, order=order))
        for order in (8, 16, 24, 32)
    )
    assert 10.0 * abs(quadrature_scan[3] - quadrature_scan[2]) < abs(
        quadrature_scan[1] - quadrature_scan[0]
    )

    undersized = solve_constrained_current_projection(
        mesh,
        raw_current,
        base_order=3,
        raw_divergence=0.04
        * ng.CoefficientFunction(4.0 + 2.0 * (cylindrical_radius - 2.0) / cylindrical_radius),
        terminal_order=0,
        bonus_integration_order=10,
    )
    assert undersized.post_projection_divergence_relative_norm > 1.0e-3

    paired_rank = analyze_divergence_constraint_rank(
        mesh,
        hdiv_order=2,
        terminal_order=1,
        element_index=0,
    )
    oversized_rank = analyze_divergence_constraint_rank(
        mesh,
        hdiv_order=2,
        terminal_order=2,
        element_index=0,
    )
    assert paired_rank.rows == paired_rank.rank == 4
    assert paired_rank.retained_singular_value_ratio > 1.0e-10
    assert oversized_rank.rows == 10
    assert oversized_rank.rank == 4
    assert oversized_rank.first_discarded_singular_value_ratio < 1.0e-10
