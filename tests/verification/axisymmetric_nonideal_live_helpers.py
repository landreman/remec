"""Shared live-table checks kept outside pytest's test-module load scopes."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from remec.analytic_equilibria import ZhengShape, solve_zheng_equilibrium
from remec.solvers.axisymmetric_nonideal import (
    _ZhengContinuationContext,
    run_zheng_nonideal_continuation,
)
from remec.solvers.continuation import ContinuationStage, ContinuationStageResult

_TABLE = Path(__file__).with_name("axisymmetric_nonideal_refinement.csv")
_RESTARTS = Path(__file__).with_name("axisymmetric_nonideal_ladder_restart_states.csv")
_ACCEPTANCE_TABLE = Path(__file__).with_name("axisymmetric_nonideal_continuation.csv")
FIXED_PRESSURE_STAGES = (
    ContinuationStage(1.0, 0.060, 0.120),
    ContinuationStage(1.0, 0.030, 0.060),
    ContinuationStage(1.0, 0.015, 0.030),
    ContinuationStage(1.0, 0.0075, 0.015),
    ContinuationStage(1.0, 0.00375, 0.0075),
    ContinuationStage(1.0, 0.001875, 0.00375),
)
ACCEPTANCE_STAGES = (
    ContinuationStage(0.6, 0.06, 0.12),
    ContinuationStage(0.8, 0.03, 0.06),
    ContinuationStage(1.0, 0.015, 0.03),
)


def check_fixed_pressure_segment(start: int, stop: int) -> None:
    """Recompute one bounded, overlapping segment of the ADR-0006 ladder."""
    with _TABLE.open(newline="") as table_file:
        records = [
            {key: float(value) for key, value in row.items() if key != "study"}
            for row in csv.DictReader(table_file)
            if row["study"] == "fixed_pressure_regularization"
        ]
    rows = run_zheng_nonideal_continuation(
        plasma_current=0.8e6,
        stages=FIXED_PRESSURE_STAGES[start:stop],
        maxh=0.32,
        polynomial_order=2,
        require_decreasing_projection_correction=False,
    ).stages

    for row, record in zip(rows, records[start:stop], strict=True):
        assert row.nonideal_to_analytic_relative_l2_error == pytest.approx(
            record["nonideal_to_analytic_relative_l2_error"], rel=1.0e-5
        )
        assert row.projection_correction_relative_norm == pytest.approx(
            record["projection_correction_relative_norm"], rel=1.0e-5
        )
        assert row.toroidal_flux_relative_error < 1.0e-10


def check_fixed_pressure_restart(stage_index: int, restart_column: int) -> None:
    """Recompute one fine ladder row from the preceding converged magnetic state."""
    with _TABLE.open(newline="") as table_file:
        records = [
            {key: float(value) for key, value in row.items() if key != "study"}
            for row in csv.DictReader(table_file)
            if row["study"] == "fixed_pressure_regularization"
        ]
    restart = np.loadtxt(
        _RESTARTS,
        delimiter=",",
        skiprows=1,
        usecols=restart_column,
        dtype=float,
    )
    equilibrium = solve_zheng_equilibrium(
        shape=ZhengShape(0.70, 0.49, 1.7, 0.125),
        poloidal_beta=0.40,
        plasma_current=0.8e6,
    )
    context = _ZhengContinuationContext(equilibrium, maxh=0.32, polynomial_order=2)
    row = context.solve_stage(FIXED_PRESSURE_STAGES[stage_index], restart)
    record = records[stage_index]

    assert row.nonideal_to_analytic_relative_l2_error == pytest.approx(
        record["nonideal_to_analytic_relative_l2_error"], rel=1.0e-5
    )
    assert row.projection_correction_relative_norm == pytest.approx(
        record["projection_correction_relative_norm"], rel=1.0e-5
    )
    assert row.current_profile_error < 1.0e-10
    assert row.projected_current_profile_error < 1.0e-10
    assert row.toroidal_flux_relative_error < 1.0e-10


def _acceptance_records(profile_index: int) -> dict[float, dict[str, float]]:
    with _ACCEPTANCE_TABLE.open(newline="") as table_file:
        return {
            float(row["pressure_amplitude"]): {
                key: float(value) for key, value in row.items() if key != "profile_index" and value
            }
            for row in csv.DictReader(table_file)
            if int(row["profile_index"]) == profile_index
        }


def _check_acceptance_row(row: ContinuationStageResult, record: dict[str, float]) -> None:
    for field in (
        "nonideal_to_analytic_relative_l2_error",
        "ideal_fem_to_analytic_relative_l2_error",
        "nonideal_to_ideal_fem_relative_l2_difference",
        "projection_correction_relative_norm",
        "target_total_current",
        "target_toroidal_flux",
    ):
        assert getattr(row, field) == pytest.approx(record[field], rel=1.0e-5)
    assert row.pressure_profile_error < 1.0e-10
    assert row.current_profile_error < 1.0e-10
    assert row.projected_current_profile_error < 1.0e-10
    assert row.toroidal_flux_relative_error < 1.0e-10


def check_acceptance_cold_start(profile_index: int, plasma_current: float) -> None:
    """Recompute the first acceptance row from its analytic initial field."""
    row = run_zheng_nonideal_continuation(
        plasma_current=plasma_current,
        stages=(ACCEPTANCE_STAGES[0],),
        maxh=0.18,
        polynomial_order=2,
    ).stages[0]
    _check_acceptance_row(row, _acceptance_records(profile_index)[0.6])


def check_acceptance_restart(
    profile_index: int,
    plasma_current: float,
    stage_index: int,
) -> None:
    """Recompute one acceptance row from its accepted nonlinear checkpoint."""
    profile_tag = "08" if profile_index == 0 else "10"
    restart = np.loadtxt(
        Path(__file__).with_name(
            f"axisymmetric_nonideal_acceptance_{profile_tag}_restart_states.csv"
        ),
        delimiter=",",
        skiprows=1,
        usecols=stage_index + 1,
        dtype=float,
    )
    equilibrium = solve_zheng_equilibrium(
        shape=ZhengShape(0.70, 0.49, 1.7, 0.125),
        poloidal_beta=0.40,
        plasma_current=plasma_current,
    )
    context = _ZhengContinuationContext(equilibrium, maxh=0.18, polynomial_order=2)
    row = context.solve_stage(ACCEPTANCE_STAGES[stage_index], restart)
    record = _acceptance_records(profile_index)[ACCEPTANCE_STAGES[stage_index].pressure_amplitude]
    _check_acceptance_row(row, record)


def check_refinement_restart(maxh: float, state_filename: str) -> None:
    """Recompute one fine projection row from its accepted magnetic checkpoint."""
    restart = np.loadtxt(
        Path(__file__).with_name(state_filename),
        delimiter=",",
        skiprows=1,
        usecols=1,
        dtype=float,
    )
    equilibrium = solve_zheng_equilibrium(
        shape=ZhengShape(0.70, 0.49, 1.7, 0.125),
        poloidal_beta=0.40,
        plasma_current=0.8e6,
    )
    context = _ZhengContinuationContext(equilibrium, maxh=maxh, polynomial_order=2)
    _check_refinement_row(context, restart, maxh)


def check_refinement_coordinate_restart(maxh: float, state_filename: str) -> None:
    """Remap a fine checkpoint by topological-node coordinates, then recompute its row."""
    records = np.loadtxt(
        Path(__file__).with_name(state_filename),
        delimiter=",",
        skiprows=1,
        dtype=float,
    )
    equilibrium = solve_zheng_equilibrium(
        shape=ZhengShape(0.70, 0.49, 1.7, 0.125),
        poloidal_beta=0.40,
        plasma_current=0.8e6,
    )
    context = _ZhengContinuationContext(equilibrium, maxh=maxh, polynomial_order=2)
    source_coordinates = records[:, 1:3]
    source_values = records[:, 3:5]
    vertex_points = {vertex.nr: tuple(map(float, vertex.point)) for vertex in context.mesh.vertices}
    restart = np.zeros(context.ndof + context.toroidal_ndof, dtype=float)

    def interpolate(radius: float, height: float) -> np.ndarray:
        offset = source_coordinates - (radius, height)
        distance_squared = np.sum(offset**2, axis=1)
        closest = int(np.argmin(distance_squared))
        if distance_squared[closest] < 1.0e-20:
            return source_values[closest]
        nearest = np.argpartition(distance_squared, 12)[:12]
        scale = float(np.sqrt(np.max(distance_squared[nearest])))
        radial = offset[nearest, 0] / scale
        vertical = offset[nearest, 1] / scale
        design = np.column_stack(
            (
                np.ones(len(nearest)),
                radial,
                vertical,
                radial**2,
                radial * vertical,
                vertical**2,
            )
        )
        weight = 1.0 / (0.1 + np.sqrt(distance_squared[nearest]) / scale)
        coefficients = np.linalg.lstsq(
            design * weight[:, None],
            source_values[nearest] * weight[:, None],
            rcond=None,
        )[0]
        return coefficients[0]

    for kind, entities in ((0, context.mesh.vertices), (1, context.mesh.edges)):
        for entity in entities:
            if kind == 0:
                radius, height = vertex_points[entity.nr]
            else:
                endpoints = [vertex_points[vertex.nr] for vertex in entity.vertices]
                radius = sum(point[0] for point in endpoints) / 2.0
                height = sum(point[1] for point in endpoints) / 2.0
            psi_value, toroidal_value = interpolate(radius, height)
            psi_dof = context.scalar_space.GetDofNrs(entity)[0]
            toroidal_dof = context.toroidal_space.GetDofNrs(entity)[0]
            restart[psi_dof] = psi_value
            restart[context.ndof + toroidal_dof] = toroidal_value
    for dof, free in enumerate(context.scalar_space.FreeDofs()):
        if not free:
            restart[dof] = 0.0
    _check_refinement_row(context, restart, maxh)


def _check_refinement_row(
    context: _ZhengContinuationContext,
    restart: np.ndarray,
    maxh: float,
) -> None:
    row = context.solve_stage(ContinuationStage(1.0, 0.015, 0.030), restart)
    with _TABLE.open(newline="") as table_file:
        record = next(
            {key: float(value) for key, value in source.items() if key != "study"}
            for source in csv.DictReader(table_file)
            if source["study"] == "projection_refinement" and float(source["maxh"]) == maxh
        )

    assert row.projection_correction_relative_norm == pytest.approx(
        record["projection_correction_relative_norm"], rel=1.0e-5
    )
    assert row.nonideal_to_analytic_relative_l2_error == pytest.approx(
        record["nonideal_to_analytic_relative_l2_error"], rel=1.0e-5
    )
    assert row.current_profile_error < 1.0e-10
    assert row.projected_current_profile_error < 1.0e-10
    assert row.toroidal_flux_relative_error < 1.0e-10
