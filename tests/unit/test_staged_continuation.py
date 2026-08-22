"""Contracts for natural continuation of the complete reduced Picard state."""

from __future__ import annotations

import io
import json
from dataclasses import replace

import numpy as np
import pytest

from remec.common import JsonEventLogger
from remec.solvers.axisymmetric_nonideal import run_zheng_nonideal_continuation
from remec.solvers.continuation import (
    ContinuationAcceptanceError,
    ContinuationStage,
    ContinuationStageResult,
    StagedContinuationOptions,
    StagedContinuationSolver,
)


def _schedule() -> tuple[ContinuationStage, ...]:
    return (
        ContinuationStage(0.5, 0.08, 0.20),
        ContinuationStage(0.75, 0.04, 0.10),
        ContinuationStage(1.0, 0.02, 0.05),
    )


class _StageSolver:
    def __init__(self, stage: ContinuationStage, builds: list[ContinuationStage]) -> None:
        self.stage = stage
        self.builds = builds
        builds.append(stage)

    def solve(self, initial_state: np.ndarray) -> ContinuationStageResult:
        index = self.builds.index(self.stage) + 1
        expected_initial = np.asarray((float(index - 1), 2.0 * float(index - 1)))
        assert np.array_equal(initial_state, expected_initial)
        return ContinuationStageResult(
            stage=self.stage,
            state=(float(index), 2.0 * float(index)),
            nonlinear_iterations=4,
            m1_relative_residual=1.0e-13,
            m3_relative_residual=2.0e-13,
            m3b_relative_residual=3.0e-13,
            m4a_relative_residual=4.0e-13,
            fixed_point_residual_norm=1.0e-12,
            pressure_profile_error=2.0e-12,
            current_profile_error=3.0e-12,
            projected_current_profile_error=4.0e-12,
            target_total_current=0.5 * self.stage.pressure_amplitude,
            projection_correction_relative_norm=1.0e-3 / index,
            nonideal_to_analytic_relative_l2_error=2.0e-2 / index,
            ideal_fem_to_analytic_relative_l2_error=5.0e-4,
            nonideal_to_ideal_fem_relative_l2_difference=2.0e-2 / index,
            minimum_current_layer_cells=8.0,
            minimum_pressure_layer_cells=9.0,
            rejected_acceleration_attempts=index - 1,
        )


def test_natural_continuation_hands_off_state_restarts_history_and_checkpoints() -> None:
    """§14.4 reuses each solution but constructs a fresh accelerated solver per stage."""
    builds: list[ContinuationStage] = []
    checkpoints: list[tuple[int, tuple[float, ...]]] = []
    solver = StagedContinuationSolver(
        lambda stage: _StageSolver(stage, builds),
        options=StagedContinuationOptions(_schedule(), checkpoint_cadence=2),
        checkpoint_writer=lambda index, result: checkpoints.append((index, result.state)),
    )
    result = solver.solve(np.asarray((0.0, 0.0)))

    assert builds == list(_schedule())
    assert result.final_state == (3.0, 6.0)
    assert [row.rejected_acceleration_attempts for row in result.stages] == [0, 1, 2]
    assert checkpoints == [(2, (2.0, 4.0)), (3, (3.0, 6.0))]


def test_stage_provenance_records_parameters_rejections_and_checkpoint_cadence() -> None:
    """Every accepted §14.4 point carries deterministic structured run provenance."""
    builds: list[ContinuationStage] = []
    stream = io.StringIO()
    result = StagedContinuationSolver(
        lambda stage: _StageSolver(stage, builds),
        options=StagedContinuationOptions(_schedule(), checkpoint_cadence=2),
        checkpoint_writer=lambda _index, _result: None,
        logger=JsonEventLogger(stream),
    ).solve(np.asarray((0.0, 0.0)))
    records = [json.loads(line) for line in stream.getvalue().splitlines()]

    assert [record["event"] for record in records] == [
        "continuation_started",
        "continuation_stage_completed",
        "continuation_stage_completed",
        "continuation_stage_completed",
        "continuation_completed",
    ]
    assert all(record["configuration_digest"] == result.configuration_digest for record in records)
    assert [record["rejected_acceleration_attempts"] for record in records[1:-1]] == [0, 1, 2]
    assert [record["checkpointed"] for record in records[1:-1]] == [False, True, True]


def test_axisymmetric_run_requires_a_nonempty_schedule() -> None:
    """The concrete run driver rejects an absent continuation path before meshing."""
    with pytest.raises(ValueError, match="at least one continuation"):
        run_zheng_nonideal_continuation(plasma_current=0.8e6, stages=())


@pytest.mark.parametrize(
    "stages, message",
    [
        ((_schedule()[1], _schedule()[0]), "pressure amplitude"),
        ((ContinuationStage(0.5, 0.04, 0.2), ContinuationStage(1.0, 0.08, 0.1)), "D_u"),
        ((ContinuationStage(0.5, 0.08, 0.1), ContinuationStage(1.0, 0.04, 0.2)), "anisotropy"),
    ],
)
def test_schedule_rejects_reversals(stages: tuple[ContinuationStage, ...], message: str) -> None:
    """Natural continuation raises pressure and decreases both regularizers."""
    with pytest.raises(ValueError, match=message):
        StagedContinuationOptions(stages)


@pytest.mark.parametrize(
    "mutation, gate",
    [
        ({"pressure_profile_error": 1.0e-4}, "pressure profile"),
        ({"current_profile_error": 1.0e-4}, "current profile"),
        ({"projected_current_profile_error": 1.0e-4}, "projected current profile"),
        ({"nonideal_to_analytic_relative_l2_error": 3.0e-2}, "regularization bias"),
        ({"projection_correction_relative_norm": 2.0e-3}, "projection correction"),
    ],
)
def test_profile_and_cross_stage_mutations_are_rejected(
    mutation: dict[str, float], gate: str
) -> None:
    """A small nonlinear residual cannot hide a broken profile or trend gate."""
    builds: list[ContinuationStage] = []

    class Mutated(_StageSolver):
        def solve(self, initial_state: np.ndarray) -> ContinuationStageResult:
            row = super().solve(initial_state)
            if len(builds) == 3:
                row = replace(row, **mutation)
            return row

    solver = StagedContinuationSolver(
        lambda stage: Mutated(stage, builds),
        options=StagedContinuationOptions(_schedule()),
    )
    with pytest.raises(ContinuationAcceptanceError, match=gate):
        solver.solve(np.asarray((0.0, 0.0)))
