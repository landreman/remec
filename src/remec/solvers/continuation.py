"""Natural continuation driver for note equations ``(M1)``--``(M4b)``."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import isfinite
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from remec.common.logging import JsonEventLogger
from remec.common.serialization import configuration_digest

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ContinuationStage:
    """One natural-continuation point in pressure, ``D_u``, and ``epsilon_kappa``."""

    pressure_amplitude: float
    current_diffusivity: float
    perpendicular_ratio: float

    def __post_init__(self) -> None:
        for name in ("pressure_amplitude", "current_diffusivity", "perpendicular_ratio"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.pressure_amplitude > 1.0:
            raise ValueError("pressure_amplitude must not exceed the target amplitude one")
        if self.perpendicular_ratio > 1.0:
            raise ValueError("perpendicular_ratio must lie in (0, 1]")


@dataclass(frozen=True, slots=True)
class ContinuationStageResult:
    """Serializable fields and independent gates returned by one converged stage."""

    stage: ContinuationStage
    state: tuple[float, ...]
    nonlinear_iterations: int
    m1_relative_residual: float
    m3_relative_residual: float
    m3b_relative_residual: float
    m4a_relative_residual: float
    fixed_point_residual_norm: float
    pressure_profile_error: float
    current_profile_error: float
    projected_current_profile_error: float
    target_total_current: float
    projection_correction_relative_norm: float
    nonideal_to_analytic_relative_l2_error: float
    ideal_fem_to_analytic_relative_l2_error: float
    nonideal_to_ideal_fem_relative_l2_difference: float
    minimum_current_layer_cells: float
    minimum_pressure_layer_cells: float
    rejected_acceleration_attempts: int = 0


class ContinuationStageSolver(Protocol):
    """A fresh, stage-local nonlinear solver whose acceleration history starts empty."""

    def solve(self, initial_state: FloatArray) -> ContinuationStageResult: ...


ContinuationStageFactory = Callable[[ContinuationStage], ContinuationStageSolver]
CheckpointWriter = Callable[[int, ContinuationStageResult], None]


@dataclass(frozen=True, slots=True)
class StagedContinuationOptions:
    """Validated natural-continuation schedule and cross-stage acceptance gates."""

    stages: Sequence[ContinuationStage]
    checkpoint_cadence: int = 1
    residual_tolerance: float = 1.0e-8
    profile_tolerance: float = 1.0e-10
    minimum_layer_cells: float = 6.0
    maximum_state_growth_factor: float = 10.0
    require_decreasing_regularization_bias: bool = True
    require_decreasing_projection_correction: bool = True

    def __post_init__(self) -> None:
        stages = tuple(self.stages)
        if not stages or not all(isinstance(stage, ContinuationStage) for stage in stages):
            raise ValueError("stages must contain at least one ContinuationStage")
        if isinstance(self.checkpoint_cadence, bool) or self.checkpoint_cadence < 1:
            raise ValueError("checkpoint_cadence must be a positive integer")
        for name in (
            "residual_tolerance",
            "profile_tolerance",
            "minimum_layer_cells",
            "maximum_state_growth_factor",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.maximum_state_growth_factor <= 1.0:
            raise ValueError("maximum_state_growth_factor must exceed one")
        for coarse, fine in pairwise(stages):
            if fine.pressure_amplitude < coarse.pressure_amplitude:
                raise ValueError("pressure amplitude must be non-decreasing")
            if fine.current_diffusivity > coarse.current_diffusivity:
                raise ValueError("D_u must be non-increasing")
            if fine.perpendicular_ratio > coarse.perpendicular_ratio:
                raise ValueError("anisotropy continuation must not increase epsilon_kappa")
        object.__setattr__(self, "stages", stages)


@dataclass(frozen=True, slots=True)
class StagedContinuationResult:
    """All accepted continuation stages and the final free magnetic state."""

    stages: tuple[ContinuationStageResult, ...]
    final_state: tuple[float, ...]
    configuration_digest: str


class ContinuationAcceptanceError(RuntimeError):
    """Raised when a converged inner solve violates a cross-stage physical gate."""


class StagedContinuationSolver:
    """Run natural continuation with a new nonlinear solver at every stage."""

    def __init__(
        self,
        stage_factory: ContinuationStageFactory,
        *,
        options: StagedContinuationOptions,
        checkpoint_writer: CheckpointWriter | None = None,
        logger: JsonEventLogger | None = None,
    ) -> None:
        self.stage_factory = stage_factory
        self.options = options
        self.checkpoint_writer = checkpoint_writer
        self.logger = logger
        self.configuration_digest = configuration_digest({"continuation": options})

    def solve(self, initial_state: FloatArray) -> StagedContinuationResult:
        """Run natural §14.4 continuation and enforce every cross-stage gate.

        A new stage solver is constructed on every pass, which keeps Anderson history
        local to a single parameter point.  Only the converged free magnetic state is
        handed to the next point; fixed harmonic coefficients and essential traces stay
        in the stage factory's backend adapter.
        """
        state = np.asarray(initial_state, dtype=float)
        if state.ndim != 1 or state.size == 0 or not np.all(np.isfinite(state)):
            raise ValueError("initial_state must be a finite non-empty vector")
        accepted: list[ContinuationStageResult] = []
        if self.logger is not None:
            self.logger.info(
                "continuation_started",
                configuration_digest=self.configuration_digest,
                stage_count=len(self.options.stages),
                checkpoint_cadence=self.options.checkpoint_cadence,
            )
        for stage_index, stage in enumerate(self.options.stages, start=1):
            row = self.stage_factory(stage).solve(np.array(state, copy=True))
            if row.stage != stage:
                raise ValueError("stage solver returned diagnostics for a different stage")
            next_state = _state(row.state)
            if next_state.shape != state.shape or not np.all(np.isfinite(next_state)):
                raise ValueError("every continuation state must preserve the initial shape")
            self._check_stage(row, accepted[-1] if accepted else None)
            old_norm = float(np.linalg.norm(state))
            new_norm = float(np.linalg.norm(next_state))
            growth_reference = max(1.0, old_norm)
            if new_norm > self.options.maximum_state_growth_factor * growth_reference:
                raise ContinuationAcceptanceError(
                    "continuation state norm growth exceeded the configured acceptance factor"
                )
            accepted.append(row)
            state = next_state
            checkpointed = self.checkpoint_writer is not None and (
                stage_index % self.options.checkpoint_cadence == 0
                or stage_index == len(self.options.stages)
            )
            if checkpointed:
                assert self.checkpoint_writer is not None
                self.checkpoint_writer(stage_index, row)
            if self.logger is not None:
                self.logger.info(
                    "continuation_stage_completed",
                    configuration_digest=self.configuration_digest,
                    stage_index=stage_index,
                    pressure_amplitude=stage.pressure_amplitude,
                    current_diffusivity=stage.current_diffusivity,
                    perpendicular_ratio=stage.perpendicular_ratio,
                    nonlinear_iterations=row.nonlinear_iterations,
                    rejected_acceleration_attempts=row.rejected_acceleration_attempts,
                    fixed_point_residual_norm=row.fixed_point_residual_norm,
                    pressure_profile_error=row.pressure_profile_error,
                    current_profile_error=row.current_profile_error,
                    projected_current_profile_error=row.projected_current_profile_error,
                    nonideal_to_analytic_relative_l2_error=(
                        row.nonideal_to_analytic_relative_l2_error
                    ),
                    checkpointed=checkpointed,
                )
        if self.logger is not None:
            self.logger.info(
                "continuation_completed",
                configuration_digest=self.configuration_digest,
                stage_count=len(accepted),
                converged=True,
            )
        return StagedContinuationResult(
            tuple(accepted),
            tuple(float(value) for value in state),
            self.configuration_digest,
        )

    def _check_stage(
        self,
        row: ContinuationStageResult,
        previous: ContinuationStageResult | None,
    ) -> None:
        """Apply independent residual, profile, layer, and continuation-trend gates."""
        scalar_values = (
            row.m1_relative_residual,
            row.m3_relative_residual,
            row.m3b_relative_residual,
            row.m4a_relative_residual,
            row.fixed_point_residual_norm,
            row.pressure_profile_error,
            row.current_profile_error,
            row.projected_current_profile_error,
            abs(row.target_total_current),
            row.projection_correction_relative_norm,
            row.nonideal_to_analytic_relative_l2_error,
            row.ideal_fem_to_analytic_relative_l2_error,
            row.nonideal_to_ideal_fem_relative_l2_difference,
            row.minimum_current_layer_cells,
            row.minimum_pressure_layer_cells,
        )
        if any(not isfinite(value) or value < 0.0 for value in scalar_values):
            raise ValueError("continuation diagnostics must be finite and non-negative")
        if row.nonlinear_iterations < 1 or row.rejected_acceleration_attempts < 0:
            raise ValueError("iteration and rejection counts are invalid")
        gates = (
            ("M1 residual", row.m1_relative_residual, self.options.residual_tolerance),
            ("M3 residual", row.m3_relative_residual, self.options.residual_tolerance),
            ("M3b residual", row.m3b_relative_residual, self.options.residual_tolerance),
            ("M4a residual", row.m4a_relative_residual, self.options.residual_tolerance),
            (
                "fixed-point residual",
                row.fixed_point_residual_norm,
                self.options.residual_tolerance,
            ),
            ("pressure profile", row.pressure_profile_error, self.options.profile_tolerance),
            ("current profile", row.current_profile_error, self.options.profile_tolerance),
            (
                "projected current profile",
                row.projected_current_profile_error,
                self.options.profile_tolerance,
            ),
        )
        for name, value, tolerance in gates:
            if value > tolerance:
                raise ContinuationAcceptanceError(
                    f"{name} error {value:.3e} exceeds {tolerance:.3e}"
                )
        if row.minimum_current_layer_cells < self.options.minimum_layer_cells:
            raise ContinuationAcceptanceError("current layer resolution is below the gate")
        if row.minimum_pressure_layer_cells < self.options.minimum_layer_cells:
            raise ContinuationAcceptanceError("pressure layer resolution is below the gate")
        if previous is None:
            return
        if (
            self.options.require_decreasing_regularization_bias
            and row.nonideal_to_analytic_relative_l2_error
            >= previous.nonideal_to_analytic_relative_l2_error
        ):
            raise ContinuationAcceptanceError(
                "regularization bias did not decrease between continuation stages"
            )
        if (
            self.options.require_decreasing_projection_correction
            and row.projection_correction_relative_norm
            > previous.projection_correction_relative_norm
        ):
            raise ContinuationAcceptanceError(
                "projection correction increased between continuation stages"
            )


def _state(values: tuple[float, ...]) -> FloatArray:
    """Return one stage state as a finite NumPy vector."""
    return np.asarray(values, dtype=float)
