"""Damped segregated iteration for note equations (M1)--(M4b)."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from remec.common.logging import JsonEventLogger
from remec.common.norms import block_l2_norms
from remec.common.serialization import configuration_digest
from remec.profiles import PressureProfile, ToroidalCurrentProfile
from remec.solvers._anderson import AndersonAccelerator

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PicardOptions:
    """Scalar under-relaxation and independent DESIGN §13.2 convergence gates."""

    magnetic_scale: float
    damping: float = 0.2
    max_iterations: int = 200
    residual_tolerance: float = 1.0e-8
    state_update_tolerance: float = 1.0e-8
    pressure_profile_tolerance: float = 1.0e-10
    current_profile_tolerance: float = 1.0e-10
    invariant_tolerance: float = 1.0e-10
    floor_sensitivity_tolerance: float = 0.01
    minimum_layer_cells: float = 6.0
    anderson_depth: int = 0
    anderson_regularization: float = 1.0e-12
    anderson_condition_limit: float = 1.0e12

    def __post_init__(self) -> None:
        if not isfinite(self.damping) or not 0.0 < self.damping <= 1.0:
            raise ValueError("damping must be finite and lie in (0, 1]")
        if isinstance(self.max_iterations, bool) or self.max_iterations < 1:
            raise ValueError("max_iterations must be at least one")
        for name in (
            "residual_tolerance",
            "state_update_tolerance",
            "pressure_profile_tolerance",
            "current_profile_tolerance",
            "invariant_tolerance",
            "magnetic_scale",
            "floor_sensitivity_tolerance",
            "minimum_layer_cells",
            "anderson_regularization",
            "anderson_condition_limit",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.anderson_depth, bool)
            or not isinstance(self.anderson_depth, int)
            or self.anderson_depth < 0
        ):
            raise ValueError("anderson_depth must be a non-negative integer")
        if self.anderson_condition_limit <= 1.0:
            raise ValueError("anderson_condition_limit must exceed one")


@dataclass(frozen=True, slots=True)
class ReferencePotentialStep:
    r"""Frozen-field ``(M4a)`` result used to construct the one shared ``s``."""

    reference_potential: FloatArray
    m4a_relative_residual: float


@dataclass(frozen=True, slots=True)
class ConstrainedCurrentStep:
    r"""Bordered ``(M3)``--``(M3b)`` result with independent ``(M2)`` moments.

    ``independent_cumulative_current`` must be reconstructed from the physical (M2)
    field with the mollified shell-moment evaluator. It must never reuse the bordered
    solve's constraint rows or matrices.
    """

    utilde: FloatArray
    g_coefficients: FloatArray
    raw_current: FloatArray
    shell_edges: FloatArray
    independent_cumulative_current: FloatArray
    m3_relative_residual: float
    m3b_relative_residual: float


@dataclass(frozen=True, slots=True)
class CurrentProjectionStep:
    r"""Paired-divergence projection result preserving every ``(M3b)`` moment.

    ``independent_cumulative_current`` is a post-projection physical shell integral,
    not the projection saddle's algebraic constraint vector.
    """

    projected_current: FloatArray
    independent_cumulative_current: FloatArray
    divergence_relative_residual: float
    projection_correction_relative_norm: float


@dataclass(frozen=True, slots=True)
class MagneticStep:
    r"""Compatible ``(M1)`` candidate state and its divergence/flux invariants."""

    candidate_magnetic_state: FloatArray
    m1_linear_relative_residual: float
    magnetic_divergence_relative_residual: float
    toroidal_flux_relative_error: float


@dataclass(frozen=True, slots=True)
class PicardSafetyStep:
    """Independent DESIGN §5.5--§5.6 floor, bounds, and layer diagnostics."""

    pressure_minimum: float
    pressure_maximum: float
    minimum_magnetic_magnitude: float
    maximum_floor_sensitivity: float
    minimum_current_layer_cells: float
    minimum_pressure_layer_cells: float


class PicardCycleOperators(Protocol):
    r"""Backend adapters for the note §9 ``(M4a)->(M3b)->(M2)->(M1)`` cycle.

    ``magnetic_state`` contains only free magnetic coefficients; fixed harmonic-flux
    coefficients and essential traces remain owned by the backend adapter and are not
    damped. All three methods that depend on the level-set coordinate receive the same array
    instance constructed once by :meth:`build_normalized_volume`.  Implementations
    may wrap NGSolve fields internally, but the nonlinear driver only sees flattened,
    backend-independent state and diagnostic arrays.
    """

    def solve_reference_potential(self, magnetic_state: FloatArray) -> ReferencePotentialStep: ...

    def build_normalized_volume(self, reference_potential: FloatArray) -> FloatArray: ...

    def pressure_profile_realization_error(
        self,
        normalized_volume: FloatArray,
        pressure: FloatArray,
    ) -> float: ...

    def solve_constrained_current(
        self,
        magnetic_state: FloatArray,
        pressure: FloatArray,
        normalized_volume: FloatArray,
        current_profile: ToroidalCurrentProfile,
    ) -> ConstrainedCurrentStep: ...

    def project_current(
        self,
        raw_current: FloatArray,
        normalized_volume: FloatArray,
        shell_edges: FloatArray,
        target_cumulative_current: FloatArray,
    ) -> CurrentProjectionStep: ...

    def solve_magnetics(self, projected_current: FloatArray) -> MagneticStep: ...

    def assess_safety(
        self,
        magnetic_state: FloatArray,
        reference_potential: FloatArray,
        normalized_volume: FloatArray,
        pressure: FloatArray,
        current_step: ConstrainedCurrentStep,
        projection_step: CurrentProjectionStep,
    ) -> PicardSafetyStep:
        """Independently monitor pressure bounds, floors, and both resolved layers."""
        ...


@dataclass(frozen=True, slots=True)
class PicardIteration:
    """All independent convergence gates and diagnostics for one accepted cycle."""

    iteration: int
    damping: float
    update_method: str
    anderson_history_size: int
    anderson_condition_number: float | None
    anderson_history_restarted: bool
    anderson_rejection_reason: str | None
    m1_relative_residual: float
    fixed_point_residual_norm: float
    m3_relative_residual: float
    m3b_relative_residual: float
    m4a_relative_residual: float
    state_update_norm: float
    pressure_profile_error: float
    current_profile_error: float
    projected_current_profile_error: float
    current_divergence_relative_residual: float
    magnetic_divergence_relative_residual: float
    toroidal_flux_relative_error: float
    projection_correction_relative_norm: float
    pressure_bounds_error: float
    minimum_magnetic_magnitude: float
    maximum_floor_sensitivity: float
    minimum_current_layer_cells: float
    minimum_pressure_layer_cells: float


@dataclass(frozen=True, slots=True)
class PicardResult:
    """Converged backend-independent fields and complete nonlinear history."""

    converged: bool
    iterations: int
    configuration_digest: str
    magnetic_state: tuple[float, ...]
    reference_potential: tuple[float, ...]
    normalized_volume: tuple[float, ...]
    pressure: tuple[float, ...]
    utilde: tuple[float, ...]
    g_coefficients: tuple[float, ...]
    projected_current: tuple[float, ...]
    history: tuple[PicardIteration, ...]


class PicardConvergenceError(RuntimeError):
    """Raised when one or more physical convergence gates remain open."""

    def __init__(
        self,
        failed_gates: tuple[str, ...],
        history: tuple[PicardIteration, ...],
    ) -> None:
        self.failed_gates = failed_gates
        self.history = history
        joined = ", ".join(failed_gates)
        super().__init__(f"Picard failed to converge; open gates: {joined}")


def _array(value: FloatArray, name: str, *, shape: tuple[int, ...] | None = None) -> FloatArray:
    """Copy one finite non-empty float array and optionally require its shape."""
    array = np.asarray(value, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite non-empty array")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return np.array(array, dtype=float, copy=True)


def _nonnegative(value: float, name: str) -> float:
    """Validate one finite non-negative physical diagnostic."""
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _maximum_absolute_error(measured: FloatArray, target: FloatArray, name: str) -> float:
    """Return one independently measured cumulative-profile infinity norm."""
    values = _array(measured, name, shape=target.shape)
    return float(np.max(np.abs(values - target)))


class DampedPicardSolver:
    r"""Scalar-under-relaxed segregated solver for ``(M1)``--``(M4b)``.

    Each iteration performs exactly the DESIGN §13.1 order: solve ``(M4a)``, build
    one normalized label ``s=V_chi(chi)/V_omega``, set ``p=p_0(s)``, solve the
    bordered ``(M3)``--``(M3b)`` system, project the reconstructed ``(M2)`` current
    while retaining all shell moments, solve ``(M1)``, then accept
    ``A <- A + alpha (A_candidate-A)``.  Convergence is the conjunction of every
    equation, state-update, profile, divergence, and flux gate; no concatenated raw
    coefficient norm can certify the solve.
    """

    def __init__(
        self,
        operators: PicardCycleOperators,
        *,
        pressure_profile: PressureProfile,
        toroidal_current_profile: ToroidalCurrentProfile,
        options: PicardOptions,
        logger: JsonEventLogger | None = None,
    ) -> None:
        self.operators = operators
        self.pressure_profile = pressure_profile
        self.toroidal_current_profile = toroidal_current_profile
        self.options = options
        self.logger = logger
        self.pressure_profile.validate()
        self.toroidal_current_profile.validate()
        self._minimum_profile_pressure = float(self.pressure_profile.value(1.0))
        self._maximum_profile_pressure = float(self.pressure_profile.value(0.0))
        self.configuration_digest = configuration_digest(
            {
                "nonlinear": self.options,
                "pressure_profile_type": type(pressure_profile).__name__,
                "toroidal_current_profile_type": type(toroidal_current_profile).__name__,
            }
        )

    def _failed_gates(self, row: PicardIteration) -> tuple[str, ...]:
        options = self.options
        gates = (
            ("M1 residual", row.m1_relative_residual, options.residual_tolerance),
            (
                "fixed-point residual",
                row.fixed_point_residual_norm,
                options.state_update_tolerance,
            ),
            ("M3 residual", row.m3_relative_residual, options.residual_tolerance),
            ("M3b residual", row.m3b_relative_residual, options.residual_tolerance),
            ("M4a residual", row.m4a_relative_residual, options.residual_tolerance),
            ("state update", row.state_update_norm, options.state_update_tolerance),
            (
                "pressure profile",
                row.pressure_profile_error,
                options.pressure_profile_tolerance,
            ),
            ("current profile", row.current_profile_error, options.current_profile_tolerance),
            (
                "projected current profile",
                row.projected_current_profile_error,
                options.current_profile_tolerance,
            ),
            (
                "current divergence",
                row.current_divergence_relative_residual,
                options.invariant_tolerance,
            ),
            (
                "magnetic divergence",
                row.magnetic_divergence_relative_residual,
                options.invariant_tolerance,
            ),
            ("toroidal flux", row.toroidal_flux_relative_error, options.invariant_tolerance),
            (
                "pressure bounds",
                row.pressure_bounds_error,
                options.pressure_profile_tolerance,
            ),
            (
                "floor sensitivity",
                row.maximum_floor_sensitivity,
                options.floor_sensitivity_tolerance,
            ),
        )
        failed = [name for name, value, tolerance in gates if value > tolerance]
        if row.minimum_current_layer_cells < options.minimum_layer_cells:
            failed.append("current layer resolution")
        if row.minimum_pressure_layer_cells < options.minimum_layer_cells:
            failed.append("pressure layer resolution")
        return tuple(failed)

    def _magnetic_norm(self, values: FloatArray) -> float:
        """Return the physically scaled §13.2 norm of one free magnetic block."""
        return block_l2_norms(
            {"magnetic": values.reshape(-1)},
            scales={"magnetic": self.options.magnetic_scale},
        ).scaled_blocks["magnetic"]

    def solve(self, initial_magnetic_state: FloatArray) -> PicardResult:
        """Run accepted damped/Anderson cycles until every DESIGN §13.2 gate passes."""
        magnetic = _array(initial_magnetic_state, "initial_magnetic_state")
        accelerator = (
            AndersonAccelerator(
                depth=self.options.anderson_depth,
                damping=self.options.damping,
                regularization=self.options.anderson_regularization,
                condition_limit=self.options.anderson_condition_limit,
            )
            if self.options.anderson_depth
            else None
        )
        history: list[PicardIteration] = []
        if self.logger is not None:
            self.logger.info(
                "picard_solve_started",
                configuration_digest=self.configuration_digest,
                damping=self.options.damping,
                anderson_depth=self.options.anderson_depth,
                max_iterations=self.options.max_iterations,
            )

        last_fields: (
            tuple[
                ReferencePotentialStep,
                FloatArray,
                FloatArray,
                ConstrainedCurrentStep,
                CurrentProjectionStep,
            ]
            | None
        ) = None
        for iteration in range(1, self.options.max_iterations + 1):
            verified_magnetic = magnetic.copy()
            reference_step = self.operators.solve_reference_potential(verified_magnetic.copy())
            reference = _array(reference_step.reference_potential, "reference_potential")
            normalized_volume = np.asarray(
                self.operators.build_normalized_volume(reference), dtype=float
            )
            if (
                normalized_volume.size == 0
                or not np.all(np.isfinite(normalized_volume))
                or np.any(normalized_volume < 0.0)
                or np.any(normalized_volume > 1.0)
            ):
                raise ValueError("normalized volume must be a finite non-empty array in [0, 1]")
            pressure = _array(
                np.asarray(self.pressure_profile.value(normalized_volume), dtype=float),
                "pressure",
                shape=normalized_volume.shape,
            )
            pressure_error = _nonnegative(
                self.operators.pressure_profile_realization_error(
                    normalized_volume,
                    pressure,
                ),
                "pressure_profile_error",
            )

            current_step = self.operators.solve_constrained_current(
                verified_magnetic.copy(),
                pressure,
                normalized_volume,
                self.toroidal_current_profile,
            )
            shell_edges = _array(current_step.shell_edges, "shell_edges")
            if (
                shell_edges.ndim != 1
                or len(shell_edges) < 2
                or shell_edges[0] != 0.0
                or shell_edges[-1] != 1.0
                or np.any(np.diff(shell_edges) <= 0.0)
            ):
                raise ValueError("shell_edges must strictly partition normalized volume [0, 1]")
            target_current = _array(
                np.asarray(
                    self.toroidal_current_profile.enclosed_current(shell_edges), dtype=float
                ),
                "target_cumulative_current",
                shape=shell_edges.shape,
            )
            current_profile_error = _maximum_absolute_error(
                current_step.independent_cumulative_current,
                target_current,
                "independent_cumulative_current",
            )
            raw_current = _array(current_step.raw_current, "raw_current")
            projection_step = self.operators.project_current(
                raw_current,
                normalized_volume,
                shell_edges,
                target_current,
            )
            projected_current = _array(
                projection_step.projected_current,
                "projected_current",
            )
            projected_profile_error = _maximum_absolute_error(
                projection_step.independent_cumulative_current,
                target_current,
                "projected_independent_cumulative_current",
            )
            safety_step = self.operators.assess_safety(
                verified_magnetic.copy(),
                reference,
                normalized_volume,
                pressure,
                current_step,
                projection_step,
            )
            magnetic_step = self.operators.solve_magnetics(projected_current)
            candidate = _array(
                magnetic_step.candidate_magnetic_state,
                "candidate_magnetic_state",
                shape=verified_magnetic.shape,
            )
            fixed_point_residual = self._magnetic_norm(candidate - verified_magnetic)
            if accelerator is None:
                accepted = verified_magnetic + self.options.damping * (
                    candidate - verified_magnetic
                )
                update_method = "damped"
                anderson_history_size = 0
                anderson_condition_number = None
                anderson_history_restarted = False
                anderson_rejection_reason = None
            else:
                acceleration = accelerator.update(verified_magnetic, candidate)
                accepted = acceleration.state
                update_method = acceleration.method
                anderson_history_size = acceleration.history_size
                anderson_condition_number = acceleration.condition_number
                anderson_history_restarted = acceleration.restarted
                anderson_rejection_reason = acceleration.rejection_reason
                if acceleration.rejection_reason is not None and self.logger is not None:
                    self.logger.info(
                        "anderson_step_rejected",
                        configuration_digest=self.configuration_digest,
                        iteration=iteration,
                        reason=acceleration.rejection_reason,
                        history_size=acceleration.history_size,
                        condition_number=acceleration.condition_number,
                        fallback="damped_picard",
                    )
            state_update = self._magnetic_norm(accepted - verified_magnetic)
            pressure_minimum = float(safety_step.pressure_minimum)
            pressure_maximum = float(safety_step.pressure_maximum)
            if not isfinite(pressure_minimum) or not isfinite(pressure_maximum):
                raise ValueError("measured pressure bounds must be finite")
            if pressure_minimum > pressure_maximum:
                raise ValueError("measured pressure minimum must not exceed its maximum")
            pressure_bounds_error = max(
                0.0,
                self._minimum_profile_pressure - pressure_minimum,
                pressure_maximum - self._maximum_profile_pressure,
            )
            row = PicardIteration(
                iteration=iteration,
                damping=self.options.damping,
                update_method=update_method,
                anderson_history_size=anderson_history_size,
                anderson_condition_number=anderson_condition_number,
                anderson_history_restarted=anderson_history_restarted,
                anderson_rejection_reason=anderson_rejection_reason,
                m1_relative_residual=_nonnegative(
                    magnetic_step.m1_linear_relative_residual,
                    "m1_linear_relative_residual",
                ),
                fixed_point_residual_norm=fixed_point_residual,
                m3_relative_residual=_nonnegative(
                    current_step.m3_relative_residual,
                    "m3_relative_residual",
                ),
                m3b_relative_residual=_nonnegative(
                    current_step.m3b_relative_residual,
                    "m3b_relative_residual",
                ),
                m4a_relative_residual=_nonnegative(
                    reference_step.m4a_relative_residual,
                    "m4a_relative_residual",
                ),
                state_update_norm=state_update,
                pressure_profile_error=pressure_error,
                current_profile_error=current_profile_error,
                projected_current_profile_error=projected_profile_error,
                current_divergence_relative_residual=_nonnegative(
                    projection_step.divergence_relative_residual,
                    "current_divergence_relative_residual",
                ),
                magnetic_divergence_relative_residual=_nonnegative(
                    magnetic_step.magnetic_divergence_relative_residual,
                    "magnetic_divergence_relative_residual",
                ),
                toroidal_flux_relative_error=_nonnegative(
                    magnetic_step.toroidal_flux_relative_error,
                    "toroidal_flux_relative_error",
                ),
                projection_correction_relative_norm=_nonnegative(
                    projection_step.projection_correction_relative_norm,
                    "projection_correction_relative_norm",
                ),
                pressure_bounds_error=pressure_bounds_error,
                minimum_magnetic_magnitude=_nonnegative(
                    safety_step.minimum_magnetic_magnitude,
                    "minimum_magnetic_magnitude",
                ),
                maximum_floor_sensitivity=_nonnegative(
                    safety_step.maximum_floor_sensitivity,
                    "maximum_floor_sensitivity",
                ),
                minimum_current_layer_cells=_nonnegative(
                    safety_step.minimum_current_layer_cells,
                    "minimum_current_layer_cells",
                ),
                minimum_pressure_layer_cells=_nonnegative(
                    safety_step.minimum_pressure_layer_cells,
                    "minimum_pressure_layer_cells",
                ),
            )
            history.append(row)
            magnetic = accepted
            last_fields = (
                reference_step,
                reference,
                np.array(normalized_volume, copy=True),
                current_step,
                projection_step,
            )
            failed_gates = self._failed_gates(row)
            if self.logger is not None:
                self.logger.info(
                    "picard_iteration",
                    configuration_digest=self.configuration_digest,
                    iteration=iteration,
                    damping=self.options.damping,
                    update_method=row.update_method,
                    anderson_history_size=row.anderson_history_size,
                    anderson_condition_number=row.anderson_condition_number,
                    anderson_history_restarted=row.anderson_history_restarted,
                    anderson_rejection_reason=row.anderson_rejection_reason,
                    accepted=True,
                    converged=not failed_gates,
                    failed_gates=list(failed_gates),
                    m1_relative_residual=row.m1_relative_residual,
                    fixed_point_residual_norm=row.fixed_point_residual_norm,
                    m3_relative_residual=row.m3_relative_residual,
                    m3b_relative_residual=row.m3b_relative_residual,
                    m4a_relative_residual=row.m4a_relative_residual,
                    state_update_norm=row.state_update_norm,
                    pressure_profile_error=row.pressure_profile_error,
                    current_profile_error=row.current_profile_error,
                    projected_current_profile_error=row.projected_current_profile_error,
                )
            if not failed_gates:
                assert last_fields is not None
                _, final_reference, final_s, final_current, final_projection = last_fields
                result = PicardResult(
                    converged=True,
                    iterations=iteration,
                    configuration_digest=self.configuration_digest,
                    magnetic_state=tuple(float(value) for value in verified_magnetic.reshape(-1)),
                    reference_potential=tuple(
                        float(value) for value in final_reference.reshape(-1)
                    ),
                    normalized_volume=tuple(float(value) for value in final_s.reshape(-1)),
                    pressure=tuple(float(value) for value in pressure.reshape(-1)),
                    utilde=tuple(
                        float(value) for value in _array(final_current.utilde, "utilde").reshape(-1)
                    ),
                    g_coefficients=tuple(
                        float(value)
                        for value in _array(final_current.g_coefficients, "g_coefficients").reshape(
                            -1
                        )
                    ),
                    projected_current=tuple(
                        float(value)
                        for value in _array(
                            final_projection.projected_current,
                            "projected_current",
                        ).reshape(-1)
                    ),
                    history=tuple(history),
                )
                if self.logger is not None:
                    self.logger.info(
                        "picard_solve_completed",
                        configuration_digest=self.configuration_digest,
                        iterations=iteration,
                        converged=True,
                    )
                return result

        failed_gates = self._failed_gates(history[-1])
        if self.logger is not None:
            self.logger.info(
                "picard_solve_completed",
                configuration_digest=self.configuration_digest,
                iterations=self.options.max_iterations,
                converged=False,
                failed_gates=list(failed_gates),
            )
        raise PicardConvergenceError(failed_gates, tuple(history))
