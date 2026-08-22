"""Manufactured coupled-map verification for milestone 5.2 damped Picard."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from math import pi
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from remec.common import JsonEventLogger, configuration_digest
from remec.geometry.axisymmetric import AxisymmetricRZDomain
from remec.profiles import (
    AnalyticPressureProfile,
    AnalyticToroidalCurrentProfile,
    ToroidalCurrentProfile,
)
from remec.solvers.axisymmetric import (
    AxisymmetricGradShafranovCoefficients,
    AxisymmetricGradShafranovSolver,
    AxisymmetricProfileClosure,
)
from remec.solvers.picard import (
    ConstrainedCurrentStep,
    CurrentProjectionStep,
    DampedPicardSolver,
    MagneticStep,
    PicardConvergenceError,
    PicardOptions,
    PicardSafetyStep,
    ReferencePotentialStep,
)

_TABLE = Path(__file__).with_name("picard_damping_convergence.csv")
_ANDERSON_TABLE = Path(__file__).with_name("picard_anderson_convergence.csv")


@dataclass
class _ManufacturedAxisymmetricCycle:
    r"""Analytic reduced block map with the note's complete Picard data flow.

    The magnetic fixed-point map is ``A_candidate = 2 - 2 A``.  It is unstable
    without relaxation; scalar damping gives error propagation
    ``e[k+1] = (1 - 3 alpha) e[k]``.  The remaining blocks evaluate nonconstant
    ``s``, ``p_0(s)``, the bordered (M3)--(M3b) shell targets, a nonidentity
    moment-preserving
    projection, and the (M1) update in that order.
    """

    pressure_bias: float = 0.0
    current_bias: float = 0.0
    projected_current_bias: float = 0.0
    floor_sensitivity: float = 0.0
    current_layer_cells: float = 8.0
    pressure_layer_cells: float = 9.0
    safety_pressure_bias: float = 0.0
    shared_s_ids: list[tuple[str, int]] = field(default_factory=list)
    shared_s_fields: list[NDArray[np.float64]] = field(default_factory=list)
    verified_magnetic_inputs: list[NDArray[np.float64]] = field(default_factory=list)
    reference_solutions: list[NDArray[np.float64]] = field(default_factory=list)
    normalized_volume_inputs: list[NDArray[np.float64]] = field(default_factory=list)
    complete_magnetic_inputs: list[NDArray[np.float64]] = field(default_factory=list)
    complete_magnetic_state: NDArray[np.float64] = field(
        default_factory=lambda: np.asarray((1.75, 0.0, -0.25, 0.5), dtype=float)
    )

    def solve_reference_potential(
        self, magnetic_state: NDArray[np.float64]
    ) -> ReferencePotentialStep:
        self.verified_magnetic_inputs.append(magnetic_state.copy())
        self.complete_magnetic_state[1:2] = magnetic_state
        self.complete_magnetic_inputs.append(self.complete_magnetic_state.copy())
        reference_potential = np.asarray((float(magnetic_state[0]),), dtype=float)
        self.reference_solutions.append(reference_potential.copy())
        return ReferencePotentialStep(
            reference_potential=reference_potential,
            m4a_relative_residual=2.0e-14,
        )

    def build_normalized_volume(
        self, reference_potential: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        self.normalized_volume_inputs.append(reference_potential.copy())
        base = np.linspace(0.0, 1.0, 9)
        antisymmetric_shape = base * (1.0 - base) * (base - 0.5)
        normalized_volume = np.asarray(
            base + 0.08 * np.tanh(float(reference_potential[0])) * antisymmetric_shape,
            dtype=float,
        )
        self.shared_s_fields.append(normalized_volume)
        return normalized_volume

    def pressure_profile_realization_error(
        self,
        normalized_volume: NDArray[np.float64],
        pressure: NDArray[np.float64],
    ) -> float:
        self.shared_s_ids.append(("pressure", id(normalized_volume)))
        independently_measured = 2.0 - normalized_volume + self.pressure_bias
        return float(np.max(np.abs(independently_measured - pressure)))

    def solve_constrained_current(
        self,
        magnetic_state: NDArray[np.float64],
        pressure: NDArray[np.float64],
        normalized_volume: NDArray[np.float64],
        current_profile: ToroidalCurrentProfile,
    ) -> ConstrainedCurrentStep:
        self.shared_s_ids.append(("current", id(normalized_volume)))
        assert pressure.shape == normalized_volume.shape
        shell_edges = np.linspace(0.0, 1.0, 5)
        target_cumulative_current = np.asarray(
            current_profile.enclosed_current(shell_edges), dtype=float
        )
        measured = target_cumulative_current + self.current_bias
        measured[0] = 0.0
        pressure_mean = float(np.mean(pressure))
        return ConstrainedCurrentStep(
            utilde=np.asarray((0.1 * magnetic_state[0] + 0.01 * pressure_mean,), dtype=float),
            g_coefficients=np.asarray((0.2, 0.4, 0.6, 0.8, 1.0), dtype=float),
            raw_current=np.asarray(
                (float(magnetic_state[0]) + 0.1 * pressure_mean, pressure_mean),
                dtype=float,
            ),
            shell_edges=shell_edges,
            independent_cumulative_current=measured,
            m3_relative_residual=3.0e-14,
            m3b_relative_residual=4.0e-14,
        )

    def project_current(
        self,
        raw_current: NDArray[np.float64],
        normalized_volume: NDArray[np.float64],
        shell_edges: NDArray[np.float64],
        target_cumulative_current: NDArray[np.float64],
    ) -> CurrentProjectionStep:
        self.shared_s_ids.append(("projection", id(normalized_volume)))
        projected_moments = target_cumulative_current + self.projected_current_bias
        projected_moments[0] = 0.0
        projected_current = raw_current.copy()
        projected_current[0] += 0.25
        return CurrentProjectionStep(
            projected_current=projected_current,
            independent_cumulative_current=projected_moments,
            divergence_relative_residual=5.0e-14,
            projection_correction_relative_norm=6.0e-4,
        )

    def solve_magnetics(self, projected_current: NDArray[np.float64]) -> MagneticStep:
        return MagneticStep(
            candidate_magnetic_state=np.asarray(
                (2.3 - 2.0 * float(projected_current[0] - 0.25),), dtype=float
            ),
            m1_linear_relative_residual=6.0e-14,
            magnetic_divergence_relative_residual=7.0e-14,
            toroidal_flux_relative_error=8.0e-14,
        )

    def assess_safety(
        self,
        magnetic_state: NDArray[np.float64],
        reference_potential: NDArray[np.float64],
        normalized_volume: NDArray[np.float64],
        pressure: NDArray[np.float64],
        current_step: ConstrainedCurrentStep,
        projection_step: CurrentProjectionStep,
    ) -> PicardSafetyStep:
        del reference_potential, normalized_volume, current_step, projection_step
        return PicardSafetyStep(
            pressure_minimum=float(np.min(pressure)),
            pressure_maximum=float(np.max(pressure)) + self.safety_pressure_bias,
            minimum_magnetic_magnitude=1.0 + abs(float(magnetic_state[0])),
            maximum_floor_sensitivity=self.floor_sensitivity,
            minimum_current_layer_cells=self.current_layer_cells,
            minimum_pressure_layer_cells=self.pressure_layer_cells,
        )


def _solver(
    operators: _ManufacturedAxisymmetricCycle,
    *,
    damping: float,
    max_iterations: int = 80,
    anderson_depth: int = 0,
    anderson_condition_limit: float = 1.0e5,
    logger: JsonEventLogger | None = None,
) -> DampedPicardSolver:
    return DampedPicardSolver(
        operators,
        pressure_profile=AnalyticPressureProfile(
            lambda s: 2.0 - s,
            lambda s: -1.0 + 0.0 * s,
        ),
        toroidal_current_profile=AnalyticToroidalCurrentProfile(
            lambda s: 0.4 * s * (2.0 - s),
            lambda s: 0.8 * (1.0 - s),
        ),
        options=PicardOptions(
            magnetic_scale=1.0,
            damping=damping,
            max_iterations=max_iterations,
            residual_tolerance=1.0e-11,
            state_update_tolerance=1.0e-11,
            pressure_profile_tolerance=1.0e-10,
            current_profile_tolerance=1.0e-10,
            invariant_tolerance=1.0e-10,
            anderson_depth=anderson_depth,
            anderson_condition_limit=anderson_condition_limit,
        ),
        logger=logger,
    )


def _recorded_rows() -> dict[float, dict[str, float]]:
    with _TABLE.open(newline="") as table_file:
        return {
            float(row["damping"]): {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(table_file)
        }


def _anderson_rows() -> dict[int, dict[str, float]]:
    with _ANDERSON_TABLE.open(newline="") as table_file:
        return {
            int(row["anderson_depth"]): {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(table_file)
        }


@pytest.mark.parametrize("damping", [0.2, 0.3, 0.4])
def test_damped_picard_converges_at_the_manufactured_linear_rate(damping: float) -> None:
    r"""The complete segregated cycle converges with rate ``abs(1-3 alpha)``."""
    operators = _ManufacturedAxisymmetricCycle()
    result = _solver(operators, damping=damping).solve(np.asarray((0.0,), dtype=float))
    recorded = _recorded_rows()[damping]

    assert result.converged
    assert result.magnetic_state == pytest.approx((2.0 / 3.0,))
    assert result.iterations == int(recorded["iterations"])
    assert result.history[-1].fixed_point_residual_norm == pytest.approx(
        recorded["final_fixed_point_residual_norm"], rel=1.0e-10, abs=1.0e-15
    )
    assert result.history[-1].state_update_norm == pytest.approx(
        recorded["final_state_update_norm"], rel=1.0e-10, abs=1.0e-15
    )
    observed_factor = (
        result.history[-1].fixed_point_residual_norm / result.history[-2].fixed_point_residual_norm
    )
    assert recorded["expected_contraction_factor"] == pytest.approx(abs(1.0 - 3.0 * damping))
    assert observed_factor == pytest.approx(abs(1.0 - 3.0 * damping), abs=2.0e-4)
    assert observed_factor == pytest.approx(recorded["observed_contraction_factor"], abs=2.0e-4)

    final = result.history[-1]
    assert final.m1_relative_residual == pytest.approx(6.0e-14)
    assert final.pressure_profile_error < 1.0e-10
    assert final.current_profile_error < 1.0e-10
    assert final.projected_current_profile_error < 1.0e-10
    assert final.m3_relative_residual < 1.0e-11
    assert final.m3b_relative_residual < 1.0e-11
    assert final.m4a_relative_residual < 1.0e-11
    assert final.current_divergence_relative_residual < 1.0e-10
    assert final.magnetic_divergence_relative_residual < 1.0e-10
    assert final.toroidal_flux_relative_error < 1.0e-10
    assert final.projection_correction_relative_norm == pytest.approx(6.0e-4)
    assert final.minimum_magnetic_magnitude == pytest.approx(1.0 + abs(result.magnetic_state[0]))
    assert final.maximum_floor_sensitivity == 0.0
    assert final.minimum_current_layer_cells >= 6.0
    assert final.minimum_pressure_layer_cells >= 6.0

    grouped_ids: dict[int, set[str]] = {}
    for name, object_id in operators.shared_s_ids:
        grouped_ids.setdefault(object_id, set()).add(name)
    assert all(names == {"pressure", "current", "projection"} for names in grouped_ids.values())
    assert len(grouped_ids) == result.iterations
    assert all(
        np.array_equal(volume_input, reference)
        for volume_input, reference in zip(
            operators.normalized_volume_inputs,
            operators.reference_solutions,
            strict=True,
        )
    )
    assert any(
        not np.array_equal(field, operators.shared_s_fields[0])
        for field in operators.shared_s_fields[1:]
    )
    assert np.array_equal(
        np.asarray(result.magnetic_state),
        operators.verified_magnetic_inputs[-1],
    )
    final_magnetic = operators.verified_magnetic_inputs[-1]
    final_reference = operators.reference_solutions[-1]
    final_s = operators.shared_s_fields[-1]
    final_pressure = 2.0 - final_s
    final_pressure_mean = float(np.mean(final_pressure))
    expected_utilde = np.asarray((0.1 * final_magnetic[0] + 0.01 * final_pressure_mean,))
    expected_g = np.asarray((0.2, 0.4, 0.6, 0.8, 1.0))
    expected_projected_current = np.asarray(
        (
            final_magnetic[0] + 0.1 * final_pressure_mean + 0.25,
            final_pressure_mean,
        )
    )
    assert np.array_equal(np.asarray(result.reference_potential), final_reference)
    assert np.array_equal(np.asarray(result.normalized_volume), final_s)
    assert np.array_equal(np.asarray(result.pressure), final_pressure)
    assert np.array_equal(np.asarray(result.utilde), expected_utilde)
    assert np.array_equal(np.asarray(result.g_coefficients), expected_g)
    assert np.array_equal(
        np.asarray(result.projected_current),
        expected_projected_current,
    )


@pytest.mark.parametrize("depth", [0, 1, 2, 5])
def test_anderson_accelerates_the_complete_picard_cycle_without_bypassing_gates(
    depth: int,
) -> None:
    """DESIGN §13.3 rows reach one fixed point without changing physical invariants."""
    operators = _ManufacturedAxisymmetricCycle()
    result = _solver(
        operators,
        damping=0.3,
        anderson_depth=depth,
        max_iterations=20,
    ).solve(np.asarray((0.0,), dtype=float))
    recorded = _anderson_rows()[depth]

    assert result.converged
    assert result.iterations == int(recorded["iterations"])
    assert result.magnetic_state == pytest.approx((2.0 / 3.0,), abs=1.0e-11)
    assert sum(row.update_method == "anderson" for row in result.history) == int(
        recorded["anderson_steps"]
    )
    assert sum(row.anderson_history_restarted for row in result.history) == int(
        recorded["history_restarts"]
    )
    assert recorded["final_fixed_point_residual_norm"] < 1.0e-11
    assert recorded["final_state_update_norm"] < 1.0e-11
    assert result.history[-1].fixed_point_residual_norm == pytest.approx(
        recorded["final_fixed_point_residual_norm"], rel=1.0e-3, abs=1.0e-16
    )
    assert result.history[-1].state_update_norm == pytest.approx(
        recorded["final_state_update_norm"], rel=1.0e-3, abs=1.0e-16
    )
    assert result.history[-1].pressure_profile_error < 1.0e-10
    assert result.history[-1].current_profile_error < 1.0e-10
    assert result.history[-1].projected_current_profile_error < 1.0e-10
    assert result.history[-1].current_divergence_relative_residual < 1.0e-10
    assert result.history[-1].magnetic_divergence_relative_residual < 1.0e-10
    assert result.history[-1].toroidal_flux_relative_error < 1.0e-10
    assert all(
        complete[0] == 1.75 and tuple(complete[-2:]) == (-0.25, 0.5)
        for complete in operators.complete_magnetic_inputs
    )


def test_anderson_rejects_ill_conditioned_history_and_logs_damped_fallback() -> None:
    """A rank-deficient least-squares history restarts instead of accepting acceleration."""
    stream = io.StringIO()
    solver = _solver(
        _ManufacturedAxisymmetricCycle(),
        damping=0.3,
        anderson_depth=5,
        anderson_condition_limit=1.0e5,
        logger=JsonEventLogger(stream),
    )
    result = solver.solve(np.asarray((0.0,), dtype=float))
    records = [json.loads(line) for line in stream.getvalue().splitlines()]

    rejected = [record for record in records if record["event"] == "anderson_step_rejected"]
    assert rejected
    assert rejected[0]["reason"] == "rank_deficient_history"
    assert any(row.update_method == "damped_fallback" for row in result.history)
    assert result.converged


def test_picard_log_provenance_and_accepted_history_are_live() -> None:
    """§13.3 records every accepted damping decision with reproducible provenance."""
    stream = io.StringIO()
    solver = _solver(
        _ManufacturedAxisymmetricCycle(),
        damping=0.3,
        logger=JsonEventLogger(stream),
    )
    result = solver.solve(np.asarray((0.0,), dtype=float))
    records = [json.loads(line) for line in stream.getvalue().splitlines()]

    assert result.configuration_digest == configuration_digest(
        {
            "nonlinear": solver.options,
            "pressure_profile_type": type(solver.pressure_profile).__name__,
            "toroidal_current_profile_type": type(solver.toroidal_current_profile).__name__,
        }
    )
    assert len(result.configuration_digest) == 64
    assert all(character in "0123456789abcdef" for character in result.configuration_digest)
    assert [record["event"] for record in records] == [
        "picard_solve_started",
        *["picard_iteration"] * result.iterations,
        "picard_solve_completed",
    ]
    assert all(record["configuration_digest"] == result.configuration_digest for record in records)
    assert records[0]["damping"] == pytest.approx(0.3)
    assert [record["iteration"] for record in records[1:-1]] == list(
        range(1, result.iterations + 1)
    )
    assert all(record["damping"] == pytest.approx(0.3) for record in records[1:-1])
    assert all(record["accepted"] is True for record in records[1:-1])
    assert [row.iteration for row in result.history] == list(range(1, result.iterations + 1))
    assert all(row.damping == pytest.approx(0.3) for row in result.history)
    assert records[-1]["iterations"] == result.iterations
    assert records[-1]["converged"] is True


def test_undamped_cycle_and_profile_mutations_cannot_report_convergence() -> None:
    """Damping and both independent profile gates are necessary for acceptance."""
    with pytest.raises(PicardConvergenceError, match="failed to converge"):
        _solver(_ManufacturedAxisymmetricCycle(), damping=1.0, max_iterations=10).solve(
            np.asarray((0.0,), dtype=float)
        )

    for operators, gate in (
        (_ManufacturedAxisymmetricCycle(pressure_bias=1.0e-4), "pressure profile"),
        (_ManufacturedAxisymmetricCycle(current_bias=1.0e-4), "current profile"),
        (
            _ManufacturedAxisymmetricCycle(projected_current_bias=1.0e-4),
            "projected current profile",
        ),
    ):
        with pytest.raises(PicardConvergenceError) as caught:
            _solver(operators, damping=0.3, max_iterations=30).solve(
                np.asarray((0.0,), dtype=float)
            )
        assert gate in caught.value.failed_gates


def test_floor_bounds_and_layer_safety_are_independent_convergence_gates() -> None:
    """§5.5--§5.6 safety failures cannot hide behind converged equation blocks."""
    cases = (
        (_ManufacturedAxisymmetricCycle(floor_sensitivity=0.02), "floor sensitivity"),
        (_ManufacturedAxisymmetricCycle(current_layer_cells=5.0), "current layer resolution"),
        (_ManufacturedAxisymmetricCycle(pressure_layer_cells=5.0), "pressure layer resolution"),
        (_ManufacturedAxisymmetricCycle(safety_pressure_bias=1.0e-4), "pressure bounds"),
    )
    for operators, gate in cases:
        with pytest.raises(PicardConvergenceError) as caught:
            _solver(operators, damping=0.3, max_iterations=30).solve(
                np.asarray((0.0,), dtype=float)
            )
        assert gate in caught.value.failed_gates


@dataclass
class _Milestone51ReducedAdapter:
    """Coarse open-loop adapter executing the real 5.1 closure and M1 solve."""

    closure: AxisymmetricProfileClosure
    domain: AxisymmetricRZDomain
    magnetic_solver: AxisymmetricGradShafranovSolver
    closure_evaluations: int = 0
    magnetic_solves: int = 0
    coefficient_history: list[tuple[float, float]] = field(default_factory=list)
    frozen_coefficients: AxisymmetricGradShafranovCoefficients | None = None

    def solve_reference_potential(
        self, magnetic_state: NDArray[np.float64]
    ) -> ReferencePotentialStep:
        return ReferencePotentialStep(magnetic_state.copy(), 0.0)

    def build_normalized_volume(
        self, reference_potential: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        base = np.linspace(0.0, 1.0, 9)
        antisymmetric_shape = base * (1.0 - base) * (base - 0.5)
        return np.asarray(
            base + 0.08 * np.tanh(float(reference_potential[0])) * antisymmetric_shape,
            dtype=float,
        )

    def pressure_profile_realization_error(
        self,
        normalized_volume: NDArray[np.float64],
        pressure: NDArray[np.float64],
    ) -> float:
        evaluation = self.closure.evaluate(
            normalized_volume,
            d_normalized_volume_d_flux=np.ones_like(normalized_volume),
            mean_inverse_radius_squared=np.ones_like(normalized_volume),
        )
        self.closure_evaluations += 1
        return float(np.max(np.abs(np.asarray(evaluation.pressure) - pressure)))

    def solve_constrained_current(
        self,
        magnetic_state: NDArray[np.float64],
        pressure: NDArray[np.float64],
        normalized_volume: NDArray[np.float64],
        current_profile: ToroidalCurrentProfile,
    ) -> ConstrainedCurrentStep:
        del current_profile
        shell_edges = np.linspace(0.0, 1.0, 5)
        evaluation = self.closure.evaluate(
            shell_edges,
            d_normalized_volume_d_flux=np.ones_like(shell_edges),
            mean_inverse_radius_squared=np.ones_like(shell_edges),
        )
        self.closure_evaluations += 1
        frozen_sources = self.closure.evaluate(
            float(np.mean(normalized_volume)),
            d_normalized_volume_d_flux=1.0,
            mean_inverse_radius_squared=1.0,
        )
        self.closure_evaluations += 1
        self.frozen_coefficients = AxisymmetricGradShafranovCoefficients(
            pressure_flux_derivative=float(frozen_sources.pressure_flux_derivative),
            toroidal_field_drive=float(frozen_sources.toroidal_field_drive),
        )
        pressure_mean = float(np.mean(pressure))
        drive_mean = float(np.mean(np.asarray(evaluation.toroidal_field_drive)))
        return ConstrainedCurrentStep(
            utilde=np.asarray((0.01 * pressure_mean,), dtype=float),
            g_coefficients=np.linspace(0.0, 0.2, len(shell_edges)),
            raw_current=np.asarray(
                (float(magnetic_state[0]) + 0.01 * pressure_mean, drive_mean), dtype=float
            ),
            shell_edges=shell_edges,
            independent_cumulative_current=np.asarray(
                evaluation.target_enclosed_current, dtype=float
            ),
            m3_relative_residual=0.0,
            m3b_relative_residual=0.0,
        )

    def project_current(
        self,
        raw_current: NDArray[np.float64],
        normalized_volume: NDArray[np.float64],
        shell_edges: NDArray[np.float64],
        target_cumulative_current: NDArray[np.float64],
    ) -> CurrentProjectionStep:
        del normalized_volume, shell_edges
        projected = raw_current.copy()
        projected[0] *= 0.9
        return CurrentProjectionStep(
            projected_current=projected,
            independent_cumulative_current=target_cumulative_current.copy(),
            divergence_relative_residual=0.0,
            projection_correction_relative_norm=0.1,
        )

    def solve_magnetics(self, projected_current: NDArray[np.float64]) -> MagneticStep:
        del projected_current
        if self.frozen_coefficients is None:
            raise RuntimeError("profile closure must be evaluated before the magnetic solve")
        solution = self.magnetic_solver.solve_with_flux(
            self.domain,
            self.frozen_coefficients,
        )
        result = solution.result
        self.coefficient_history.append(
            (
                float(self.frozen_coefficients.pressure_flux_derivative),
                float(self.frozen_coefficients.toroidal_field_drive),
            )
        )
        self.magnetic_solves += 1
        return MagneticStep(
            candidate_magnetic_state=np.asarray((solution.flux_at(1.5, 0.5),), dtype=float),
            m1_linear_relative_residual=result.free_dof_relative_residual_norm,
            magnetic_divergence_relative_residual=0.0,
            toroidal_flux_relative_error=0.0,
        )

    def assess_safety(
        self,
        magnetic_state: NDArray[np.float64],
        reference_potential: NDArray[np.float64],
        normalized_volume: NDArray[np.float64],
        pressure: NDArray[np.float64],
        current_step: ConstrainedCurrentStep,
        projection_step: CurrentProjectionStep,
    ) -> PicardSafetyStep:
        del reference_potential, normalized_volume, current_step, projection_step
        return PicardSafetyStep(
            pressure_minimum=float(np.min(pressure)),
            pressure_maximum=float(np.max(pressure)),
            minimum_magnetic_magnitude=1.0 + abs(float(magnetic_state[0])),
            maximum_floor_sensitivity=0.0,
            minimum_current_layer_cells=8.0,
            minimum_pressure_layer_cells=8.0,
        )


def test_picard_protocol_executes_milestone_51_closure_and_grad_shafranov_block() -> None:
    """A coarse open-loop cycle wires the real closure into the real M1 solve."""
    pressure_profile = AnalyticPressureProfile(lambda s: 0.2 * (1.0 - s), lambda s: -0.2 + 0.0 * s)
    current_profile = AnalyticToroidalCurrentProfile(lambda s: 0.1 * s, lambda s: 0.1 + 0.0 * s)
    adapter = _Milestone51ReducedAdapter(
        closure=AxisymmetricProfileClosure(
            pressure_profile,
            current_profile,
            total_volume=1.0,
        ),
        domain=AxisymmetricRZDomain((1.0, 2.0), (0.0, 1.0), 0.4),
        magnetic_solver=AxisymmetricGradShafranovSolver(polynomial_order=1),
    )
    result = DampedPicardSolver(
        adapter,
        pressure_profile=pressure_profile,
        toroidal_current_profile=current_profile,
        options=PicardOptions(
            magnetic_scale=0.1,
            damping=0.5,
            max_iterations=40,
            residual_tolerance=1.0e-10,
            state_update_tolerance=1.0e-8,
            pressure_profile_tolerance=1.0e-10,
            current_profile_tolerance=1.0e-10,
            invariant_tolerance=1.0e-10,
        ),
    ).solve(np.asarray((0.0,), dtype=float))

    assert result.converged
    assert adapter.magnetic_solves == result.iterations
    assert adapter.closure_evaluations == 3 * result.iterations
    assert adapter.coefficient_history == pytest.approx(
        [(-0.2, 0.2 + 0.2 * pi)] * result.iterations
    )
    assert result.history[-1].m1_relative_residual < 1.0e-10
    assert result.history[-1].pressure_profile_error < 1.0e-10
    assert result.history[-1].current_profile_error < 1.0e-10
    assert result.history[-1].projected_current_profile_error < 1.0e-10
