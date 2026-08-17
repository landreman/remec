"""Manufactured coupled-map verification for milestone 5.2 damped Picard."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

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

    def solve_reference_potential(
        self, magnetic_state: NDArray[np.float64]
    ) -> ReferencePotentialStep:
        return ReferencePotentialStep(
            reference_potential=np.asarray((float(magnetic_state[0]),), dtype=float),
            m4a_relative_residual=2.0e-14,
        )

    def build_normalized_volume(
        self, reference_potential: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        base = np.linspace(0.0, 1.0, 9)
        del reference_potential
        normalized_volume = np.asarray(base, dtype=float)
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
        ),
    )


def _recorded_rows() -> dict[float, dict[str, float]]:
    with _TABLE.open(newline="") as table_file:
        return {
            float(row["damping"]): {key: float(value) for key, value in row.items()}
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
    assert final.maximum_floor_sensitivity == 0.0
    assert final.minimum_current_layer_cells >= 6.0
    assert final.minimum_pressure_layer_cells >= 6.0

    grouped_ids: dict[int, set[str]] = {}
    for name, object_id in operators.shared_s_ids:
        grouped_ids.setdefault(object_id, set()).add(name)
    assert all(names == {"pressure", "current", "projection"} for names in grouped_ids.values())
    assert len(grouped_ids) == result.iterations


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
    """Coarse protocol adapter that executes the real 5.1 closure and M1 solve."""

    closure: AxisymmetricProfileClosure
    domain: AxisymmetricRZDomain
    magnetic_solver: AxisymmetricGradShafranovSolver
    closure_evaluations: int = 0
    magnetic_solves: int = 0

    def solve_reference_potential(
        self, magnetic_state: NDArray[np.float64]
    ) -> ReferencePotentialStep:
        return ReferencePotentialStep(magnetic_state.copy(), 0.0)

    def build_normalized_volume(
        self, reference_potential: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        del reference_potential
        return np.linspace(0.0, 1.0, 9)

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
        del normalized_volume, current_profile
        shell_edges = np.linspace(0.0, 1.0, 5)
        evaluation = self.closure.evaluate(
            shell_edges,
            d_normalized_volume_d_flux=np.ones_like(shell_edges),
            mean_inverse_radius_squared=np.ones_like(shell_edges),
        )
        self.closure_evaluations += 1
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
        result = self.magnetic_solver.solve(
            self.domain,
            AxisymmetricGradShafranovCoefficients(
                pressure_flux_derivative=-0.02,
                toroidal_field_drive=0.2 + 0.05 * float(projected_current[0]),
            ),
        )
        self.magnetic_solves += 1
        return MagneticStep(
            candidate_magnetic_state=np.asarray(
                (self.magnetic_solver.flux_at(1.5, 0.5),), dtype=float
            ),
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
    """A coarse cycle wires the real normalized closure into the real reduced M1 solve."""
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
    assert adapter.closure_evaluations == 2 * result.iterations
    assert result.history[-1].m1_relative_residual < 1.0e-10
    assert result.history[-1].pressure_profile_error < 1.0e-10
    assert result.history[-1].current_profile_error < 1.0e-10
    assert result.history[-1].projected_current_profile_error < 1.0e-10
