"""Manufactured coupled-map verification for milestone 5.2 damped Picard."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from remec.profiles import (
    AnalyticPressureProfile,
    AnalyticToroidalCurrentProfile,
    ToroidalCurrentProfile,
)
from remec.solvers.picard import (
    ConstrainedCurrentStep,
    CurrentProjectionStep,
    DampedPicardSolver,
    MagneticStep,
    PicardConvergenceError,
    PicardOptions,
    ReferencePotentialStep,
)

_TABLE = Path(__file__).with_name("picard_damping_convergence.csv")


@dataclass
class _ManufacturedAxisymmetricCycle:
    r"""Analytic reduced block map with the note's complete Picard data flow.

    The magnetic fixed-point map is ``A_candidate = 2 - 2 A``.  It is unstable
    without relaxation; scalar damping gives error propagation
    ``e[k+1] = (1 - 3 alpha) e[k]``.  The remaining blocks evaluate nonconstant
    ``s(A)``, ``p_0(s)``, the bordered (M3)--(M3b) shell targets, a moment-preserving
    projection, and the (M1) update in that order.
    """

    pressure_bias: float = 0.0
    current_bias: float = 0.0
    projected_current_bias: float = 0.0
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
        perturbation = 0.08 * np.tanh(reference_potential[0]) * base * (1.0 - base)
        normalized_volume = np.asarray(base + perturbation, dtype=float)
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
        return ConstrainedCurrentStep(
            utilde=np.asarray((0.1 * magnetic_state[0],), dtype=float),
            g_coefficients=np.asarray((0.2, 0.4, 0.6, 0.8, 1.0), dtype=float),
            raw_current=np.asarray((float(magnetic_state[0]),), dtype=float),
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
        return CurrentProjectionStep(
            projected_current=raw_current.copy(),
            independent_cumulative_current=projected_moments,
            divergence_relative_residual=5.0e-14,
            projection_correction_relative_norm=6.0e-4,
        )

    def solve_magnetics(self, projected_current: NDArray[np.float64]) -> MagneticStep:
        return MagneticStep(
            candidate_magnetic_state=np.asarray(
                (2.0 - 2.0 * float(projected_current[0]),), dtype=float
            ),
            m1_linear_relative_residual=6.0e-14,
            magnetic_divergence_relative_residual=7.0e-14,
            toroidal_flux_relative_error=8.0e-14,
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
    assert result.history[-1].m1_relative_residual == pytest.approx(
        recorded["final_m1_relative_residual"], rel=1.0e-10, abs=1.0e-15
    )
    assert result.history[-1].state_update_norm == pytest.approx(
        recorded["final_state_update_norm"], rel=1.0e-10, abs=1.0e-15
    )
    observed_factor = (
        result.history[-1].m1_relative_residual / result.history[-2].m1_relative_residual
    )
    assert recorded["expected_contraction_factor"] == pytest.approx(abs(1.0 - 3.0 * damping))
    assert observed_factor == pytest.approx(abs(1.0 - 3.0 * damping), abs=2.0e-4)
    assert observed_factor == pytest.approx(recorded["observed_contraction_factor"], abs=2.0e-4)

    final = result.history[-1]
    assert final.pressure_profile_error < 1.0e-10
    assert final.current_profile_error < 1.0e-10
    assert final.projected_current_profile_error < 1.0e-10
    assert final.m3_relative_residual < 1.0e-11
    assert final.m3b_relative_residual < 1.0e-11
    assert final.m4a_relative_residual < 1.0e-11
    assert final.current_divergence_relative_residual < 1.0e-10
    assert final.magnetic_divergence_relative_residual < 1.0e-10
    assert final.toroidal_flux_relative_error < 1.0e-10

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
