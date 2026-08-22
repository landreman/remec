"""Shaped non-ideal continuation against independent Zheng ideal equilibria."""

from __future__ import annotations

import csv
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from remec.analytic_equilibria import ZhengShape, solve_zheng_equilibrium
from remec.solvers.axisymmetric_nonideal import (
    _ZhengContinuationContext,
    run_zheng_nonideal_continuation,
)
from remec.solvers.continuation import ContinuationStage

_TABLE = Path(__file__).with_name("axisymmetric_nonideal_continuation.csv")
_STAGES = (
    ContinuationStage(0.6, 0.06, 0.12),
    ContinuationStage(0.8, 0.03, 0.06),
    ContinuationStage(1.0, 0.015, 0.03),
)


def _recorded() -> dict[tuple[int, float], dict[str, float]]:
    with _TABLE.open(newline="") as table_file:
        return {
            (int(row["profile_index"]), float(row["pressure_amplitude"])): {
                key: float(value) for key, value in row.items() if key != "profile_index"
            }
            for row in csv.DictReader(table_file)
        }


def test_shaped_nonideal_continuation_sentinel() -> None:
    """Cheap real-block sentinel realizes profiles and reports the independent error split."""
    result = run_zheng_nonideal_continuation(
        plasma_current=0.8e6,
        stages=(_STAGES[0],),
        maxh=0.32,
        polynomial_order=2,
    )
    row = result.stages[0]
    assert row.nonideal_to_analytic_relative_l2_error > row.ideal_fem_to_analytic_relative_l2_error
    assert row.ideal_fem_to_analytic_relative_l2_error < 0.01
    assert row.pressure_profile_error < 1.0e-10
    assert row.current_profile_error < 1.0e-10
    assert row.projected_current_profile_error < 1.0e-10


def test_axisymmetric_m4a_anisotropy_is_live() -> None:
    """Replacing the in-plane M4a tensor by its isotropic part is conspicuous."""
    equilibrium = solve_zheng_equilibrium(
        shape=ZhengShape(0.70, 0.49, 1.7, 0.125),
        poloidal_beta=0.40,
        plasma_current=0.8e6,
    )
    context = _ZhengContinuationContext(equilibrium, maxh=0.32, polynomial_order=2)
    stage = _STAGES[0]
    psi, toroidal_field = context.fields_from_state(context.initial_state(stage))
    magnetic_field = context.magnetic_field(psi, toroidal_field)
    _, anisotropic_map, _, _ = context.solve_reference_potential(
        magnetic_field, stage.perpendicular_ratio
    )
    _, isotropic_map, _, _ = context.solve_reference_potential(magnetic_field, 1.0)
    anisotropic_s = anisotropic_map.quadrature_normalized_volume
    isotropic_s = isotropic_map.quadrature_normalized_volume
    relative_difference = float(
        np.linalg.norm(anisotropic_s - isotropic_s) / np.linalg.norm(isotropic_s)
    )
    assert relative_difference > 1.0e-4


@pytest.mark.slow
@pytest.mark.parametrize("plasma_current", [0.8e6, 1.0e6])
def test_shaped_nonideal_continuation_realizes_two_i0_targets_and_ideal_limit(
    plasma_current: float,
) -> None:
    r"""Finite ``(M4a)``/``(M3)`` bias falls while ``(M4b)``/``(M3b)`` stay exact."""
    profile_index = 0 if plasma_current == 0.8e6 else 1
    result = run_zheng_nonideal_continuation(
        plasma_current=plasma_current,
        stages=_STAGES,
        maxh=0.18,
        polynomial_order=2,
    )
    rows = result.stages

    assert all(row.pressure_profile_error < 1.0e-10 for row in rows)
    assert all(row.current_profile_error < 1.0e-10 for row in rows)
    assert all(row.projected_current_profile_error < 1.0e-10 for row in rows)
    assert all(row.minimum_current_layer_cells >= 6.0 for row in rows)
    assert all(row.minimum_pressure_layer_cells >= 6.0 for row in rows)
    assert all(
        fine.projection_correction_relative_norm < coarse.projection_correction_relative_norm
        for coarse, fine in pairwise(rows)
    )
    assert all(
        fine.nonideal_to_analytic_relative_l2_error < coarse.nonideal_to_analytic_relative_l2_error
        for coarse, fine in pairwise(rows)
    )
    assert rows[-1].nonideal_to_analytic_relative_l2_error < 0.30
    assert rows[-1].ideal_fem_to_analytic_relative_l2_error < 0.01

    recorded = _recorded()
    for row in rows:
        expected = recorded[profile_index, row.stage.pressure_amplitude]
        assert row.nonideal_to_analytic_relative_l2_error == pytest.approx(
            expected["nonideal_to_analytic_relative_l2_error"], rel=0.08
        )
        assert row.ideal_fem_to_analytic_relative_l2_error == pytest.approx(
            expected["ideal_fem_to_analytic_relative_l2_error"], rel=0.08
        )
        assert row.nonideal_to_ideal_fem_relative_l2_difference == pytest.approx(
            expected["nonideal_to_ideal_fem_relative_l2_difference"], rel=0.08
        )
        assert row.projection_correction_relative_norm == pytest.approx(
            expected["projection_correction_relative_norm"], rel=0.08
        )
        assert row.target_total_current == pytest.approx(expected["target_total_current"])
