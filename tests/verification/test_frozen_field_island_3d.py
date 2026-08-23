"""Live and remote verification of note equations (M4a)--(M4b) on a 3D island."""

from __future__ import annotations

import pytest

from remec.solvers.frozen_field_island import FrozenFieldIslandConfig, FrozenFieldIslandSolver


@pytest.mark.sentinel("frozen_field_island_3d")
def test_fast_frozen_field_island_row_runs_the_production_m4_path() -> None:
    """The smallest live row measures M4a/M4b, geometry, layers, and an island control."""
    result = FrozenFieldIslandSolver().solve(
        FrozenFieldIslandConfig(
            epsilon_kappa=1.0e-2,
            epsilon_1=1.0e-3,
            polynomial_order=1,
            max_element_size=0.45,
            geometry_order=4,
        )
    )

    assert result.diagnostics["equations"] == "M4a-M4b"
    assert result.diagnostics["free_dof_relative_residual"] < 1.0e-10
    assert result.diagnostics["geometry_maximum_relative_error"] < 5.0e-3
    assert result.diagnostics["minimum_mapped_jacobian_ratio"] > 0.02
    assert result.diagnostics["b_floor_relative_activity"] < 1.0e-12
    assert result.diagnostics["island_flattening_width"] <= result.critical_width
    assert result.diagnostics["integrable_flattening_width"] <= result.critical_width
    assert result.diagnostics["isotropic_flattening_width"] <= result.critical_width


@pytest.mark.exhaustive
@pytest.mark.sentinel("frozen_field_island_3d")
def test_full_frozen_field_island_ladder_recomputes_every_diagnostic() -> None:
    """The remote ladder crosses the max(w_island,w_c) threshold with live solves."""
    rows = [
        FrozenFieldIslandSolver().solve(
            FrozenFieldIslandConfig(
                epsilon_kappa=epsilon_kappa,
                epsilon_1=1.0e-3,
                polynomial_order=2,
                max_element_size=0.3,
                geometry_order=4,
                refinements=refinements,
            )
        )
        for epsilon_kappa, refinements in (
            (1.0e-2, 0),
            (1.0e-3, 0),
            (1.0e-4, 1),
            (1.0e-6, 2),
        )
    ]

    assert all(row.diagnostics["pollution_ratio"] < 0.1 for row in rows)
    assert all(row.diagnostics["layer_cells"] >= 6.0 for row in rows)
    assert rows[-1].diagnostics["island_flattening_width"] == pytest.approx(
        rows[-1].island_width, rel=0.15
    )
    assert rows[1].diagnostics["island_pressure_drop"] == pytest.approx(
        rows[1].diagnostics["integrable_pressure_drop"], rel=0.15
    )
    assert rows[-1].diagnostics["coarea_spike_ratio"] > 1.5
    assert rows[-1].diagnostics["critical_safeguard_samples"] > 0
