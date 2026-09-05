"""Live and remote verification of note equations (M4a)--(M4b) on a 3D island."""

from __future__ import annotations

import csv
from pathlib import Path

import ngsolve as ng
import numpy as np
import pytest

from remec.fem.spaces import make_periodic_tetrahedral_de_rham_sequence
from remec.geometry import GradedAnnulus, PeriodicCylinder3D
from remec.solvers.frozen_field_island import FrozenFieldIslandConfig, FrozenFieldIslandSolver

_ASPECT_TABLE_PATH = Path(__file__).with_name("frozen_field_island_aspect_scan.csv")
_COST_TABLE_PATH = Path(__file__).with_name("frozen_field_island_cost.csv")
_OVERLAY_PATH = Path(__file__).with_name("frozen_field_island_overlay.png")


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
            angular_cells=12,
            axial_cells=2,
            volume_levels=25,
            ray_samples=97,
        )
    )

    assert result.diagnostics["equations"] == "M4a-M4b"
    assert result.diagnostics["free_dof_relative_residual"] < 1.0e-10
    assert result.diagnostics["pollution_ratio"] < 0.1
    assert result.diagnostics["total_power_relative_error"] < 5.0e-2
    assert result.diagnostics["divergence_theorem_relative_error"] < 1.0e-10
    assert result.diagnostics["geometry_maximum_relative_error"] < 5.0e-3
    assert result.diagnostics["minimum_mapped_jacobian_ratio"] > 0.02
    assert result.diagnostics["b_floor_relative_activity"] < 1.0e-12
    assert result.diagnostics["mesh_family"] == "graded-extruded-tetrahedral"
    assert result.diagnostics["mollifier_element_size_mode"] == "level-set-normal"
    assert result.diagnostics["layer_cells"] >= 6.0
    assert result.diagnostics["island_flattening_width"] == 0.0
    assert result.diagnostics["integrable_flattening_width"] == 0.0
    assert result.diagnostics["isotropic_flattening_width"] == 0.0
    assert result.diagnostics["subcritical_flattening_width"] == 0.0
    assert result.diagnostics["conservative_flux_relative_correction"] < 1.0
    assert result.diagnostics["conservative_flux_divergence_relative_error"] < 0.2
    assert result.diagnostics["volume_averaged_dp_ds"] == pytest.approx(-1.0, abs=1.0e-12)
    assert (
        abs(
            result.diagnostics["island_pressure_drop"]
            - result.diagnostics["integrable_pressure_drop"]
        )
        > 5.0e-5
    )
    assert (
        abs(
            result.diagnostics["island_pressure_drop"]
            - result.diagnostics["isotropic_pressure_drop"]
        )
        > 1.0e-6
    )


@pytest.mark.sentinel("frozen_field_island_3d")
def test_graded_extruded_mesh_preserves_periodic_tetrahedral_geometry_contract() -> None:
    """ADR 0011 splitting keeps exact wall geometry, periodicity, and tetrahedra."""
    cylinder = PeriodicCylinder3D(geometry_order=4)
    target = GradedAnnulus(0.74339, 0.06, 0.03)
    bundle = cylinder.build_graded_mesh((target,), angular_cells=12, axial_cells=2)
    metrics = cylinder.measure_geometry(bundle)

    assert bundle.periodic_identification_count == 1
    assert set(bundle.boundary_names) == {"wall", "periodic_lower", "periodic_upper"}
    assert bundle.element_types == ("TET",)
    assert metrics.maximum_relative_error < 5.0e-3
    assert metrics.minimum_mapped_jacobian_ratio > 0.02
    assert bundle.maximum_aspect_ratio > 1.0


@pytest.mark.sentinel("frozen_field_island_3d")
def test_graded_tetrahedral_split_preserves_hdiv_divergence_theorem() -> None:
    """ADR 0011's diagonal pattern makes interior HDiv normal traces conforming."""
    cylinder = PeriodicCylinder3D(geometry_order=2)
    bundle = cylinder.build_graded_mesh(
        (GradedAnnulus(0.74339, 0.06, 0.03),), angular_cells=12, axial_cells=2
    )
    mesh = bundle._mesh
    space = make_periodic_tetrahedral_de_rham_sequence(mesh, order=1).hdiv
    trial, test = space.TnT()
    mass = ng.BilinearForm(space)
    mass += ng.InnerProduct(trial, test) * ng.dx
    source = ng.CoefficientFunction((0.5 * ng.x, 0.5 * ng.y, 0.0))
    load = ng.LinearForm(space)
    load += ng.InnerProduct(source, test) * ng.dx
    mass.Assemble()
    load.Assemble()
    flux = ng.GridFunction(space)
    flux.vec.data = mass.mat.Inverse(space.FreeDofs(), inverse="umfpack") * load.vec

    integration_order = 10
    divergence_power = float(ng.Integrate(ng.div(flux), mesh, order=integration_order))
    normal = ng.specialcf.normal(3)
    boundary_power = float(ng.Integrate(flux * normal, mesh, ng.BND, order=integration_order))
    assert boundary_power == pytest.approx(divergence_power, abs=2.0e-10)


@pytest.mark.sentinel("frozen_field_island_3d")
def test_graded_mesh_preserves_periodic_trace_and_curved_derham_identity() -> None:
    """ADR 0011 keeps the Section-16.2 periodic and curved de Rham gates live."""
    cylinder = PeriodicCylinder3D(geometry_order=3)
    bundle = cylinder.build_graded_mesh(
        (GradedAnnulus(0.74339, 0.06, 0.03),), angular_cells=12, axial_cells=2
    )
    mesh = bundle._mesh
    sequence = make_periodic_tetrahedral_de_rham_sequence(mesh, order=2)

    scalar = ng.GridFunction(sequence.h1)
    scalar.Set(1.0 + ng.x + 0.1 * ng.sin(ng.z))
    assert float(scalar(mesh(0.2, -0.1, 0.0))) == pytest.approx(
        float(scalar(mesh(0.2, -0.1, cylinder.periodic_length))), abs=2.0e-10
    )

    potential = ng.GridFunction(sequence.hcurl)
    potential.vec.FV().NumPy()[:] = np.random.default_rng(63011).standard_normal(
        sequence.hcurl.ndof
    )
    trial, test = sequence.hdiv.TnT()
    mass = ng.BilinearForm(sequence.hdiv)
    mass += ng.InnerProduct(trial, test) * ng.dx
    load = ng.LinearForm(sequence.hdiv)
    load += ng.InnerProduct(ng.curl(potential), test) * ng.dx
    mass.Assemble()
    load.Assemble()
    magnetic = ng.GridFunction(sequence.hdiv)
    magnetic.vec.data = (
        mass.mat.Inverse(sequence.hdiv.FreeDofs(), inverse="sparsecholesky") * load.vec
    )
    curl_norm = float(ng.sqrt(ng.Integrate(magnetic * magnetic, mesh, order=10)))
    divergence_relative = float(
        ng.sqrt(ng.Integrate(ng.div(magnetic) ** 2, mesh, order=10))
        / max(curl_norm, np.finfo(float).tiny)
    )
    assert divergence_relative < 128.0 * np.finfo(float).eps * 4**3


def test_frozen_field_cost_table_has_required_provenance_and_live_gates() -> None:
    """The script-generated cost deliverable records both solver paths and physics gates."""
    with _COST_TABLE_PATH.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "source_commit",
        "elements",
        "h1_dofs",
        "polynomial_order",
        "assembly_seconds",
        "factorization_solve_seconds",
        "peak_memory_megabytes",
        "linear_solver_path",
        "iteration_count",
        "conservative_flux_relative_correction",
        "conservative_flux_divergence_relative_error",
        "volume_averaged_dp_ds",
    }
    assert required <= set(rows[0])
    assert len({row["source_commit"] for row in rows}) == 1
    assert [
        (float(row["epsilon_kappa"]), int(row["angular_cells"]), int(row["axial_cells"]))
        for row in rows
    ] == [
        (1.0e-2, 12, 2),
        (1.0e-3, 24, 4),
        (1.0e-4, 24, 4),
        (1.0e-4, 36, 6),
        (1.0e-4, 48, 8),
        (1.0e-5, 36, 4),
    ]
    assert {row["linear_solver_path"] for row in rows} == {
        "direct:sparsecholesky",
        "iterative:cg-h1amg",
    }
    assert all(float(row["pollution_ratio"]) < 0.1 for row in rows)
    assert all(float(row["layer_cells"]) >= 6.0 for row in rows)
    assert all(float(row["total_power_relative_error"]) < 5.0e-2 for row in rows)
    assert _OVERLAY_PATH.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert _OVERLAY_PATH.stat().st_size > 100_000


@pytest.mark.slow
def test_graded_mesh_aspect_scan_controls_pollution_and_iterations() -> None:
    """ADR 0011 chooses tangential stretch from live pollution and CG measurements."""
    rows = [
        FrozenFieldIslandSolver().solve(
            FrozenFieldIslandConfig(
                epsilon_kappa=1.0e-2,
                epsilon_1=1.0e-3,
                polynomial_order=2,
                angular_cells=angular_cells,
                axial_cells=axial_cells,
                axial_spacing_amplitude=amplitude,
                direct_dof_threshold=1,
            )
        )
        for angular_cells, axial_cells, amplitude in ((24, 4, 0.0), (24, 4, 0.5))
    ]
    aspects = [float(row.diagnostics["maximum_target_aspect_ratio"]) for row in rows]
    pollution = [float(row.diagnostics["pollution_ratio"]) for row in rows]
    iterations = [int(row.diagnostics["iteration_count"]) for row in rows]

    assert aspects[1] > 1.1 * aspects[0]
    assert max(pollution) < 0.1
    assert abs(pollution[1] - pollution[0]) / pollution[0] < 0.01
    assert 0 < iterations[0] <= iterations[1]
    assert rows[0].diagnostics["elements"] == rows[1].diagnostics["elements"]
    assert rows[0].diagnostics["h1_dofs"] == rows[1].diagnostics["h1_dofs"]
    assert all(row.diagnostics["linear_solver_path"] == "iterative:cg-h1amg" for row in rows)

    with _ASPECT_TABLE_PATH.open(newline="", encoding="utf-8") as stream:
        recorded = list(csv.DictReader(stream))
    assert [
        (
            int(row["angular_cells"]),
            int(row["axial_cells"]),
            float(row["axial_spacing_amplitude"]),
        )
        for row in recorded
    ] == [
        (24, 4, 0.0),
        (24, 4, 0.5),
    ]
    for actual, expected in zip(rows, recorded, strict=True):
        assert actual.diagnostics["elements"] == int(expected["elements"])
        assert actual.diagnostics["h1_dofs"] == int(expected["h1_dofs"])
        assert actual.diagnostics["maximum_target_aspect_ratio"] == pytest.approx(
            float(expected["maximum_target_aspect_ratio"]), rel=2.0e-10
        )
        assert actual.diagnostics["pollution_ratio"] == pytest.approx(
            float(expected["pollution_ratio"]), rel=2.0e-6
        )
        # H1-AMG coarsening differs by one CG step between the reference macOS
        # build and the canonical Linux wheel; the monotone iteration gate above
        # remains exact and is the stretch-selection contract.
        assert abs(actual.diagnostics["iteration_count"] - int(expected["iteration_count"])) <= 1


def _exhaustive_row(epsilon_kappa: float, angular_cells: int, axial_cells: int):
    return FrozenFieldIslandSolver().solve(
        FrozenFieldIslandConfig(
            epsilon_kappa=epsilon_kappa,
            epsilon_1=1.0e-3,
            polynomial_order=2,
            geometry_order=4,
            angular_cells=angular_cells,
            axial_cells=axial_cells,
        )
    )


@pytest.mark.exhaustive
@pytest.mark.sentinel("frozen_field_island_3d")
@pytest.mark.parametrize(
    ("epsilon_kappa", "angular_cells", "axial_cells"),
    ((1.0e-2, 12, 2), (1.0e-3, 24, 4), (1.0e-4, 24, 4)),
)
def test_independent_frozen_field_rows_recompute_live_gates(
    epsilon_kappa: float, angular_cells: int, axial_cells: int
) -> None:
    """Independent ladder rows are separate remote-shardable pytest nodes."""
    row = _exhaustive_row(epsilon_kappa, angular_cells, axial_cells)
    assert row.diagnostics["pollution_ratio"] < 0.1
    assert row.diagnostics["layer_cells"] >= 6.0
    assert row.diagnostics["total_power_relative_error"] < 5.0e-2
    assert row.diagnostics["integrable_flattening_width"] == 0.0
    assert row.diagnostics["isotropic_flattening_width"] == 0.0
    assert row.diagnostics["subcritical_flattening_width"] == 0.0
    assert row.diagnostics["volume_averaged_dp_ds"] == pytest.approx(-1.0, abs=1.0e-12)
    if epsilon_kappa == 1.0e-3:
        assert row.diagnostics["island_flattening_width"] == 0.0
    if epsilon_kappa == 1.0e-4:
        assert row.diagnostics["island_pressure_drop"] < (
            0.98 * row.diagnostics["integrable_pressure_drop"]
        )


@pytest.mark.exhaustive
@pytest.mark.sentinel("frozen_field_island_3d")
def test_frozen_field_refinement_pair_converges_and_localizes_vchi() -> None:
    """The only coupled rows gate refinement motion and island-local Vchi structure."""
    coarse = _exhaustive_row(1.0e-4, 36, 6)
    fine = _exhaustive_row(1.0e-4, 48, 8)
    assert fine.diagnostics["island_flattening_width"] == pytest.approx(
        coarse.diagnostics["island_flattening_width"], rel=0.1
    )
    assert fine.diagnostics["coarea_spike_ratio"] > 1.15
    assert fine.diagnostics["volume_plateau_ratio"] > 1.1
    assert fine.diagnostics["island_level_mollifier_width"] > 0.0
    assert (
        fine.diagnostics["conservative_flux_relative_correction"]
        < coarse.diagnostics["conservative_flux_relative_correction"]
    )


@pytest.mark.exhaustive
@pytest.mark.sentinel("frozen_field_island_3d")
def test_lower_decade_flattening_is_threshold_robust() -> None:
    """At epsilon_kappa=1e-5 saturation is not tuned to the 0.97 definition."""
    row = _exhaustive_row(1.0e-5, 36, 4)
    assert row.diagnostics["pollution_ratio"] < 0.1
    assert row.diagnostics["layer_cells"] >= 6.0
    for key in (
        "island_flattening_width_fraction_095",
        "island_flattening_width_fraction_097",
        "island_flattening_width_fraction_099",
    ):
        assert row.diagnostics[key] == pytest.approx(row.island_width, rel=0.15)
