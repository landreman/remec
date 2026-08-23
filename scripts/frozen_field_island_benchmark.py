"""Regenerate milestone-6.3 aspect, cost, and Poincare/isobar artifacts.

This script is the only source of the checked-in CSV tables and overlay. The exhaustive
test independently recomputes every asserted diagnostic through the public solver.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from remec.diagnostics.poincare import trace_poincare
from remec.fem._frozen_field_island import _solve_m4
from remec.geometry import GradedAnnulus, PeriodicCylinder3D
from remec.profiles import TabulatedPressureProfile
from remec.reiman_greenside import ReimanGreensideField
from remec.solvers.frozen_field_island import (
    FrozenFieldIslandConfig,
    FrozenFieldIslandSolver,
    critical_layer_width,
    exact_island_width,
)

ROOT = Path(__file__).resolve().parents[1]
VERIFICATION = ROOT / "tests" / "verification"
ASPECT_TABLE = VERIFICATION / "frozen_field_island_aspect_scan.csv"
COST_TABLE = VERIFICATION / "frozen_field_island_cost.csv"
OVERLAY = VERIFICATION / "frozen_field_island_overlay.png"

ASPECT_CONFIGS = ((24, 4, 0.0), (24, 4, 0.5))
LADDER_CONFIGS = (
    (1.0e-2, 12, 2),
    (1.0e-3, 24, 4),
    (1.0e-4, 24, 4),
    (1.0e-4, 36, 6),
    (1.0e-4, 48, 8),
    (1.0e-5, 36, 4),
)


def _solve(epsilon_kappa: float, angular_cells: int, axial_cells: int, *, force_cg: bool):
    return FrozenFieldIslandSolver().solve(
        FrozenFieldIslandConfig(
            epsilon_kappa=epsilon_kappa,
            epsilon_1=1.0e-3,
            polynomial_order=2,
            geometry_order=4,
            angular_cells=angular_cells,
            axial_cells=axial_cells,
            direct_dof_threshold=1 if force_cg else 20_000,
        )
    )


def _solve_cost_row(index: int, queue: Any) -> None:
    """Solve one cost row in a fresh process so peak RSS is row-local."""
    epsilon_kappa, angular_cells, axial_cells = LADDER_CONFIGS[index]
    result = _solve(epsilon_kappa, angular_cells, axial_cells, force_cg=False)
    queue.put(dict(result.diagnostics))


def regenerate_aspect_scan() -> None:
    """Write the ADR-0011 aspect/pollution/iteration scan from live solves."""
    fields = (
        "angular_cells",
        "axial_cells",
        "axial_spacing_amplitude",
        "elements",
        "h1_dofs",
        "maximum_target_aspect_ratio",
        "pollution_ratio",
        "linear_solver_path",
        "iteration_count",
    )
    with ASPECT_TABLE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for angular_cells, axial_cells, amplitude in ASPECT_CONFIGS:
            result = FrozenFieldIslandSolver().solve(
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
            diagnostics = result.diagnostics
            writer.writerow(
                {
                    "angular_cells": angular_cells,
                    "axial_cells": axial_cells,
                    "axial_spacing_amplitude": amplitude,
                    "elements": diagnostics["elements"],
                    "h1_dofs": diagnostics["h1_dofs"],
                    "maximum_target_aspect_ratio": (
                        f"{float(diagnostics['maximum_target_aspect_ratio']):.16e}"
                    ),
                    "pollution_ratio": f"{float(diagnostics['pollution_ratio']):.16e}",
                    "linear_solver_path": diagnostics["linear_solver_path"],
                    "iteration_count": diagnostics["iteration_count"],
                }
            )


def regenerate_cost_ladder(selected_indices: tuple[int, ...] | None = None) -> None:
    """Write the required M4a/M4b cost and physics ladder from live solves."""
    required = (
        "elements",
        "h1_dofs",
        "polynomial_order",
        "assembly_seconds",
        "factorization_solve_seconds",
        "peak_memory_megabytes",
        "linear_solver_path",
        "iteration_count",
    )
    physics = (
        "maximum_target_aspect_ratio",
        "pollution_ratio",
        "layer_cells",
        "total_power_relative_error",
        "conservative_flux_relative_correction",
        "conservative_flux_divergence_relative_error",
        "volume_averaged_dp_ds",
        "island_flattening_width",
        "island_flattening_width_fraction_095",
        "island_flattening_width_fraction_097",
        "island_flattening_width_fraction_099",
        "integrable_flattening_width",
        "isotropic_flattening_width",
        "subcritical_flattening_width",
        "island_pressure_drop",
        "integrable_pressure_drop",
        "coarea_spike_ratio",
        "volume_plateau_ratio",
        "island_level_volume_coordinate",
        "island_level_mollifier_width",
        "critical_safeguard_samples",
        "minimum_mollifier_width",
        "maximum_mollifier_width",
    )
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    fields = ("source_commit", "epsilon_kappa", "angular_cells", "axial_cells", *required, *physics)
    rows: dict[tuple[float, int, int], dict[str, object]] = {}
    if selected_indices is not None and COST_TABLE.exists():
        with COST_TABLE.open(newline="", encoding="utf-8") as stream:
            for existing in csv.DictReader(stream):
                key = (
                    float(existing["epsilon_kappa"]),
                    int(existing["angular_cells"]),
                    int(existing["axial_cells"]),
                )
                if existing.get("source_commit") != source_commit:
                    raise RuntimeError(
                        "existing cost rows have different provenance; regenerate the whole ladder"
                    )
                rows[key] = dict(existing)
    for index, (epsilon_kappa, angular_cells, axial_cells) in enumerate(LADDER_CONFIGS):
        if selected_indices is None or index in selected_indices:
            context = mp.get_context("spawn")
            queue = context.Queue()
            process = context.Process(target=_solve_cost_row, args=(index, queue))
            process.start()
            diagnostics = queue.get()
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"cost-row process {index} exited with {process.exitcode}")
            row: dict[str, object] = {
                "source_commit": source_commit,
                "epsilon_kappa": f"{epsilon_kappa:.16e}",
                "angular_cells": angular_cells,
                "axial_cells": axial_cells,
            }
            for name in required:
                value = diagnostics[name]
                row[name] = f"{value:.16e}" if isinstance(value, float) else value
            for name in physics:
                value = diagnostics[name]
                row[name] = f"{value:.16e}" if isinstance(value, float) else value
            rows[(epsilon_kappa, angular_cells, axial_cells)] = row
    missing = [configuration for configuration in LADDER_CONFIGS if configuration not in rows]
    if missing:
        raise RuntimeError(f"cost table is missing configurations: {missing}")
    with COST_TABLE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for configuration in LADDER_CONFIGS:
            writer.writerow(rows[configuration])


def regenerate_overlay() -> None:
    """Overlay live M4b isobars and Poincare points on the Phi=0 section."""
    matplotlib_cache = Path(tempfile.gettempdir()) / "remec-matplotlib"
    matplotlib_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = FrozenFieldIslandConfig(
        epsilon_kappa=1.0e-4,
        epsilon_1=1.0e-3,
        polynomial_order=2,
        geometry_order=4,
        angular_cells=24,
        axial_cells=4,
        direct_dof_threshold=100_000,
    )
    model = ReimanGreensideField(epsilon_1=config.epsilon_1)
    resonance = model.resonance_radius(0.5)
    island_width = exact_island_width(epsilon_1=config.epsilon_1, t1=model.t1)
    critical_width = critical_layer_width(
        epsilon_kappa=config.epsilon_kappa,
        resonance_radius=resonance,
        t1=model.t1,
        major_radius=model.major_radius,
    )
    cylinder = PeriodicCylinder3D(geometry_order=config.geometry_order)
    bundle = cylinder.build_graded_mesh(
        (
            GradedAnnulus(
                resonance,
                0.6 * max(island_width, critical_width),
                critical_width / config.min_layer_cells,
            ),
        ),
        angular_cells=config.angular_cells,
        axial_cells=config.axial_cells,
        background_radial_width=config.background_radial_width,
    )
    profile = TabulatedPressureProfile((0.0, 1.0), (1.0, 0.0))
    state = _solve_m4(
        bundle._mesh,
        model,
        config,
        profile,
        perpendicular_conductivity=config.epsilon_kappa,
    )

    coordinates = np.linspace(-0.99, 0.99, 241)
    x, y = np.meshgrid(coordinates, coordinates, indexing="xy")
    inside = x * x + y * y < 0.99**2
    pressure = np.full_like(x, np.nan)
    chi = np.asarray(
        state.field(bundle._mesh(x[inside], y[inside], np.zeros(np.count_nonzero(inside)))),
        dtype=float,
    ).reshape(-1)
    pressure[inside] = np.asarray(
        profile.value(state.volume_map.evaluate_volume_coordinate(chi)), dtype=float
    )

    seed_radii = np.concatenate(
        (
            np.linspace(0.12, resonance - 0.8 * island_width, 6),
            np.linspace(resonance - 0.55 * island_width, resonance + 0.55 * island_width, 13),
            np.linspace(resonance + 0.8 * island_width, 0.94, 5),
        )
    )
    seeds = np.column_stack((seed_radii, np.zeros_like(seed_radii)))
    trace = trace_poincare(model, seeds, turns=48)
    theta = np.mod(trace.theta_unwrapped, 2.0 * np.pi)
    poincare_x = trace.radii * np.cos(theta)
    poincare_y = trace.radii * np.sin(theta)

    figure, axes = plt.subplots(1, 2, figsize=(13.2, 6.4), constrained_layout=True)
    for axis in axes:
        axis.scatter(poincare_x, poincare_y, s=2.0, color="0.55", alpha=0.65, label="Poincare")
        contours = axis.contour(x, y, pressure, levels=24, cmap="viridis", linewidths=1.0)
        axis.clabel(contours, contours.levels[::4], fontsize=7, fmt="%.2f")
        axis.set(aspect="equal", xlabel="x", ylabel="y")
    axes[0].add_patch(plt.Circle((0.0, 0.0), 1.0, fill=False, color="black", linewidth=1.2))
    axes[0].set_title("Full Phi=0 section")
    axes[0].legend(loc="upper right")
    axes[1].set(xlim=(0.58, 0.90), ylim=(-0.22, 0.22), title="m=2 island O-point zoom")
    figure.suptitle("M4b isobars on Reiman–Greenside Poincare section")
    figure.savefig(OVERLAY, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aspect-scan", action="store_true")
    parser.add_argument("--cost-ladder", action="store_true")
    parser.add_argument("--cost-row", action="append", type=int, default=[])
    parser.add_argument("--overlay", action="store_true")
    arguments = parser.parse_args()
    run_all = not (
        arguments.aspect_scan or arguments.cost_ladder or arguments.cost_row or arguments.overlay
    )
    if run_all or arguments.aspect_scan:
        regenerate_aspect_scan()
    if run_all or arguments.cost_ladder:
        regenerate_cost_ladder()
    elif arguments.cost_row:
        regenerate_cost_ladder(tuple(arguments.cost_row))
    if run_all or arguments.overlay:
        regenerate_overlay()


if __name__ == "__main__":
    main()
