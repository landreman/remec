"""Comparison study for constrained note ``(M3)`` gradient variants."""

from __future__ import annotations

import csv
from dataclasses import replace
from math import cos, log, pi, sin, sqrt
from pathlib import Path
from typing import Any

import ngsolve as ng
import numpy as np
import pytest

from remec import AnalyticToroidalCurrentProfile, RuntimeOptions
from remec.geometry.slab import Slab2D
from remec.solvers.current_continuity import (
    ConstrainedCurrentContinuitySolver,
    FrozenCurrentConstraintGeometry,
    FrozenCurrentContinuityCoefficients,
)

_DU_TABLE = Path(__file__).with_name("m3_gradient_du_limit.csv")
_MISALIGNMENT_TABLE = Path(__file__).with_name("m3_gradient_misalignment.csv")


def _recorded_du_rows() -> dict[tuple[str, float], dict[str, float]]:
    """Read deterministic fields from the checked-in fixed-state comparison scan."""
    with _DU_TABLE.open(newline="") as table_file:
        return {
            (row["variant"], float(row["current_diffusivity"])): {
                name: float(row[name])
                for name in (
                    "epsilon_j",
                    "cross_variant_relative_l2",
                    "cross_variant_over_epsilon_j",
                    "multiplier_current_l2",
                    "regularizing_toroidal_current_l2",
                    "layer_fwhm",
                    "radial_turning_points",
                    "parallel_noise_transfer",
                    "minimum_field_magnitude",
                    "minimum_shell_radial_cells",
                    "minimum_shell_mollifier_widths",
                )
            }
            for row in csv.DictReader(table_file)
        }


def _recorded_misalignment_rows() -> dict[tuple[str, str], dict[str, float]]:
    """Read the checked-in field/mesh misalignment comparison."""
    with _MISALIGNMENT_TABLE.open(newline="") as table_file:
        return {
            (row["variant"], row["alignment"]): {
                name: float(row[name])
                for name in (
                    "mesh_field_misalignment_degrees",
                    "coarse_elements",
                    "fine_elements",
                    "coarse_to_fine_relative_l2",
                    "cross_variant_relative_l2",
                    "misalignment_amplification",
                    "multiplier_current_l2",
                    "minimum_shell_radial_cells",
                    "minimum_shell_mollifier_widths",
                )
            }
            for row in csv.DictReader(table_file)
        }


def _resonant_comparison_case(
    diffusivity: float,
) -> tuple[
    FrozenCurrentContinuityCoefficients,
    FrozenCurrentConstraintGeometry,
    AnalyticToroidalCurrentProfile,
]:
    r"""Return one fixed resonant ``(M2)``--``(M3b)`` comparison state.

    ``B_y=6(x-1/2)`` makes the fundamental periodic harmonic resonant at
    ``x=1/2``.  The frozen drive contains a smaller fifth harmonic so the study can
    separately measure layer smearing and parallel grid-noise damping.  Neither the
    drive nor the normalized cumulative target ``I_0(s)`` depends on ``D_u`` or on
    the selected regularization gradient.
    """
    magnetic_field = ng.CoefficientFunction((0.0, 6.0 * (ng.x - 0.5), 2.0))
    pressure_gradient = ng.CoefficientFunction((1.0, 0.0, 0.0))
    magnetic_floor = 1.0e-8
    safe_magnitude = ng.sqrt(ng.InnerProduct(magnetic_field, magnetic_field) + magnetic_floor**2)
    frozen_drive = ng.sin(2.0 * ng.pi * ng.y) + 0.05 * ng.sin(10.0 * ng.pi * ng.y)
    drive_direction = ng.Cross(magnetic_field, pressure_gradient)
    magnetic_magnitude_gradient = (
        frozen_drive
        * safe_magnitude**3
        * drive_direction
        / (2.0 * ng.InnerProduct(drive_direction, drive_direction))
    )
    coefficients = FrozenCurrentContinuityCoefficients(
        magnetic_field=magnetic_field,
        pressure_gradient=pressure_gradient,
        magnetic_magnitude_gradient=magnetic_magnitude_gradient,
        current_diffusivity=diffusivity,
        magnetic_floor=magnetic_floor,
        vacuum_permeability=1.0e-8,
    )
    geometry = FrozenCurrentConstraintGeometry(
        level_set=1.0 - ng.x,
        level_set_gradient=ng.CoefficientFunction((-1.0, 0.0, 0.0)),
        toroidal_angle_gradient=ng.CoefficientFunction(
            (0.4, 0.3 * ng.sin(2.0 * ng.pi * ng.y), 1.0)
        ),
    )
    profile = AnalyticToroidalCurrentProfile(
        enclosed_current_function=lambda s: (
            0.04 * np.asarray(s) + 0.01 * np.asarray(s) * (1.0 - np.asarray(s))
        ),
        derivative_function=lambda s: 0.05 - 0.02 * np.asarray(s),
    )
    profile.validate()
    return coefficients, geometry, profile


def _misaligned_comparison_case(
    diffusivity: float,
    *,
    angle_degrees: float,
) -> tuple[
    FrozenCurrentContinuityCoefficients,
    FrozenCurrentConstraintGeometry,
    AnalyticToroidalCurrentProfile,
]:
    r"""Return a fixed state with a controlled in-plane field/mesh angle.

    The structured triangular mesh has edge directions at 0, 45, and 90 degrees.
    At 22.5 degrees this constant field bisects the first two directions, giving the
    largest possible nearest-edge angular misalignment for that mesh family; zero
    degrees is its aligned control.  The explicit frozen ``(M3)`` drive is independent
    of ``D_u`` and of ``grad_r``.
    """
    angle = angle_degrees * pi / 180.0
    magnetic_field = ng.CoefficientFunction((cos(angle), sin(angle), 1.5))
    pressure_gradient = ng.CoefficientFunction((0.4, 0.0, 0.0))
    magnetic_floor = 1.0e-10
    safe_magnitude = ng.sqrt(ng.InnerProduct(magnetic_field, magnetic_field) + magnetic_floor**2)
    frozen_drive = ng.sin(2.0 * ng.pi * ng.x) * ng.sin(2.0 * ng.pi * ng.y)
    drive_direction = ng.Cross(magnetic_field, pressure_gradient)
    magnetic_magnitude_gradient = (
        frozen_drive
        * safe_magnitude**3
        * drive_direction
        / (2.0 * ng.InnerProduct(drive_direction, drive_direction))
    )
    coefficients = FrozenCurrentContinuityCoefficients(
        magnetic_field=magnetic_field,
        pressure_gradient=pressure_gradient,
        magnetic_magnitude_gradient=magnetic_magnitude_gradient,
        current_diffusivity=diffusivity,
        magnetic_floor=magnetic_floor,
        vacuum_permeability=0.2,
    )
    geometry = FrozenCurrentConstraintGeometry(
        level_set=1.0 - ng.x,
        level_set_gradient=ng.CoefficientFunction((-1.0, 0.0, 0.0)),
        toroidal_angle_gradient=ng.CoefficientFunction((0.7, -0.2, 1.1)),
    )
    profile = AnalyticToroidalCurrentProfile(
        enclosed_current_function=lambda s: (
            0.04 * np.asarray(s) + 0.01 * np.asarray(s) * (1.0 - np.asarray(s))
        ),
        derivative_function=lambda s: 0.05 - 0.02 * np.asarray(s),
    )
    profile.validate()
    return coefficients, geometry, profile


def _solve_case(
    case: tuple[
        FrozenCurrentContinuityCoefficients,
        FrozenCurrentConstraintGeometry,
        AnalyticToroidalCurrentProfile,
    ],
    variant: str,
    *,
    subdivisions: tuple[int, int],
) -> tuple[ConstrainedCurrentContinuitySolver, Any]:
    """Solve one comparison row with the production constrained kernel."""
    coefficients, geometry, profile = case
    solver = ConstrainedCurrentContinuitySolver(
        polynomial_order=2,
        runtime=RuntimeOptions(regularization_gradient=variant),  # type: ignore[arg-type]
        quadrature_order=6,
        volume_levels=17,
    )
    result = solver.solve(
        Slab2D(
            maxh=1.0 / subdivisions[1],
            subdivisions=subdivisions,
            periodic_y=True,
        ),
        coefficients,
        geometry,
        profile,
        shell_edges=np.linspace(0.0, 1.0, 5),
        edge_value=0.2,
    )
    return solver, result


def _relative_l2_difference(
    first: ConstrainedCurrentContinuitySolver,
    second: ConstrainedCurrentContinuitySolver,
) -> float:
    """Evaluate two physical ``u`` fields on one common order-16 quadrature rule."""
    first_solution = first._solution()
    second_solution = second._solution()
    mesh = second_solution.mesh()
    element_types = {element.type for element in mesh.Elements(ng.VOL)}
    rules = {element_type: ng.IntegrationRule(element_type, 16) for element_type in element_types}
    weights = np.asarray(
        [
            float(point.weight) * float(mesh.GetTrafo(element)(point).measure)
            for element in mesh.Elements(ng.VOL)
            for point in rules[element.type]
        ]
    )
    mapped_points = mesh.MapToAllElements(rules, ng.VOL)
    x_coordinates = np.asarray(ng.x(mapped_points), dtype=float).reshape(-1)
    y_coordinates = np.asarray(ng.y(mapped_points), dtype=float).reshape(-1)
    first_points = first_solution.mesh()(x_coordinates, y_coordinates)
    first_values = np.asarray(first_solution.grid_function()(first_points), dtype=float).reshape(-1)
    second_values = np.asarray(second_solution.grid_function()(mapped_points), dtype=float).reshape(
        -1
    )
    difference_norm = sqrt(float(np.dot(weights, (second_values - first_values) ** 2)))
    reference_norm = sqrt(float(np.dot(weights, first_values**2)))
    return difference_norm / reference_norm


def _harmonic_amplitude(
    solver: ConstrainedCurrentContinuitySolver,
    x_coordinate: float,
    harmonic: int,
) -> float:
    r"""Return one Fourier amplitude of physical ``J_parallel/B`` along the field."""
    y_coordinates = np.arange(64, dtype=float) / 64.0
    values = np.asarray(
        [
            solver.parallel_current_over_field_at(x_coordinate, float(y_coordinate))
            for y_coordinate in y_coordinates
        ]
    )
    values -= np.mean(values)
    sine_component = (
        2.0 * float(np.dot(values, np.sin(2.0 * pi * harmonic * y_coordinates))) / len(values)
    )
    cosine_component = (
        2.0 * float(np.dot(values, np.cos(2.0 * pi * harmonic * y_coordinates))) / len(values)
    )
    return sqrt(sine_component**2 + cosine_component**2)


def _interpolated_crossing(
    first_x: float,
    first_value: float,
    second_x: float,
    second_value: float,
    target: float,
) -> float:
    """Linearly locate a sampled half-maximum crossing."""
    fraction = (target - first_value) / (second_value - first_value)
    return first_x + fraction * (second_x - first_x)


def _resonant_layer_observables(
    solver: ConstrainedCurrentContinuitySolver,
) -> tuple[float, int, float]:
    r"""Measure FWHM, radial turning points, and fifth-harmonic noise transfer."""
    x_coordinates = np.linspace(0.08, 0.92, 169)
    amplitudes = np.asarray(
        [_harmonic_amplitude(solver, float(x_coordinate), 1) for x_coordinate in x_coordinates]
    )
    peak = int(np.argmax(amplitudes))
    half_maximum = 0.5 * amplitudes[peak]
    left = int(np.flatnonzero(amplitudes[:peak] < half_maximum)[-1])
    right = peak + 1 + int(np.flatnonzero(amplitudes[peak + 1 :] < half_maximum)[0])
    left_crossing = _interpolated_crossing(
        float(x_coordinates[left]),
        float(amplitudes[left]),
        float(x_coordinates[left + 1]),
        float(amplitudes[left + 1]),
        float(half_maximum),
    )
    right_crossing = _interpolated_crossing(
        float(x_coordinates[right - 1]),
        float(amplitudes[right - 1]),
        float(x_coordinates[right]),
        float(amplitudes[right]),
        float(half_maximum),
    )
    slopes = np.diff(amplitudes)
    turning_points = int(np.count_nonzero(slopes[:-1] * slopes[1:] < 0.0))
    noise_transfer = _harmonic_amplitude(solver, 0.7, 5) / _harmonic_amplitude(solver, 0.7, 1)
    return right_crossing - left_crossing, turning_points, noise_transfer


@pytest.fixture(scope="module")
def resonant_scan() -> dict[float, dict[str, tuple[ConstrainedCurrentContinuitySolver, Any]]]:
    """Cache the fixed-frozen-state ``D_u`` scan shared by the acceptance tests."""
    return {
        diffusivity: {
            variant: _solve_case(
                _resonant_comparison_case(diffusivity),
                variant,
                subdivisions=(24, 16),
            )
            for variant in ("perpendicular", "full")
        }
        for diffusivity in (0.04, 0.02, 0.01)
    }


def test_constrained_comparison_reports_cost_and_invariant_diagnostics(
    resonant_scan: dict[float, dict[str, tuple[ConstrainedCurrentContinuitySolver, Any]]],
) -> None:
    r"""The shared-``I_0`` study exposes cost plus DESIGN §5 invariants 4--6."""
    required_diagnostics = {
        "a_assemblies",
        "a_factorizations",
        "a_factorization_reuses",
        "a_response_solves",
        "a_assembly_wall_seconds",
        "linear_form_assembly_wall_seconds",
        "diagnostic_assembly_wall_seconds",
        "factorization_and_response_wall_seconds",
        "bordered_solve_wall_seconds",
        "diagnostics_wall_seconds",
        "total_wall_seconds",
        "minimum_field_magnitude",
        "floor_activity_l2",
    }

    for variant in ("perpendicular", "full"):
        _, result = resonant_scan[0.02][variant]

        assert required_diagnostics <= result.diagnostics.keys()
        assert result.constraint_relative_residual_norm < 1.0e-10
        assert result.independent_cumulative_current == pytest.approx(
            result.target_cumulative_current,
            abs=1.0e-10,
        )
        assert result.diagnostics["minimum_shell_radial_cells"] >= 3.0
        assert result.diagnostics["minimum_shell_mollifier_widths"] >= 2.0
        assert result.diagnostics["floor_activity_l2"] < 1.0e-12
        assert result.diagnostics["a_assembly_wall_seconds"] > 0.0
        assert result.diagnostics["linear_form_assembly_wall_seconds"] > 0.0
        assert result.diagnostics["diagnostic_assembly_wall_seconds"] > 0.0
        assert result.diagnostics["factorization_and_response_wall_seconds"] > 0.0
        assert result.diagnostics["diagnostics_wall_seconds"] > 0.0
        assert (
            result.diagnostics["total_wall_seconds"]
            > result.diagnostics["bordered_solve_wall_seconds"]
        )
        assert result.diagnostics["a_assemblies"] == 1.0
        assert result.diagnostics["a_factorizations"] == 1.0
        assert result.diagnostics["a_factorization_reuses"] == 4.0
        assert result.diagnostics["a_response_solves"] == 5.0
        assert result.diagnostics["regularizing_toroidal_current_l2"] > 1.0e-4

    coefficients, geometry, profile = _resonant_comparison_case(0.02)
    _, active_floor_result = _solve_case(
        (replace(coefficients, magnetic_floor=2.0), geometry, profile),
        "perpendicular",
        subdivisions=(16, 16),
    )
    assert active_floor_result.diagnostics["minimum_field_magnitude"] == pytest.approx(
        2.0,
        abs=1.0e-12,
    )
    assert active_floor_result.diagnostics["floor_activity_l2"] > 0.25


def test_fixed_state_variants_are_o_epsilon_j_and_have_one_common_limit(
    resonant_scan: dict[float, dict[str, tuple[ConstrainedCurrentContinuitySolver, Any]]],
) -> None:
    r"""At fixed ``(B,p,s,I_0,drive)``, disagreement is ``O(epsilon_J)`` and vanishes."""
    recorded = _recorded_du_rows()
    differences: list[float] = []
    multiplier_norms: dict[str, list[float]] = {"perpendicular": [], "full": []}
    for diffusivity in (0.04, 0.02, 0.01):
        perpendicular, perpendicular_result = resonant_scan[diffusivity]["perpendicular"]
        full, full_result = resonant_scan[diffusivity]["full"]
        difference = _relative_l2_difference(perpendicular, full)
        epsilon_j = diffusivity / 2.0  # min |B|=2 and the slab reference length is one.
        differences.append(difference)

        assert 0.8 < difference / epsilon_j < 1.3
        for result in (perpendicular_result, full_result):
            assert result.constraint_relative_residual_norm < 1.0e-10
            assert result.independent_cumulative_current == pytest.approx(
                result.target_cumulative_current,
                abs=1.0e-10,
            )
            assert result.diagnostics["maximum_shell_mean_utilde"] < 1.0e-12
            assert result.diagnostics["floor_activity_l2"] < 1.0e-12
        for variant, result in (
            ("perpendicular", perpendicular_result),
            ("full", full_result),
        ):
            expected = recorded[variant, diffusivity]
            assert epsilon_j == pytest.approx(expected["epsilon_j"], abs=1.0e-14)
            assert difference == pytest.approx(
                expected["cross_variant_relative_l2"],
                rel=5.0e-6,
            )
            assert difference / epsilon_j == pytest.approx(
                expected["cross_variant_over_epsilon_j"],
                rel=5.0e-6,
            )
            assert result.diagnostics["multiplier_current_l2"] == pytest.approx(
                expected["multiplier_current_l2"],
                rel=5.0e-6,
            )
            assert result.diagnostics["regularizing_toroidal_current_l2"] == pytest.approx(
                expected["regularizing_toroidal_current_l2"],
                rel=5.0e-6,
            )
            multiplier_norms[variant].append(result.diagnostics["multiplier_current_l2"])

    convergence_rate = log(differences[-2] / differences[-1]) / log(2.0)
    assert differences == sorted(differences, reverse=True)
    assert differences[-1] < 0.3 * differences[0]
    assert convergence_rate > 0.8
    for norms in multiplier_norms.values():
        assert norms == sorted(norms, reverse=True)
        assert norms[-1] < 0.3 * norms[0]


def test_resonant_layer_records_smearing_oscillation_and_parallel_noise(
    resonant_scan: dict[float, dict[str, tuple[ConstrainedCurrentContinuitySolver, Any]]],
) -> None:
    r"""Both variants resolve one monotone layer; full grad damps the injected noise."""
    recorded = _recorded_du_rows()
    observables = {
        (variant, diffusivity): _resonant_layer_observables(resonant_scan[diffusivity][variant][0])
        for diffusivity in (0.04, 0.02, 0.01)
        for variant in ("perpendicular", "full")
    }
    perpendicular_width, perpendicular_turns, perpendicular_noise = observables[
        "perpendicular", 0.02
    ]
    full_width, full_turns, full_noise = observables["full", 0.02]

    assert perpendicular_width * 24.0 >= 6.0
    assert full_width * 24.0 >= 6.0
    assert abs(perpendicular_width - full_width) / perpendicular_width < 0.03
    assert perpendicular_turns == 1
    assert full_turns == 1
    assert full_noise < perpendicular_noise
    for (variant, diffusivity), (width, turns, noise) in observables.items():
        expected = recorded[variant, diffusivity]
        assert width * 24.0 >= 6.0
        assert width == pytest.approx(expected["layer_fwhm"], rel=5.0e-3)
        assert turns == int(expected["radial_turning_points"])
        assert noise == pytest.approx(expected["parallel_noise_transfer"], rel=5.0e-3)


def test_field_misalignment_sensitivity_has_an_aligned_control() -> None:
    r"""A 22.5-degree field/mesh mismatch is compared with a zero-degree control."""
    recorded = _recorded_misalignment_rows()
    solutions: dict[
        str,
        dict[str, dict[str, tuple[ConstrainedCurrentContinuitySolver, Any]]],
    ] = {}
    for alignment, angle_degrees in (("aligned", 0.0), ("misaligned", 22.5)):
        solutions[alignment] = {}
        for variant in ("perpendicular", "full"):
            solutions[alignment][variant] = {
                "coarse": _solve_case(
                    _misaligned_comparison_case(0.01, angle_degrees=angle_degrees),
                    variant,
                    subdivisions=(20, 20),
                ),
                "fine": _solve_case(
                    _misaligned_comparison_case(0.01, angle_degrees=angle_degrees),
                    variant,
                    subdivisions=(28, 28),
                ),
            }

    sensitivities = {
        (alignment, variant): _relative_l2_difference(
            solutions[alignment][variant]["coarse"][0],
            solutions[alignment][variant]["fine"][0],
        )
        for alignment in ("aligned", "misaligned")
        for variant in ("perpendicular", "full")
    }
    cross_variant = {
        alignment: _relative_l2_difference(
            solutions[alignment]["perpendicular"]["fine"][0],
            solutions[alignment]["full"]["fine"][0],
        )
        for alignment in ("aligned", "misaligned")
    }

    for alignment, angle_degrees in (("aligned", 0.0), ("misaligned", 22.5)):
        for variant in ("perpendicular", "full"):
            result = solutions[alignment][variant]["fine"][1]
            expected = recorded[variant, alignment]
            amplification = sensitivities["misaligned", variant] / sensitivities["aligned", variant]
            assert result.constraint_relative_residual_norm < 1.0e-10
            assert result.diagnostics["minimum_shell_radial_cells"] >= 3.0
            assert result.diagnostics["minimum_shell_mollifier_widths"] >= 2.0
            assert expected["mesh_field_misalignment_degrees"] == angle_degrees
            assert solutions[alignment][variant]["coarse"][0]._solution().mesh().ne == int(
                expected["coarse_elements"]
            )
            assert solutions[alignment][variant]["fine"][0]._solution().mesh().ne == int(
                expected["fine_elements"]
            )
            assert sensitivities[alignment, variant] == pytest.approx(
                expected["coarse_to_fine_relative_l2"],
                rel=5.0e-3,
            )
            assert cross_variant[alignment] == pytest.approx(
                expected["cross_variant_relative_l2"],
                rel=5.0e-3,
            )
            assert amplification == pytest.approx(
                expected["misalignment_amplification"],
                rel=5.0e-3,
            )
            assert result.diagnostics["multiplier_current_l2"] == pytest.approx(
                expected["multiplier_current_l2"],
                rel=5.0e-3,
            )
            assert result.diagnostics["minimum_shell_radial_cells"] == pytest.approx(
                expected["minimum_shell_radial_cells"],
                rel=5.0e-3,
            )
            assert result.diagnostics["minimum_shell_mollifier_widths"] == pytest.approx(
                expected["minimum_shell_mollifier_widths"],
                rel=5.0e-3,
            )
