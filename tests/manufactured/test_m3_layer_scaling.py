"""Resonant-layer verification for note equation (M3) and Eq. ``layer_width``."""

from __future__ import annotations

import csv
from math import cos, log, pi, sin, sqrt
from pathlib import Path

import ngsolve as ng
import pytest

from remec import RuntimeOptions
from remec.geometry.slab import Slab2D
from remec.solvers.current_continuity import (
    CurrentContinuitySolver,
    FrozenCurrentContinuityCoefficients,
    PrescribedCurrentProfile,
)

_DIFFUSIVITIES = (0.005, 0.01, 0.02, 0.04)
_NORMAL_ELEMENT_WIDTH = 1.0 / 64.0
_RATE_TABLE = Path(__file__).with_name("m3_layer_scaling.csv")
_REFINEMENT_TABLE = Path(__file__).with_name("m3_layer_mesh_refinement.csv")


def _recorded_rows() -> dict[tuple[str, float], dict[str, float]]:
    """Read the checked-in M3 resonant-layer width scan."""
    with _RATE_TABLE.open(newline="") as table_file:
        rows = csv.DictReader(table_file)
        return {
            (row["variant"], float(row["current_diffusivity"])): {
                name: float(row[name])
                for name in (
                    "elements",
                    "measured_layer_width",
                    "theoretical_inner_scale",
                    "fwhm_to_inner_scale",
                    "cells_across_layer",
                    "fitted_exponent",
                )
            }
            for row in rows
        }


def _recorded_refinement_rows() -> dict[tuple[int, int], dict[str, float]]:
    """Read the checked-in mesh-independence check for the thinnest M3 layer."""
    with _REFINEMENT_TABLE.open(newline="") as table_file:
        rows = csv.DictReader(table_file)
        return {
            (int(row["nx"]), int(row["ny"])): {
                name: float(row[name])
                for name in ("elements", "measured_layer_width", "cells_across_layer")
            }
            for row in rows
        }


def _theoretical_inner_scale(diffusivity: float) -> float:
    r"""Return the unit-coefficient estimate ``(D_u / (40 pi))**(1/3)``."""
    return (diffusivity / (40.0 * pi)) ** (1.0 / 3.0)


def _layer_case(
    diffusivity: float,
) -> tuple[FrozenCurrentContinuityCoefficients, PrescribedCurrentProfile]:
    r"""Return the note-``layer_equation`` reduction of (M3).

    With ``B_y = 20 (x - 1/2)`` and the fundamental periodic harmonic
    ``exp(i 2 pi y)``, the retained balance is

    ``i 40 pi (x - 1/2) u_hat - D_u u_hat'' = h_hat``.

    Thus Eq. ``layer_width`` predicts ``delta proportional to D_u**(1/3)``.
    The explicit ``grad(B)`` input injects the fixed resonant M3 drive, as permitted
    by the frozen verification interface; a tiny nondimensional ``mu0`` isolates the
    reduced balance whose omitted correction is smaller by the order stated in the
    note's layer analysis.
    """
    shear = 20.0
    magnetic_field = ng.CoefficientFunction((0.0, shear * (ng.x - 0.5), 10.0))
    pressure_gradient = ng.CoefficientFunction((1.0, 0.0, 0.0))
    magnetic_floor = 1.0e-8
    safe_magnitude = ng.sqrt(ng.InnerProduct(magnetic_field, magnetic_field) + magnetic_floor**2)
    resonant_drive = ng.sin(2.0 * ng.pi * ng.y)
    drive_direction = ng.Cross(magnetic_field, pressure_gradient)
    prescribed_magnitude_gradient = (
        resonant_drive
        * safe_magnitude**3
        * drive_direction
        / (2.0 * ng.InnerProduct(drive_direction, drive_direction))
    )
    coefficients = FrozenCurrentContinuityCoefficients(
        magnetic_field=magnetic_field,
        pressure_gradient=pressure_gradient,
        magnetic_magnitude_gradient=prescribed_magnitude_gradient,
        current_diffusivity=diffusivity,
        magnetic_floor=magnetic_floor,
        vacuum_permeability=1.0e-8,
    )
    profile = PrescribedCurrentProfile(
        identifier="resonant-layer-constant-f-v1",
        value=0.2,
        pressure_derivative=0.0,
        perpendicular_gradient_divergence=0.0,
        full_gradient_divergence=0.0,
    )
    return coefficients, profile


def _harmonic_amplitude(solver: CurrentContinuitySolver, x_coordinate: float) -> float:
    r"""Return the physical-(M2) ``J_parallel/B`` harmonic amplitude."""
    y_coordinates = [index / 65.0 for index in range(65)]
    values = [
        solver.parallel_current_over_field_at(x_coordinate, y_coordinate) - 0.2
        for y_coordinate in y_coordinates
    ]
    sine_component = (
        2.0
        * sum(
            value * sin(2.0 * pi * y_coordinate)
            for value, y_coordinate in zip(values, y_coordinates, strict=True)
        )
        / len(values)
    )
    cosine_component = (
        2.0
        * sum(
            value * cos(2.0 * pi * y_coordinate)
            for value, y_coordinate in zip(values, y_coordinates, strict=True)
        )
        / len(values)
    )
    return sqrt(sine_component**2 + cosine_component**2)


def _interpolated_crossing(
    first_x: float,
    first_value: float,
    second_x: float,
    second_value: float,
    target: float,
) -> float:
    """Linearly locate one half-maximum crossing between adjacent samples."""
    fraction = (target - first_value) / (second_value - first_value)
    return first_x + fraction * (second_x - first_x)


def _layer_fwhm(solver: CurrentContinuitySolver) -> float:
    """Measure the full width at half maximum of the resonant M2 current harmonic."""
    sample_count = 281
    x_coordinates = [0.15 + 0.7 * index / (sample_count - 1) for index in range(sample_count)]
    amplitudes = [_harmonic_amplitude(solver, x_coordinate) for x_coordinate in x_coordinates]
    peak_index = max(range(sample_count), key=amplitudes.__getitem__)
    assert 0 < peak_index < sample_count - 1
    half_maximum = 0.5 * amplitudes[peak_index]
    left_candidates = [index for index in range(peak_index) if amplitudes[index] < half_maximum]
    right_candidates = [
        index for index in range(peak_index + 1, sample_count) if amplitudes[index] < half_maximum
    ]
    assert left_candidates, "layer FWHM has no left half-maximum crossing in the sample window"
    assert right_candidates, "layer FWHM has no right half-maximum crossing in the sample window"
    left_below = max(left_candidates)
    right_below = min(right_candidates)
    left_crossing = _interpolated_crossing(
        x_coordinates[left_below],
        amplitudes[left_below],
        x_coordinates[left_below + 1],
        amplitudes[left_below + 1],
        half_maximum,
    )
    right_crossing = _interpolated_crossing(
        x_coordinates[right_below - 1],
        amplitudes[right_below - 1],
        x_coordinates[right_below],
        amplitudes[right_below],
        half_maximum,
    )
    return right_crossing - left_crossing


def _log_log_slope(abscissae: list[float], ordinates: list[float]) -> float:
    """Return the least-squares power-law exponent."""
    log_abscissae = [log(value) for value in abscissae]
    log_ordinates = [log(value) for value in ordinates]
    mean_abscissa = sum(log_abscissae) / len(log_abscissae)
    mean_ordinate = sum(log_ordinates) / len(log_ordinates)
    return sum(
        (x_value - mean_abscissa) * (y_value - mean_ordinate)
        for x_value, y_value in zip(log_abscissae, log_ordinates, strict=True)
    ) / sum((value - mean_abscissa) ** 2 for value in log_abscissae)


@pytest.mark.parametrize("variant", ["perpendicular", "full"])
def test_resonant_m3_layer_width_scales_as_diffusivity_to_one_third(variant: str) -> None:
    r"""Eq. ``layer_width`` gives ``delta ~ D_u**(1/3)`` for both M3 variants."""
    slab = Slab2D(
        maxh=1.0 / 16.0,
        subdivisions=(64, 16),
        periodic_y=True,
    )
    recorded_rows = _recorded_rows()
    widths: list[float] = []
    for diffusivity in _DIFFUSIVITIES:
        coefficients, profile = _layer_case(diffusivity)
        solver = CurrentContinuitySolver(
            polynomial_order=3,
            runtime=RuntimeOptions(regularization_gradient=variant),  # type: ignore[arg-type]
            stabilization="supg",
        )
        result = solver.solve_utilde(
            slab,
            coefficients,
            profile,
            boundary="left|right",
        )
        width = _layer_fwhm(solver)
        resolution = solver.assess_layer_resolution(
            layer_width=width,
            normal_element_width=_NORMAL_ELEMENT_WIDTH,
        )
        expected = recorded_rows[variant, diffusivity]

        assert solver._solution().mesh().ne == expected["elements"]
        assert width == pytest.approx(expected["measured_layer_width"], rel=0.05)
        assert expected["theoretical_inner_scale"] == pytest.approx(
            _theoretical_inner_scale(diffusivity), rel=1.0e-8
        )
        assert width / _theoretical_inner_scale(diffusivity) == pytest.approx(
            expected["fwhm_to_inner_scale"], rel=0.05
        )
        assert resolution.cells_across_layer == pytest.approx(
            expected["cells_across_layer"], rel=0.05
        )
        assert resolution.resolved
        assert resolution.cells_across_layer >= 6.0
        assert result.free_dof_relative_residual_norm < 1.0e-11
        assert result.diagnostics["minimum_field_magnitude"] == pytest.approx(10.0, rel=1.0e-3)
        assert result.diagnostics["floor_activity_l2"] < 1.0e-12
        widths.append(width)

    fitted_exponent = _log_log_slope(list(_DIFFUSIVITIES), widths)
    assert widths == sorted(widths)
    assert abs(fitted_exponent - 1.0 / 3.0) < 0.04
    assert fitted_exponent == pytest.approx(
        recorded_rows[variant, _DIFFUSIVITIES[0]]["fitted_exponent"], abs=0.02
    )


def test_thinnest_resonant_layer_fwhm_is_mesh_independent() -> None:
    """The six-cell FWHM verdict is unchanged by one layer-aligned h-refinement."""
    diffusivity = _DIFFUSIVITIES[0]
    coefficients, profile = _layer_case(diffusivity)
    recorded_rows = _recorded_refinement_rows()
    measured_widths: dict[tuple[int, int], float] = {}

    for subdivisions in ((64, 16), (96, 24)):
        nx, ny = subdivisions
        solver = CurrentContinuitySolver(
            polynomial_order=3,
            runtime=RuntimeOptions(regularization_gradient="perpendicular"),
            stabilization="supg",
        )
        solver.solve_utilde(
            Slab2D(maxh=1.0 / ny, subdivisions=subdivisions, periodic_y=True),
            coefficients,
            profile,
            boundary="left|right",
        )
        measured_width = _layer_fwhm(solver)
        expected = recorded_rows[subdivisions]
        measured_widths[subdivisions] = measured_width

        assert solver._solution().mesh().ne == expected["elements"]
        assert measured_width == pytest.approx(expected["measured_layer_width"], rel=0.02)
        assert nx * measured_width == pytest.approx(expected["cells_across_layer"], rel=0.02)

    relative_change = (
        abs(measured_widths[64, 16] - measured_widths[96, 24]) / measured_widths[96, 24]
    )
    assert relative_change < 0.01


@pytest.mark.parametrize("variant", ["perpendicular", "full"])
def test_resonant_utilde_layer_retains_direct_u_and_m2_cross_checks(variant: str) -> None:
    r"""Preferred utilde and direct (M3) agree in physical u and reconstructed (M2)."""
    slab = Slab2D(
        maxh=1.0 / 16.0,
        subdivisions=(64, 16),
        periodic_y=True,
    )
    coefficients, profile = _layer_case(0.01)
    runtime = RuntimeOptions(regularization_gradient=variant)  # type: ignore[arg-type]
    direct = CurrentContinuitySolver(
        polynomial_order=3,
        runtime=runtime,
        stabilization="supg",
    )
    direct.solve(
        slab,
        coefficients,
        boundary="left|right",
        boundary_value=profile.value,
    )
    transformed = CurrentContinuitySolver(
        polynomial_order=3,
        runtime=runtime,
        stabilization="supg",
    )
    transformed.solve_utilde(
        slab,
        coefficients,
        profile,
        boundary="left|right",
    )

    for solver in (direct, transformed):
        for x_coordinate in (0.35, 0.50, 0.65):
            assert solver.solution_at(x_coordinate, 0.0) == pytest.approx(
                solver.solution_at(x_coordinate, 1.0), abs=1.0e-11
            )

    sample_points = (
        (0.35, 0.125),
        (0.45, 0.375),
        (0.50, 0.625),
        (0.65, 0.875),
    )
    for x_coordinate, y_coordinate in sample_points:
        assert transformed.solution_at(x_coordinate, y_coordinate) == pytest.approx(
            direct.solution_at(x_coordinate, y_coordinate), abs=1.0e-10
        )
        assert transformed.current_at(x_coordinate, y_coordinate) == pytest.approx(
            direct.current_at(x_coordinate, y_coordinate), abs=1.0e-9
        )
        assert transformed.parallel_current_over_field_at(
            x_coordinate, y_coordinate
        ) == pytest.approx(
            direct.parallel_current_over_field_at(x_coordinate, y_coordinate), abs=1.0e-10
        )
