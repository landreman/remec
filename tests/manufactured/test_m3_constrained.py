"""Manufactured contracts for the constrained note ``(M3)``--``(M3b)`` solve."""

from __future__ import annotations

import csv
from math import log, pi, sqrt
from pathlib import Path

import ngsolve as ng
import numpy as np
import pytest

from remec import AnalyticToroidalCurrentProfile, RuntimeOptions
from remec.geometry.slab import Slab2D
from remec.solvers.current_continuity import (
    ConstrainedCurrentContinuitySolver,
    CurrentContinuitySolver,
    FrozenCurrentConstraintGeometry,
    FrozenCurrentContinuityCoefficients,
    PrescribedCurrentProfile,
)

_RATE_TABLE = Path(__file__).with_name("m3_constrained_rates.csv")
_DU_TABLE = Path(__file__).with_name("m3_constrained_du_scan.csv")


def _geometry() -> FrozenCurrentConstraintGeometry:
    r"""Return a periodic toroidal surrogate with shared ``s=V_chi/V_omega``."""
    return FrozenCurrentConstraintGeometry(
        level_set=1.0 - ng.x,
        level_set_gradient=ng.CoefficientFunction((-1.0, 0.0, 0.0)),
        toroidal_angle_gradient=ng.CoefficientFunction((0.0, 0.0, 2.0 * pi)),
    )


def _coefficients(diffusivity: float) -> FrozenCurrentContinuityCoefficients:
    r"""Return a frozen (M3) case whose exact homogeneous ``utilde`` is zero."""
    return FrozenCurrentContinuityCoefficients(
        magnetic_field=ng.CoefficientFunction((0.0, 1.0, 2.0)),
        pressure_gradient=ng.CoefficientFunction((0.0, 0.0, 0.0)),
        magnetic_magnitude_gradient=ng.CoefficientFunction((0.0, 0.0, 0.0)),
        current_diffusivity=diffusivity,
        magnetic_floor=1.0e-12,
        vacuum_permeability=1.0,
    )


def _profile(kind: int) -> AnalyticToroidalCurrentProfile:
    if kind == 1:
        profile = AnalyticToroidalCurrentProfile(
            enclosed_current_function=lambda s: 0.6 * np.asarray(s) ** 2,
            derivative_function=lambda s: 1.2 * np.asarray(s),
        )
    else:
        profile = AnalyticToroidalCurrentProfile(
            enclosed_current_function=lambda s: 0.2 * np.asarray(s) + 0.5 * np.asarray(s) ** 3,
            derivative_function=lambda s: 0.2 + 1.5 * np.asarray(s) ** 2,
        )
    profile.validate()
    return profile


def _regularized_gradient(gradient: object, magnetic_field: object, variant: str) -> object:
    """Return the independent constant-field ``grad_r`` used by the manufactured oracle."""
    safe_magnitude = ng.sqrt(ng.InnerProduct(magnetic_field, magnetic_field) + 1.0e-24)
    direction = magnetic_field / safe_magnitude
    if variant == "full":
        return gradient
    return gradient - direction * ng.InnerProduct(direction, gradient)


def _coupled_manufactured_case(
    variant: str,
    *,
    diffusivity: float = 0.03,
) -> tuple[
    FrozenCurrentContinuityCoefficients,
    FrozenCurrentConstraintGeometry,
    AnalyticToroidalCurrentProfile,
    object,
    float,
]:
    r"""Manufacture every unknown-``G`` coupling in ``(M2)``--``(M3b)``.

    ``G(s)=g_0+g_1 s`` and a nonzero periodic ``utilde`` make
    ``-G' B.grad(s)``, ``-(mu0 G/B**2) B.grad(p)``, the diamagnetic current, and
    ``-D_u grad_r(utilde)`` all nonzero.  The injected magnitude-gradient numerator
    transcribes the exact strong (M3) drive independently of the bordered assembly.
    """
    magnetic_field = ng.CoefficientFunction((0.25, 1.0, 1.7))
    pressure_gradient = ng.CoefficientFunction((0.4, 0.0, 0.0))
    toroidal_gradient = ng.CoefficientFunction((0.7, -0.2, 1.1))
    magnetic_floor = 1.0e-12
    safe_magnitude = ng.sqrt(ng.InnerProduct(magnetic_field, magnetic_field) + magnetic_floor**2)
    amplitude = 0.12
    modulation = 0.3
    g0 = 0.18
    g1 = 0.34
    exact_utilde = (
        amplitude * ng.sin(ng.pi * ng.x) * (1.0 + modulation * ng.cos(2.0 * ng.pi * ng.y))
    )
    exact_g = g0 + g1 * ng.x
    exact_u = exact_g + exact_utilde
    exact_gradient = ng.CoefficientFunction((exact_utilde.Diff(ng.x), exact_utilde.Diff(ng.y), 0.0))
    regularized = _regularized_gradient(exact_gradient, magnetic_field, variant)
    diffusion_divergence = regularized[0].Diff(ng.x) + regularized[1].Diff(ng.y)
    b_dot_grad_p = ng.InnerProduct(magnetic_field, pressure_gradient)
    strong_left = (
        ng.InnerProduct(magnetic_field, exact_gradient)
        - diffusivity * diffusion_divergence
        + exact_utilde * b_dot_grad_p / safe_magnitude**2
        - diffusivity * ng.InnerProduct(regularized, pressure_gradient) / safe_magnitude**2
    )
    required_drive = (
        strong_left + g1 * magnetic_field[0] + exact_g * b_dot_grad_p / safe_magnitude**2
    )
    drive_direction = ng.Cross(magnetic_field, pressure_gradient)
    magnitude_gradient = (
        required_drive
        * safe_magnitude**3
        * drive_direction
        / (2.0 * ng.InnerProduct(drive_direction, drive_direction))
    )
    coefficients = FrozenCurrentContinuityCoefficients(
        magnetic_field=magnetic_field,
        pressure_gradient=pressure_gradient,
        magnetic_magnitude_gradient=magnitude_gradient,
        current_diffusivity=diffusivity,
        magnetic_floor=magnetic_floor,
        vacuum_permeability=1.0,
    )
    geometry = FrozenCurrentConstraintGeometry(
        level_set=1.0 - ng.x,
        level_set_gradient=ng.CoefficientFunction((-1.0, 0.0, 0.0)),
        toroidal_angle_gradient=toroidal_gradient,
    )
    magnetic_components = np.asarray((0.25, 1.0, 1.7))
    toroidal_components = np.asarray((0.7, -0.2, 1.1))
    pressure_components = np.asarray((0.4, 0.0, 0.0))
    safe_magnitude_value = sqrt(float(np.dot(magnetic_components, magnetic_components)) + 1.0e-24)
    b_dot_phi = float(np.dot(magnetic_components, toroidal_components))
    diamagnetic_dot_phi = float(
        np.dot(np.cross(magnetic_components, pressure_components), toroidal_components)
        / safe_magnitude_value**2
    )
    regularizing_weight = 0.7
    if variant == "perpendicular":
        regularizing_weight -= (
            float(np.dot(toroidal_components, magnetic_components))
            * float(magnetic_components[0])
            / safe_magnitude_value**2
        )

    def enclosed_current(s: object) -> object:
        coordinate = np.asarray(s)
        integral_u = (
            g0 * coordinate
            + 0.5 * g1 * coordinate**2
            + amplitude * (1.0 - np.cos(pi * coordinate)) / pi
        )
        return (
            b_dot_phi * integral_u
            + diamagnetic_dot_phi * coordinate
            - diffusivity * regularizing_weight * amplitude * np.sin(pi * coordinate)
        ) / (2.0 * pi)

    def current_derivative(s: object) -> object:
        coordinate = np.asarray(s)
        mean_u = g0 + g1 * coordinate + amplitude * np.sin(pi * coordinate)
        return (
            b_dot_phi * mean_u
            + diamagnetic_dot_phi
            - diffusivity * regularizing_weight * amplitude * pi * np.cos(pi * coordinate)
        ) / (2.0 * pi)

    profile = AnalyticToroidalCurrentProfile(
        enclosed_current_function=enclosed_current,
        derivative_function=current_derivative,
    )
    profile.validate()
    return coefficients, geometry, profile, exact_u, g0 + g1


@pytest.mark.parametrize("variant", ["perpendicular", "full"])
def test_two_distinct_i0_profiles_realize_independent_m2_currents(variant: str) -> None:
    r"""Two ``I_0(s)`` inputs return their own independently reconstructed (M2) currents."""
    edges = np.linspace(0.0, 1.0, 5)
    realized: list[np.ndarray] = []
    digests: list[str] = []
    for profile_kind in (1, 2):
        profile = _profile(profile_kind)
        solver = ConstrainedCurrentContinuitySolver(
            polynomial_order=2,
            runtime=RuntimeOptions(regularization_gradient=variant),  # type: ignore[arg-type]
        )
        result = solver.solve(
            Slab2D(maxh=1.0 / 24.0, subdivisions=(24, 24), periodic_y=True),
            _coefficients(0.04),
            _geometry(),
            profile,
            shell_edges=edges,
        )
        expected = np.asarray(profile.enclosed_current(edges), dtype=float)
        measured = np.asarray(result.independent_cumulative_current)

        assert result.m3_relative_residual_norm < 1.0e-10
        assert result.constraint_relative_residual_norm < 1.0e-10
        assert result.regularization_gradient == variant
        assert measured == pytest.approx(expected, abs=1.0e-10)
        assert result.target_cumulative_current == pytest.approx(expected, abs=1.0e-14)
        checkpoint = result.checkpoint_state()
        assert checkpoint.shell_edges == tuple(edges)
        assert checkpoint.g_coefficients == result.g_coefficients
        assert checkpoint.shell_constraint_residuals == pytest.approx(
            result.shell_constraint_residuals
        )
        realized.append(measured)
        digests.append(result.configuration_digest)

    assert np.max(np.abs(realized[0] - realized[1])) > 0.05
    assert digests[0] != digests[1]


@pytest.mark.parametrize("variant", ["perpendicular", "full"])
def test_fixed_i0_du_scan_has_a_regular_multiplier_limit(variant: str) -> None:
    r"""At fixed ``I_0``, ``D_u G' grad_r(s)`` vanishes without moving (M3b)."""
    edges = np.linspace(0.0, 1.0, 5)
    multiplier_norms: list[float] = []
    with _DU_TABLE.open(newline="") as table_file:
        recorded = {
            float(row["current_diffusivity"]): row
            for row in csv.DictReader(table_file)
            if row["variant"] == variant
        }
    for diffusivity in (0.08, 0.04, 0.02):
        result = ConstrainedCurrentContinuitySolver(
            polynomial_order=2,
            runtime=RuntimeOptions(regularization_gradient=variant),  # type: ignore[arg-type]
        ).solve(
            Slab2D(maxh=1.0 / 24.0, subdivisions=(24, 24), periodic_y=True),
            _coefficients(diffusivity),
            _geometry(),
            _profile(1),
            shell_edges=edges,
        )
        assert result.constraint_relative_residual_norm < 1.0e-10
        assert result.diagnostics["maximum_shell_mean_utilde"] < 1.0e-10
        multiplier_norm = result.diagnostics["multiplier_current_l2"]
        assert multiplier_norm == pytest.approx(
            float(recorded[diffusivity]["multiplier_current_l2"]), rel=5.0e-10
        )
        assert result.schur_condition_number == pytest.approx(
            float(recorded[diffusivity]["schur_condition_number"]), rel=5.0e-10
        )
        multiplier_norms.append(multiplier_norm)

    assert np.asarray(multiplier_norms[:-1]) / np.asarray(multiplier_norms[1:]) == pytest.approx(
        [2.0, 2.0], rel=0.05
    )


@pytest.mark.parametrize("variant", ["perpendicular", "full"])
def test_constrained_manufactured_h_p_and_shell_scans_match_rate_table(variant: str) -> None:
    r"""Coupled (M3)--(M3b) errors converge in h/p and remain stable when N doubles."""
    with _RATE_TABLE.open(newline="") as table_file:
        rows = [row for row in csv.DictReader(table_file) if row["variant"] == variant]
    coefficients, geometry, profile, exact_u, edge_value = _coupled_manufactured_case(variant)
    measured: dict[tuple[str, int, int, int], dict[str, float]] = {}
    for row in rows:
        order = int(row["polynomial_order"])
        subdivisions = int(row["subdivisions"])
        shell_count = int(row["shell_count"])
        solver = ConstrainedCurrentContinuitySolver(
            polynomial_order=order,
            runtime=RuntimeOptions(regularization_gradient=variant),  # type: ignore[arg-type]
        )
        result = solver.solve(
            Slab2D(
                maxh=1.0 / subdivisions,
                subdivisions=(subdivisions, subdivisions),
                periodic_y=True,
            ),
            coefficients,
            geometry,
            profile,
            shell_edges=np.linspace(0.0, 1.0, shell_count + 1),
            edge_value=edge_value,
        )
        internal = solver._solution()
        error = sqrt(
            float(
                ng.Integrate(
                    (internal.grid_function() - exact_u) ** 2,
                    internal.mesh(),
                    order=20,
                )
            )
        )
        point_value = solver.solution_at(0.37, 0.23)
        key = row["scan"], order, shell_count, subdivisions
        measured[key] = {
            "error": error,
            "subdivisions": float(subdivisions),
            "point_value": point_value,
        }

        assert internal.mesh().ne == int(row["elements"])
        assert error == pytest.approx(float(row["physical_u_l2_error"]), rel=5.0e-3)
        assert result.m3_relative_residual_norm < 1.0e-10
        assert result.constraint_relative_residual_norm < 1.0e-10
        assert result.diagnostics["a_factorizations"] == 1.0
        assert result.diagnostics["g_advection_coupling_l2"] > 0.02
        assert result.diagnostics["g_reaction_coupling_l2"] > 0.005
        assert result.diagnostics["diamagnetic_toroidal_current_l2"] > 0.005
        assert result.diagnostics["regularizing_toroidal_current_l2"] > 1.0e-4
        assert solver.utilde_at(0.0, 0.3) == pytest.approx(0.0, abs=1.0e-12)
        assert solver.g_at(1.0, 0.3) == pytest.approx(edge_value, abs=1.0e-12)
        if row["scan"] == "N":
            assert point_value == pytest.approx(float(row["solution_at_0p37_0p23"]), rel=5.0e-6)
            assert result.diagnostics["minimum_shell_radial_cells"] == pytest.approx(
                float(row["minimum_shell_radial_cells"]), rel=5.0e-6
            )
            assert result.diagnostics["minimum_shell_mollifier_widths"] == pytest.approx(
                float(row["minimum_shell_mollifier_widths"]), rel=5.0e-6
            )

    h_rows = sorted(
        (value for (scan, _, _, _), value in measured.items() if scan == "h"),
        key=lambda value: value["subdivisions"],
    )
    h_rate = log(h_rows[0]["error"] / h_rows[1]["error"]) / log(
        h_rows[1]["subdivisions"] / h_rows[0]["subdivisions"]
    )
    assert h_rate > 1.8
    p_errors = [measured[("p", order, 4, 24)]["error"] for order in (1, 2, 3)]
    assert p_errors[1] < 0.65 * p_errors[0]
    assert p_errors[2] <= 1.001 * p_errors[1]
    n_values = [measured[("N", 2, shells, 32)]["point_value"] for shells in (4, 8)]
    assert abs(n_values[1] - n_values[0]) / abs(n_values[1]) < 1.0e-5


@pytest.mark.parametrize("variant", ["perpendicular", "full"])
def test_two_legacy_f_shifts_with_one_edge_value_cancel_from_physical_u(variant: str) -> None:
    r"""The old ``u=F(p)+utilde`` profiles remain only the required negative control."""
    pressure = ng.x * (1.0 - ng.x) * ng.y * (1.0 - ng.y)
    pressure_gradient = ng.CoefficientFunction((pressure.Diff(ng.x), pressure.Diff(ng.y), 0.0))
    magnetic_field = ng.CoefficientFunction((0.3, 1.1, 1.8))
    coefficients = FrozenCurrentContinuityCoefficients(
        magnetic_field=magnetic_field,
        pressure_gradient=pressure_gradient,
        magnetic_magnitude_gradient=ng.CoefficientFunction((0.0, 0.0, 0.0)),
        current_diffusivity=0.04,
        magnetic_floor=1.0e-12,
        vacuum_permeability=1.0,
    )
    constant = PrescribedCurrentProfile(
        identifier="legacy-constant-f-v1",
        value=0.2,
        pressure_derivative=0.0,
        perpendicular_gradient_divergence=0.0,
        full_gradient_divergence=0.0,
    )
    shaped_value = 0.2 + 0.4 * pressure
    shaped_derivative = 0.4
    shaped_gradient = shaped_derivative * pressure_gradient
    full_gradient = _regularized_gradient(shaped_gradient, magnetic_field, "full")
    perpendicular_gradient = _regularized_gradient(
        shaped_gradient,
        magnetic_field,
        "perpendicular",
    )
    shaped = PrescribedCurrentProfile(
        identifier="legacy-shaped-f-v1",
        value=shaped_value,
        pressure_derivative=shaped_derivative,
        perpendicular_gradient_divergence=(
            perpendicular_gradient[0].Diff(ng.x) + perpendicular_gradient[1].Diff(ng.y)
        ),
        full_gradient_divergence=(full_gradient[0].Diff(ng.x) + full_gradient[1].Diff(ng.y)),
    )
    runtime = RuntimeOptions(regularization_gradient=variant)  # type: ignore[arg-type]
    first = CurrentContinuitySolver(
        polynomial_order=4,
        runtime=runtime,
        stabilization="supg",
    )
    second = CurrentContinuitySolver(
        polynomial_order=4,
        runtime=runtime,
        stabilization="supg",
    )
    slab = Slab2D(maxh=0.25)
    with pytest.warns(DeprecationWarning):
        first.solve_utilde(slab, coefficients, constant)
    with pytest.warns(DeprecationWarning):
        second.solve_utilde(slab, coefficients, shaped)
    first_solution = first._solution()
    difference = sqrt(
        float(
            ng.Integrate(
                (first_solution.grid_function() - second._solution().grid_function()) ** 2,
                first_solution.mesh(),
                order=20,
            )
        )
    )

    assert difference < 1.0e-10
    assert first.solution_at(0.0, 0.5) == pytest.approx(0.2, abs=1.0e-12)
    assert second.solution_at(0.0, 0.5) == pytest.approx(0.2, abs=1.0e-12)
