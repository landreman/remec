"""Manufactured verification of the note's transformed M3 ``utilde`` equation."""

from __future__ import annotations

import csv
from math import log, sqrt
from pathlib import Path
from typing import Any

import ngsolve as ng
import pytest

from remec import RuntimeOptions
from remec.geometry.slab import Slab2D
from remec.solvers.current_continuity import (
    CurrentContinuitySolver,
    FrozenCurrentContinuityCoefficients,
    PrescribedCurrentProfile,
)

_RATE_TABLE = Path(__file__).with_name("m3_utilde_rates.csv")


def _recorded_rows() -> dict[tuple[str, int, float], dict[str, float]]:
    """Read the checked-in direct-u/utilde manufactured comparison table."""
    with _RATE_TABLE.open(newline="") as table_file:
        rows = csv.DictReader(table_file)
        return {
            (row["variant"], int(row["polynomial_order"]), float(row["maxh"])): {
                name: float(row[name])
                for name in (
                    "elements",
                    "direct_l2_error",
                    "utilde_l2_error",
                    "relative_disagreement",
                    "free_dof_relative_residual",
                    "m3_profile_advection_source_l2",
                    "m3_profile_diffusion_source_l2",
                    "direct_l2_rate",
                    "utilde_l2_rate",
                )
            }
            for row in rows
        }


def _embedded_gradient(value: Any) -> Any:
    return ng.CoefficientFunction((value[0], value[1], 0.0))


def _regularized_gradient(gradient: Any, direction: Any, variant: str) -> Any:
    if variant == "full":
        return gradient
    return gradient - direction * ng.InnerProduct(direction, gradient)


def _manufactured_case(
    variant: str,
    *,
    quadratic_pressure: bool = False,
    magnetic_floor: float = 1.0e-8,
) -> tuple[FrozenCurrentContinuityCoefficients, PrescribedCurrentProfile, Any, Any]:
    r"""Manufacture every strong (M3) term for ``u = F(p) + utilde``."""
    magnetic_field = ng.CoefficientFunction((1.0 + ng.x, 0.5 - ng.y, 2.0 + ng.x * ng.y))
    if quadratic_pressure:
        pressure = ng.x**2 + ng.x * ng.y + ng.y**2
        profile_value = 0.2 + 0.3 * pressure
    else:
        pressure = ng.x + ng.y
        profile_value = 0.25 + 0.3 * pressure
    pressure_gradient = ng.CoefficientFunction((pressure.Diff(ng.x), pressure.Diff(ng.y), 0.0))
    current_diffusivity = 0.2
    vacuum_permeability = 0.7
    safe_magnitude = ng.sqrt(ng.InnerProduct(magnetic_field, magnetic_field) + magnetic_floor**2)
    direction = magnetic_field / safe_magnitude
    profile_gradient = 0.3 * pressure_gradient
    perpendicular_profile_gradient = _regularized_gradient(
        profile_gradient,
        direction,
        "perpendicular",
    )
    profile = PrescribedCurrentProfile(
        value=profile_value,
        pressure_derivative=0.3,
        perpendicular_gradient_divergence=(
            perpendicular_profile_gradient[0].Diff(ng.x)
            + perpendicular_profile_gradient[1].Diff(ng.y)
        ),
        full_gradient_divergence=(profile_gradient[0].Diff(ng.x) + profile_gradient[1].Diff(ng.y)),
    )
    exact_utilde = ng.sin(ng.pi * ng.x) * ng.sin(ng.pi * ng.y)
    exact_u = profile_value + exact_utilde
    exact_gradient = ng.CoefficientFunction((exact_u.Diff(ng.x), exact_u.Diff(ng.y), 0.0))
    regularized_gradient = _regularized_gradient(exact_gradient, direction, variant)
    diffusion_divergence = current_diffusivity * (
        regularized_gradient[0].Diff(ng.x) + regularized_gradient[1].Diff(ng.y)
    )
    required_drive = (
        ng.InnerProduct(magnetic_field, exact_gradient)
        - diffusion_divergence
        + vacuum_permeability
        * exact_u
        * ng.InnerProduct(magnetic_field, pressure_gradient)
        / safe_magnitude**2
        - vacuum_permeability
        * current_diffusivity
        * ng.InnerProduct(regularized_gradient, pressure_gradient)
        / safe_magnitude**2
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
        current_diffusivity=current_diffusivity,
        magnetic_floor=magnetic_floor,
        vacuum_permeability=vacuum_permeability,
    )
    return coefficients, profile, exact_u, exact_utilde


def _l2_norm(solver: CurrentContinuitySolver, expression: Any) -> float:
    solution = solver._solution()
    return sqrt(float(ng.Integrate(expression**2, solution.mesh(), order=20)))


@pytest.mark.parametrize("variant", ["perpendicular", "full"])
@pytest.mark.parametrize("polynomial_order", [1, 2, 3])
def test_utilde_matches_direct_u_and_retains_manufactured_order(
    variant: str,
    polynomial_order: int,
) -> None:
    r"""Eq. ``utilde_equation`` agrees with direct (M3) at order p+1 for both grad_r."""
    coefficients, profile, exact_u, _ = _manufactured_case(variant)
    runtime = RuntimeOptions(regularization_gradient=variant)  # type: ignore[arg-type]
    mesh_sizes = (0.0625, 0.03125) if polynomial_order == 1 else (0.125, 0.0625)
    recorded_rows = _recorded_rows()
    direct_errors: list[float] = []
    utilde_errors: list[float] = []
    for maxh in mesh_sizes:
        slab = Slab2D(maxh=maxh)
        direct = CurrentContinuitySolver(
            polynomial_order=polynomial_order,
            runtime=runtime,
            stabilization="supg",
        )
        direct_result = direct.solve(slab, coefficients, boundary_value=profile.value)
        transformed = CurrentContinuitySolver(
            polynomial_order=polynomial_order,
            runtime=runtime,
            stabilization="supg",
        )
        result = transformed.solve_utilde(slab, coefficients, profile)
        direct_field = direct._solution().grid_function()
        transformed_field = transformed._solution().grid_function()
        direct_error = _l2_norm(direct, direct_field - exact_u)
        transformed_error = _l2_norm(transformed, transformed_field - exact_u)
        disagreement = _l2_norm(transformed, transformed_field - direct_field)
        relative_disagreement = disagreement / max(_l2_norm(direct, direct_field), 1.0)
        expected = recorded_rows[variant, polynomial_order, maxh]

        assert transformed._solution().mesh().ne == expected["elements"]
        assert direct_error == pytest.approx(expected["direct_l2_error"], rel=0.05)
        assert transformed_error == pytest.approx(expected["utilde_l2_error"], rel=0.05)
        assert relative_disagreement < 1.0e-10
        assert result.diagnostics["m3_profile_advection_source_l2"] == pytest.approx(
            expected["m3_profile_advection_source_l2"],
            rel=0.05,
        )
        assert transformed.utilde_at(0.0, 0.5) == pytest.approx(0.0, abs=1.0e-12)
        assert transformed.current_at(0.4, 0.6) == pytest.approx(
            direct.current_at(0.4, 0.6),
            abs=1.0e-10,
        )
        assert transformed.parallel_current_over_field_at(0.4, 0.6) == pytest.approx(
            direct.parallel_current_over_field_at(0.4, 0.6),
            abs=1.0e-10,
        )
        assert result.formulation == "utilde"
        assert direct_result.formulation == "direct-u"
        assert result.configuration_digest != direct_result.configuration_digest
        assert result.free_dof_relative_residual_norm < 1.0e-11
        assert result.free_dof_relative_residual_norm == pytest.approx(
            expected["free_dof_relative_residual"],
            abs=1.0e-12,
        )
        direct_errors.append(direct_error)
        utilde_errors.append(transformed_error)

    direct_rate = log(direct_errors[0] / direct_errors[1]) / log(2.0)
    utilde_rate = log(utilde_errors[0] / utilde_errors[1]) / log(2.0)
    assert direct_errors[0] > direct_errors[1]
    assert utilde_errors[0] > utilde_errors[1]
    assert direct_rate >= polynomial_order + 0.8
    assert utilde_rate >= polynomial_order + 0.8
    fine_expected = recorded_rows[variant, polynomial_order, mesh_sizes[-1]]
    assert direct_rate == pytest.approx(fine_expected["direct_l2_rate"], rel=0.05)
    assert utilde_rate == pytest.approx(fine_expected["utilde_l2_rate"], rel=0.05)


def test_utilde_profile_sources_are_transcribed_for_the_selected_gradient() -> None:
    r"""Both extra Eq. ``utilde_equation`` sources use the selected grad_r."""
    measured_diffusion_norms: dict[str, float] = {}
    for variant in ("perpendicular", "full"):
        coefficients, profile, _, _ = _manufactured_case(variant, quadratic_pressure=True)
        runtime = RuntimeOptions(regularization_gradient=variant)  # type: ignore[arg-type]
        direct = CurrentContinuitySolver(
            polynomial_order=3,
            runtime=runtime,
            stabilization="supg",
        )
        direct.solve(Slab2D(maxh=0.125), coefficients, boundary_value=profile.value)
        transformed = CurrentContinuitySolver(
            polynomial_order=3,
            runtime=runtime,
            stabilization="supg",
        )
        result = transformed.solve_utilde(Slab2D(maxh=0.125), coefficients, profile)
        solution = transformed._solution()
        mesh = solution.mesh()
        safe_magnitude = ng.sqrt(
            ng.InnerProduct(coefficients.magnetic_field, coefficients.magnetic_field)
            + coefficients.magnetic_floor**2
        )
        direction = coefficients.magnetic_field / safe_magnitude
        profile_gradient = profile.pressure_derivative * coefficients.pressure_gradient
        regularized_profile_gradient = _regularized_gradient(
            profile_gradient,
            direction,
            variant,
        )
        expected_advection = -profile.pressure_derivative * ng.InnerProduct(
            coefficients.magnetic_field,
            coefficients.pressure_gradient,
        )
        expected_diffusion = coefficients.current_diffusivity * (
            regularized_profile_gradient[0].Diff(ng.x) + regularized_profile_gradient[1].Diff(ng.y)
        )
        b_dot_grad_p = ng.InnerProduct(
            coefficients.magnetic_field,
            coefficients.pressure_gradient,
        )
        expected_reaction = (
            -coefficients.vacuum_permeability * profile.value * b_dot_grad_p / safe_magnitude**2
        )
        expected_final_correction = (
            coefficients.vacuum_permeability
            * coefficients.current_diffusivity
            * ng.InnerProduct(regularized_profile_gradient, coefficients.pressure_gradient)
            / safe_magnitude**2
        )
        expected_advection_l2 = sqrt(float(ng.Integrate(expected_advection**2, mesh, order=20)))
        expected_diffusion_l2 = sqrt(float(ng.Integrate(expected_diffusion**2, mesh, order=20)))
        expected_reaction_l2 = sqrt(float(ng.Integrate(expected_reaction**2, mesh, order=20)))
        expected_final_correction_l2 = sqrt(
            float(ng.Integrate(expected_final_correction**2, mesh, order=20))
        )
        disagreement = _l2_norm(
            transformed,
            solution.grid_function() - direct._solution().grid_function(),
        )

        assert disagreement < 1.0e-10
        assert result.diagnostics["m3_profile_advection_source_l2"] == pytest.approx(
            expected_advection_l2,
            rel=1.0e-12,
        )
        assert result.diagnostics["m3_profile_diffusion_source_l2"] == pytest.approx(
            expected_diffusion_l2,
            rel=1.0e-12,
        )
        assert result.diagnostics["m3_profile_reaction_source_l2"] == pytest.approx(
            expected_reaction_l2,
            rel=1.0e-12,
        )
        assert result.diagnostics["m3_profile_final_correction_source_l2"] == pytest.approx(
            expected_final_correction_l2, rel=1.0e-12
        )
        assert expected_advection_l2 > 0.1
        assert expected_diffusion_l2 > 0.01
        assert expected_reaction_l2 > 0.01
        assert expected_final_correction_l2 > 1.0e-3
        measured_diffusion_norms[variant] = expected_diffusion_l2

    assert measured_diffusion_norms["perpendicular"] != pytest.approx(
        measured_diffusion_norms["full"],
        rel=0.05,
    )


@pytest.mark.parametrize("variant", ["perpendicular", "full"])
def test_utilde_matches_direct_u_when_the_magnetic_floor_is_active(variant: str) -> None:
    r"""The Galerkin ``utilde_equation`` shift matches direct (M3) for active B_safe floors."""
    for magnetic_floor in (1.0e-8, 0.1, 1.0):
        coefficients, profile, _, _ = _manufactured_case(
            variant,
            quadratic_pressure=True,
            magnetic_floor=magnetic_floor,
        )
        runtime = RuntimeOptions(regularization_gradient=variant)  # type: ignore[arg-type]
        direct = CurrentContinuitySolver(
            polynomial_order=3,
            runtime=runtime,
            stabilization="supg",
        )
        direct.solve(Slab2D(maxh=0.25), coefficients, boundary_value=profile.value)
        transformed = CurrentContinuitySolver(
            polynomial_order=3,
            runtime=runtime,
            stabilization="supg",
        )
        transformed.solve_utilde(Slab2D(maxh=0.25), coefficients, profile)
        direct_field = direct._solution().grid_function()
        transformed_field = transformed._solution().grid_function()
        relative_disagreement = _l2_norm(
            transformed,
            transformed_field - direct_field,
        ) / max(_l2_norm(direct, direct_field), 1.0)

        assert relative_disagreement < 1.0e-10
