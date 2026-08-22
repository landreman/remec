"""Manufactured verification for the DESIGN §9.1 SUPG form of note equation (M3)."""

from __future__ import annotations

import csv
from itertools import pairwise
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
)

_RATE_TABLE = Path(__file__).with_name("m3_supg_rates.csv")


def _recorded_rows() -> dict[tuple[str, int, float], dict[str, float]]:
    """Read the checked-in M3 SUPG convergence and diagnostic record."""
    with _RATE_TABLE.open(newline="") as table_file:
        rows = csv.DictReader(table_file)
        return {
            (row["variant"], int(row["polynomial_order"]), float(row["maxh"])): {
                name: float(row[name])
                for name in (
                    "elements",
                    "l2_error",
                    "l2_rate",
                    "free_dof_relative_residual",
                    "m3_supg_stabilization_norm",
                    "m3_supg_strong_residual_l2",
                    "m3_supg_tau_min",
                    "m3_supg_tau_max",
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


def _manufactured_coefficients(
    variant: str,
    *,
    magnetic_field: Any,
    pressure_gradient: Any,
    exact_u: Any,
    current_diffusivity: float,
    vacuum_permeability: float,
    magnetic_floor: float = 1.0e-8,
) -> FrozenCurrentContinuityCoefficients:
    r"""Inject the exact complete (M3) residual through its physical drive numerator."""
    safe_magnitude = ng.sqrt(ng.InnerProduct(magnetic_field, magnetic_field) + magnetic_floor**2)
    direction = magnetic_field / safe_magnitude
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
    return FrozenCurrentContinuityCoefficients(
        magnetic_field=magnetic_field,
        pressure_gradient=pressure_gradient,
        magnetic_magnitude_gradient=magnitude_gradient,
        current_diffusivity=current_diffusivity,
        magnetic_floor=magnetic_floor,
        vacuum_permeability=vacuum_permeability,
    )


def _all_terms_case(variant: str) -> tuple[FrozenCurrentContinuityCoefficients, Any]:
    magnetic_field = ng.CoefficientFunction((1.0 + ng.x, 0.5 - ng.y, 2.0 + ng.x * ng.y))
    pressure_gradient = ng.CoefficientFunction((1.0 + ng.y, 2.0 + ng.x, 0.0))
    exact_u = ng.sin(ng.pi * ng.x) * ng.sin(ng.pi * ng.y)
    coefficients = _manufactured_coefficients(
        variant,
        magnetic_field=magnetic_field,
        pressure_gradient=pressure_gradient,
        exact_u=exact_u,
        current_diffusivity=0.2,
        vacuum_permeability=0.7,
    )
    return coefficients, exact_u


def _l2_error(solver: CurrentContinuitySolver, exact_u: Any) -> float:
    solution = solver._solution()
    return sqrt(
        float(ng.Integrate((solution.grid_function() - exact_u) ** 2, solution.mesh(), order=20))
    )


@pytest.mark.parametrize("variant", ["perpendicular", "full"])
@pytest.mark.parametrize("polynomial_order", [1, 2, 3])
def test_supg_all_terms_manufactured_convergence(
    variant: str,
    polynomial_order: int,
) -> None:
    r"""The complete stabilized (M3) form converges at L2 order p+1 for both grad_r."""
    coefficients, exact_u = _all_terms_case(variant)
    errors: list[float] = []
    recorded_rows = _recorded_rows()
    mesh_sizes = (0.125, 0.0625, 0.03125) if polynomial_order == 1 else (0.25, 0.125, 0.0625)
    for maxh in mesh_sizes:
        solver = CurrentContinuitySolver(
            polynomial_order=polynomial_order,
            runtime=RuntimeOptions(regularization_gradient=variant),  # type: ignore[arg-type]
            stabilization="supg",
        )
        result = solver.solve(Slab2D(maxh=maxh), coefficients)
        observed_error = _l2_error(solver, exact_u)
        expected = recorded_rows[variant, polynomial_order, maxh]
        assert solver._solution().mesh().ne == expected["elements"]
        assert observed_error == pytest.approx(expected["l2_error"], rel=0.05, abs=1.0e-12)
        for name in (
            "m3_supg_stabilization_norm",
            "m3_supg_strong_residual_l2",
            "m3_supg_tau_min",
            "m3_supg_tau_max",
        ):
            assert result.diagnostics[name] == pytest.approx(expected[name], rel=0.05, abs=1.0e-12)
        assert result.free_dof_relative_residual_norm < 1.0e-11
        errors.append(observed_error)

    rates = [log(coarse / fine) / log(2.0) for coarse, fine in pairwise(errors)]
    assert errors[0] > errors[1] > errors[2]
    assert rates[-1] >= polynomial_order + 0.8
    assert rates[-1] == pytest.approx(
        recorded_rows[variant, polynomial_order, mesh_sizes[-1]]["l2_rate"], rel=0.05
    )


@pytest.mark.parametrize("variant", ["perpendicular", "full"])
@pytest.mark.parametrize("stabilization", ["none", "supg"])
def test_aligned_advection_manufactured_case_runs_with_supg_on_and_off(
    variant: str,
    stabilization: str,
) -> None:
    r"""A smooth aligned-advection (M3) case is accurate with either stabilization mode."""
    magnetic_field = ng.CoefficientFunction((1.0, 0.0, 0.0))
    pressure_gradient = ng.CoefficientFunction((0.0, 1.0, 0.0))
    exact_u = ng.sin(ng.pi * ng.x)
    coefficients = _manufactured_coefficients(
        variant,
        magnetic_field=magnetic_field,
        pressure_gradient=pressure_gradient,
        exact_u=exact_u,
        current_diffusivity=1.0e-3,
        vacuum_permeability=1.0,
    )
    solver = CurrentContinuitySolver(
        polynomial_order=3,
        runtime=RuntimeOptions(regularization_gradient=variant),  # type: ignore[arg-type]
        stabilization=stabilization,  # type: ignore[arg-type]
    )
    result = solver.solve(Slab2D(maxh=0.125), coefficients, boundary_value=exact_u)

    assert _l2_error(solver, exact_u) < 2.0e-5
    if stabilization == "supg":
        assert result.diagnostics["m3_supg_stabilization_norm"] > 0.0
    else:
        assert result.diagnostics["m3_supg_stabilization_norm"] == 0.0


@pytest.mark.parametrize("variant", ["perpendicular", "full"])
def test_transverse_diffusion_manufactured_case(variant: str) -> None:
    r"""SUPG retains the transverse -div(D_u grad_r u) part of strong (M3)."""
    magnetic_field = ng.CoefficientFunction((1.0, 0.0, 0.0))
    pressure_gradient = ng.CoefficientFunction((0.0, 0.0, 1.0))
    exact_u = ng.sin(ng.pi * ng.y)
    coefficients = _manufactured_coefficients(
        variant,
        magnetic_field=magnetic_field,
        pressure_gradient=pressure_gradient,
        exact_u=exact_u,
        current_diffusivity=0.1,
        vacuum_permeability=1.0,
    )
    solver = CurrentContinuitySolver(
        polynomial_order=3,
        runtime=RuntimeOptions(regularization_gradient=variant),  # type: ignore[arg-type]
        stabilization="supg",
    )
    result = solver.solve(Slab2D(maxh=0.125), coefficients, boundary_value=exact_u)

    assert _l2_error(solver, exact_u) < 2.0e-5
    expected_residual = {"perpendicular": 0.004035, "full": 0.004371}[variant]
    assert result.diagnostics["m3_supg_strong_residual_l2"] == pytest.approx(
        expected_residual, rel=0.05
    )
    expected_stabilization = {
        "perpendicular": 3.6411e-7,
        "full": 2.1310e-7,
    }[variant]
    assert result.diagnostics["m3_supg_stabilization_norm"] == pytest.approx(
        expected_stabilization, rel=0.05
    )


@pytest.mark.parametrize("variant", ["perpendicular", "full"])
def test_supg_stays_bounded_for_a_nearly_out_of_plane_field(variant: str) -> None:
    r"""The M3 diffusive tau scale stays finite as the in-plane field tends to zero."""
    exact_u = ng.sin(ng.pi * ng.x) * ng.sin(ng.pi * ng.y)
    pressure_gradient = ng.CoefficientFunction((0.0, 1.0, 0.0))
    tau_maxima: list[float] = []
    for in_plane_fraction in (1.0, 0.01, 0.001):
        magnetic_field = ng.CoefficientFunction(
            (
                in_plane_fraction,
                0.0,
                sqrt(1.0 - in_plane_fraction**2),
            )
        )
        coefficients = _manufactured_coefficients(
            variant,
            magnetic_field=magnetic_field,
            pressure_gradient=pressure_gradient,
            exact_u=exact_u,
            current_diffusivity=0.5,
            vacuum_permeability=1.0,
        )
        runtime = RuntimeOptions(regularization_gradient=variant)  # type: ignore[arg-type]
        unstabilized_solver = CurrentContinuitySolver(
            polynomial_order=2,
            runtime=runtime,
            stabilization="none",
        )
        unstabilized_solver.solve(Slab2D(maxh=0.125), coefficients)
        stabilized_solver = CurrentContinuitySolver(
            polynomial_order=2,
            runtime=runtime,
            stabilization="supg",
        )
        stabilized = stabilized_solver.solve(Slab2D(maxh=0.125), coefficients)
        unstabilized_error = _l2_error(unstabilized_solver, exact_u)
        stabilized_error = _l2_error(stabilized_solver, exact_u)

        tau_maxima.append(stabilized.diagnostics["m3_supg_tau_max"])
        assert stabilized_error < 2.0 * unstabilized_error

    assert max(tau_maxima) < 0.02
    assert max(tau_maxima) < 2.0 * min(tau_maxima)


@pytest.mark.parametrize("variant", ["perpendicular", "full"])
def test_final_current_viscosity_term_is_conspicuous(variant: str) -> None:
    r"""The complete (M3) solve resolves a deliberately large D_u grad_r(u).grad(p) term."""
    magnetic_field = ng.CoefficientFunction((1.0 + ng.x, 0.5 - ng.y, 2.0 + ng.x * ng.y))
    pressure_gradient = ng.CoefficientFunction((2.0 + 2.0 * ng.y, 4.0 + 2.0 * ng.x, 0.0))
    exact_u = ng.sin(ng.pi * ng.x) * ng.sin(ng.pi * ng.y)
    coefficients = _manufactured_coefficients(
        variant,
        magnetic_field=magnetic_field,
        pressure_gradient=pressure_gradient,
        exact_u=exact_u,
        current_diffusivity=0.4,
        vacuum_permeability=2.0,
    )
    solver = CurrentContinuitySolver(
        polynomial_order=3,
        runtime=RuntimeOptions(regularization_gradient=variant),  # type: ignore[arg-type]
        stabilization="supg",
    )
    result = solver.solve(Slab2D(maxh=0.0625), coefficients)

    assert result.diagnostics["m3_final_correction_l2"] > 0.1
    assert _l2_error(solver, exact_u) < 2.0e-5
