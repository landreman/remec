"""Contracts for the frozen-field direct-u form of note equations (M2)--(M3)."""

from __future__ import annotations

import io
import json
from math import isfinite, sqrt
from typing import Any

import ngsolve as ng
import pytest

from remec import Normalization, RuntimeOptions
from remec.common import CheckpointMetadata, JsonEventLogger
from remec.common.serialization import configuration_digest
from remec.fem._current_continuity import solve_frozen_current_continuity
from remec.geometry.slab import Slab2D
from remec.solvers.current_continuity import (
    CurrentContinuitySolver,
    FrozenCurrentContinuityCoefficients,
)


def _frozen_coefficients() -> FrozenCurrentContinuityCoefficients:
    magnetic_field = ng.CoefficientFunction((1.0 + ng.x, 0.5 + ng.y, 2.0 + ng.x * ng.y))
    pressure_gradient = ng.CoefficientFunction((1.0 + ng.y, 2.0 + ng.x, 0.0))
    magnitude = ng.sqrt(ng.InnerProduct(magnetic_field, magnetic_field))
    magnitude_gradient = ng.CoefficientFunction((magnitude.Diff(ng.x), magnitude.Diff(ng.y), 0.0))
    return FrozenCurrentContinuityCoefficients(
        magnetic_field=magnetic_field,
        pressure_gradient=pressure_gradient,
        magnetic_magnitude_gradient=magnitude_gradient,
        current_diffusivity=0.2,
        magnetic_floor=1.0e-8,
        vacuum_permeability=0.7,
    )


def _embedded_gradient(value: Any) -> Any:
    return ng.CoefficientFunction((value[0], value[1], 0.0))


def _regularized_gradient(gradient: Any, direction: Any, variant: str) -> Any:
    if variant == "full":
        return gradient
    return gradient - direction * ng.InnerProduct(direction, gradient)


@pytest.mark.parametrize("variant", ["perpendicular", "full"])
def test_direct_u_solution_satisfies_independently_assembled_m3_weak_form(
    variant: str,
) -> None:
    """(M3) contains advection, matching diffusion/reaction, and the full frozen drive."""
    slab = Slab2D(maxh=0.25)
    coefficients = _frozen_coefficients()
    solution = solve_frozen_current_continuity(
        slab,
        polynomial_order=2,
        coefficients=coefficients,
        runtime=RuntimeOptions(regularization_gradient=variant),  # type: ignore[arg-type]
    )
    field = solution.grid_function()
    space = field.space
    trial, test = space.TnT()
    trial_gradient = _embedded_gradient(ng.grad(trial))
    test_gradient = _embedded_gradient(ng.grad(test))
    magnetic_field = coefficients.magnetic_field
    pressure_gradient = coefficients.pressure_gradient
    safe_magnitude = ng.sqrt(
        ng.InnerProduct(magnetic_field, magnetic_field) + coefficients.magnetic_floor**2
    )
    direction = magnetic_field / safe_magnitude
    regularized_trial = _regularized_gradient(trial_gradient, direction, variant)
    regularized_test = _regularized_gradient(test_gradient, direction, variant)
    b_dot_grad_p = ng.InnerProduct(magnetic_field, pressure_gradient)
    drive = (
        2.0
        * ng.InnerProduct(
            magnetic_field,
            ng.Cross(pressure_gradient, coefficients.magnetic_magnitude_gradient),
        )
        / safe_magnitude**3
    )

    expected_operator = ng.BilinearForm(space)
    expected_operator += (
        test * ng.InnerProduct(magnetic_field, trial_gradient)
        + coefficients.current_diffusivity * ng.InnerProduct(regularized_test, regularized_trial)
        + coefficients.vacuum_permeability * b_dot_grad_p * test * trial / safe_magnitude**2
        - coefficients.vacuum_permeability
        * coefficients.current_diffusivity
        * ng.InnerProduct(regularized_trial, pressure_gradient)
        * test
        / safe_magnitude**2
    ) * ng.dx(bonus_intorder=12)
    expected_source = ng.LinearForm(space)
    expected_source += drive * test * ng.dx(bonus_intorder=12)
    expected_operator.Assemble()
    expected_source.Assemble()

    residual = expected_source.vec.CreateVector()
    residual.data = expected_operator.mat * field.vec - expected_source.vec
    free_residual = ng.Projector(space.FreeDofs(), True) * residual
    free_source = ng.Projector(space.FreeDofs(), True) * expected_source.vec
    relative_residual = float(ng.Norm(free_residual)) / max(1.0, float(ng.Norm(free_source)))

    assert relative_residual < 1.0e-11
    assert solution.free_dof_relative_residual_norm < 1.0e-11
    assert solution.diagnostics["m3_drive_l2"] > 1.0e-3
    assert solution.diagnostics["m3_reaction_l2"] > 1.0e-3
    assert solution.diagnostics["m3_final_correction_l2"] > 1.0e-3


@pytest.mark.parametrize("variant", ["perpendicular", "full"])
def test_m2_current_and_parallel_diagnostic_use_the_selected_gradient(variant: str) -> None:
    """(M2) and J_parallel/B use exactly the gradient selected for the (M3) solve."""
    runtime = RuntimeOptions(regularization_gradient=variant)  # type: ignore[arg-type]
    solver = CurrentContinuitySolver(polynomial_order=2, runtime=runtime)
    solver.solve(Slab2D(maxh=0.25), _frozen_coefficients())

    x_coordinate, y_coordinate = 0.4, 0.6
    u_value = solver.solution_at(x_coordinate, y_coordinate)
    du_dx, du_dy = solver.solution_gradient_at(x_coordinate, y_coordinate)
    magnetic_field = (1.4, 1.1, 2.24)
    pressure_gradient = (1.6, 2.4, 0.0)
    safe_magnitude = sqrt(sum(component**2 for component in magnetic_field) + 1.0e-16)
    direction = tuple(component / safe_magnitude for component in magnetic_field)
    full_gradient = (du_dx, du_dy, 0.0)
    if variant == "full":
        regularized_gradient = full_gradient
    else:
        parallel_gradient = sum(a * b for a, b in zip(direction, full_gradient, strict=True))
        regularized_gradient = tuple(
            gradient - direction_component * parallel_gradient
            for gradient, direction_component in zip(full_gradient, direction, strict=True)
        )
    cross_field_pressure = (
        -magnetic_field[2] * pressure_gradient[1],
        magnetic_field[2] * pressure_gradient[0],
        magnetic_field[0] * pressure_gradient[1] - magnetic_field[1] * pressure_gradient[0],
    )
    expected_current = tuple(
        u_value * magnetic + diamagnetic / safe_magnitude**2 - 0.2 * regularized
        for magnetic, diamagnetic, regularized in zip(
            magnetic_field, cross_field_pressure, regularized_gradient, strict=True
        )
    )
    expected_parallel = (
        u_value
        if variant == "perpendicular"
        else u_value
        - 0.2 * sum(a * b for a, b in zip(direction, full_gradient, strict=True)) / safe_magnitude
    )

    assert solver.current_at(x_coordinate, y_coordinate) == pytest.approx(
        expected_current, abs=1.0e-12
    )
    assert solver.parallel_current_over_field_at(x_coordinate, y_coordinate) == pytest.approx(
        expected_parallel, abs=1.0e-12
    )


def test_gradient_variant_is_in_digest_logs_and_checkpoint_metadata() -> None:
    """The restart-critical M3 runtime choice is never implicit or absent from provenance."""
    perpendicular = RuntimeOptions(regularization_gradient="perpendicular")
    full = RuntimeOptions(regularization_gradient="full")
    assert configuration_digest(perpendicular) != configuration_digest(full)

    metadata = CheckpointMetadata.create(
        normalization=Normalization(reference_length=1.0, reference_field=1.0),
        runtime=full,
        state_names=("u",),
        git_commit="abc123",
        platform="test-platform",
    )
    assert metadata.configuration["runtime"]["regularization_gradient"] == "full"

    stream = io.StringIO()
    solver = CurrentContinuitySolver(
        polynomial_order=2,
        runtime=full,
        logger=JsonEventLogger(stream),
    )
    result = solver.solve(Slab2D(maxh=0.25), _frozen_coefficients())
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [record["event"] for record in records] == ["m3_solve_started", "m3_solve_completed"]
    assert all(record["regularization_gradient"] == "full" for record in records)
    assert all(record["configuration_digest"] == result.configuration_digest for record in records)
    assert len(result.configuration_digest) == 64
    assert all(character in "0123456789abcdef" for character in result.configuration_digest)
    assert solver.diagnostics() == result.diagnostics


def test_direct_u_solver_preserves_the_prescribed_boundary_value() -> None:
    """(M3) realizes the fixed-boundary value u=F(p_b) while solving free DOFs."""
    solver = CurrentContinuitySolver(polynomial_order=2)
    result = solver.solve(
        Slab2D(maxh=0.25),
        _frozen_coefficients(),
        boundary_value=0.3,
    )

    assert solver.solution_at(0.0, 0.5) == pytest.approx(0.3, abs=1.0e-12)
    assert solver.solution_at(1.0, 0.5) == pytest.approx(0.3, abs=1.0e-12)
    assert result.free_dof_relative_residual_norm < 1.0e-11


def test_m3_options_and_coefficients_reject_invalid_physics() -> None:
    """Invalid gradient switches and non-positive M3 coefficients fail before assembly."""
    with pytest.raises(ValueError, match="regularization_gradient"):
        RuntimeOptions(regularization_gradient="diagonal")  # type: ignore[arg-type]
    base = _frozen_coefficients()
    for field_name, value in (
        ("current_diffusivity", 0.0),
        ("magnetic_floor", -1.0),
        ("vacuum_permeability", float("nan")),
    ):
        values = {
            "magnetic_field": base.magnetic_field,
            "pressure_gradient": base.pressure_gradient,
            "magnetic_magnitude_gradient": base.magnetic_magnitude_gradient,
            "current_diffusivity": base.current_diffusivity,
            "magnetic_floor": base.magnetic_floor,
            "vacuum_permeability": base.vacuum_permeability,
        }
        values[field_name] = value
        with pytest.raises(ValueError, match=field_name):
            FrozenCurrentContinuityCoefficients(**values)

    assert isfinite(base.current_diffusivity)
