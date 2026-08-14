"""Unit contracts for the DESIGN §9.1 SUPG stabilization of note equation (M3)."""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from math import sqrt
from typing import Any

import ngsolve as ng
import pytest

import remec.fem._current_continuity as current_continuity_fem
from remec import RuntimeOptions
from remec.common import JsonEventLogger
from remec.geometry.slab import Slab2D
from remec.solvers.current_continuity import (
    CurrentContinuitySolver,
    FrozenCurrentContinuityCoefficients,
)


def _coefficients() -> FrozenCurrentContinuityCoefficients:
    magnetic_field = ng.CoefficientFunction((1.0 + ng.x, 0.5 - ng.y, 2.0 + ng.x * ng.y))
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


def _supg_parameter() -> Callable[..., float]:
    function: Any = current_continuity_fem.supg_stabilization_parameter
    return function


def test_supg_parameter_has_the_advection_diffusion_limits_and_order_scaling() -> None:
    r"""The centralized M3 parameter blends h/(2|B|) and h²/(4D_u) scales."""
    parameter = _supg_parameter()

    advection_limit = parameter(
        element_size_along_field=0.2,
        magnetic_magnitude=2.0,
        transverse_diffusion=0.0,
        polynomial_order=1,
    )
    diffusion_limit = parameter(
        element_size_along_field=0.2,
        magnetic_magnitude=0.0,
        transverse_diffusion=0.1,
        polynomial_order=1,
    )
    blended = parameter(
        element_size_along_field=0.2,
        magnetic_magnitude=2.0,
        transverse_diffusion=0.1,
        polynomial_order=1,
    )
    higher_order = parameter(
        element_size_along_field=0.2,
        magnetic_magnitude=2.0,
        transverse_diffusion=0.1,
        polynomial_order=2,
    )
    nearly_out_of_plane = parameter(
        element_size_along_field=200.0,
        diffusive_element_size=0.2,
        magnetic_magnitude=2.0,
        transverse_diffusion=0.1,
        polynomial_order=1,
    )

    assert advection_limit == pytest.approx(0.05)
    assert diffusion_limit == pytest.approx(0.1)
    assert blended == pytest.approx(1.0 / sqrt(20.0**2 + 10.0**2))
    assert 0.0 < higher_order < blended
    assert nearly_out_of_plane == pytest.approx(diffusion_limit, rel=1.0e-5)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("element_size_along_field", 0.0),
        ("magnetic_magnitude", -1.0),
        ("transverse_diffusion", -1.0),
        ("polynomial_order", 0),
        ("diffusive_element_size", 0.0),
    ],
)
def test_supg_parameter_rejects_invalid_inputs(field_name: str, value: float) -> None:
    """Invalid stabilization scales fail before an M3 form is assembled."""
    arguments: dict[str, float | int] = {
        "element_size_along_field": 0.2,
        "magnetic_magnitude": 2.0,
        "transverse_diffusion": 0.1,
        "polynomial_order": 2,
    }
    arguments[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        _supg_parameter()(**arguments)


@pytest.mark.parametrize("variant", ["perpendicular", "full"])
def test_supg_contribution_is_separate_and_the_switch_is_provenance(variant: str) -> None:
    """SUPG contributes a separately reported M3 residual for both gradient variants."""
    runtime = RuntimeOptions(regularization_gradient=variant)  # type: ignore[arg-type]
    stream = io.StringIO()
    stabilized_solver = CurrentContinuitySolver(
        polynomial_order=2,
        runtime=runtime,
        stabilization="supg",
        logger=JsonEventLogger(stream),
    )
    stabilized = stabilized_solver.solve(Slab2D(maxh=0.25), _coefficients())
    unstabilized = CurrentContinuitySolver(
        polynomial_order=2,
        runtime=runtime,
        stabilization="none",
    ).solve(Slab2D(maxh=0.25), _coefficients())

    assert stabilized.stabilization == "supg"
    assert unstabilized.stabilization == "none"
    assert stabilized.configuration_digest != unstabilized.configuration_digest
    assert stabilized.diagnostics["m3_supg_stabilization_norm"] > 0.0
    assert stabilized.diagnostics["m3_supg_strong_residual_l2"] > 0.0
    assert stabilized.diagnostics["m3_supg_tau_max"] > 0.0
    assert unstabilized.diagnostics["m3_supg_stabilization_norm"] == 0.0
    assert unstabilized.diagnostics["m3_supg_strong_residual_l2"] == 0.0
    assert unstabilized.diagnostics["m3_supg_tau_max"] == 0.0

    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert all(record["stabilization"] == "supg" for record in records)
    assert records[-1]["m3_supg_stabilization_norm"] == pytest.approx(
        stabilized.diagnostics["m3_supg_stabilization_norm"]
    )


def test_current_continuity_rejects_an_unknown_stabilization() -> None:
    """The public M3 solver admits only the documented off/SUPG choices."""
    with pytest.raises(ValueError, match="stabilization"):
        CurrentContinuitySolver(stabilization="artificial-diffusion")  # type: ignore[arg-type]


def test_gridfunction_magnetic_field_requires_and_uses_an_explicit_gradient() -> None:
    r"""GridFunction-backed B cannot silently erase the M3 projector derivative."""
    mesh = Slab2D(maxh=0.25).build_mesh()._mesh
    field_space = ng.H1(mesh, order=3, dim=3)
    magnetic_field = ng.GridFunction(field_space)
    exact_field = ng.CoefficientFunction((1.0 + ng.x, 0.5 - ng.y, 2.0 + ng.x * ng.y))
    magnetic_field.Set(exact_field)

    with pytest.raises(ValueError, match="magnetic_field_gradient"):
        current_continuity_fem._resolve_magnetic_field_gradient(
            mesh,
            magnetic_field,
            supplied_gradient=None,
        )

    native_gradient = ng.grad(magnetic_field)
    resolved_gradient = current_continuity_fem._resolve_magnetic_field_gradient(
        mesh,
        magnetic_field,
        supplied_gradient=native_gradient,
    )
    magnetic_floor = 1.0e-8
    safe_magnitude = ng.sqrt(ng.InnerProduct(magnetic_field, magnetic_field) + magnetic_floor**2)
    direction = magnetic_field / safe_magnitude
    computed = current_continuity_fem._normalized_direction_gradient(
        magnetic_field,
        resolved_gradient,
        safe_magnitude,
    )
    exact_safe_magnitude = ng.sqrt(ng.InnerProduct(exact_field, exact_field) + magnetic_floor**2)
    exact_direction = exact_field / exact_safe_magnitude
    exact = ng.CoefficientFunction(
        tuple(
            exact_direction[component].Diff(coordinate)
            for component in range(3)
            for coordinate in (ng.x, ng.y)
        ),
        dims=(3, 2),
    )
    error_squared = ng.Integrate(
        sum(
            (computed[component, coordinate] - exact[component, coordinate]) ** 2
            for component in range(3)
            for coordinate in range(2)
        ),
        mesh,
        order=20,
    )

    assert error_squared < 1.0e-20
    assert (
        ng.Integrate(
            sum(
                computed[component, coordinate] ** 2
                for component in range(3)
                for coordinate in range(2)
            ),
            mesh,
            order=20,
        )
        > 0.1
    )

    scalar_space = ng.H1(mesh, order=3)
    scalar_field = ng.GridFunction(scalar_space)
    scalar_field.Set(ng.x**2 + ng.x * ng.y + ng.y**2)
    computed_divergence = current_continuity_fem._regularized_gradient_divergence(
        scalar_field,
        direction,
        "perpendicular",
        computed,
    )
    scalar_gradient = ng.grad(scalar_field)
    scalar_hessian = scalar_field.Operator("hesse")
    expected_divergence: Any = 0.0
    for row, coordinate in enumerate((ng.x, ng.y)):
        for column in range(2):
            projector = (1.0 if row == column else 0.0) - exact_direction[row] * exact_direction[
                column
            ]
            expected_divergence += projector * scalar_hessian[row, column]
            expected_divergence += projector.Diff(coordinate) * scalar_gradient[column]
    divergence_error_squared = ng.Integrate(
        (computed_divergence - expected_divergence) ** 2,
        mesh,
        order=20,
    )

    assert divergence_error_squared < 1.0e-20
