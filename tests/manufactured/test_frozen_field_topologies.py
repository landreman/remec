"""Closed-field and analytic-island tests for note equation (M4a)."""

from __future__ import annotations

import csv
from itertools import pairwise
from math import isfinite, log
from pathlib import Path

import ngsolve as ng
import pytest

from remec.fem._anisotropic_diffusion import solve_frozen_field_anisotropic_diffusion
from remec.geometry.slab import Slab2D

_CLOSED_FIELD_TABLE = Path(__file__).with_name("closed_field_anisotropy_scan.csv")
_ISLAND_RATE_TABLE = Path(__file__).with_name("analytic_island_rates.csv")


def _recorded_rows(path: Path) -> list[dict[str, str]]:
    """Read one machine-readable frozen-field verification table."""
    with path.open(newline="") as table_file:
        return list(csv.DictReader(table_file))


def _laplacian_eigenvalue(flux: ng.CoefficientFunction, mesh: ng.Mesh) -> float:
    """Derive ``lambda`` from ``-Delta(flux) = lambda*flux``."""
    laplacian = flux.Diff(ng.x).Diff(ng.x) + flux.Diff(ng.y).Diff(ng.y)
    flux_l2_squared = float(ng.Integrate(flux**2, mesh, order=20))
    return -float(ng.Integrate(laplacian * flux, mesh, order=20)) / flux_l2_squared


def _safe_direction(
    raw_field: ng.CoefficientFunction, field_floor: float
) -> ng.CoefficientFunction:
    """Independently construct the smooth DESIGN section 6 small-B direction."""
    safe_norm = ng.sqrt(ng.InnerProduct(raw_field, raw_field) + field_floor**2)
    return raw_field / safe_norm


def _closed_field_data(
    field_floor: float,
) -> tuple[ng.CoefficientFunction, ng.CoefficientFunction, ng.CoefficientFunction]:
    """Return the translated Sovinec flux, its tangent field, and a positive source."""
    flux = ng.sin(ng.pi * ng.x) * ng.sin(ng.pi * ng.y)
    gradient = ng.CoefficientFunction((flux.Diff(ng.x), flux.Diff(ng.y)))
    raw_field = ng.CoefficientFunction((gradient[1], -gradient[0]))
    return flux, raw_field, _safe_direction(raw_field, field_floor)


def _island_data(
    field_floor: float,
    *,
    parallel_conductivity: float,
    perpendicular_conductivity: float,
) -> tuple[
    ng.CoefficientFunction,
    ng.CoefficientFunction,
    ng.CoefficientFunction,
    ng.CoefficientFunction,
    ng.CoefficientFunction,
]:
    """Return an independent one-island field and manufactured M4a data."""
    island_strength = 1.0
    flux = (
        0.5 * (ng.y - 0.5) ** 2 + island_strength * ng.cos(2.0 * ng.pi * ng.x) / (2.0 * ng.pi) ** 2
    )
    raw_field = ng.CoefficientFunction((flux.Diff(ng.y), -flux.Diff(ng.x)))
    direction = _safe_direction(raw_field, field_floor)
    exact = ng.sin(ng.pi * ng.x) * ng.sin(ng.pi * ng.y)
    exact_gradient = ng.CoefficientFunction((exact.Diff(ng.x), exact.Diff(ng.y)))
    parallel_gradient = ng.InnerProduct(direction, exact_gradient)
    flux_vector = (
        perpendicular_conductivity * exact_gradient
        + (parallel_conductivity - perpendicular_conductivity) * direction * parallel_gradient
    )
    source = -(flux_vector[0].Diff(ng.x) + flux_vector[1].Diff(ng.y))
    return flux, raw_field, direction, exact, source


@pytest.mark.parametrize("perpendicular_conductivity", [1.0e-1, 1.0e-2, 1.0e-3])
def test_closed_field_finite_anisotropy_axis(perpendicular_conductivity: float) -> None:
    """(M4a) resolves the analytic finite-kappa_perp response on closed field lines."""
    slab = Slab2D.unit_square(maxh=0.125)
    field_floor = 1.0e-6
    flux, raw_field, direction = _closed_field_data(field_floor)
    source = flux
    solution = solve_frozen_field_anisotropic_diffusion(
        slab,
        polynomial_order=3,
        source=source,
        raw_field=raw_field,
        parallel_conductivity=1.0,
        perpendicular_conductivity=perpendicular_conductivity,
        field_floor=field_floor,
    )

    source_laplacian_eigenvalue = _laplacian_eigenvalue(flux, solution._mesh)
    exact_amplitude = 1.0 / (source_laplacian_eigenvalue * perpendicular_conductivity)
    exact = exact_amplitude * flux
    central_amplitude = float(solution._field(solution._mesh(0.5, 0.5)))
    effective_perpendicular_diffusivity = 1.0 / (source_laplacian_eigenvalue * central_amplitude)
    numerical_perpendicular_diffusivity = (
        effective_perpendicular_diffusivity - perpendicular_conductivity
    )
    relative_l2_error = float(
        ng.sqrt(ng.Integrate((solution._field - exact) ** 2, solution._mesh, order=20))
        / ng.sqrt(ng.Integrate(exact**2, solution._mesh, order=20))
    )
    tangency_l2_squared = float(
        ng.Integrate(
            ng.InnerProduct(direction, ng.CoefficientFunction((flux.Diff(ng.x), flux.Diff(ng.y))))
            ** 2,
            solution._mesh,
            order=20,
        )
    )
    recorded = next(
        row
        for row in _recorded_rows(_CLOSED_FIELD_TABLE)
        if int(row["polynomial_order"]) == 3
        and float(row["maxh"]) == 0.125
        and float(row["perpendicular_conductivity"]) == perpendicular_conductivity
    )

    assert central_amplitude == pytest.approx(float(recorded["central_amplitude"]), rel=1.0e-5)
    assert central_amplitude == pytest.approx(exact_amplitude, rel=1.5e-3)
    assert numerical_perpendicular_diffusivity == pytest.approx(
        float(recorded["numerical_perpendicular_diffusivity"]), rel=0.05, abs=1.0e-12
    )
    assert numerical_perpendicular_diffusivity > 0.0
    assert relative_l2_error < 6.0e-4
    assert tangency_l2_squared < 1.0e-12
    assert solution.free_dof_relative_residual_norm < 1.0e-11
    assert solution.energy_diagnostics.parallel < 0.02 * solution.energy_diagnostics.total
    assert solution.energy_diagnostics.perpendicular > 0.0


@pytest.mark.slow
def test_closed_field_full_order_resolution_anisotropy_scan() -> None:
    """(M4a) reproduces the scheduled p-h-kappa_perp regression cube."""
    rows = _recorded_rows(_CLOSED_FIELD_TABLE)
    measured: dict[tuple[int, float, float], float] = {}

    for row in rows:
        polynomial_order = int(row["polynomial_order"])
        maxh = float(row["maxh"])
        perpendicular_conductivity = float(row["perpendicular_conductivity"])
        flux, raw_field, _ = _closed_field_data(1.0e-6)
        solution = solve_frozen_field_anisotropic_diffusion(
            Slab2D.unit_square(maxh=maxh),
            polynomial_order=polynomial_order,
            source=flux,
            raw_field=raw_field,
            parallel_conductivity=1.0,
            perpendicular_conductivity=perpendicular_conductivity,
            field_floor=1.0e-6,
        )
        central_amplitude = float(solution._field(solution._mesh(0.5, 0.5)))
        source_laplacian_eigenvalue = _laplacian_eigenvalue(flux, solution._mesh)
        effective_perpendicular_diffusivity = 1.0 / (
            source_laplacian_eigenvalue * central_amplitude
        )
        numerical_perpendicular_diffusivity = (
            effective_perpendicular_diffusivity - perpendicular_conductivity
        )
        key = (polynomial_order, maxh, perpendicular_conductivity)
        measured[key] = numerical_perpendicular_diffusivity

        assert solution._mesh.ne == int(row["elements"])
        assert central_amplitude == pytest.approx(float(row["central_amplitude"]), rel=1.0e-5)
        assert effective_perpendicular_diffusivity == pytest.approx(
            float(row["effective_perpendicular_diffusivity"]), rel=1.0e-5
        )
        assert numerical_perpendicular_diffusivity > 0.0
        assert solution.free_dof_relative_residual_norm <= 1.0e-11

    polynomial_orders = sorted({key[0] for key in measured})
    mesh_sizes = sorted({key[1] for key in measured}, reverse=True)
    perpendicular_conductivities = sorted({key[2] for key in measured}, reverse=True)
    assert len(measured) == (
        len(polynomial_orders) * len(mesh_sizes) * len(perpendicular_conductivities)
    )
    for perpendicular_conductivity in perpendicular_conductivities:
        for polynomial_order in polynomial_orders:
            by_refinement = [
                measured[polynomial_order, maxh, perpendicular_conductivity] for maxh in mesh_sizes
            ]
            assert all(fine < coarse for coarse, fine in pairwise(by_refinement))
        for maxh in mesh_sizes:
            by_order = [
                measured[polynomial_order, maxh, perpendicular_conductivity]
                for polynomial_order in polynomial_orders
            ]
            assert all(higher < lower for lower, higher in pairwise(by_order))


@pytest.mark.parametrize("polynomial_order", [1, 2, 3])
def test_analytic_island_manufactured_convergence(polynomial_order: int) -> None:
    """(M4a) retains L2 order p+1 and energy order p through island O/X nulls."""
    mesh_sizes = (0.2, 0.1, 0.05)
    parallel_conductivity = 10.0
    perpendicular_conductivity = 1.0
    field_floor = 0.05
    errors: list[tuple[float, float]] = []
    recorded = {
        (int(row["polynomial_order"]), float(row["maxh"])): row
        for row in _recorded_rows(_ISLAND_RATE_TABLE)
    }

    for maxh in mesh_sizes:
        flux, raw_field, direction, exact, source = _island_data(
            field_floor,
            parallel_conductivity=parallel_conductivity,
            perpendicular_conductivity=perpendicular_conductivity,
        )
        solution = solve_frozen_field_anisotropic_diffusion(
            Slab2D.unit_square(maxh=maxh),
            polynomial_order=polynomial_order,
            source=source,
            raw_field=raw_field,
            parallel_conductivity=parallel_conductivity,
            perpendicular_conductivity=perpendicular_conductivity,
            field_floor=field_floor,
        )
        gradient_error = ng.grad(solution._field) - ng.CoefficientFunction(
            (exact.Diff(ng.x), exact.Diff(ng.y))
        )
        parallel_error = ng.InnerProduct(direction, gradient_error)
        energy_error_squared = ng.Integrate(
            perpendicular_conductivity * ng.InnerProduct(gradient_error, gradient_error)
            + (parallel_conductivity - perpendicular_conductivity) * parallel_error**2,
            solution._mesh,
            order=20,
        )
        l2_error_squared = ng.Integrate((solution._field - exact) ** 2, solution._mesh, order=20)
        observed = (float(ng.sqrt(l2_error_squared)), float(ng.sqrt(energy_error_squared)))
        errors.append(observed)
        expected = recorded[polynomial_order, maxh]
        assert solution._mesh.ne == int(expected["elements"])
        assert observed[0] == pytest.approx(float(expected["l2_error"]), rel=0.05)
        assert observed[1] == pytest.approx(float(expected["energy_error"]), rel=0.05)

        o_point = float(flux(solution._mesh(0.5, 0.5)))
        x_point = float(flux(solution._mesh(0.0, 0.5)))
        assert o_point < x_point
        assert all(
            isfinite(float(direction(solution._mesh(x_coordinate, 0.5))[component]))
            for x_coordinate in (0.0, 0.5, 1.0)
            for component in (0, 1)
        )
        assert solution.field_direction_diagnostics.floor == field_floor
        assert solution.field_direction_diagnostics.floor_activity_l2_squared == pytest.approx(
            float(expected["floor_activity_l2_squared"]), rel=1.0e-4
        )
        assert solution.free_dof_relative_residual_norm < 1.0e-11
        boundary_trace_norm_squared = sum(
            ng.Integrate(
                solution._field**2,
                solution._mesh,
                definedon=solution._mesh.Boundaries(boundary_name),
            )
            for boundary_name in Slab2D.unit_square(maxh=maxh).boundary_regions()
        )
        assert boundary_trace_norm_squared < 1.0e-24

    l2_rate = log(errors[-2][0] / errors[-1][0]) / log(mesh_sizes[-2] / mesh_sizes[-1])
    energy_rate = log(errors[-2][1] / errors[-1][1]) / log(mesh_sizes[-2] / mesh_sizes[-1])
    assert l2_rate >= polynomial_order + 0.8
    assert energy_rate >= polynomial_order - 0.2
    assert all(fine[0] < coarse[0] for coarse, fine in pairwise(errors))
    assert all(fine[1] < coarse[1] for coarse, fine in pairwise(errors))
