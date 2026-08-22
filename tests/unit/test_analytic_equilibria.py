"""Contracts for analytic Solov'ev Grad-Shafranov equilibria."""

from __future__ import annotations

import numpy as np
import pytest

from remec.analytic_equilibria import (
    CerfonFreidbergBoundary,
    CerfonFreidbergShape,
    ZhengShape,
    solve_cerfon_freidberg,
    solve_zheng_equilibrium,
)
from remec.geometry.axisymmetric import AxisymmetricFluxContourDomain


def test_zheng_figure_1_coefficients_derivatives_and_integrals() -> None:
    """Zheng Eqs. (14)--(20) realize the requested shape, current, and beta."""
    equilibrium = solve_zheng_equilibrium(
        shape=ZhengShape(0.70, 0.49, 1.7, 0.125),
        poloidal_beta=0.40,
        plasma_current=1.0e6,
    )
    radius = np.linspace(0.25, 1.25, 19)
    height = np.linspace(-0.7, 0.7, 19)
    step = 1.0e-5
    radial_fd = (
        equilibrium.flux(radius + step, height) - equilibrium.flux(radius - step, height)
    ) / (2.0 * step)
    vertical_fd = (
        equilibrium.flux(radius, height + step) - equilibrium.flux(radius, height - step)
    ) / (2.0 * step)
    assert equilibrium.radial_derivative(radius, height) == pytest.approx(
        radial_fd, rel=1.0e-8, abs=1.0e-9
    )
    assert equilibrium.vertical_derivative(radius, height) == pytest.approx(
        vertical_fd, rel=1.0e-8, abs=1.0e-9
    )
    assert equilibrium.delta_star(radius, height) == pytest.approx(
        equilibrium.a1 * radius**2 - equilibrium.a2,
        rel=2.0e-13,
        abs=2.0e-13,
    )
    figures = equilibrium.figure_of_merit_integrals()
    assert figures.plasma_current == pytest.approx(1.0e6, rel=2.0e-10)
    assert figures.poloidal_beta == pytest.approx(0.40, rel=2.0e-10)
    contour = equilibrium.boundary_contour(samples=257)
    assert np.max(np.abs(equilibrium.flux(contour.radius, contour.height))) < 2.0e-10
    assert contour.signed_area > 0.0


@pytest.mark.parametrize(
    ("boundary", "shape", "source_parameter"),
    [
        (
            CerfonFreidbergBoundary.SMOOTH,
            CerfonFreidbergShape(0.32, 1.7, 0.33),
            -0.155,
        ),
        (
            CerfonFreidbergBoundary.DOUBLE_NULL,
            CerfonFreidbergShape(0.78, 2.0, 0.35),
            0.0,
        ),
    ],
)
def test_cerfon_freidberg_constraints_operator_and_contour(
    boundary: CerfonFreidbergBoundary,
    shape: CerfonFreidbergShape,
    source_parameter: float,
) -> None:
    """Cerfon--Freidberg Eqs. (5)--(12) satisfy the GS operator and boundary data."""
    equilibrium = solve_cerfon_freidberg(
        shape=shape,
        source_parameter=source_parameter,
        boundary=boundary,
    )
    radius = np.linspace(1.0 - 0.7 * shape.inverse_aspect_ratio, 1.3, 17)
    height = np.linspace(-0.6 * shape.elongation, 0.6 * shape.elongation, 17)
    assert equilibrium.delta_star(radius, height) == pytest.approx(
        (1.0 - source_parameter) * radius**2 + source_parameter,
        rel=2.0e-12,
        abs=2.0e-12,
    )
    assert equilibrium.maximum_constraint_residual() < 2.0e-11
    contour = equilibrium.boundary_contour(samples=321)
    contour_residual = np.max(np.abs(equilibrium.flux(contour.radius, contour.height)))
    assert contour_residual < 2.0e-9
    assert contour.signed_area > 0.0
    integrals = equilibrium.figure_of_merit_integrals(radial_order=18, angular_order=64)
    assert integrals.normalized_volume > 0.0
    assert integrals.normalized_perimeter > 0.0
    assert np.isfinite(integrals.flux_volume_integral)
    if boundary is CerfonFreidbergBoundary.SMOOTH:
        beta_t, beta = integrals.beta_values(
            q_star=1.57,
            inverse_aspect_ratio=shape.inverse_aspect_ratio,
        )
        assert beta_t == pytest.approx(0.05, abs=1.0e-3)
        assert 0.0 < beta < beta_t
    if boundary is CerfonFreidbergBoundary.DOUBLE_NULL:
        upper_xpoint = equilibrium.upper_xpoint
        assert upper_xpoint is not None
        assert equilibrium.radial_derivative(*upper_xpoint) == pytest.approx(0.0, abs=2.0e-11)
        assert equilibrium.vertical_derivative(*upper_xpoint) == pytest.approx(0.0, abs=2.0e-11)
        assert len(contour.corner_indices) == 2


def test_flux_contour_domain_has_one_constant_flux_wall() -> None:
    """A sampled analytic contour produces a curved R-Z mesh with one named wall."""
    equilibrium = solve_cerfon_freidberg(
        shape=CerfonFreidbergShape(0.32, 1.7, 0.33),
        source_parameter=-0.155,
        boundary=CerfonFreidbergBoundary.SMOOTH,
    )
    domain = AxisymmetricFluxContourDomain(
        equilibrium.boundary_contour(samples=129),
        maxh=0.25,
        geometry_order=3,
    )
    mesh = domain.build_mesh()._mesh
    assert mesh.dim == 2
    assert mesh.ne > 0
    assert set(mesh.GetBoundaries()) == {"wall"}
    assert domain.boundary_regions() == {"flux_surface": "wall"}
    assert domain.metadata()["geometry_order"] == 3
    assert domain.metadata()["toroidal_discretization"] is None
    assert domain.metadata()["boundary_flux"] == pytest.approx(0.0)
    assert mesh.ngmesh.dim == 2
