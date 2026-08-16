"""Harmonic magnetic-flux construction for note equation (M1)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import pi, sqrt
from typing import Any

import numpy as np
from numpy.typing import NDArray

from remec.geometry.solid_torus import AnalyticSolidTorus

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class HarmonicFluxField:
    """Normalized harmonic component and (M1) diagnostics on a solid torus."""

    field: Any
    normalization: float
    major_radius: float
    weak_curl_relative_residual: float
    weak_divergence_relative_residual: float
    boundary_normal_relative_norm: float
    sampled_magnetic_magnitude_minimum: float
    sampled_magnetic_magnitude_maximum: float

    def poloidal_normal_component(
        self,
        x_coordinate: FloatArray,
        z_coordinate: FloatArray,
    ) -> FloatArray:
        """Evaluate ``B_h . e_y`` on the positive-y poloidal cut."""
        del z_coordinate
        return self.normalization / x_coordinate


def _dual_l2_norm(space: Any, residual: Any, mesh: Any, *, integration_order: int) -> float:
    """Return the mass-Riesz norm of a residual tested on ``space``."""
    import ngsolve as ng  # type: ignore[import-untyped]

    trial, test = space.TnT()
    mass = ng.BilinearForm(space)
    mass += ng.InnerProduct(trial, test) * ng.dx(bonus_intorder=integration_order)
    load = ng.LinearForm(space)
    load += ng.InnerProduct(residual, test) * ng.dx(bonus_intorder=integration_order)
    mass.Assemble()
    load.Assemble()
    riesz = mass.mat.Inverse(space.FreeDofs(), inverse="sparsecholesky") * load.vec
    squared_norm = float(ng.InnerProduct(load.vec, riesz))
    return sqrt(max(squared_norm, 0.0))


def _quadrature_extrema(
    mesh: Any,
    coefficient: Any,
    *,
    integration_order: int,
) -> tuple[float, float]:
    """Return deterministic volume-quadrature extrema of a scalar coefficient."""
    import ngsolve as ng

    element_types = {element.type for element in mesh.Elements(ng.VOL)}
    rules = {
        element_type: ng.IntegrationRule(element_type, integration_order)
        for element_type in element_types
    }
    points = mesh.MapToAllElements(rules, ng.VOL)
    values = np.asarray(coefficient(points), dtype=float).reshape(-1)
    return float(np.min(values)), float(np.max(values))


def build_analytic_torus_harmonic_field(
    mesh: Any,
    torus: AnalyticSolidTorus,
    *,
    test_order: int = 2,
) -> HarmonicFluxField:
    r"""Construct the unit-flux harmonic field required by note equation ``(M1)``.

    On a circular solid torus linking the cylindrical axis,

    ``B_h = C grad(phi) = C (-y, x, 0)/(x^2+y^2)``

    is curl-free, divergence-free, and tangent to the exact boundary.  Its raw
    positive-y flux through the ``phi=0`` poloidal disk is
    ``2*pi*(R-sqrt(R^2-a^2))``; ``C`` is its reciprocal.  The returned weak residuals
    are mass-Riesz norms of the analytic curl and divergence tested on finite-element
    spaces, while the boundary-normal diagnostic records curved-geometry error.
    """
    if not isinstance(torus, AnalyticSolidTorus):
        raise TypeError("torus must be an AnalyticSolidTorus")
    if getattr(mesh, "dim", None) != 3:
        raise ValueError("mesh must be three-dimensional")
    if isinstance(test_order, bool) or not isinstance(test_order, int):
        raise TypeError("test_order must be an integer")
    if test_order < 1:
        raise ValueError("test_order must be positive")

    import ngsolve as ng

    radius_squared = ng.x**2 + ng.y**2
    raw_flux = 2.0 * pi * (torus.major_radius - sqrt(torus.major_radius**2 - torus.minor_radius**2))
    normalization = 1.0 / raw_flux
    field = ng.CoefficientFunction(
        (
            -normalization * ng.y / radius_squared,
            normalization * ng.x / radius_squared,
            0.0,
        )
    )
    divergence = field[0].Diff(ng.x) + field[1].Diff(ng.y) + field[2].Diff(ng.z)
    curl = ng.CoefficientFunction(
        (
            field[2].Diff(ng.y) - field[1].Diff(ng.z),
            field[0].Diff(ng.z) - field[2].Diff(ng.x),
            field[1].Diff(ng.x) - field[0].Diff(ng.y),
        )
    )
    integration_order = 2 * max(test_order, torus.geometry_order) + 8
    field_norm = float(
        ng.sqrt(
            ng.Integrate(
                ng.InnerProduct(field, field),
                mesh,
                order=integration_order,
            )
        )
    )
    scale = max(field_norm, np.finfo(float).tiny)
    weak_divergence_norm = _dual_l2_norm(
        ng.H1(mesh, order=test_order),
        divergence,
        mesh,
        integration_order=integration_order,
    )
    weak_curl_norm = _dual_l2_norm(
        ng.VectorH1(mesh, order=test_order),
        curl,
        mesh,
        integration_order=integration_order,
    )
    boundary_normal_norm = float(
        ng.sqrt(
            ng.Integrate(
                (field * ng.specialcf.normal(3)) ** 2,
                mesh,
                ng.BND,
                order=integration_order,
            )
        )
    )
    magnitude = ng.sqrt(ng.InnerProduct(field, field))
    sampled_minimum, sampled_maximum = _quadrature_extrema(
        mesh,
        magnitude,
        integration_order=integration_order,
    )
    return HarmonicFluxField(
        field=field,
        normalization=normalization,
        major_radius=torus.major_radius,
        weak_curl_relative_residual=weak_curl_norm / scale,
        weak_divergence_relative_residual=weak_divergence_norm / scale,
        boundary_normal_relative_norm=boundary_normal_norm / scale,
        sampled_magnetic_magnitude_minimum=sampled_minimum,
        sampled_magnetic_magnitude_maximum=sampled_maximum,
    )


def poloidal_cut_flux(
    torus: AnalyticSolidTorus,
    normal_component: Callable[[FloatArray, FloatArray], FloatArray],
    *,
    quadrature_order: int = 24,
) -> float:
    r"""Integrate ``field . e_y`` across the positive-y poloidal cut.

    The cut is the disk ``y=0`` centered at ``(R,0,0)``. Tensor-product Gauss
    quadrature in polar coordinates deliberately evaluates the physical field rather
    than reusing the analytic normalization formula.
    """
    if not isinstance(torus, AnalyticSolidTorus):
        raise TypeError("torus must be an AnalyticSolidTorus")
    if not callable(normal_component):
        raise TypeError("normal_component must be callable")
    if isinstance(quadrature_order, bool) or not isinstance(quadrature_order, int):
        raise TypeError("quadrature_order must be an integer")
    if quadrature_order < 2:
        raise ValueError("quadrature_order must be at least two")

    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    radii = 0.5 * torus.minor_radius * (nodes + 1.0)
    radial_weights = 0.5 * torus.minor_radius * weights
    angles = pi * (nodes + 1.0)
    angular_weights = pi * weights
    radial_grid, angular_grid = np.meshgrid(radii, angles, indexing="ij")
    cut_weights = np.outer(radial_weights, angular_weights) * radial_grid
    x_coordinates = torus.major_radius + radial_grid * np.cos(angular_grid)
    z_coordinates = radial_grid * np.sin(angular_grid)
    values = np.asarray(normal_component(x_coordinates, z_coordinates), dtype=float)
    if values.shape != radial_grid.shape:
        raise ValueError("normal_component must preserve the input array shape")
    if not np.all(np.isfinite(values)):
        raise ValueError("field produced non-finite values on the poloidal cut")
    return float(np.sum(values * cut_weights))
