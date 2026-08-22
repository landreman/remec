r"""Benchmark the axisymmetric Grad-Shafranov solver against Zheng's analytic solution.

Reference: S. B. Zheng, A. J. Wootton and E. R. Solano, "Analytical tokamak
equilibrium for shaped plasmas", Phys. Plasmas **3**, 1176 (1996)
(``docs/Zheng_1996_PoP_Analytic_Grad_Shafranov_solutions.md``).  Equation numbers
below are Zheng's.

Zheng closes the Grad-Shafranov equation with the Solov'ev-type choice (3),

``-mu0 dP/dPsi = A1``,  ``F dF/dPsi = A2``,  both constant,

for which (2) reduces to ``Delta* Psi = A1 R^2 - A2`` with the exact five-term
solution (14)

``Psi = c1 + c2 R^2 + c3 (R^4 - 4 R^2 Z^2) + c4 (R^2 ln R - Z^2)
        + A1 R^4 / 8 - A2 Z^2 / 2``.

The six constants ``c1..c4, A1, A2`` are fixed by (15)-(18) (``Psi=0`` at the
inboard, outboard and top shape points, and ``dPsi/dR=0`` at the top point) plus
the plasma current (19) and poloidal beta (20).  This script solves that system
for the Fig. 1 spherical-tokamak point design

``R0=0.70 m, a=0.49 m, kappa=1.7, delta=0.125, beta_pol=0.40, Ip=1 MA``

and then benchmarks the finite-element solver against it.

**What is benchmarked.**  ``remec``'s reduced axisymmetric kernel solves note
``(M1)``/``GS_recovered``, ``-Delta* psi = mu0 R^2 p'(psi) + I I'(psi)``, in the
true R-Z weak form

``int grad(psi).grad(v) / R  dR dZ = int (mu0 R p'(psi) + I I'(psi) / R) v dR dZ``.

Matching that against Zheng's reduction gives ``p' = -A1/mu0`` and ``I I' = A2``,
so the Zheng equilibrium is a legitimate exact solution of the shipped weak form
with *constant* profile coefficients.  The FEM domain here is a rectangle in
(R,Z) that encloses the plasma; the analytic (14) supplies the Dirichlet data on
all four sides.

**Why this file assembles its own system.**
``remec.solvers.axisymmetric.AxisymmetricGradShafranovSolver`` currently hard-codes
*homogeneous* Dirichlet data, and the rectangle boundary is not a flux surface, so
the Zheng benchmark cannot be posed through the public API as-is.  Note also that
the constant right-hand side ``A1 R^2 - A2`` has the trivial particular solution
``A1 R^4/8 - A2 Z^2/2``: on a rectangle *all* of the interesting structure of the
Zheng equilibrium (the four ``Delta*``-harmonic terms ``c1..c4``) enters through the
boundary data.  A homogeneous-Dirichlet benchmark on this geometry would therefore
test almost nothing.

``solve_with_dirichlet_data`` below reproduces
``remec.fem._axisymmetric.solve_axisymmetric_grad_shafranov`` line for line except
for the boundary lift (the pattern recorded in ``docs/dev_notes.md`` for milestone
3.1: apply the constrained inverse to ``linear_form.vec - mat * field.vec``).  Two
checks tie it back to the shipped solver:

``--check`` (on by default) verifies that

1. with zero boundary data this file and ``AxisymmetricGradShafranovSolver`` return
   the *same* discrete solution to round-off, so the operator being benchmarked is
   the shipped operator; and
2. discrete superposition holds, ``psi_h = psi_shipped_homogeneous + psi_harmonic_lift``,
   which is the sense in which the shipped solver carries the source term of this
   benchmark.

This script is the committed seed for milestone 5.4 (``docs/STATUS.md``): that
milestone promotes the analytic machinery below (coefficient solve, self-checks,
shape recovery from ``Psi_h``) into an analytic-equilibrium module of the main
package and replaces this rectangle-plus-Dirichlet-lift geometry with shaped R-Z
domains bounded by a ``Psi = const`` contour of the analytic solution, on which the
shipped homogeneous-Dirichlet solver applies unchanged (smooth and X-point
boundaries; see the 5.4 row in ``docs/STATUS.md``).  The Dirichlet-lift assembly
below is therefore transitional, not a proposed public-API change.

Usage::

    python scripts/zheng_grad_shafranov_benchmark.py
    python scripts/zheng_grad_shafranov_benchmark.py --orders 1 2 3 --maxh 0.1 0.05 0.025
    python scripts/zheng_grad_shafranov_benchmark.py --show
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from itertools import pairwise
from math import log, pi
from pathlib import Path
from typing import Any

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from remec.fem._axisymmetric import AxisymmetricGradShafranovCoefficients
from remec.geometry.axisymmetric import AxisymmetricRZDomain
from remec.solvers.axisymmetric import AxisymmetricGradShafranovSolver

_MU0 = 4.0e-7 * pi

# Zheng Fig. 1: the USTX spherical-tokamak point design.
_FIGURE_1 = {
    "major_radius": 0.70,
    "minor_radius": 0.49,
    "elongation": 1.7,
    "triangularity": 0.125,
    "poloidal_beta": 0.40,
    "plasma_current": 1.0e6,
}

# Rectangle enclosing the Fig. 1 plasma (0.21 < R < 1.19, |Z| < 0.833) and bounded
# away from the cylindrical axis, where the 1/R weight of the weak form is singular.
_RADIAL_BOUNDS = (0.15, 1.35)
_VERTICAL_BOUNDS = (-1.0, 1.0)


# --------------------------------------------------------------------------------
# Analytic equilibrium
# --------------------------------------------------------------------------------


@dataclass(frozen=True)
class ZhengShape:
    """The four-parameter up-down-symmetric boundary shape of Zheng's text."""

    major_radius: float
    minor_radius: float
    elongation: float
    triangularity: float

    @property
    def inner_radius(self) -> float:
        """Innermost equatorial point ``R_i = R0 - a``."""
        return self.major_radius - self.minor_radius

    @property
    def outer_radius(self) -> float:
        """Outermost equatorial point ``R_o = R0 + a``."""
        return self.major_radius + self.minor_radius

    @property
    def top_radius(self) -> float:
        """Radius of the highest point, ``R_t = R0 - delta a``."""
        return self.major_radius - self.triangularity * self.minor_radius

    @property
    def top_height(self) -> float:
        """Height of the highest point, ``Z_t = kappa a``."""
        return self.elongation * self.minor_radius


@dataclass(frozen=True)
class ZhengEquilibrium:
    r"""Zheng's exact Grad-Shafranov solution, Eq. (14).

    ``Psi = c1 + c2 R^2 + c3 (R^4 - 4 R^2 Z^2) + c4 (R^2 ln R - Z^2)
            + A1 R^4/8 - A2 Z^2/2``

    satisfies ``Delta* Psi = A1 R^2 - A2`` identically, with
    ``-mu0 dP/dPsi = A1`` and ``F dF/dPsi = A2`` (Eq. 3).
    """

    shape: ZhengShape
    c1: float
    c2: float
    c3: float
    c4: float
    a1: float
    a2: float

    # -- pointwise evaluation ----------------------------------------------------

    def flux(self, radius: Any, height: Any) -> Any:
        """Evaluate Eq. (14); works for NumPy arrays and NGSolve coefficients."""
        logarithm = np.log if not _is_ngsolve(radius) else _ngsolve().log
        return (
            self.c1
            + self.c2 * radius**2
            + self.c3 * (radius**4 - 4.0 * radius**2 * height**2)
            + self.c4 * (radius**2 * logarithm(radius) - height**2)
            + self.a1 / 8.0 * radius**4
            - self.a2 / 2.0 * height**2
        )

    def flux_radial_derivative(self, radius: Any, height: Any) -> Any:
        """``dPsi/dR`` of Eq. (14); the ``c4`` term differentiates to ``2R ln R + R``."""
        logarithm = np.log if not _is_ngsolve(radius) else _ngsolve().log
        return (
            2.0 * self.c2 * radius
            + self.c3 * (4.0 * radius**3 - 8.0 * radius * height**2)
            + self.c4 * (2.0 * radius * logarithm(radius) + radius)
            + self.a1 / 2.0 * radius**3
        )

    def flux_vertical_derivative(self, radius: Any, height: Any) -> Any:
        """``dPsi/dZ`` of Eq. (14)."""
        return -8.0 * self.c3 * radius**2 * height - 2.0 * self.c4 * height - self.a2 * height

    def delta_star_flux(self, radius: Any) -> Any:
        """The exact ``Delta* Psi = A1 R^2 - A2`` of Eq. (4)."""
        return self.a1 * radius**2 - self.a2

    # -- Eq. (14) is linear in Z^2, which makes the plasma region explicit --------

    def _midplane_flux(self, radius: NDArray[np.float64]) -> NDArray[np.float64]:
        """``alpha(R) = Psi(R, 0)``, the ``Z``-independent part of Eq. (14)."""
        return np.asarray(self.flux(radius, 0.0), dtype=float)

    def _vertical_curvature(self, radius: NDArray[np.float64]) -> NDArray[np.float64]:
        """``beta(R)`` in ``Psi = alpha(R) + beta(R) Z^2``; note Eq. (14) has no ``Z^4``."""
        return -4.0 * self.c3 * radius**2 - self.c4 - self.a2 / 2.0

    def boundary_height(self, radius: NDArray[np.float64]) -> NDArray[np.float64]:
        """Half-height of the ``Psi=0`` plasma boundary, ``Z_b = sqrt(-alpha/beta)``."""
        squared = -self._midplane_flux(radius) / self._vertical_curvature(radius)
        return np.sqrt(np.maximum(squared, 0.0))

    def minimum_boundary_height_squared(self, quadrature_nodes: int = 400) -> float:
        """Smallest ``Z_b^2``; negative values mean the ``Psi=0`` contour does not close."""
        radius, _ = _chebyshev_nodes(self.shape, quadrature_nodes)
        return float(np.min(-self._midplane_flux(radius) / self._vertical_curvature(radius)))

    # -- integral diagnostics, Eqs. (19)-(20) ------------------------------------

    def plasma_current(self, quadrature_nodes: int = 400) -> float:
        r"""Eq. (19), ``Ip = -int (R^2 A1 - A2)/(mu0 R) dR dZ`` over the plasma.

        The integrand is ``Z``-independent, so the inner integral is just ``2 Z_b(R)``.
        """
        radius, weights = _chebyshev_nodes(self.shape, quadrature_nodes)
        height = self.boundary_height(radius)
        integrand = -2.0 * height * (radius**2 * self.a1 - self.a2) / (_MU0 * radius)
        return float(np.sum(weights * integrand))

    def flux_integral(self, quadrature_nodes: int = 400) -> float:
        r"""``int Psi dR dZ`` over the plasma, needed by Eq. (20).

        With ``Psi = alpha + beta Z^2`` and ``alpha = -beta Z_b^2``, the inner
        integral collapses to ``2 alpha Z_b + 2 beta Z_b^3/3 = -4 beta Z_b^3/3``.
        """
        radius, weights = _chebyshev_nodes(self.shape, quadrature_nodes)
        height = self.boundary_height(radius)
        integrand = -(4.0 / 3.0) * self._vertical_curvature(radius) * height**3
        return float(np.sum(weights * integrand))

    def poloidal_beta(self, quadrature_nodes: int = 400) -> float:
        """Eq. (20), ``beta_pol = -8 pi A1 int Psi / (mu0^2 Ip^2)``."""
        current = self.plasma_current(quadrature_nodes)
        return float(
            -8.0 * pi * self.a1 * self.flux_integral(quadrature_nodes) / (_MU0 * current) ** 2
        )

    def magnetic_axis(self) -> tuple[float, float]:
        """Locate the on-midplane extremum of ``alpha(R)`` and return ``(R_axis, Psi_axis)``."""
        radius = np.linspace(self.shape.inner_radius, self.shape.outer_radius, 20001)[1:-1]
        return _refined_extremum(radius, self._midplane_flux(radius))

    # -- profiles ----------------------------------------------------------------

    def pressure(self, flux: Any) -> Any:
        """``P = -A1 Psi / mu0``, normalized to vanish on the ``Psi=0`` boundary."""
        return -self.a1 * flux / _MU0

    def toroidal_current_density(self, radius: Any) -> Any:
        """``J_phi = -(R^2 A1 - A2)/(mu0 R)`` from Eqs. (2)-(3)."""
        return -(radius**2 * self.a1 - self.a2) / (_MU0 * radius)

    def summary(self) -> str:
        """One printable block of the derived constants and their check values."""
        axis_radius, axis_flux = self.magnetic_axis()
        return "\n".join(
            (
                f"  c1 = {self.c1:+.12e}",
                f"  c2 = {self.c2:+.12e}",
                f"  c3 = {self.c3:+.12e}",
                f"  c4 = {self.c4:+.12e}",
                f"  A1 = {self.a1:+.12e}",
                f"  A2 = {self.a2:+.12e}",
                f"  Ip                = {self.plasma_current():.9e} A",
                f"  beta_pol          = {self.poloidal_beta():.9f}",
                f"  magnetic axis     = R {axis_radius:.9f} m, Psi {axis_flux:.9e} Wb/rad",
                f"  Shafranov shift   = {axis_radius - self.shape.major_radius:.9f} m",
                f"  on-axis pressure  = {float(self.pressure(axis_flux)):.6e} Pa",
                f"  min Z_b^2         = {self.minimum_boundary_height_squared():+.3e}",
            )
        )


def _chebyshev_nodes(
    shape: ZhengShape, count: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Gauss-Legendre nodes in ``theta`` for ``R = R_mid + a cos(theta)``.

    ``Z_b(R)`` has square-root endpoint behaviour because ``alpha(R)`` has simple
    zeros at ``R_i`` and ``R_o``; this substitution makes the integrand of Eqs.
    (19)-(20) smooth in ``theta``, so plain Gauss-Legendre converges spectrally.
    """
    nodes, weights = leggauss(count)
    angle = 0.5 * pi * (nodes + 1.0)
    angle_weights = 0.5 * pi * weights
    midpoint = 0.5 * (shape.inner_radius + shape.outer_radius)
    half_width = 0.5 * (shape.outer_radius - shape.inner_radius)
    radius = midpoint + half_width * np.cos(angle)
    return radius, angle_weights * half_width * np.sin(angle)


def _refined_extremum(
    abscissa: NDArray[np.float64], values: NDArray[np.float64]
) -> tuple[float, float]:
    """Parabolic refinement of the discrete maximum of a sampled unimodal curve."""
    index = int(np.argmax(values))
    if index in (0, values.size - 1):
        return float(abscissa[index]), float(values[index])
    left, centre, right = values[index - 1 : index + 2]
    step = abscissa[index] - abscissa[index - 1]
    offset = 0.5 * (left - right) / (left - 2.0 * centre + right)
    return (
        float(abscissa[index] + offset * step),
        float(centre - 0.25 * (left - right) * offset),
    )


def _shape_coefficients(shape: ZhengShape, a1: float, a2: float) -> NDArray[np.float64]:
    """Solve Eqs. (15)-(18) for ``c1..c4`` at given ``A1``, ``A2``.

    Rows are, in order: ``Psi=0`` at ``(R_i,0)``, at ``(R_o,0)``, at ``(R_t,Z_t)``,
    and ``dPsi/dR=0`` at ``(R_t,Z_t)`` after dividing Eq. (18) through by ``R_t``.
    """
    inner, outer = shape.inner_radius, shape.outer_radius
    top_radius, top_height = shape.top_radius, shape.top_height
    matrix = np.array(
        (
            (1.0, inner**2, inner**4, inner**2 * log(inner)),
            (1.0, outer**2, outer**4, outer**2 * log(outer)),
            (
                1.0,
                top_radius**2,
                top_radius**2 * (top_radius**2 - 4.0 * top_height**2),
                top_radius**2 * log(top_radius) - top_height**2,
            ),
            (
                0.0,
                2.0,
                4.0 * (top_radius**2 - 2.0 * top_height**2),
                2.0 * log(top_radius) + 1.0,
            ),
        )
    )
    rhs = np.array(
        (
            -a1 / 8.0 * inner**4,
            -a1 / 8.0 * outer**4,
            -a1 / 8.0 * top_radius**4 + a2 / 2.0 * top_height**2,
            -a1 / 2.0 * top_radius**2,
        )
    )
    return np.linalg.solve(matrix, rhs)


def _equilibrium_at(shape: ZhengShape, a1: float, a2: float) -> ZhengEquilibrium:
    """Build the Eq. (14) solution whose shape points are pinned by Eqs. (15)-(18)."""
    c1, c2, c3, c4 = _shape_coefficients(shape, a1, a2)
    return ZhengEquilibrium(shape, float(c1), float(c2), float(c3), float(c4), a1, a2)


def solve_zheng_coefficients(
    *,
    major_radius: float,
    minor_radius: float,
    elongation: float,
    triangularity: float,
    poloidal_beta: float,
    plasma_current: float,
    bisection_steps: int = 200,
) -> ZhengEquilibrium:
    r"""Solve Zheng's full system (15)-(20) for ``c1..c4, A1, A2``.

    Eqs. (15)-(18) are linear in ``c1..c4`` at fixed ``(A1, A2)``, and Zheng's text
    notes that multiplying all six constants by ``alpha_I`` rescales ``Ip`` while
    leaving both the shape and ``beta_pol`` unchanged.  The nonlinear system
    therefore collapses to one scalar root find: fix ``A1 = -1`` (negative, so that
    ``P = -A1 Psi/mu0 > 0`` inside, where ``Psi > 0``), find the ``A2`` reproducing
    the requested ``beta_pol``, and then rescale everything to the requested ``Ip``.
    """
    shape = ZhengShape(major_radius, minor_radius, elongation, triangularity)

    def beta_residual(a2: float) -> float:
        """``beta_pol(A2) - beta_pol_target`` along the unit-``A1`` branch."""
        candidate = _equilibrium_at(shape, -1.0, a2)
        if candidate.minimum_boundary_height_squared() < -1.0e-12:
            return float("nan")
        return candidate.poloidal_beta() - poloidal_beta

    lower, upper = _bracket_beta_root(beta_residual)
    for _ in range(bisection_steps):
        middle = 0.5 * (lower + upper)
        if beta_residual(lower) * beta_residual(middle) <= 0.0:
            upper = middle
        else:
            lower = middle
    unit = _equilibrium_at(shape, -1.0, 0.5 * (lower + upper))

    scale = plasma_current / unit.plasma_current()
    return _equilibrium_at(shape, -1.0 * scale, unit.a2 * scale)


def _bracket_beta_root(residual: Any) -> tuple[float, float]:
    """Bracket the ``beta_pol`` root on the physical ``A2 > 0`` branch.

    ``beta_pol`` falls monotonically with ``A2`` there; below some positive ``A2``
    the ``Psi=0`` contour stops closing and the residual is NaN.
    """
    candidates = np.geomspace(1.0e-4, 1.0e3, 200)
    values = np.array([residual(float(value)) for value in candidates])
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        raise RuntimeError("no closed-boundary A2 branch found")
    signs = np.sign(values[finite])
    changes = np.flatnonzero(signs[:-1] * signs[1:] < 0.0)
    if changes.size == 0:
        raise RuntimeError("requested poloidal beta is not attained on the A2 > 0 branch")
    first = int(changes[0])
    return float(candidates[finite[first]]), float(candidates[finite[first + 1]])


def _is_ngsolve(value: Any) -> bool:
    """Distinguish NGSolve coefficient arguments from NumPy ones."""
    return type(value).__module__.startswith("ngsolve")


def _ngsolve() -> Any:
    """Import NGSolve lazily, mirroring the deferred import in ``remec.fem``."""
    import ngsolve as ng  # type: ignore[import-untyped]

    return ng


def verify_analytic_solution(equilibrium: ZhengEquilibrium) -> dict[str, float]:
    """Finite-difference check that Eq. (14) satisfies Eq. (4) and its stated gradient.

    Guards the transcription of Eq. (14) and of the hand-differentiated gradient
    used for the weighted-energy error norm.
    """
    rng = np.random.default_rng(20260818)
    radius = rng.uniform(0.2, 1.3, 400)
    height = rng.uniform(-0.9, 0.9, 400)
    step = 1.0e-5

    def flux(r: NDArray[np.float64], z: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(equilibrium.flux(r, z), dtype=float)

    d_radius = (flux(radius + step, height) - flux(radius - step, height)) / (2.0 * step)
    d_height = (flux(radius, height + step) - flux(radius, height - step)) / (2.0 * step)
    scale = np.max(np.abs(flux(radius, height)))

    # Delta* Psi = R d/dR( (1/R) dPsi/dR ) + d2Psi/dZ2, by central differences of
    # the analytic first derivatives.
    def scaled_radial(r: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(equilibrium.flux_radial_derivative(r, height), dtype=float) / r

    delta_star = radius * (scaled_radial(radius + step) - scaled_radial(radius - step)) / (
        2.0 * step
    ) + (
        np.asarray(equilibrium.flux_vertical_derivative(radius, height + step), dtype=float)
        - np.asarray(equilibrium.flux_vertical_derivative(radius, height - step), dtype=float)
    ) / (2.0 * step)
    exact_delta_star = np.asarray(equilibrium.delta_star_flux(radius), dtype=float)

    return {
        "radial_gradient": float(
            np.max(np.abs(d_radius - equilibrium.flux_radial_derivative(radius, height))) / scale
        ),
        "vertical_gradient": float(
            np.max(np.abs(d_height - equilibrium.flux_vertical_derivative(radius, height))) / scale
        ),
        "delta_star": float(
            np.max(np.abs(delta_star - exact_delta_star)) / np.max(np.abs(exact_delta_star))
        ),
    }


# --------------------------------------------------------------------------------
# Finite-element solve
# --------------------------------------------------------------------------------


@dataclass(frozen=True)
class DirichletSolution:
    """One inhomogeneous-Dirichlet GS solve plus the NGSolve objects it owns."""

    mesh: Any
    space: Any
    flux: Any
    polynomial_order: int
    elements: int
    degrees_of_freedom: int
    free_dof_relative_residual_norm: float

    def flux_at(self, radius: float, height: float) -> float:
        """Evaluate the computed flux at one physical point."""
        return float(self.flux(self.mesh(radius, height)))

    def flux_on(
        self, radii: NDArray[np.float64], heights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Evaluate the computed flux on a batch of physical points."""
        radii, heights = np.broadcast_arrays(np.asarray(radii, float), np.asarray(heights, float))
        values = np.asarray(self.flux(self.mesh(radii.ravel(), heights.ravel())), dtype=float)
        return values.reshape(radii.shape)


def solve_with_dirichlet_data(
    domain: AxisymmetricRZDomain,
    *,
    polynomial_order: int,
    coefficients: AxisymmetricGradShafranovCoefficients,
    boundary_flux: Any,
) -> DirichletSolution:
    r"""``remec``'s R-Z Grad-Shafranov weak form with inhomogeneous Dirichlet data.

    Identical to ``remec.fem._axisymmetric.solve_axisymmetric_grad_shafranov`` --

    ``int grad(psi).grad(v)/R dR dZ = int (mu0 R p'(psi) + I I'(psi)/R) v dR dZ``

    -- except that the boundary degrees of freedom are first set to
    ``boundary_flux`` and the constrained inverse is then applied to the lift
    residual ``linear_form.vec - mat * flux.vec`` (``docs/dev_notes.md``,
    milestone 3.1).  With ``boundary_flux = 0`` it reproduces the shipped solver
    exactly; ``--check`` asserts that.
    """
    ng = _ngsolve()

    mesh = domain.build_mesh()._mesh
    space = ng.H1(mesh, order=polynomial_order, dirichlet="bottom|right|top|left")
    trial, test = space.TnT()
    quadrature = ng.dx(bonus_intorder=6)
    bilinear_form = ng.BilinearForm(space)
    bilinear_form += (ng.InnerProduct(ng.grad(trial), ng.grad(test)) / ng.x).Compile() * quadrature
    linear_form = ng.LinearForm(space)
    source = (
        coefficients.mu0 * ng.x * coefficients.pressure_flux_derivative
        + coefficients.toroidal_field_drive / ng.x
    )
    linear_form += (source * test).Compile() * quadrature
    free_dofs = space.FreeDofs()

    with ng.TaskManager():
        bilinear_form.Assemble()
        linear_form.Assemble()
        flux = ng.GridFunction(space)
        flux.Set(boundary_flux, ng.BND)
        residual = linear_form.vec.CreateVector()
        residual.data = linear_form.vec - bilinear_form.mat * flux.vec
        inverse = bilinear_form.mat.Inverse(free_dofs, inverse="umfpack")
        flux.vec.data += inverse * residual
        residual.data = linear_form.vec - bilinear_form.mat * flux.vec
        free_residual = ng.Projector(free_dofs, True) * residual
        free_load = ng.Projector(free_dofs, True) * linear_form.vec
        relative_residual = float(ng.Norm(free_residual)) / max(1.0e-300, float(ng.Norm(free_load)))

    return DirichletSolution(
        mesh,
        space,
        flux,
        polynomial_order,
        mesh.ne,
        space.ndof,
        relative_residual,
    )


def grad_shafranov_coefficients(
    equilibrium: ZhengEquilibrium,
) -> AxisymmetricGradShafranovCoefficients:
    """Map Zheng's Eq. (3) constants onto the note-``(M1)`` source record.

    ``remec`` writes the source as ``mu0 R^2 p'(psi) + I I'(psi)``; Zheng writes
    ``-A1 R^2 + A2``.  Hence ``p' = -A1/mu0`` (Eq. 3's ``-mu0 dP/dPsi = A1``) and
    ``I I' = A2`` (Eq. 3's ``F dF/dPsi = A2``).
    """
    return AxisymmetricGradShafranovCoefficients(
        pressure_flux_derivative=-equilibrium.a1 / _MU0,
        toroidal_field_drive=equilibrium.a2,
        mu0=_MU0,
    )


def _bisect_zero(
    evaluate: Any,
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    *,
    steps: int = 60,
) -> NDArray[np.float64]:
    """Vectorized bisection for a sign change between ``lower`` and ``upper``."""
    lower, upper = np.array(lower, float), np.array(upper, float)
    lower_sign = np.sign(evaluate(lower))
    for _ in range(steps):
        middle = 0.5 * (lower + upper)
        same_side = np.sign(evaluate(middle)) == lower_sign
        lower = np.where(same_side, middle, lower)
        upper = np.where(same_side, upper, middle)
    return 0.5 * (lower + upper)


def recover_shape(
    solution: DirichletSolution, equilibrium: ZhengEquilibrium, *, samples: int = 401
) -> ZhengShape:
    """Recover ``R0, a, kappa, delta`` from the ``Psi_h = 0`` contour.

    This is the physically meaningful half of the benchmark: the four Fig. 1 shape
    parameters are re-measured from the computed flux exactly as Zheng defines
    them, rather than being read back from the coefficients that produced it.
    ``Psi_h`` is inverted by bisection -- on the midplane for the two equatorial
    extrema, then in ``Z`` at fixed ``R`` for the boundary height, whose maximum
    over ``R`` gives the top point.
    """
    axis_radius, _ = equilibrium.magnetic_axis()

    def midplane(radius: NDArray[np.float64]) -> NDArray[np.float64]:
        return solution.flux_on(radius, np.zeros_like(radius))

    inner = float(
        _bisect_zero(
            midplane,
            np.array([_RADIAL_BOUNDS[0] + 1.0e-9]),
            np.array([axis_radius]),
        )[0]
    )
    outer = float(
        _bisect_zero(
            midplane,
            np.array([_RADIAL_BOUNDS[1] - 1.0e-9]),
            np.array([axis_radius]),
        )[0]
    )

    def boundary_height(radius: NDArray[np.float64]) -> NDArray[np.float64]:
        """Invert ``Psi_h(R, Z) = 0`` in ``Z`` above the midplane."""
        return _bisect_zero(
            lambda height: solution.flux_on(radius, height),
            np.zeros_like(radius),
            np.full_like(radius, _VERTICAL_BOUNDS[1] - 1.0e-9),
        )

    # Both bisection endpoints must straddle Psi_h = 0, so trim the sampled band
    # away from the two equatorial points where Z_b -> 0.  The top point is a flat
    # maximum, so its *radius* -- and hence the triangularity -- needs a refined
    # bracket; one parabolic pass on a fixed grid would leave an O(dR^2) search
    # error well above the order-3 discretization error.
    margin = 1.0e-4 * (outer - inner)
    lower, upper = inner + margin, outer - margin
    top_radius = top_height = 0.0
    for refinement in range(3):
        count = samples if refinement == 0 else 81
        radius = np.linspace(lower, upper, count)
        top_radius, top_height = _refined_extremum(radius, boundary_height(radius))
        spacing = radius[1] - radius[0]
        lower = max(inner + margin, top_radius - 2.0 * spacing)
        upper = min(outer - margin, top_radius + 2.0 * spacing)

    major_radius = 0.5 * (inner + outer)
    minor_radius = 0.5 * (outer - inner)
    return ZhengShape(
        major_radius,
        minor_radius,
        top_height / minor_radius,
        (major_radius - top_radius) / minor_radius,
    )


@dataclass(frozen=True)
class ConvergenceRow:
    """One (order, maxh) entry of the benchmark table."""

    polynomial_order: int
    maxh: float
    elements: int
    degrees_of_freedom: int
    l2_error: float
    relative_l2_error: float
    weighted_energy_error: float
    relative_weighted_energy_error: float
    free_dof_relative_residual_norm: float
    axis_radius_error: float
    axis_flux_relative_error: float
    major_radius_error: float
    minor_radius_error: float
    elongation_error: float
    triangularity_error: float


def _measure(
    equilibrium: ZhengEquilibrium,
    solution: DirichletSolution,
    maxh: float,
) -> ConvergenceRow:
    """Measure L2 and weighted-energy errors plus the magnetic-axis error."""
    ng = _ngsolve()

    exact = equilibrium.flux(ng.x, ng.y)
    exact_gradient = ng.CoefficientFunction(
        (
            equilibrium.flux_radial_derivative(ng.x, ng.y),
            equilibrium.flux_vertical_derivative(ng.x, ng.y),
        )
    )
    order = 2 * solution.polynomial_order + 8
    l2_error = float(
        ng.sqrt(ng.Integrate((solution.flux - exact) ** 2, solution.mesh, order=order))
    )
    l2_norm = float(ng.sqrt(ng.Integrate(exact**2, solution.mesh, order=order)))
    gradient_error = ng.grad(solution.flux) - exact_gradient
    energy_error = float(
        ng.sqrt(
            ng.Integrate(
                ng.InnerProduct(gradient_error, gradient_error) / ng.x, solution.mesh, order=order
            )
        )
    )
    energy_norm = float(
        ng.sqrt(
            ng.Integrate(
                ng.InnerProduct(exact_gradient, exact_gradient) / ng.x, solution.mesh, order=order
            )
        )
    )

    exact_axis_radius, exact_axis_flux = equilibrium.magnetic_axis()
    radius = np.linspace(equilibrium.shape.inner_radius, equilibrium.shape.outer_radius, 2001)
    sampled = solution.flux_on(radius, np.zeros_like(radius))
    axis_radius, axis_flux = _refined_extremum(radius, sampled)
    recovered = recover_shape(solution, equilibrium)
    requested = equilibrium.shape

    return ConvergenceRow(
        solution.polynomial_order,
        maxh,
        solution.elements,
        solution.degrees_of_freedom,
        l2_error,
        l2_error / l2_norm,
        energy_error,
        energy_error / energy_norm,
        solution.free_dof_relative_residual_norm,
        abs(axis_radius - exact_axis_radius),
        abs(axis_flux - exact_axis_flux) / abs(exact_axis_flux),
        abs(recovered.major_radius - requested.major_radius),
        abs(recovered.minor_radius - requested.minor_radius),
        abs(recovered.elongation - requested.elongation),
        abs(recovered.triangularity - requested.triangularity),
    )


def run_convergence_study(
    equilibrium: ZhengEquilibrium,
    *,
    orders: tuple[int, ...],
    mesh_sizes: tuple[float, ...],
) -> tuple[list[ConvergenceRow], dict[tuple[int, float], DirichletSolution]]:
    """Solve the benchmark on every (order, maxh) pair and measure the errors."""
    ng = _ngsolve()
    coefficients = grad_shafranov_coefficients(equilibrium)
    rows: list[ConvergenceRow] = []
    solutions: dict[tuple[int, float], DirichletSolution] = {}
    for order in orders:
        for maxh in mesh_sizes:
            domain = AxisymmetricRZDomain(_RADIAL_BOUNDS, _VERTICAL_BOUNDS, maxh)
            solution = solve_with_dirichlet_data(
                domain,
                polynomial_order=order,
                coefficients=coefficients,
                boundary_flux=equilibrium.flux(ng.x, ng.y),
            )
            solutions[order, maxh] = solution
            rows.append(_measure(equilibrium, solution, maxh))
            row = rows[-1]
            print(
                f"  order {order}  maxh {maxh:<8.5g} elements {row.elements:>6d} "
                f"ndof {row.degrees_of_freedom:>7d} "
                f"L2 {row.l2_error:.4e}  energy {row.weighted_energy_error:.4e}"
            )
    return rows, solutions


def observed_rates(rows: list[ConvergenceRow]) -> dict[int, tuple[list[float], list[float]]]:
    """Successive-refinement rates ``log2(e_coarse/e_fine)`` per polynomial order."""
    rates: dict[int, tuple[list[float], list[float]]] = {}
    for order in sorted({row.polynomial_order for row in rows}):
        ordered = sorted(
            (row for row in rows if row.polynomial_order == order), key=lambda r: -r.maxh
        )
        l2 = [log(a.l2_error / b.l2_error) / log(a.maxh / b.maxh) for a, b in pairwise(ordered)]
        energy = [
            log(a.weighted_energy_error / b.weighted_energy_error) / log(a.maxh / b.maxh)
            for a, b in pairwise(ordered)
        ]
        rates[order] = (l2, energy)
    return rates


# --------------------------------------------------------------------------------
# Cross-checks against the shipped solver
# --------------------------------------------------------------------------------


def check_against_shipped_solver(
    equilibrium: ZhengEquilibrium,
    *,
    polynomial_order: int,
    maxh: float,
) -> dict[str, float]:
    """Tie this file's assembly back to ``AxisymmetricGradShafranovSolver``.

    ``identity`` is the largest pointwise disagreement between the shipped solver
    and this file at zero boundary data -- if it is at round-off, the operator
    benchmarked above is the shipped operator.  ``superposition`` checks the exact
    discrete identity ``psi_h = psi_shipped + psi_lift``, where ``psi_shipped``
    carries the Zheng source with zero boundary data and ``psi_lift`` is the
    ``Delta*``-harmonic extension of the analytic boundary data.
    """
    ng = _ngsolve()
    domain = AxisymmetricRZDomain(_RADIAL_BOUNDS, _VERTICAL_BOUNDS, maxh)
    coefficients = grad_shafranov_coefficients(equilibrium)
    zero_source = AxisymmetricGradShafranovCoefficients(
        pressure_flux_derivative=0.0, toroidal_field_drive=0.0, mu0=_MU0
    )
    boundary_flux = equilibrium.flux(ng.x, ng.y)

    shipped = AxisymmetricGradShafranovSolver(polynomial_order=polynomial_order).solve_with_flux(
        domain, coefficients
    )
    homogeneous = solve_with_dirichlet_data(
        domain,
        polynomial_order=polynomial_order,
        coefficients=coefficients,
        boundary_flux=ng.CoefficientFunction(0.0),
    )
    lift = solve_with_dirichlet_data(
        domain,
        polynomial_order=polynomial_order,
        coefficients=zero_source,
        boundary_flux=boundary_flux,
    )
    direct = solve_with_dirichlet_data(
        domain,
        polynomial_order=polynomial_order,
        coefficients=coefficients,
        boundary_flux=boundary_flux,
    )

    rng = np.random.default_rng(11235)
    radii = rng.uniform(_RADIAL_BOUNDS[0] + 1.0e-3, _RADIAL_BOUNDS[1] - 1.0e-3, 200)
    heights = rng.uniform(_VERTICAL_BOUNDS[0] + 1.0e-3, _VERTICAL_BOUNDS[1] - 1.0e-3, 200)

    shipped_values = np.array([shipped.flux_at(float(r), float(z)) for r, z in zip(radii, heights)])
    homogeneous_values = np.array(
        [homogeneous.flux_at(float(r), float(z)) for r, z in zip(radii, heights)]
    )
    lift_values = np.array([lift.flux_at(float(r), float(z)) for r, z in zip(radii, heights)])
    direct_values = np.array([direct.flux_at(float(r), float(z)) for r, z in zip(radii, heights)])

    scale = float(np.max(np.abs(direct_values)))
    return {
        "identity": float(np.max(np.abs(shipped_values - homogeneous_values))) / scale,
        "superposition": float(
            np.max(np.abs(direct_values - shipped_values - lift_values)) / scale
        ),
        "shipped_residual": shipped.result.free_dof_relative_residual_norm,
    }


# --------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------


def write_rate_table(rows: list[ConvergenceRow], path: Path) -> None:
    """Write the measured benchmark table next to the figure."""
    with path.open("w", newline="") as table_file:
        writer = csv.writer(table_file)
        writer.writerow(
            (
                "polynomial_order",
                "maxh",
                "elements",
                "degrees_of_freedom",
                "l2_error",
                "relative_l2_error",
                "weighted_energy_error",
                "relative_weighted_energy_error",
                "free_dof_relative_residual_norm",
                "axis_radius_error",
                "axis_flux_relative_error",
                "major_radius_error",
                "minor_radius_error",
                "elongation_error",
                "triangularity_error",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.polynomial_order,
                    repr(row.maxh),
                    row.elements,
                    row.degrees_of_freedom,
                    repr(row.l2_error),
                    repr(row.relative_l2_error),
                    repr(row.weighted_energy_error),
                    repr(row.relative_weighted_energy_error),
                    repr(row.free_dof_relative_residual_norm),
                    repr(row.axis_radius_error),
                    repr(row.axis_flux_relative_error),
                    repr(row.major_radius_error),
                    repr(row.minor_radius_error),
                    repr(row.elongation_error),
                    repr(row.triangularity_error),
                )
            )


def _sample(solution: DirichletSolution, resolution: int) -> tuple[Any, Any, Any]:
    """Evaluate a computed flux on a regular grid just inside the FEM rectangle."""
    radius = np.linspace(_RADIAL_BOUNDS[0] + 1.0e-9, _RADIAL_BOUNDS[1] - 1.0e-9, resolution)
    height = np.linspace(_VERTICAL_BOUNDS[0] + 1.0e-9, _VERTICAL_BOUNDS[1] - 1.0e-9, resolution)
    grid_radius, grid_height = np.meshgrid(radius, height)
    return grid_radius, grid_height, solution.flux_on(grid_radius, grid_height)


def make_figure(
    equilibrium: ZhengEquilibrium,
    rows: list[ConvergenceRow],
    finest: DirichletSolution,
    *,
    resolution: int,
) -> Figure:
    """Six panels: the Fig. 1 reproduction, the FEM solution, and the error behaviour."""
    grid_radius, grid_height, computed = _sample(finest, resolution)
    exact = np.asarray(equilibrium.flux(grid_radius, grid_height), dtype=float)
    _, axis_flux = equilibrium.magnetic_axis()

    figure, axes = plt.subplots(2, 3, figsize=(16.5, 10.0))

    boundary_radius = np.linspace(
        equilibrium.shape.inner_radius, equilibrium.shape.outer_radius, 601
    )
    boundary_height = equilibrium.boundary_height(boundary_radius)

    def draw_plasma(axis: Any, colour: str = "white") -> None:
        """Overlay the analytic ``Psi=0`` plasma boundary and its shape points."""
        axis.plot(boundary_radius, boundary_height, "-", color=colour, lw=1.6)
        axis.plot(boundary_radius, -boundary_height, "-", color=colour, lw=1.6)
        axis.plot(
            (
                equilibrium.shape.inner_radius,
                equilibrium.shape.outer_radius,
                equilibrium.shape.top_radius,
            ),
            (0.0, 0.0, equilibrium.shape.top_height),
            "o",
            ms=4.0,
            color=colour,
            mfc="none",
        )

    levels = np.linspace(0.0, axis_flux, 13)[1:]
    panel = axes[0, 0]
    filled = panel.contourf(grid_radius, grid_height, exact, levels=40, cmap="viridis")
    panel.contour(grid_radius, grid_height, exact, levels=levels, colors="white", linewidths=0.7)
    draw_plasma(panel)
    figure.colorbar(filled, ax=panel, shrink=0.9, label="Wb/rad")
    panel.set(
        title="Analytic $\\Psi$ (Zheng Eq. 14), Fig. 1 parameters",
        xlabel="R [m]",
        ylabel="Z [m]",
        aspect="equal",
    )

    panel = axes[0, 1]
    filled = panel.contourf(grid_radius, grid_height, computed, levels=40, cmap="viridis")
    panel.contour(grid_radius, grid_height, computed, levels=levels, colors="white", linewidths=0.7)
    draw_plasma(panel)
    figure.colorbar(filled, ax=panel, shrink=0.9, label="Wb/rad")
    panel.set(
        title=(f"FEM $\\Psi_h$ (order {finest.polynomial_order}, {finest.elements} elements)"),
        xlabel="R [m]",
        ylabel="Z [m]",
        aspect="equal",
    )

    panel = axes[0, 2]
    difference = computed - exact
    limit = float(np.max(np.abs(difference))) or 1.0
    filled = panel.contourf(
        grid_radius, grid_height, difference, levels=40, cmap="RdBu_r", vmin=-limit, vmax=limit
    )
    draw_plasma(panel, colour="0.25")
    figure.colorbar(filled, ax=panel, shrink=0.9, label="Wb/rad")
    panel.set(
        title=f"$\\Psi_h-\\Psi$  (max $|\\cdot|$ = {limit:.2e})",
        xlabel="R [m]",
        ylabel="Z [m]",
        aspect="equal",
    )

    panel = axes[1, 0]
    midplane = np.linspace(_RADIAL_BOUNDS[0] + 1.0e-9, _RADIAL_BOUNDS[1] - 1.0e-9, 401)
    panel.plot(midplane, equilibrium.flux(midplane, 0.0), "k-", lw=2.0, label="analytic")
    panel.plot(
        midplane[::8],
        finest.flux_on(midplane[::8], np.zeros_like(midplane[::8])),
        "o",
        ms=4.0,
        mfc="none",
        color="tab:red",
        label="FEM",
    )
    panel.axhline(0.0, color="0.7", lw=0.8)
    for marker in (equilibrium.shape.inner_radius, equilibrium.shape.outer_radius):
        panel.axvline(marker, color="0.7", lw=0.8, ls=":")
    panel.set(title="Midplane $\\Psi(R,0)$", xlabel="R [m]", ylabel="$\\Psi$ [Wb/rad]")
    panel.legend()

    for panel, attribute, name, offset in (
        (axes[1, 1], "relative_l2_error", "relative $L^2$ error", 1),
        (axes[1, 2], "relative_weighted_energy_error", "relative weighted-energy error", 0),
    ):
        for order in sorted({row.polynomial_order for row in rows}):
            ordered = sorted(
                (row for row in rows if row.polynomial_order == order), key=lambda r: -r.maxh
            )
            mesh_sizes = np.array([row.maxh for row in ordered])
            errors = np.array([getattr(row, attribute) for row in ordered])
            line = panel.loglog(mesh_sizes, errors, "o-", label=f"order {order}")[0]
            reference = errors[0] * (mesh_sizes / mesh_sizes[0]) ** (order + offset)
            panel.loglog(
                mesh_sizes,
                reference,
                "--",
                color=line.get_color(),
                lw=0.9,
                label=f"$h^{{{order + offset}}}$",
            )
        panel.set(title=name, xlabel="maxh [m]", ylabel="relative error")
        panel.grid(True, which="both", alpha=0.3)
        panel.legend(fontsize=8, ncol=2)

    figure.suptitle(
        "remec axisymmetric Grad-Shafranov vs. Zheng (1996) analytic equilibrium: "
        "$R_0$=0.70 m, $a$=0.49 m, $\\kappa$=1.7, $\\delta$=0.125, "
        "$\\beta_{pol}$=0.40, $I_p$=1 MA",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return figure


def parse_arguments() -> argparse.Namespace:
    """Command-line settings for the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--orders", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--maxh", type=float, nargs="+", default=(0.1, 0.05, 0.025))
    parser.add_argument("--output-dir", type=Path, default=Path("scratch"))
    parser.add_argument("--resolution", type=int, default=201)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--no-check",
        dest="check",
        action="store_false",
        help="Skip the consistency checks against the shipped solver.",
    )
    return parser.parse_args()


def main() -> int:
    """Solve Zheng's constants, run the FEM benchmark, and report."""
    arguments = parse_arguments()
    orders = tuple(arguments.orders)
    mesh_sizes = tuple(sorted(arguments.maxh, reverse=True))

    print("Zheng (1996) Fig. 1 equilibrium -- solving Eqs. (15)-(20)")
    equilibrium = solve_zheng_coefficients(**_FIGURE_1)
    print(equilibrium.summary())
    requested = (
        f"  requested: Ip {_FIGURE_1['plasma_current']:.6e} A, "
        f"beta_pol {_FIGURE_1['poloidal_beta']}"
    )
    print(requested)

    print("\nAnalytic self-check (central differences vs. Eqs. 4 and 14)")
    for name, value in verify_analytic_solution(equilibrium).items():
        print(f"  {name:<20s} {value:.3e}")

    print(
        f"\nFEM benchmark on R in {_RADIAL_BOUNDS}, Z in {_VERTICAL_BOUNDS}; "
        "Dirichlet data from Eq. (14)"
    )
    rows, solutions = run_convergence_study(equilibrium, orders=orders, mesh_sizes=mesh_sizes)

    print("\nObserved convergence rates (successive refinement)")
    for order, (l2_rates, energy_rates) in observed_rates(rows).items():
        formatted_l2 = ", ".join(f"{rate:.2f}" for rate in l2_rates)
        formatted_energy = ", ".join(f"{rate:.2f}" for rate in energy_rates)
        print(
            f"  order {order}:  L2 [{formatted_l2}] (expect {order + 1})"
            f"   weighted energy [{formatted_energy}] (expect {order})"
        )

    print(f"\nRecovery from the computed flux on the finest mesh (maxh {mesh_sizes[-1]:g})")
    print(
        f"  {'order':>5s} {'|dR_axis|':>11s} {'Psi_axis rel':>13s} {'|dR0|':>11s} "
        f"{'|da|':>11s} {'|dkappa|':>11s} {'|ddelta|':>11s}"
    )
    for order in orders:
        row = next(r for r in rows if r.polynomial_order == order and r.maxh == mesh_sizes[-1])
        print(
            f"  {order:>5d} {row.axis_radius_error:>11.3e} "
            f"{row.axis_flux_relative_error:>13.3e} {row.major_radius_error:>11.3e} "
            f"{row.minor_radius_error:>11.3e} {row.elongation_error:>11.3e} "
            f"{row.triangularity_error:>11.3e}"
        )

    if arguments.check:
        print("\nConsistency with remec.solvers.axisymmetric.AxisymmetricGradShafranovSolver")
        checks = check_against_shipped_solver(equilibrium, polynomial_order=2, maxh=mesh_sizes[0])
        for name, value in checks.items():
            print(f"  {name:<20s} {value:.3e}")

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = arguments.output_dir / "zheng_grad_shafranov_rates.csv"
    write_rate_table(rows, table_path)
    print(f"\nWrote {table_path}")

    finest = solutions[max(orders), mesh_sizes[-1]]
    figure = make_figure(equilibrium, rows, finest, resolution=arguments.resolution)
    if arguments.show:
        plt.show()
    else:
        figure_path = arguments.output_dir / "zheng_grad_shafranov_benchmark.png"
        figure.savefig(figure_path, dpi=140)
        print(f"Wrote {figure_path}")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
