r"""Analytic Solov'ev equilibria for the ideal axisymmetric ``(M1)`` benchmark.

The classes in this module implement Zheng et al. (1996), Eqs. (14)--(20), and
Cerfon--Freidberg (2010), Eqs. (5)--(12).  Both solve coefficient constraints,
evaluate :math:`\psi` and its first/second derivatives, extract a
:math:`\psi=0` contour, and evaluate the papers' figure-of-merit integrals.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from math import asin, cos, isfinite, log, pi
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

_MU0 = 4.0e-7 * pi
_Term = tuple[float, int, int, bool]


def _is_ngsolve(value: Any) -> bool:
    """Return whether ``value`` is an NGSolve expression without importing NGSolve."""
    return type(value).__module__.startswith("ngsolve")


def _logarithm(value: Any) -> Any:
    """Apply NumPy or NGSolve logarithm according to the expression backend."""
    if _is_ngsolve(value):
        import ngsolve as ng  # type: ignore[import-untyped]

        return ng.log(value)
    return np.log(value)


def _falling_factorial(power: int, derivative: int) -> int:
    """Return ``power!/(power-derivative)!``, or zero past a polynomial's degree."""
    if derivative > power:
        return 0
    result = 1
    for value in range(power - derivative + 1, power + 1):
        result *= value
    return result


def _radial_factor(radius: Any, power: int, logarithmic: bool, derivative: int) -> Any:
    """Evaluate ``d^derivative/dR^derivative [R^power (log R)^logarithmic]``."""
    if not logarithmic:
        factor = _falling_factorial(power, derivative)
        return factor * radius ** (power - derivative) if factor else 0.0 * radius
    logarithm = _logarithm(radius)
    if derivative == 0:
        return radius**power * logarithm
    if derivative == 1:
        return radius ** (power - 1) * (power * logarithm + 1.0)
    if derivative == 2:
        return radius ** (power - 2) * (power * (power - 1) * logarithm + 2 * power - 1)
    raise ValueError("analytic equilibria expose derivatives only through second order")


def _evaluate_terms(
    terms: Sequence[_Term],
    radius: Any,
    height: Any,
    *,
    radial_derivative: int = 0,
    vertical_derivative: int = 0,
) -> Any:
    """Evaluate a polynomial/logarithmic Solov'ev expansion or one derivative."""
    value: Any = 0.0 * radius + 0.0 * height
    for coefficient, radial_power, vertical_power, logarithmic in terms:
        vertical_factor = _falling_factorial(vertical_power, vertical_derivative)
        if vertical_factor == 0:
            continue
        value += (
            coefficient
            * _radial_factor(radius, radial_power, logarithmic, radial_derivative)
            * vertical_factor
            * height ** (vertical_power - vertical_derivative)
        )
    return value


@dataclass(frozen=True, slots=True)
class FluxContour:
    """One counter-clockwise constant-flux contour sampled without a duplicate endpoint."""

    radius: NDArray[np.float64]
    height: NDArray[np.float64]
    corner_indices: tuple[int, ...] = ()
    parameterizations: tuple[Callable[[float], tuple[float, float]], ...] = ()

    def __post_init__(self) -> None:
        radius = np.asarray(self.radius, dtype=float)
        height = np.asarray(self.height, dtype=float)
        if radius.ndim != 1 or height.shape != radius.shape or radius.size < 8:
            raise ValueError("a flux contour needs equally sized one-dimensional coordinate arrays")
        if not np.all(np.isfinite(radius)) or not np.all(np.isfinite(height)):
            raise ValueError("flux-contour coordinates must be finite")
        if np.any(radius <= 0.0):
            raise ValueError("an axisymmetric flux contour must have R > 0")
        if any(index < 0 or index >= radius.size for index in self.corner_indices):
            raise ValueError("flux-contour corner indices are out of range")
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "height", height)
        if self.signed_area <= 0.0:
            raise ValueError("flux-contour samples must be counter-clockwise")

    @property
    def signed_area(self) -> float:
        """Return the oriented poloidal area enclosed by the contour."""
        return 0.5 * float(
            np.sum(self.radius * np.roll(self.height, -1) - np.roll(self.radius, -1) * self.height)
        )

    @property
    def radial_bounds(self) -> tuple[float, float]:
        """Return the sampled minimum and maximum cylindrical radius."""
        return float(np.min(self.radius)), float(np.max(self.radius))

    @property
    def vertical_bounds(self) -> tuple[float, float]:
        """Return the sampled minimum and maximum vertical coordinate."""
        return float(np.min(self.height)), float(np.max(self.height))

    def curve_segments(self) -> tuple[tuple[NDArray[np.float64], NDArray[np.float64]], ...]:
        """Split the closed contour at physical corners for geometry construction."""
        count = self.radius.size
        if not self.corner_indices:
            return (
                (
                    np.append(self.radius, self.radius[0]),
                    np.append(self.height, self.height[0]),
                ),
            )
        corners = tuple(sorted(set(self.corner_indices)))
        segments: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
        for start, stop in zip(corners, corners[1:] + (corners[0] + count,)):
            indices = np.arange(start, stop + 1) % count
            segments.append((self.radius[indices], self.height[indices]))
        return tuple(segments)


@dataclass(frozen=True, slots=True)
class ZhengShape:
    """The four up-down-symmetric shape parameters in Zheng Eqs. (15)--(18)."""

    major_radius: float
    minor_radius: float
    elongation: float
    triangularity: float

    def __post_init__(self) -> None:
        values = (
            self.major_radius,
            self.minor_radius,
            self.elongation,
            self.triangularity,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("Zheng shape parameters must be finite")
        if self.major_radius <= self.minor_radius or self.minor_radius <= 0.0:
            raise ValueError("Zheng shape requires R0 > a > 0")
        if self.elongation <= 0.0:
            raise ValueError("Zheng elongation must be positive")
        if abs(self.triangularity) >= 1.0:
            raise ValueError("Zheng triangularity magnitude must be below one")

    @property
    def inner_radius(self) -> float:
        """Return ``R_i = R0-a``."""
        return self.major_radius - self.minor_radius

    @property
    def outer_radius(self) -> float:
        """Return ``R_o = R0+a``."""
        return self.major_radius + self.minor_radius

    @property
    def top_radius(self) -> float:
        """Return ``R_t = R0-delta*a``."""
        return self.major_radius - self.triangularity * self.minor_radius

    @property
    def top_height(self) -> float:
        """Return ``Z_t = kappa*a``."""
        return self.elongation * self.minor_radius


@dataclass(frozen=True, slots=True)
class ZhengFigureOfMeritIntegrals:
    """Zheng Eqs. (19)--(20) evaluated over the analytic plasma contour."""

    plasma_current: float
    poloidal_beta: float
    flux_integral: float
    poloidal_area: float
    toroidal_volume: float
    poloidal_perimeter: float


@dataclass(frozen=True, slots=True)
class ZhengEquilibrium:
    r"""Zheng's exact solution of note ``GS_recovered`` (paper Eq. 14).

    ``Psi = c1+c2 R^2+c3(R^4-4R^2Z^2)+c4(R^2 log R-Z^2)
            + A1 R^4/8-A2 Z^2/2`` and ``Delta*Psi=A1 R^2-A2``.
    """

    shape: ZhengShape
    c1: float
    c2: float
    c3: float
    c4: float
    a1: float
    a2: float

    @property
    def _terms(self) -> tuple[_Term, ...]:
        return (
            (self.c1, 0, 0, False),
            (self.c2, 2, 0, False),
            (self.c3, 4, 0, False),
            (-4.0 * self.c3, 2, 2, False),
            (self.c4, 2, 0, True),
            (-self.c4, 0, 2, False),
            (self.a1 / 8.0, 4, 0, False),
            (-self.a2 / 2.0, 0, 2, False),
        )

    def flux(self, radius: Any, height: Any) -> Any:
        """Evaluate Zheng Eq. (14) for arrays or NGSolve coefficient functions."""
        return _evaluate_terms(self._terms, radius, height)

    def radial_derivative(self, radius: Any, height: Any) -> Any:
        """Evaluate ``partial Psi/partial R`` from Zheng Eq. (14)."""
        return _evaluate_terms(self._terms, radius, height, radial_derivative=1)

    def vertical_derivative(self, radius: Any, height: Any) -> Any:
        """Evaluate ``partial Psi/partial Z`` from Zheng Eq. (14)."""
        return _evaluate_terms(self._terms, radius, height, vertical_derivative=1)

    def radial_second_derivative(self, radius: Any, height: Any) -> Any:
        """Evaluate ``partial^2 Psi/partial R^2`` from Zheng Eq. (14)."""
        return _evaluate_terms(self._terms, radius, height, radial_derivative=2)

    def vertical_second_derivative(self, radius: Any, height: Any) -> Any:
        """Evaluate ``partial^2 Psi/partial Z^2`` from Zheng Eq. (14)."""
        return _evaluate_terms(self._terms, radius, height, vertical_derivative=2)

    def delta_star(self, radius: Any, height: Any) -> Any:
        r"""Evaluate ``Delta*Psi=Psi_RR-Psi_R/R+Psi_ZZ=A1 R^2-A2``."""
        return (
            self.radial_second_derivative(radius, height)
            - self.radial_derivative(radius, height) / radius
            + self.vertical_second_derivative(radius, height)
        )

    def _midplane_flux(self, radius: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(self.flux(radius, 0.0), dtype=float)

    def _vertical_coefficient(self, radius: NDArray[np.float64]) -> NDArray[np.float64]:
        return -4.0 * self.c3 * radius**2 - self.c4 - self.a2 / 2.0

    def boundary_height(self, radius: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the positive height of the exact ``Psi=0`` contour at ``R``."""
        squared = -self._midplane_flux(radius) / self._vertical_coefficient(radius)
        if np.min(squared) < -1.0e-10:
            raise ValueError("the requested Zheng coefficients do not form a closed contour")
        return np.sqrt(np.maximum(squared, 0.0))

    def _minimum_boundary_height_squared(self, quadrature_nodes: int = 240) -> float:
        radius, _ = _zheng_quadrature(self.shape, quadrature_nodes)
        return float(np.min(-self._midplane_flux(radius) / self._vertical_coefficient(radius)))

    def magnetic_axis(self) -> tuple[float, float]:
        """Recover the midplane magnetic-axis radius and flux from analytic ``Psi``."""
        radius = np.linspace(self.shape.inner_radius, self.shape.outer_radius, 20001)[1:-1]
        values = self._midplane_flux(radius)
        return _refined_absolute_extremum(radius, values)

    def boundary_contour(self, *, samples: int = 257) -> FluxContour:
        """Extract the exact ``Psi=0`` boundary in counter-clockwise order."""
        if samples < 32:
            raise ValueError("at least 32 samples are required for a shaped contour")
        upper_count = samples // 2
        lower_count = samples - upper_count
        upper_radius = np.linspace(
            self.shape.outer_radius, self.shape.inner_radius, upper_count, endpoint=False
        )
        lower_radius = np.linspace(
            self.shape.inner_radius, self.shape.outer_radius, lower_count, endpoint=False
        )
        radius = np.concatenate((upper_radius, lower_radius))
        height = np.concatenate(
            (self.boundary_height(upper_radius), -self.boundary_height(lower_radius))
        )

        parameterizations: list[Callable[[float], tuple[float, float]]] = []
        for start in np.linspace(0.0, 2.0 * pi, 17)[:-1]:

            def parameterization(
                parameter: float,
                start_angle: float = float(start),
            ) -> tuple[float, float]:
                angle = start_angle + min(max(float(parameter), 0.0), 1.0) * (pi / 8.0)
                cylindrical_radius = self.shape.major_radius + self.shape.minor_radius * np.cos(
                    angle
                )
                boundary_height = float(self.boundary_height(np.asarray([cylindrical_radius]))[0])
                if abs(np.sin(angle)) < 1.0e-14:
                    boundary_height = 0.0
                return cylindrical_radius, float(np.copysign(boundary_height, np.sin(angle)))

            parameterizations.append(parameterization)
        return FluxContour(radius, height, parameterizations=tuple(parameterizations))

    def figure_of_merit_integrals(
        self, *, quadrature_nodes: int = 320
    ) -> ZhengFigureOfMeritIntegrals:
        """Evaluate Zheng Eqs. (19)--(20), area, toroidal volume, and perimeter."""
        radius, weights = _zheng_quadrature(self.shape, quadrature_nodes)
        height = self.boundary_height(radius)
        vertical_coefficient = self._vertical_coefficient(radius)
        current_integrand = -2.0 * height * (radius**2 * self.a1 - self.a2) / (_MU0 * radius)
        plasma_current = float(np.sum(weights * current_integrand))
        flux_integrand = -(4.0 / 3.0) * vertical_coefficient * height**3
        flux_integral = float(np.sum(weights * flux_integrand))
        poloidal_beta = float(-8.0 * pi * self.a1 * flux_integral / (_MU0 * plasma_current) ** 2)
        poloidal_area = float(np.sum(weights * 2.0 * height))
        toroidal_volume = float(np.sum(weights * 4.0 * pi * radius * height))
        contour = self.boundary_contour(samples=max(1024, 4 * quadrature_nodes))
        poloidal_perimeter = _polygon_perimeter(contour)
        return ZhengFigureOfMeritIntegrals(
            plasma_current,
            poloidal_beta,
            flux_integral,
            poloidal_area,
            toroidal_volume,
            poloidal_perimeter,
        )


def _zheng_quadrature(
    shape: ZhengShape, count: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Gauss nodes under ``R=R0+a cos(theta)`` to regularize contour endpoints."""
    nodes, weights = leggauss(count)
    angle = 0.5 * pi * (nodes + 1.0)
    radius = shape.major_radius + shape.minor_radius * np.cos(angle)
    transformed_weights = 0.5 * pi * weights * shape.minor_radius * np.sin(angle)
    return radius, transformed_weights


def _zheng_shape_coefficients(shape: ZhengShape, a1: float, a2: float) -> NDArray[np.float64]:
    """Solve Zheng Eqs. (15)--(18) for ``c1..c4`` at fixed ``A1,A2``."""
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
            -a1 * inner**4 / 8.0,
            -a1 * outer**4 / 8.0,
            -a1 * top_radius**4 / 8.0 + a2 * top_height**2 / 2.0,
            -a1 * top_radius**2 / 2.0,
        )
    )
    return np.linalg.solve(matrix, rhs)


def _zheng_at(shape: ZhengShape, a1: float, a2: float) -> ZhengEquilibrium:
    coefficients = _zheng_shape_coefficients(shape, a1, a2)
    return ZhengEquilibrium(
        shape,
        float(coefficients[0]),
        float(coefficients[1]),
        float(coefficients[2]),
        float(coefficients[3]),
        a1,
        a2,
    )


def solve_zheng_equilibrium(
    *,
    shape: ZhengShape,
    poloidal_beta: float,
    plasma_current: float,
    bisection_steps: int = 120,
) -> ZhengEquilibrium:
    """Solve Zheng Eqs. (15)--(20) for the six analytic coefficients."""
    if not isfinite(poloidal_beta) or poloidal_beta <= 0.0:
        raise ValueError("poloidal_beta must be finite and positive")
    if not isfinite(plasma_current) or plasma_current <= 0.0:
        raise ValueError("plasma_current must be finite and positive")

    def residual(a2: float) -> float:
        candidate = _zheng_at(shape, -1.0, a2)
        if candidate._minimum_boundary_height_squared() < -1.0e-12:
            return float("nan")
        return (
            candidate.figure_of_merit_integrals(quadrature_nodes=160).poloidal_beta - poloidal_beta
        )

    candidates = np.geomspace(1.0e-4, 1.0e3, 200)
    values = np.asarray([residual(float(candidate)) for candidate in candidates])
    finite_indices = np.flatnonzero(np.isfinite(values))
    changes = np.flatnonzero(
        np.sign(values[finite_indices][:-1]) * np.sign(values[finite_indices][1:]) < 0.0
    )
    if changes.size == 0:
        raise ValueError("the requested Zheng beta has no closed-boundary coefficient root")
    index = int(changes[0])
    lower = float(candidates[finite_indices[index]])
    upper = float(candidates[finite_indices[index + 1]])
    lower_value = residual(lower)
    for _ in range(bisection_steps):
        middle = 0.5 * (lower + upper)
        middle_value = residual(middle)
        if lower_value * middle_value <= 0.0:
            upper = middle
        else:
            lower, lower_value = middle, middle_value
    unit = _zheng_at(shape, -1.0, 0.5 * (lower + upper))
    scale = plasma_current / unit.figure_of_merit_integrals().plasma_current
    return _zheng_at(shape, -scale, unit.a2 * scale)


class CerfonFreidbergBoundary(str, Enum):
    """The symmetric smooth and double-null constraint sets in paper Eqs. (10),(12)."""

    SMOOTH = "smooth"
    DOUBLE_NULL = "double_null"


@dataclass(frozen=True, slots=True)
class CerfonFreidbergShape:
    """Dimensionless shape inputs ``epsilon,kappa,delta`` in paper Eq. (9)."""

    inverse_aspect_ratio: float
    elongation: float
    triangularity: float

    def __post_init__(self) -> None:
        values = (self.inverse_aspect_ratio, self.elongation, self.triangularity)
        if not all(isfinite(value) for value in values):
            raise ValueError("Cerfon--Freidberg shape parameters must be finite")
        if not 0.0 < self.inverse_aspect_ratio < 1.0:
            raise ValueError("inverse_aspect_ratio must lie strictly between zero and one")
        if self.elongation <= 0.0:
            raise ValueError("elongation must be positive")
        if abs(self.triangularity) >= np.sin(1.0):
            raise ValueError("smooth-reference triangularity must satisfy |delta| < sin(1)")

    @property
    def inner_radius(self) -> float:
        return 1.0 - self.inverse_aspect_ratio

    @property
    def outer_radius(self) -> float:
        return 1.0 + self.inverse_aspect_ratio

    @property
    def top_point(self) -> tuple[float, float]:
        epsilon = self.inverse_aspect_ratio
        return 1.0 - self.triangularity * epsilon, self.elongation * epsilon

    @property
    def double_null_point(self) -> tuple[float, float]:
        epsilon = self.inverse_aspect_ratio
        return (
            1.0 - 1.1 * self.triangularity * epsilon,
            1.1 * self.elongation * epsilon,
        )


_CERFON_BASES: tuple[tuple[_Term, ...], ...] = (
    ((1.0, 0, 0, False),),
    ((1.0, 2, 0, False),),
    ((1.0, 0, 2, False), (-1.0, 2, 0, True)),
    ((1.0, 4, 0, False), (-4.0, 2, 2, False)),
    (
        (2.0, 0, 4, False),
        (-9.0, 2, 2, False),
        (3.0, 4, 0, True),
        (-12.0, 2, 2, True),
    ),
    ((1.0, 6, 0, False), (-12.0, 4, 2, False), (8.0, 2, 4, False)),
    (
        (8.0, 0, 6, False),
        (-140.0, 2, 4, False),
        (75.0, 4, 2, False),
        (-15.0, 6, 0, True),
        (180.0, 4, 2, True),
        (-120.0, 2, 4, True),
    ),
)


@dataclass(frozen=True, slots=True)
class CerfonFreidbergFigureOfMeritIntegrals:
    """Raw Eq. (19)--(20) integrals and the resulting poloidal beta."""

    normalized_volume: float
    normalized_perimeter: float
    flux_volume_integral: float
    source_current_integral: float
    poloidal_beta: float

    def beta_values(self, *, q_star: float, inverse_aspect_ratio: float) -> tuple[float, float]:
        """Return ``(beta_t,beta)`` from paper Eq. (19) for a supplied ``q*``."""
        if not isfinite(q_star) or q_star <= 0.0:
            raise ValueError("q_star must be finite and positive")
        beta_t = inverse_aspect_ratio**2 * self.poloidal_beta / q_star**2
        beta = inverse_aspect_ratio**2 * self.poloidal_beta / (q_star**2 + inverse_aspect_ratio**2)
        return beta_t, beta


@dataclass(frozen=True, slots=True)
class CerfonFreidbergEquilibrium:
    r"""Cerfon--Freidberg Eq. (8), an exact solution of note ``GS_recovered``.

    ``Delta*psi=(1-A)R^2+A`` with the seven homogeneous coefficients fixed by
    either the smooth Eq. (10) or double-null Eq. (12) boundary constraints.
    """

    shape: CerfonFreidbergShape
    source_parameter: float
    boundary: CerfonFreidbergBoundary
    coefficients: tuple[float, float, float, float, float, float, float]

    @property
    def _particular_terms(self) -> tuple[_Term, ...]:
        parameter = self.source_parameter
        return (
            ((1.0 - parameter) / 8.0, 4, 0, False),
            (parameter / 2.0, 2, 0, True),
        )

    @property
    def _terms(self) -> tuple[_Term, ...]:
        terms = list(self._particular_terms)
        for coefficient, basis in zip(self.coefficients, _CERFON_BASES):
            terms.extend(
                (coefficient * value, radial, vertical, logarithmic)
                for value, radial, vertical, logarithmic in basis
            )
        return tuple(terms)

    @property
    def upper_xpoint(self) -> tuple[float, float] | None:
        """Return the upper X-point for a double-null equilibrium."""
        return (
            self.shape.double_null_point
            if self.boundary is CerfonFreidbergBoundary.DOUBLE_NULL
            else None
        )

    def flux(self, radius: Any, height: Any) -> Any:
        """Evaluate Cerfon--Freidberg Eq. (8)."""
        return _evaluate_terms(self._terms, radius, height)

    def radial_derivative(self, radius: Any, height: Any) -> Any:
        """Evaluate ``partial psi/partial R`` of paper Eq. (8)."""
        return _evaluate_terms(self._terms, radius, height, radial_derivative=1)

    def vertical_derivative(self, radius: Any, height: Any) -> Any:
        """Evaluate ``partial psi/partial Z`` of paper Eq. (8)."""
        return _evaluate_terms(self._terms, radius, height, vertical_derivative=1)

    def radial_second_derivative(self, radius: Any, height: Any) -> Any:
        return _evaluate_terms(self._terms, radius, height, radial_derivative=2)

    def vertical_second_derivative(self, radius: Any, height: Any) -> Any:
        return _evaluate_terms(self._terms, radius, height, vertical_derivative=2)

    def delta_star(self, radius: Any, height: Any) -> Any:
        r"""Evaluate ``Delta*psi=psi_RR-psi_R/R+psi_ZZ=(1-A)R^2+A``."""
        return (
            self.radial_second_derivative(radius, height)
            - self.radial_derivative(radius, height) / radius
            + self.vertical_second_derivative(radius, height)
        )

    def magnetic_axis(self) -> tuple[float, float]:
        """Recover the midplane magnetic-axis radius and flux from analytic ``psi``."""
        radius = np.linspace(self.shape.inner_radius, self.shape.outer_radius, 20001)[1:-1]
        values = np.asarray(self.flux(radius, 0.0), dtype=float)
        return _refined_absolute_extremum(radius, values)

    def _constraint_residuals(self) -> NDArray[np.float64]:
        return _cerfon_constraint_vector(
            self._terms,
            self.shape,
            self.boundary,
        )

    def maximum_constraint_residual(self) -> float:
        """Return the largest absolute residual in paper Eq. (10) or Eq. (12)."""
        return float(np.max(np.abs(self._constraint_residuals())))

    def _boundary_radius(self, angle: float) -> float:
        """Find the first ``psi=0`` ray intersection outwards from the magnetic axis."""
        axis_radius, axis_flux = self.magnetic_axis()
        cosine, sine = np.cos(angle), np.sin(angle)
        if self.upper_xpoint is not None:
            upper = self.upper_xpoint
            for xpoint in (upper, (upper[0], -upper[1])):
                x_angle = float(np.mod(np.arctan2(xpoint[1], xpoint[0] - axis_radius), 2.0 * pi))
                angle_distance = abs(float(np.angle(np.exp(1j * (angle - x_angle)))))
                if angle_distance < 2.0e-12:
                    return float(np.hypot(xpoint[0] - axis_radius, xpoint[1]))
        radial_extent = 4.0 * max(
            self.shape.inverse_aspect_ratio,
            self.shape.elongation * self.shape.inverse_aspect_ratio,
        )
        if cosine < 0.0:
            radial_extent = min(radial_extent, 0.999999 * axis_radius / -cosine)
        ray = np.linspace(0.0, radial_extent, 801)
        values = np.asarray(
            self.flux(axis_radius + ray * cosine, ray * sine),
            dtype=float,
        )
        sign = 1.0 if axis_flux >= 0.0 else -1.0
        crossing = np.flatnonzero(sign * values[1:] <= 0.0)
        if crossing.size == 0:
            candidate = int(np.argmin(np.abs(values[1:]))) + 1
            if abs(values[candidate]) > 1.0e-7 * max(1.0, abs(axis_flux)):
                raise ValueError(f"failed to close Cerfon--Freidberg contour at angle {angle}")
            return float(ray[candidate])
        upper_index = int(crossing[0]) + 1
        lower_radius, upper_radius = float(ray[upper_index - 1]), float(ray[upper_index])
        lower_value = float(values[upper_index - 1])
        for _ in range(70):
            middle = 0.5 * (lower_radius + upper_radius)
            middle_value = float(self.flux(axis_radius + middle * cosine, middle * sine))
            if lower_value * middle_value <= 0.0:
                upper_radius = middle
            else:
                lower_radius, lower_value = middle, middle_value
        return 0.5 * (lower_radius + upper_radius)

    def boundary_contour(self, *, samples: int = 321) -> FluxContour:
        """Extract the paper's smooth or double-null ``psi=0`` boundary."""
        if samples < 64:
            raise ValueError("at least 64 samples are required for a Cerfon--Freidberg contour")
        axis_radius, _ = self.magnetic_axis()
        angles = np.linspace(0.0, 2.0 * pi, samples, endpoint=False)
        corner_angles: list[float] = []
        if self.upper_xpoint is not None:
            upper = self.upper_xpoint
            corner_angles.extend(
                (
                    float(np.mod(np.arctan2(upper[1], upper[0] - axis_radius), 2.0 * pi)),
                    float(np.mod(np.arctan2(-upper[1], upper[0] - axis_radius), 2.0 * pi)),
                )
            )
            angles = np.unique(np.concatenate((angles, np.asarray(corner_angles))))
        ray = np.asarray([self._boundary_radius(float(angle)) for angle in angles])
        radius = axis_radius + ray * np.cos(angles)
        height = ray * np.sin(angles)
        corner_indices = tuple(
            int(np.argmin(np.abs(angles - corner_angle))) for corner_angle in corner_angles
        )
        if corner_angles:
            return FluxContour(radius, height, corner_indices)
        parameterizations: list[Callable[[float], tuple[float, float]]] = []
        intervals = [(0.0, 2.0 * pi)]
        subdivided_intervals: list[tuple[float, float]] = []
        for start, stop in intervals:
            edges = np.linspace(start, stop, 17)
            subdivided_intervals.extend(
                (float(left), float(right)) for left, right in pairwise(edges)
            )
        for start, stop in subdivided_intervals:

            def parameterization(
                parameter: float,
                start_angle: float = start,
                stop_angle: float = stop,
            ) -> tuple[float, float]:
                angle = start_angle + min(max(float(parameter), 0.0), 1.0) * (
                    stop_angle - start_angle
                )
                ray_distance = self._boundary_radius(float(np.mod(angle, 2.0 * pi)))
                return (
                    axis_radius + ray_distance * float(np.cos(angle)),
                    ray_distance * float(np.sin(angle)),
                )

            parameterizations.append(parameterization)
        return FluxContour(radius, height, corner_indices, tuple(parameterizations))

    def figure_of_merit_integrals(
        self,
        *,
        radial_order: int = 24,
        angular_order: int = 96,
    ) -> CerfonFreidbergFigureOfMeritIntegrals:
        """Evaluate the normalized integrals and poloidal beta in paper Eq. (19)."""
        if radial_order < 4 or angular_order < 16:
            raise ValueError("figure-of-merit quadrature orders are too small")
        radial_nodes, radial_weights = leggauss(radial_order)
        angular_nodes, angular_weights = leggauss(angular_order)
        angles = pi * (angular_nodes + 1.0)
        angular_weights = pi * angular_weights
        axis_radius, _ = self.magnetic_axis()
        normalized_volume = 0.0
        flux_volume_integral = 0.0
        source_current_integral = 0.0
        for angle, angle_weight in zip(angles, angular_weights):
            boundary_radius = self._boundary_radius(float(angle))
            ray = 0.5 * boundary_radius * (radial_nodes + 1.0)
            ray_weights = 0.5 * boundary_radius * radial_weights
            radius = axis_radius + ray * np.cos(angle)
            height = ray * np.sin(angle)
            jacobian = ray * ray_weights * angle_weight
            flux = np.asarray(self.flux(radius, height), dtype=float)
            source = (1.0 - self.source_parameter) * radius**2 + self.source_parameter
            normalized_volume += float(np.sum(radius * jacobian))
            flux_volume_integral += float(np.sum(flux * radius * jacobian))
            source_current_integral += float(np.sum(source / radius * jacobian))
        contour = self.boundary_contour(samples=max(768, 8 * angular_order))
        perimeter = _polygon_perimeter(contour)
        poloidal_beta = float(
            -2.0
            * (1.0 - self.source_parameter)
            * perimeter**2
            * flux_volume_integral
            / (normalized_volume * source_current_integral**2)
        )
        return CerfonFreidbergFigureOfMeritIntegrals(
            normalized_volume,
            perimeter,
            flux_volume_integral,
            source_current_integral,
            poloidal_beta,
        )


def _constraint_value(
    terms: Sequence[_Term],
    radius: float,
    height: float,
    radial_derivative: int = 0,
    vertical_derivative: int = 0,
) -> float:
    return float(
        _evaluate_terms(
            terms,
            radius,
            height,
            radial_derivative=radial_derivative,
            vertical_derivative=vertical_derivative,
        )
    )


def _cerfon_constraint_functions(
    shape: CerfonFreidbergShape,
    boundary: CerfonFreidbergBoundary,
) -> tuple[Callable[[Sequence[_Term]], float], ...]:
    epsilon = shape.inverse_aspect_ratio
    inner, outer = shape.inner_radius, shape.outer_radius
    top_radius, top_height = shape.top_point
    alpha = asin(shape.triangularity)
    n1 = -((1.0 + alpha) ** 2) / (epsilon * shape.elongation**2)
    n2 = (1.0 - alpha) ** 2 / (epsilon * shape.elongation**2)
    n3 = -shape.elongation / (epsilon * cos(alpha) ** 2)

    def value(radius: float, height: float) -> Callable[[Sequence[_Term]], float]:
        return lambda terms: _constraint_value(terms, radius, height)

    def derivative(
        radius: float, height: float, dr: int, dz: int
    ) -> Callable[[Sequence[_Term]], float]:
        return lambda terms: _constraint_value(terms, radius, height, dr, dz)

    def outer_curvature(terms: Sequence[_Term]) -> float:
        return _constraint_value(terms, outer, 0.0, 0, 2) + n1 * _constraint_value(
            terms, outer, 0.0, 1, 0
        )

    def inner_curvature(terms: Sequence[_Term]) -> float:
        return _constraint_value(terms, inner, 0.0, 0, 2) + n2 * _constraint_value(
            terms, inner, 0.0, 1, 0
        )

    if boundary is CerfonFreidbergBoundary.SMOOTH:

        def top_curvature(terms: Sequence[_Term]) -> float:
            return _constraint_value(terms, top_radius, top_height, 2, 0) + n3 * _constraint_value(
                terms, top_radius, top_height, 0, 1
            )

        return (
            value(outer, 0.0),
            value(inner, 0.0),
            value(top_radius, top_height),
            derivative(top_radius, top_height, 1, 0),
            outer_curvature,
            inner_curvature,
            top_curvature,
        )
    xpoint_radius, xpoint_height = shape.double_null_point
    return (
        value(outer, 0.0),
        value(inner, 0.0),
        value(xpoint_radius, xpoint_height),
        derivative(xpoint_radius, xpoint_height, 1, 0),
        derivative(xpoint_radius, xpoint_height, 0, 1),
        outer_curvature,
        inner_curvature,
    )


def _cerfon_constraint_vector(
    terms: Sequence[_Term],
    shape: CerfonFreidbergShape,
    boundary: CerfonFreidbergBoundary,
) -> NDArray[np.float64]:
    functions = _cerfon_constraint_functions(shape, boundary)
    return np.asarray([function(terms) for function in functions])


def solve_cerfon_freidberg(
    *,
    shape: CerfonFreidbergShape,
    source_parameter: float,
    boundary: CerfonFreidbergBoundary = CerfonFreidbergBoundary.SMOOTH,
) -> CerfonFreidbergEquilibrium:
    """Solve paper Eq. (10) or Eq. (12) for the seven Eq. (8) coefficients."""
    if not isfinite(source_parameter):
        raise ValueError("source_parameter must be finite")
    boundary = CerfonFreidbergBoundary(boundary)
    particular: tuple[_Term, ...] = (
        ((1.0 - source_parameter) / 8.0, 4, 0, False),
        (source_parameter / 2.0, 2, 0, True),
    )
    functions = _cerfon_constraint_functions(shape, boundary)
    matrix = np.asarray(
        [[function(basis) for basis in _CERFON_BASES] for function in functions],
        dtype=float,
    )
    rhs = -np.asarray([function(particular) for function in functions], dtype=float)
    coefficients = np.linalg.solve(matrix, rhs)
    return CerfonFreidbergEquilibrium(
        shape,
        source_parameter,
        boundary,
        tuple(float(value) for value in coefficients),  # type: ignore[arg-type]
    )


@dataclass(frozen=True, slots=True)
class SmoothFluxObservables:
    """Axis and boundary-shape measurements recovered from a computed ``Psi_h``."""

    axis_radius: float
    axis_flux: float
    major_radius: float
    minor_radius: float
    elongation: float
    triangularity: float
    boundary_flux_rms: float


def recover_smooth_flux_observables(
    *,
    mesh: Any,
    flux: Any,
    search_contour: FluxContour,
    axis_samples: int = 2001,
) -> SmoothFluxObservables:
    """Recover axis/shape observables from the computed homogeneous-Dirichlet ``Psi_h``.

    Boundary coordinates come from the computed mesh, and only vertices at which
    the supplied finite-element flux is its homogeneous boundary value are used.
    The axis is independently recovered from the finite-element midplane field.
    """
    import ngsolve as ng

    boundary_nodes = {vertex.nr for element in mesh.Elements(ng.BND) for vertex in element.vertices}
    vertex_coordinates = np.asarray(
        [tuple(mesh[ng.NodeId(ng.VERTEX, node)].point) for node in boundary_nodes]
    )
    boundary_rule = ng.IntegrationRule(ng.ET.SEGM, 20)
    mapped_boundary = mesh.MapToAllElements({ng.ET.SEGM: boundary_rule}, ng.BND)
    mapped_coordinates = np.column_stack(
        (
            np.asarray(ng.x(mapped_boundary), dtype=float).ravel(),
            np.asarray(ng.y(mapped_boundary), dtype=float).ravel(),
        )
    )
    coordinates = np.vstack((vertex_coordinates, mapped_coordinates))
    radius = coordinates[:, 0]
    height = coordinates[:, 1]
    boundary_values = np.concatenate(
        (
            np.asarray([float(flux(mesh(float(r), float(z)))) for r, z in vertex_coordinates]),
            np.asarray(flux(mapped_boundary), dtype=float).ravel(),
        )
    )
    scale = max(1.0e-300, float(np.max(np.abs(boundary_values))))
    boundary_flux_rms = float(np.sqrt(np.mean(boundary_values**2)))
    if not np.all(np.isfinite(boundary_values)) or boundary_flux_rms > 1.0e-8 * max(1.0, scale):
        raise ValueError("computed flux is not homogeneous on the supplied shaped boundary")

    inner, outer = float(np.min(radius)), float(np.max(radius))
    major_radius = 0.5 * (inner + outer)
    minor_radius = 0.5 * (outer - inner)
    top_index = int(np.argmax(height))
    top_radius = float(radius[top_index])
    top_height = float(height[top_index])

    contour_inner, contour_outer = search_contour.radial_bounds
    margin = 0.03 * (contour_outer - contour_inner)
    midplane_radius = np.linspace(
        contour_inner + margin,
        contour_outer - margin,
        axis_samples,
    )
    midplane_values = np.asarray(
        [float(flux(mesh(float(value), 0.0))) for value in midplane_radius]
    )
    axis_radius, axis_flux = _refined_absolute_extremum(midplane_radius, midplane_values)
    return SmoothFluxObservables(
        axis_radius,
        axis_flux,
        major_radius,
        minor_radius,
        top_height / minor_radius,
        (major_radius - top_radius) / minor_radius,
        boundary_flux_rms,
    )


def _refined_absolute_extremum(
    abscissa: NDArray[np.float64], values: NDArray[np.float64]
) -> tuple[float, float]:
    """Refine the largest-magnitude sampled extremum by one quadratic step."""
    index = int(np.argmax(np.abs(values)))
    if index in (0, values.size - 1):
        return float(abscissa[index]), float(values[index])
    left, centre, right = values[index - 1 : index + 2]
    step = abscissa[index] - abscissa[index - 1]
    denominator = left - 2.0 * centre + right
    if denominator == 0.0:
        return float(abscissa[index]), float(centre)
    offset = 0.5 * (left - right) / denominator
    return (
        float(abscissa[index] + offset * step),
        float(centre - 0.25 * (left - right) * offset),
    )


def _polygon_perimeter(contour: FluxContour) -> float:
    radius_difference = contour.radius - np.roll(contour.radius, -1)
    height_difference = contour.height - np.roll(contour.height, -1)
    return float(np.sum(np.hypot(radius_difference, height_difference)))


__all__ = [
    "CerfonFreidbergBoundary",
    "CerfonFreidbergEquilibrium",
    "CerfonFreidbergFigureOfMeritIntegrals",
    "CerfonFreidbergShape",
    "FluxContour",
    "SmoothFluxObservables",
    "ZhengEquilibrium",
    "ZhengFigureOfMeritIntegrals",
    "ZhengShape",
    "recover_smooth_flux_observables",
    "solve_cerfon_freidberg",
    "solve_zheng_equilibrium",
]
