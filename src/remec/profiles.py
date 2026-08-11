"""Volume profiles and the note equation ``(M4b)`` pressure transplant."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from remec.level_set import MollifiedVolumeMap, QuadratureLevelSetData

if TYPE_CHECKING:
    import ngsolve as ng  # type: ignore[import-untyped]


class InvalidProfileError(ValueError):
    """A prescribed pressure-versus-volume profile is inadmissible."""


class VolumeProfile(Protocol):
    """One-dimensional ``p_0(V)`` profile required by DESIGN.md §12.5."""

    def value(self, volume: float | ArrayLike) -> float | NDArray[np.float64]: ...

    def derivative(self, volume: float | ArrayLike) -> float | NDArray[np.float64]: ...

    def validate(self, total_volume: float, edge_value: float | None = None) -> None: ...


def _as_array(value: float | ArrayLike) -> tuple[NDArray[np.float64], bool]:
    points = np.asarray(value, dtype=float)
    return points, points.ndim == 0


def _scalar_or_array(values: NDArray[np.float64], scalar: bool) -> float | NDArray[np.float64]:
    return float(values) if scalar else values


@dataclass(frozen=True, slots=True)
class AnalyticVolumeProfile:
    """Analytic implementation of the §12.5 ``VolumeProfile`` protocol."""

    value_function: Callable[[float | NDArray[np.float64]], float | NDArray[np.float64]]
    derivative_function: Callable[[float | NDArray[np.float64]], float | NDArray[np.float64]]

    def value(self, volume: float | ArrayLike) -> float | NDArray[np.float64]:
        """Evaluate the prescribed pressure without silently clipping its volume input."""
        points, scalar = _as_array(volume)
        values = np.asarray(self.value_function(points), dtype=float)
        if values.shape != points.shape or not np.all(np.isfinite(values)):
            raise InvalidProfileError("profile values must be finite and match the volume shape")
        return _scalar_or_array(values, scalar)

    def derivative(self, volume: float | ArrayLike) -> float | NDArray[np.float64]:
        """Evaluate ``dp_0/dV`` used by the local (M4b) composition diagnostic."""
        points, scalar = _as_array(volume)
        values = np.asarray(self.derivative_function(points), dtype=float)
        if values.shape != points.shape or not np.all(np.isfinite(values)):
            raise InvalidProfileError(
                "profile derivatives must be finite and match the volume shape"
            )
        return _scalar_or_array(values, scalar)

    def validate(self, total_volume: float, edge_value: float | None = None) -> None:
        """Check §12.5 monotonicity and the optional physical edge pressure."""
        _validate_profile(self, total_volume, edge_value)


@dataclass(frozen=True, slots=True)
class TabulatedVolumeProfile:
    """Piecewise-linear, non-increasing tabulated profile (plateaus included)."""

    volumes: Sequence[float]
    pressures: Sequence[float]

    def __post_init__(self) -> None:
        volumes = np.asarray(self.volumes, dtype=float)
        pressures = np.asarray(self.pressures, dtype=float)
        if volumes.ndim != 1 or pressures.ndim != 1 or len(volumes) < 2:
            raise InvalidProfileError(
                "tabulated profile needs at least two one-dimensional samples"
            )
        if (
            len(volumes) != len(pressures)
            or not np.all(np.isfinite(volumes))
            or not np.all(np.isfinite(pressures))
        ):
            raise InvalidProfileError("tabulated profile samples must be finite and equally sized")
        if np.any(np.diff(volumes) <= 0.0):
            raise InvalidProfileError("tabulated profile volumes must be strictly increasing")
        if np.any(np.diff(pressures) > 0.0):
            raise InvalidProfileError("tabulated profile pressures must be non-increasing")
        object.__setattr__(self, "volumes", tuple(float(item) for item in volumes))
        object.__setattr__(self, "pressures", tuple(float(item) for item in pressures))

    def _check_range(self, points: NDArray[np.float64]) -> None:
        if np.any(points < self.volumes[0]) or np.any(points > self.volumes[-1]):
            raise InvalidProfileError("profile evaluated outside its tabulated volume interval")

    def value(self, volume: float | ArrayLike) -> float | NDArray[np.float64]:
        """Evaluate the monotone table, including an intentional edge plateau."""
        points, scalar = _as_array(volume)
        self._check_range(points)
        values = np.interp(points, self.volumes, self.pressures)
        return _scalar_or_array(values, scalar)

    def derivative(self, volume: float | ArrayLike) -> float | NDArray[np.float64]:
        """Return the one-sided piecewise-linear derivative of the profile table."""
        points, scalar = _as_array(volume)
        self._check_range(points)
        volumes = np.asarray(self.volumes)
        pressures = np.asarray(self.pressures)
        index = np.clip(np.searchsorted(volumes, points, side="right") - 1, 0, len(volumes) - 2)
        values = np.diff(pressures)[index] / np.diff(volumes)[index]
        return _scalar_or_array(np.asarray(values, dtype=float), scalar)

    def validate(self, total_volume: float, edge_value: float | None = None) -> None:
        """Require the table to cover exactly the map's enclosed-volume interval."""
        if not np.isclose(self.volumes[0], 0.0) or not np.isclose(self.volumes[-1], total_volume):
            raise InvalidProfileError("tabulated profile interval must be [0, total_volume]")
        if edge_value is not None and not np.isclose(self.pressures[-1], edge_value):
            raise InvalidProfileError("profile edge value differs from the required edge value")


def _validate_profile(
    profile: VolumeProfile, total_volume: float, edge_value: float | None
) -> None:
    if not isfinite(total_volume) or total_volume <= 0.0:
        raise InvalidProfileError("total volume must be finite and positive")
    probe = np.linspace(0.0, total_volume, 257)
    values = np.asarray(profile.value(probe), dtype=float)
    derivatives = np.asarray(profile.derivative(probe), dtype=float)
    if values.shape != probe.shape or derivatives.shape != probe.shape:
        raise InvalidProfileError("profile evaluation must preserve array shape")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(derivatives)):
        raise InvalidProfileError("profile values and derivatives must be finite")
    if np.any(np.diff(values) > 32.0 * np.finfo(float).eps * max(1.0, np.max(np.abs(values)))):
        raise InvalidProfileError("profile pressures must be non-increasing")
    if np.any(derivatives > 1.0e-12 * max(1.0, np.max(np.abs(derivatives)))):
        raise InvalidProfileError("profile derivative must be non-positive")
    if edge_value is not None and not np.isclose(values[-1], edge_value):
        raise InvalidProfileError("profile edge value differs from the required edge value")


@dataclass(frozen=True, slots=True)
class TransplantedProfile:
    """Note equation (M4b): ``p(r) = p_0(V_chi(chi(r)))``.

    Both factors are monotone decreasing, therefore the composed pressure is a
    monotone increasing function of ``chi`` and has exactly the prescribed volume
    distribution (up to the quadrature and interpolation of ``V_chi``).
    """

    volume_map: MollifiedVolumeMap
    profile: VolumeProfile

    def __post_init__(self) -> None:
        self.profile.validate(self.volume_map.diagnostics()["total_volume"])

    @property
    def total_volume(self) -> float:
        """Return ``V_Omega`` represented by the level-set map."""
        return self.volume_map.diagnostics()["total_volume"]

    def pressure(self, chi: float | ArrayLike) -> float | NDArray[np.float64]:
        """Evaluate the monotone (M4b) composition at level-set values ``chi``."""
        volume = self.volume_map.volume(np.asarray(chi, dtype=float))
        return self.profile.value(volume)

    def enclosed_volume(self, pressure: float | ArrayLike) -> float | NDArray[np.float64]:
        """Invert a strictly decreasing sampled ``p_0(V)`` to diagnose realization."""
        points, scalar = _as_array(pressure)
        volumes = np.linspace(0.0, self.total_volume, len(self.volume_map.levels))
        values = np.asarray(self.profile.value(volumes), dtype=float)
        if np.any(np.diff(values) >= 0.0):
            raise InvalidProfileError(
                "enclosed-volume inversion requires a strictly decreasing profile"
            )
        if np.any(points < values[-1]) or np.any(points > values[0]):
            raise InvalidProfileError("pressure evaluated outside the profile range")
        result = np.interp(points, values[::-1], volumes[::-1])
        return _scalar_or_array(result, scalar)

    def layer_cake_moments(
        self,
        chi: ArrayLike,
        weights: ArrayLike,
        *,
        test_functions: Sequence[Callable[[NDArray[np.float64]], NDArray[np.float64]]],
        quadrature_order: int = 1025,
    ) -> NDArray[np.float64]:
        """Return residuals of note equation ``(layercake)`` for smooth moments."""
        levels = np.asarray(chi, dtype=float).reshape(-1)
        sample_weights = np.asarray(weights, dtype=float).reshape(-1)
        if len(levels) != len(sample_weights) or not len(levels):
            raise ValueError("chi and weights must be non-empty arrays of equal length")
        if (
            not np.all(np.isfinite(levels))
            or not np.all(np.isfinite(sample_weights))
            or np.any(sample_weights <= 0.0)
        ):
            raise ValueError("chi must be finite and weights must be finite and positive")
        if quadrature_order < 2:
            raise ValueError("quadrature_order must be at least two")
        pressure = np.asarray(self.pressure(levels), dtype=float)
        volumes = np.linspace(0.0, self.total_volume, quadrature_order)
        target = np.asarray(self.profile.value(volumes), dtype=float)
        return np.asarray(
            [
                float(
                    np.dot(sample_weights, function(pressure))
                    - np.trapezoid(function(target), volumes)
                )
                for function in test_functions
            ],
            dtype=float,
        )

    def as_ngsolve_coefficient(self, chi: ng.CoefficientFunction) -> ng.CoefficientFunction:
        """Wrap the local (M4b) composition as a monotone NGSolve 1D ``BSpline``.

        The degree-one spline is deliberate: it exactly preserves tabulated
        monotonicity and lets NGSolve differentiate the local chain-rule term
        ``g'(chi) delta_chi`` symbolically.  Milestone 2.3 owns the nonlocal
        ``delta V_chi`` term.
        """
        import ngsolve as ng

        levels = self.volume_map.levels
        pressures = np.asarray(self.pressure(levels), dtype=float)
        spline = ng.BSpline(
            2, [float(levels[0]), *map(float, levels), float(levels[-1])], pressures.tolist()
        )
        lower = float(pressures[0])
        upper = float(pressures[-1])
        return ng.IfPos(
            chi - float(levels[0]), ng.IfPos(float(levels[-1]) - chi, spline(chi), upper), lower
        )


def extract_ngsolve_quadrature(
    mesh: ng.Mesh,
    chi: ng.CoefficientFunction,
    gradient: ng.CoefficientFunction,
    *,
    integration_order: int,
) -> QuadratureLevelSetData:
    """Extract FEM quadrature data for ``(mollified_V)`` from an NGSolve mesh.

    ``w_i=weight(ip_i)*|det J_i|`` and ``h_i=|det J_i|^(1/d)`` are evaluated at
    each mapped integration point; hence curved elements retain their local geometry
    in the M4b volume coordinate instead of being reduced to a histogram.
    """
    import ngsolve as ng

    if integration_order < 1:
        raise ValueError("integration_order must be positive")
    values: list[float] = []
    gradients: list[float] = []
    weights: list[float] = []
    sizes: list[float] = []
    for element in mesh.Elements(ng.VOL):
        rule = ng.IntegrationRule(element.type, integration_order)
        transformation = mesh.GetTrafo(element)
        for point in rule:
            mapped_point = transformation(point)
            measure = float(mapped_point.measure)
            values.append(float(chi(mapped_point)))
            gradients.append(float(np.linalg.norm(np.asarray(gradient(mapped_point), dtype=float))))
            weights.append(float(point.weight) * measure)
            sizes.append(measure ** (1.0 / mesh.dim))
    return QuadratureLevelSetData(
        values=np.asarray(values, dtype=float),
        gradient_magnitudes=np.asarray(gradients, dtype=float),
        weights=np.asarray(weights, dtype=float),
        element_sizes=np.asarray(sizes, dtype=float),
    )
