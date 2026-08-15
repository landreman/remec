"""Normalized profiles and the note equation ``(M4b)`` pressure transplant."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from remec.level_set import MollifiedVolumeMap, QuadratureLevelSetData

if TYPE_CHECKING:
    import ngsolve as ng  # type: ignore[import-untyped]


class InvalidProfileError(ValueError):
    """A normalized pressure or enclosed-current profile is inadmissible."""


class PressureProfile(Protocol):
    """One-dimensional ``p_0(s)`` profile on normalized volume ``s in [0, 1]``."""

    def value(self, normalized_volume: float | ArrayLike) -> float | NDArray[np.float64]: ...

    def derivative(self, normalized_volume: float | ArrayLike) -> float | NDArray[np.float64]: ...

    def validate(self, edge_value: float | None = None) -> None: ...


class ToroidalCurrentProfile(Protocol):
    """Cumulative enclosed-current input ``I_0(s)`` for note equation ``(M3b)``."""

    def enclosed_current(
        self, normalized_volume: float | ArrayLike
    ) -> float | NDArray[np.float64]: ...

    def derivative(self, normalized_volume: float | ArrayLike) -> float | NDArray[np.float64]: ...

    def validate(self) -> None: ...


def _as_array(value: float | ArrayLike) -> tuple[NDArray[np.float64], bool]:
    points = np.asarray(value, dtype=float)
    return points, points.ndim == 0


def _normalized_points(value: float | ArrayLike) -> tuple[NDArray[np.float64], bool]:
    points, scalar = _as_array(value)
    if not np.all(np.isfinite(points)):
        raise InvalidProfileError("normalized-volume coordinates must be finite")
    if np.any(points < 0.0) or np.any(points > 1.0):
        raise InvalidProfileError("profile evaluated outside s in [0, 1]")
    return points, scalar


def _scalar_or_array(values: NDArray[np.float64], scalar: bool) -> float | NDArray[np.float64]:
    return float(values) if scalar else values


@dataclass(frozen=True, slots=True)
class AnalyticPressureProfile:
    """Analytic non-increasing pressure profile ``p_0(s)`` from note ``(M4b)``."""

    value_function: Callable[[float | NDArray[np.float64]], float | NDArray[np.float64]]
    derivative_function: Callable[[float | NDArray[np.float64]], float | NDArray[np.float64]]

    def value(self, normalized_volume: float | ArrayLike) -> float | NDArray[np.float64]:
        """Evaluate ``p_0(s)`` without clipping coordinates outside ``[0, 1]``."""
        points, scalar = _normalized_points(normalized_volume)
        values = np.asarray(self.value_function(points), dtype=float)
        if values.shape != points.shape or not np.all(np.isfinite(values)):
            raise InvalidProfileError("profile values must be finite and match the input shape")
        return _scalar_or_array(values, scalar)

    def derivative(self, normalized_volume: float | ArrayLike) -> float | NDArray[np.float64]:
        """Evaluate ``dp_0/ds`` used by the local ``(M4b)`` composition."""
        points, scalar = _normalized_points(normalized_volume)
        values = np.asarray(self.derivative_function(points), dtype=float)
        if values.shape != points.shape or not np.all(np.isfinite(values)):
            raise InvalidProfileError(
                "profile derivatives must be finite and match the input shape"
            )
        return _scalar_or_array(values, scalar)

    def validate(self, edge_value: float | None = None) -> None:
        """Check §12.5 monotonicity, derivative, and optional ``p_0(1)=p_b``."""
        _validate_pressure_profile(self, edge_value)


@dataclass(frozen=True, slots=True)
class TabulatedPressureProfile:
    """Piecewise-linear ``p_0(s)`` table on exactly ``[0, 1]``."""

    normalized_volumes: Sequence[float]
    pressures: Sequence[float]

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.normalized_volumes, dtype=float)
        values = np.asarray(self.pressures, dtype=float)
        _validate_tabulated_arrays(coordinates, values)
        if np.any(np.diff(values) > 0.0):
            raise InvalidProfileError("tabulated profile pressures must be non-increasing")
        object.__setattr__(self, "normalized_volumes", tuple(float(item) for item in coordinates))
        object.__setattr__(self, "pressures", tuple(float(item) for item in values))

    def value(self, normalized_volume: float | ArrayLike) -> float | NDArray[np.float64]:
        """Evaluate the monotone table, including an intentional edge plateau."""
        points, scalar = _normalized_points(normalized_volume)
        values = np.interp(points, self.normalized_volumes, self.pressures)
        return _scalar_or_array(values, scalar)

    def derivative(self, normalized_volume: float | ArrayLike) -> float | NDArray[np.float64]:
        """Return the one-sided piecewise-linear ``dp_0/ds``."""
        points, scalar = _normalized_points(normalized_volume)
        coordinates = np.asarray(self.normalized_volumes)
        values = np.asarray(self.pressures)
        index = np.clip(
            np.searchsorted(coordinates, points, side="right") - 1, 0, len(coordinates) - 2
        )
        derivatives = np.diff(values)[index] / np.diff(coordinates)[index]
        return _scalar_or_array(np.asarray(derivatives, dtype=float), scalar)

    def validate(self, edge_value: float | None = None) -> None:
        """Require the optional physical edge pressure ``p_0(1)=p_b``."""
        if edge_value is not None and not np.isclose(self.pressures[-1], edge_value):
            raise InvalidProfileError("profile edge value differs from the required edge value")

    def to_record(self) -> dict[str, object]:
        """Return an unambiguous normalized-volume checkpoint record."""
        return _tabulated_record("pressure", self.normalized_volumes, self.pressures)

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> TabulatedPressureProfile:
        """Restore only the explicit normalized ``p_0(s)`` record contract."""
        coordinates, values = _read_tabulated_record(record, "pressure")
        return cls(coordinates, values)


@dataclass(frozen=True, slots=True)
class AnalyticToroidalCurrentProfile:
    """Analytic cumulative enclosed toroidal current ``I_0(s)`` from ``(M3b)``."""

    enclosed_current_function: Callable[[float | NDArray[np.float64]], float | NDArray[np.float64]]
    derivative_function: Callable[[float | NDArray[np.float64]], float | NDArray[np.float64]]

    def enclosed_current(self, normalized_volume: float | ArrayLike) -> float | NDArray[np.float64]:
        """Evaluate cumulative ``I_0(s)`` without silently clipping ``s``."""
        points, scalar = _normalized_points(normalized_volume)
        values = np.asarray(self.enclosed_current_function(points), dtype=float)
        if values.shape != points.shape or not np.all(np.isfinite(values)):
            raise InvalidProfileError("current values must be finite and match the input shape")
        return _scalar_or_array(values, scalar)

    def derivative(self, normalized_volume: float | ArrayLike) -> float | NDArray[np.float64]:
        """Evaluate ``dI_0/ds`` used for note ``(M3b)`` shell targets."""
        points, scalar = _normalized_points(normalized_volume)
        values = np.asarray(self.derivative_function(points), dtype=float)
        if values.shape != points.shape or not np.all(np.isfinite(values)):
            raise InvalidProfileError(
                "current derivatives must be finite and match the input shape"
            )
        return _scalar_or_array(values, scalar)

    def validate(self) -> None:
        """Check ``I_0(0)=0`` and consistency of the analytic derivative."""
        _validate_current_profile(self, check_derivative=True)


@dataclass(frozen=True, slots=True)
class TabulatedToroidalCurrentProfile:
    """Piecewise-linear cumulative ``I_0(s)`` table; reversed current is valid."""

    normalized_volumes: Sequence[float]
    enclosed_currents: Sequence[float]

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.normalized_volumes, dtype=float)
        values = np.asarray(self.enclosed_currents, dtype=float)
        _validate_tabulated_arrays(coordinates, values)
        if values[0] != 0.0:
            raise InvalidProfileError("cumulative toroidal current requires I_0(0)=0")
        object.__setattr__(self, "normalized_volumes", tuple(float(item) for item in coordinates))
        object.__setattr__(self, "enclosed_currents", tuple(float(item) for item in values))

    def enclosed_current(self, normalized_volume: float | ArrayLike) -> float | NDArray[np.float64]:
        """Evaluate cumulative ``I_0(s)`` including reversed-current segments."""
        points, scalar = _normalized_points(normalized_volume)
        values = np.interp(points, self.normalized_volumes, self.enclosed_currents)
        return _scalar_or_array(values, scalar)

    def derivative(self, normalized_volume: float | ArrayLike) -> float | NDArray[np.float64]:
        """Return the one-sided piecewise-linear ``dI_0/ds``."""
        points, scalar = _normalized_points(normalized_volume)
        coordinates = np.asarray(self.normalized_volumes)
        values = np.asarray(self.enclosed_currents)
        index = np.clip(
            np.searchsorted(coordinates, points, side="right") - 1, 0, len(coordinates) - 2
        )
        derivatives = np.diff(values)[index] / np.diff(coordinates)[index]
        return _scalar_or_array(np.asarray(derivatives, dtype=float), scalar)

    def validate(self) -> None:
        """Recheck the cumulative-current endpoint contract."""
        _validate_current_profile(self, check_derivative=False)

    def to_record(self) -> dict[str, object]:
        """Return an unambiguous normalized-volume checkpoint record."""
        return _tabulated_record(
            "toroidal_current", self.normalized_volumes, self.enclosed_currents
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> TabulatedToroidalCurrentProfile:
        """Restore only the explicit normalized cumulative-current contract."""
        coordinates, values = _read_tabulated_record(record, "toroidal_current")
        return cls(coordinates, values)


def _validate_tabulated_arrays(
    coordinates: NDArray[np.float64], values: NDArray[np.float64]
) -> None:
    if coordinates.ndim != 1 or values.ndim != 1 or len(coordinates) < 2:
        raise InvalidProfileError("tabulated profile needs at least two one-dimensional samples")
    if (
        len(coordinates) != len(values)
        or not np.all(np.isfinite(coordinates))
        or not np.all(np.isfinite(values))
    ):
        raise InvalidProfileError("tabulated profile samples must be finite and equally sized")
    if np.any(np.diff(coordinates) <= 0.0):
        raise InvalidProfileError("normalized-volume samples must be strictly increasing")
    if coordinates[0] != 0.0 or coordinates[-1] != 1.0:
        raise InvalidProfileError("tabulated profile coordinate must be exactly [0, 1]")


def _validate_pressure_profile(profile: PressureProfile, edge_value: float | None) -> None:
    probe = np.linspace(0.0, 1.0, 257)
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
    if isinstance(profile, AnalyticPressureProfile):
        _validate_analytic_derivative(
            values_function=profile.value,
            derivative_function=profile.derivative,
            message="profile derivative disagrees with its value function",
        )
    if edge_value is not None and not np.isclose(values[-1], edge_value):
        raise InvalidProfileError("profile edge value differs from the required edge value")


def _validate_current_profile(profile: ToroidalCurrentProfile, *, check_derivative: bool) -> None:
    probe = np.linspace(0.0, 1.0, 257)
    values = np.asarray(profile.enclosed_current(probe), dtype=float)
    derivatives = np.asarray(profile.derivative(probe), dtype=float)
    if values.shape != probe.shape or derivatives.shape != probe.shape:
        raise InvalidProfileError("profile evaluation must preserve array shape")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(derivatives)):
        raise InvalidProfileError("profile values and derivatives must be finite")
    if values[0] != 0.0:
        raise InvalidProfileError("cumulative toroidal current requires I_0(0)=0")
    if check_derivative:
        _validate_analytic_derivative(
            values_function=profile.enclosed_current,
            derivative_function=profile.derivative,
            message="current derivative disagrees with its value function",
        )


def _validate_analytic_derivative(
    *,
    values_function: Callable[[NDArray[np.float64]], float | NDArray[np.float64]],
    derivative_function: Callable[[NDArray[np.float64]], float | NDArray[np.float64]],
    message: str,
) -> None:
    coarse_probe: NDArray[np.float64] = np.linspace(0.0, 1.0, 257, dtype=np.float64)
    refined_probe: NDArray[np.float64] = np.linspace(0.0, 1.0, 1025, dtype=np.float64)
    coarse_values = np.asarray(values_function(coarse_probe), dtype=float)
    refined_values = np.asarray(values_function(refined_probe), dtype=float)
    refined_derivatives = np.asarray(derivative_function(refined_probe), dtype=float)
    finite_difference = np.gradient(refined_values, refined_probe)
    coarse_difference = np.interp(
        refined_probe, coarse_probe, np.gradient(coarse_values, coarse_probe)
    )
    resolution_error = np.abs(finite_difference - coarse_difference)
    scale = max(1.0, float(np.max(np.abs(finite_difference))))
    tolerance = 2.0 * resolution_error + 2.0e-3 * scale
    if np.any(np.abs(refined_derivatives[1:-1] - finite_difference[1:-1]) > tolerance[1:-1]):
        raise InvalidProfileError(message)


def _tabulated_record(
    profile_kind: str, coordinates: Sequence[float], values: Sequence[float]
) -> dict[str, object]:
    return {
        "profile_kind": profile_kind,
        "representation": "piecewise_linear",
        "coordinate_kind": "normalized_volume",
        "normalized_volume": list(coordinates),
        "values": list(values),
    }


def _read_tabulated_record(
    record: Mapping[str, object], expected_kind: str
) -> tuple[Sequence[float], Sequence[float]]:
    required = {
        "profile_kind",
        "representation",
        "coordinate_kind",
        "normalized_volume",
        "values",
    }
    if set(record) != required:
        raise InvalidProfileError("profile record requires an explicit coordinate_kind and fields")
    if record["profile_kind"] != expected_kind:
        raise InvalidProfileError(f"expected {expected_kind!r} profile record")
    if record["representation"] != "piecewise_linear":
        raise InvalidProfileError("unsupported profile representation")
    if record["coordinate_kind"] != "normalized_volume":
        raise InvalidProfileError("profile coordinate_kind must be 'normalized_volume'")
    coordinates = record["normalized_volume"]
    values = record["values"]
    if not isinstance(coordinates, Sequence) or isinstance(coordinates, str):
        raise InvalidProfileError("normalized_volume must be an array")
    if not isinstance(values, Sequence) or isinstance(values, str):
        raise InvalidProfileError("profile values must be an array")
    return cast(Sequence[float], coordinates), cast(Sequence[float], values)


@dataclass(frozen=True, slots=True)
class TransplantedProfile:
    """Note ``(M4b)``: ``p(r)=p_0(V_chi(chi(r))/V_omega)``.

    Both factors are monotone decreasing, so the composition is monotone increasing
    in ``chi`` and realizes the prescribed normalized-volume distribution.
    """

    volume_map: MollifiedVolumeMap
    profile: PressureProfile
    edge_pressure: float | None = None

    def __post_init__(self) -> None:
        self.profile.validate(self.edge_pressure)

    @property
    def total_volume(self) -> float:
        """Return ``V_omega`` represented by the level-set map."""
        return self.volume_map.diagnostics()["total_volume"]

    def pressure(self, chi: float | ArrayLike) -> float | NDArray[np.float64]:
        """Evaluate the shared-normalized-coordinate ``(M4b)`` composition."""
        normalized_volume = self.volume_map.evaluate_volume_coordinate(np.asarray(chi, dtype=float))
        return self.profile.value(normalized_volume)

    def enclosed_volume(
        self, chi: ArrayLike, weights: ArrayLike, pressure: float | ArrayLike
    ) -> float | NDArray[np.float64]:
        """Measure pressure-superlevel volume independently of the profile inverse."""
        points, scalar = _as_array(pressure)
        levels = np.asarray(chi, dtype=float).reshape(-1)
        sample_weights = np.asarray(weights, dtype=float).reshape(-1)
        if len(levels) != len(sample_weights) or not len(levels) or np.any(sample_weights <= 0.0):
            raise ValueError(
                "chi and weights must be non-empty arrays of equal length with positive weights"
            )
        pressure_samples = np.asarray(self.pressure(levels), dtype=float)
        profile_range = np.asarray(self.profile.value(np.array([1.0, 0.0])), dtype=float)
        if np.any(points < profile_range[0]) or np.any(points > profile_range[1]):
            raise InvalidProfileError("pressure evaluated outside the profile range")
        result = np.asarray(
            [
                float(np.sum(sample_weights[pressure_samples >= threshold]))
                for threshold in points.reshape(-1)
            ]
        ).reshape(points.shape)
        return _scalar_or_array(result, scalar)

    def layer_cake_moments(
        self,
        chi: ArrayLike,
        weights: ArrayLike,
        *,
        test_functions: Sequence[Callable[[NDArray[np.float64]], NDArray[np.float64]]],
        quadrature_order: int = 1025,
    ) -> NDArray[np.float64]:
        """Return residuals of normalized note equation ``(layercake)``."""
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
        normalized_volumes = np.linspace(0.0, 1.0, quadrature_order)
        target = np.asarray(self.profile.value(normalized_volumes), dtype=float)
        return np.asarray(
            [
                float(
                    np.dot(sample_weights, function(pressure))
                    - self.total_volume * _trapezoid(function(target), normalized_volumes)
                )
                for function in test_functions
            ],
            dtype=np.float64,
        )

    def as_ngsolve_coefficient(self, chi: ng.CoefficientFunction) -> ng.CoefficientFunction:
        """Wrap local ``(M4b)`` as a differentiable monotone NGSolve 1D spline."""
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
    """Extract FEM quadrature data for ``(mollified_V)`` from an NGSolve mesh."""
    import ngsolve as ng

    if integration_order < 1:
        raise ValueError("integration_order must be positive")
    weights: list[float] = []
    sizes: list[float] = []
    element_types = {element.type for element in mesh.Elements(ng.VOL)}
    rules = {
        element_type: ng.IntegrationRule(element_type, integration_order)
        for element_type in element_types
    }
    for element in mesh.Elements(ng.VOL):
        rule = rules[element.type]
        transformation = mesh.GetTrafo(element)
        for point in rule:
            mapped_point = transformation(point)
            measure = float(mapped_point.measure)
            weights.append(float(point.weight) * measure)
            sizes.append(measure ** (1.0 / mesh.dim))
    mapped_points = mesh.MapToAllElements(rules, ng.VOL)
    values = np.asarray(chi(mapped_points), dtype=float).reshape(-1)
    gradients = np.linalg.norm(np.asarray(gradient(mapped_points), dtype=float), axis=1)
    if len(values) != len(weights):
        raise RuntimeError(
            "NGSolve mapped quadrature ordering does not match element integration rules"
        )
    return QuadratureLevelSetData(
        values=values,
        gradient_magnitudes=gradients,
        weights=np.asarray(weights, dtype=float),
        element_sizes=np.asarray(sizes, dtype=float),
    )


def _trapezoid(values: ArrayLike, points: ArrayLike) -> float:
    """NumPy-1.24-compatible composite trapezoidal integration."""
    ordinates = np.asarray(values, dtype=np.float64)
    abscissae = np.asarray(points, dtype=np.float64)
    return float(np.sum(np.diff(abscissae) * (ordinates[:-1] + ordinates[1:]) * 0.5))
