"""Differentiable level-set volume maps for the interpretive construction."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray


class VolumeMapConsistencyWarning(RuntimeWarning):
    """A mandatory §12.3 volume-map diagnostic exceeded its configured tolerance."""


@dataclass(frozen=True, slots=True)
class QuadratureLevelSetData:
    """Quadrature samples required by the mollified ``V_chi`` construction."""

    values: NDArray[np.float64]
    gradient_magnitudes: NDArray[np.float64]
    weights: NDArray[np.float64]
    element_sizes: NDArray[np.float64]

    @property
    def total_volume(self) -> float:
        """Return the integration volume represented by the samples."""
        return float(np.sum(self.weights))


@dataclass(frozen=True, slots=True)
class _MonotonePchip:
    """Shape-preserving cubic Hermite interpolant for strictly monotone data."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    slopes: NDArray[np.float64]

    @classmethod
    def build(cls, x: NDArray[np.float64], y: NDArray[np.float64]) -> _MonotonePchip:
        """Construct the Fritsch--Carlson monotone cubic interpolant."""
        widths = np.diff(x)
        secants = np.diff(y) / widths
        if np.any(widths <= 0.0) or np.any(secants == 0.0):
            raise ValueError("PCHIP requires strictly monotone samples")
        slopes = np.empty_like(x)
        if len(x) == 2:
            slopes[:] = secants[0]
            return cls(x=x, y=y, slopes=slopes)
        slopes[0] = cls._endpoint_slope(widths[0], widths[1], secants[0], secants[1])
        slopes[-1] = cls._endpoint_slope(widths[-1], widths[-2], secants[-1], secants[-2])
        for index in range(1, len(x) - 1):
            left, right = secants[index - 1], secants[index]
            if left * right <= 0.0:
                slopes[index] = 0.0
                continue
            left_width, right_width = widths[index - 1], widths[index]
            slopes[index] = (
                3.0
                * (left_width + right_width)
                / (
                    (2.0 * right_width + left_width) / left
                    + (right_width + 2.0 * left_width) / right
                )
            )
        return cls(x=x, y=y, slopes=slopes)

    @staticmethod
    def _endpoint_slope(
        first_width: float, second_width: float, first: float, second: float
    ) -> float:
        """Return the shape-preserving one-sided Fritsch--Carlson slope."""
        slope = ((2.0 * first_width + second_width) * first - first_width * second) / (
            first_width + second_width
        )
        if slope * first <= 0.0:
            return 0.0
        if first * second < 0.0 and abs(slope) > abs(3.0 * first):
            return 3.0 * first
        return slope

    def evaluate(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate, clamping outside the tabulated endpoint interval."""
        clipped = np.clip(points, self.x[0], self.x[-1])
        index = np.clip(np.searchsorted(self.x, clipped, side="right") - 1, 0, len(self.x) - 2)
        width = self.x[index + 1] - self.x[index]
        coordinate = (clipped - self.x[index]) / width
        h00 = 2.0 * coordinate**3 - 3.0 * coordinate**2 + 1.0
        h10 = coordinate**3 - 2.0 * coordinate**2 + coordinate
        h01 = -2.0 * coordinate**3 + 3.0 * coordinate**2
        h11 = coordinate**3 - coordinate**2
        return (
            h00 * self.y[index]
            + h10 * width * self.slopes[index]
            + h01 * self.y[index + 1]
            + h11 * width * self.slopes[index + 1]
        )

    def derivative(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the derivative of the clamped cubic interpolant."""
        clipped = np.clip(points, self.x[0], self.x[-1])
        index = np.clip(np.searchsorted(self.x, clipped, side="right") - 1, 0, len(self.x) - 2)
        width = self.x[index + 1] - self.x[index]
        coordinate = (clipped - self.x[index]) / width
        dh00 = (6.0 * coordinate**2 - 6.0 * coordinate) / width
        dh10 = 3.0 * coordinate**2 - 4.0 * coordinate + 1.0
        dh01 = (-6.0 * coordinate**2 + 6.0 * coordinate) / width
        dh11 = 3.0 * coordinate**2 - 2.0 * coordinate
        return (
            dh00 * self.y[index]
            + dh10 * self.slopes[index]
            + dh01 * self.y[index + 1]
            + dh11 * self.slopes[index + 1]
        )


class MollifiedVolumeMap:
    """Mollified note equation ``(mollified_V)`` level-set volume map.

    The map evaluates ``V_chi^epsilon(level) = sum_i w_i H_epsilon_i(chi_i-level)``.
    Each width is ``epsilon_i = c h_i max(|grad chi_i|, gradient_floor)``, so the
    smoothing width is spatially uniform rather than fixed in chi-space.
    """

    def __init__(
        self,
        *,
        values: NDArray[np.float64],
        widths: NDArray[np.float64],
        weights: NDArray[np.float64],
        levels: NDArray[np.float64],
        volumes: NDArray[np.float64],
        raw_endpoint_volume_error: float,
        raw_endpoint_zero_error: float,
        minimum_gradient_fraction: float,
        floored_sample_count: int,
    ) -> None:
        self._values = values
        self._widths = widths
        self._weights = weights
        self.levels = levels
        self.volumes = volumes
        self.minimum_level = float(levels[0])
        self.maximum_level = float(levels[-1])
        self._raw_endpoint_volume_error = raw_endpoint_volume_error
        self._raw_endpoint_zero_error = raw_endpoint_zero_error
        self.minimum_gradient_fraction = minimum_gradient_fraction
        self._floored_sample_count = floored_sample_count
        self._volume_interpolant = _MonotonePchip.build(levels, volumes)
        self._inverse_interpolant = _MonotonePchip.build(volumes[::-1], levels[::-1])

    @classmethod
    def build(
        cls,
        data: QuadratureLevelSetData,
        *,
        spatial_width_cells: float = 1.5,
        levels: int = 129,
        minimum_gradient_fraction: float = 1.0e-3,
        coarea_consistency_tolerance: float = 0.1,
    ) -> MollifiedVolumeMap:
        """Build a volume-uniform monotone tabulation from quadrature samples.

        This is the differentiable default from note equation ``(mollified_V)``;
        its co-area derivative is available through :meth:`coarea_density`.
        """
        if levels < 3:
            raise ValueError("levels must be at least three")
        if not isfinite(spatial_width_cells) or spatial_width_cells <= 0.0:
            raise ValueError("spatial_width_cells must be finite and positive")
        if not isfinite(minimum_gradient_fraction) or not 0.0 < minimum_gradient_fraction <= 1.0:
            raise ValueError("minimum_gradient_fraction must be finite and in (0, 1]")
        if not isfinite(coarea_consistency_tolerance) or coarea_consistency_tolerance <= 0.0:
            raise ValueError("coarea_consistency_tolerance must be finite and positive")
        values, gradients, weights, sizes = cls._validated_arrays(data)
        maximum_gradient = float(np.max(gradients))
        gradient_floor = max(np.finfo(float).tiny, maximum_gradient * minimum_gradient_fraction)
        floored_sample_count = int(np.count_nonzero(gradients < gradient_floor))
        widths = spatial_width_cells * sizes * np.maximum(gradients, gradient_floor)
        minimum_level, maximum_level = float(np.min(values)), float(np.max(values))
        raw_levels = np.linspace(minimum_level, maximum_level, levels, dtype=np.float64)
        raw_volumes = cls._mollified_volumes(values, widths, weights, raw_levels)
        total_volume = float(np.sum(weights))
        raw_endpoint_volume_error = abs(raw_volumes[0] - total_volume)
        raw_endpoint_zero_error = abs(raw_volumes[-1])
        raw_volumes[0], raw_volumes[-1] = total_volume, 0.0
        target_volumes = np.linspace(total_volume, 0.0, levels, dtype=np.float64)
        volume_uniform_levels = np.interp(target_volumes, raw_volumes[::-1], raw_levels[::-1])
        volume_uniform_levels[0], volume_uniform_levels[-1] = minimum_level, maximum_level
        volume_map = cls(
            values=values,
            widths=widths,
            weights=weights,
            levels=volume_uniform_levels,
            volumes=target_volumes,
            raw_endpoint_volume_error=raw_endpoint_volume_error,
            raw_endpoint_zero_error=raw_endpoint_zero_error,
            minimum_gradient_fraction=minimum_gradient_fraction,
            floored_sample_count=floored_sample_count,
        )
        coarea_error = volume_map.diagnostics()["coarea_spot_relative_error"]
        if coarea_error > coarea_consistency_tolerance:
            warnings.warn(
                "tabulated volume derivative disagrees with the mollified co-area density: "
                f"relative error {coarea_error:.3g} exceeds "
                f"{coarea_consistency_tolerance:.3g}",
                VolumeMapConsistencyWarning,
                stacklevel=2,
            )
        return volume_map

    @staticmethod
    def _validated_arrays(
        data: QuadratureLevelSetData,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        arrays = tuple(
            np.asarray(array, dtype=float).reshape(-1)
            for array in (data.values, data.gradient_magnitudes, data.weights, data.element_sizes)
        )
        values, gradients, weights, sizes = arrays
        if not values.size or len({array.size for array in arrays}) != 1:
            raise ValueError("quadrature arrays must be non-empty and have equal length")
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("quadrature arrays must be finite")
        if np.any(gradients < 0.0) or np.any(weights <= 0.0) or np.any(sizes <= 0.0):
            raise ValueError("gradients must be non-negative; weights and sizes must be positive")
        if float(np.max(values)) == float(np.min(values)):
            raise ValueError("level-set values must not be constant")
        return values, gradients, weights, sizes

    @staticmethod
    def _mollified_volumes(
        values: NDArray[np.float64],
        widths: NDArray[np.float64],
        weights: NDArray[np.float64],
        levels: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Integrate the compact, moment-matched mollified Heaviside per level."""
        result = np.empty_like(levels)
        for index, level in enumerate(levels):
            argument = (values - level) / widths
            smooth_step = np.where(
                argument <= -1.0,
                0.0,
                np.where(
                    argument >= 1.0,
                    1.0,
                    0.5 * (1.0 + argument + np.sin(np.pi * argument) / np.pi),
                ),
            )
            result[index] = float(np.dot(weights, smooth_step))
        return result

    @staticmethod
    def mollified_volume(
        values: NDArray[np.float64],
        widths: NDArray[np.float64],
        weights: NDArray[np.float64],
        levels: float | NDArray[np.float64],
    ) -> float | NDArray[np.float64]:
        """Evaluate the un-tabulated mollified ``V_chi`` in ``(mollified_V)``.

        Unlike :meth:`volume`, this method intentionally does not use the
        volume-uniform PCHIP table.  It is the smooth quadrature functional
        whose directional derivative is specified by ``(V_derivatives)``.
        """
        sample_values = np.asarray(values, dtype=float).reshape(-1)
        sample_widths = np.asarray(widths, dtype=float).reshape(-1)
        sample_weights = np.asarray(weights, dtype=float).reshape(-1)
        if (
            not sample_values.size
            or len({sample_values.size, sample_widths.size, sample_weights.size}) != 1
            or not np.all(np.isfinite(sample_values))
            or not np.all(np.isfinite(sample_widths))
            or not np.all(np.isfinite(sample_weights))
            or np.any(sample_widths <= 0.0)
            or np.any(sample_weights <= 0.0)
        ):
            raise ValueError("mollified-volume samples must be finite, equally sized, and positive")
        points = np.asarray(levels, dtype=float)
        if not np.all(np.isfinite(points)):
            raise ValueError("mollified-volume levels must be finite")
        result = MollifiedVolumeMap._mollified_volumes(
            sample_values, sample_widths, sample_weights, points.reshape(-1)
        ).reshape(points.shape)
        return float(result) if points.ndim == 0 else result

    @property
    def mollifier_widths(self) -> NDArray[np.float64]:
        """Return a copy of the frozen spatially scaled widths in ``(mollified_V)``."""
        return np.array(self._widths, dtype=np.float64, copy=True)

    def volume(self, level: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
        """Return monotone ``V_chi^epsilon(level)`` with exact endpoint identities."""
        points = np.asarray(level, dtype=float)
        values = self._volume_interpolant.evaluate(points)
        values = np.where(points <= self.minimum_level, self.volumes[0], values)
        values = np.where(points >= self.maximum_level, self.volumes[-1], values)
        return float(values) if points.ndim == 0 else values

    def inverse_level(self, volume: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
        """Return the stable monotone inverse ``chi_hat(V)``."""
        points = np.asarray(volume, dtype=float)
        values = self._inverse_interpolant.evaluate(points)
        values = np.where(points <= self.volumes[-1], self.maximum_level, values)
        values = np.where(points >= self.volumes[0], self.minimum_level, values)
        return float(values) if points.ndim == 0 else values

    def coarea_density(self, level: float) -> float:
        """Return ``-dV/dchi_hat = sum_i H'_epsilon_i w_i`` from (V_derivatives)."""
        argument = (self._values - level) / self._widths
        derivative = np.where(
            np.abs(argument) < 1.0,
            (1.0 + np.cos(np.pi * argument)) / (2.0 * self._widths),
            0.0,
        )
        return float(np.dot(self._weights, derivative))

    def jvp(
        self,
        delta_chi: NDArray[np.float64],
        levels: float | NDArray[np.float64] | None = None,
    ) -> float | NDArray[np.float64]:
        """Apply the nonlocal ``(V_derivatives)`` JVP at one or more levels.

        With the gradient-scaled mollifier widths frozen at this map's
        linearization point, this applies
        ``delta V_chi(level) = sum_i H'_epsilon_i(chi_i-level) w_i delta_chi_i``.
        The direct mollified surface weights are used rather than a numerical
        derivative of the volume-uniform tabulation, avoiding the near-step
        conditioning warned about in DESIGN.md §12.3.
        """
        perturbation = np.asarray(delta_chi, dtype=float).reshape(-1)
        if perturbation.shape != self._values.shape or not np.all(np.isfinite(perturbation)):
            raise ValueError("delta_chi must be finite and match the quadrature sample shape")
        evaluation_levels = self.levels if levels is None else np.asarray(levels, dtype=float)
        points = np.asarray(evaluation_levels, dtype=float)
        if not np.all(np.isfinite(points)):
            raise ValueError("JVP levels must be finite")
        argument = (self._values[:, None] - points.reshape(-1)) / self._widths[:, None]
        surface_delta = np.where(
            np.abs(argument) < 1.0,
            (1.0 + np.cos(np.pi * argument)) / (2.0 * self._widths[:, None]),
            0.0,
        )
        result = np.dot(self._weights * perturbation, surface_delta).reshape(points.shape)
        return float(result) if points.ndim == 0 else result

    def volume_derivative(self, level: float) -> float:
        """Return the tabulated ``dV_chi^epsilon/dchi_hat`` for the §12.3 check."""
        point = np.asarray(level, dtype=float)
        if point.ndim != 0:
            raise TypeError("volume_derivative requires one scalar level")
        return float(self._volume_interpolant.derivative(point))

    def diagnostics(self) -> dict[str, float]:
        """Return endpoint, monotonicity, and smoothing diagnostics for ``V_chi``."""
        spot_level = float(self.levels[len(self.levels) // 2])
        coarea_density = self.coarea_density(spot_level)
        spline_density = -self.volume_derivative(spot_level)
        coarea_relative_error = abs(spline_density - coarea_density) / max(
            coarea_density, np.finfo(float).tiny
        )
        return {
            "total_volume": float(self.volumes[0]),
            "minimum_level": self.minimum_level,
            "maximum_level": self.maximum_level,
            "raw_endpoint_volume_error": self._raw_endpoint_volume_error,
            "raw_endpoint_zero_error": self._raw_endpoint_zero_error,
            "spline_monotonicity_margin": float(np.min(-np.diff(self.volumes))),
            "coarea_spot_level": spot_level,
            "coarea_spot_relative_error": coarea_relative_error,
            "floored_sample_count": float(self._floored_sample_count),
            "minimum_mollifier_width": float(np.min(self._widths)),
            "maximum_mollifier_width": float(np.max(self._widths)),
        }
