"""Analytic contracts for the mollified level-set volume map."""

from __future__ import annotations

import csv
from math import pi
from pathlib import Path

import numpy as np
import pytest

from remec.level_set import (
    MollifiedVolumeMap,
    QuadratureLevelSetData,
    VolumeMapConsistencyWarning,
)

_MANUFACTURED_DIRECTORY = Path(__file__).parents[1] / "verification"


def _tensor_product_data(
    dimension: int, order: int = 96
) -> tuple[QuadratureLevelSetData, np.ndarray]:
    """Return Gauss data on ``[-1, 1]^d`` and the associated coordinates."""
    nodes, weights_1d = np.polynomial.legendre.leggauss(order)
    coordinates = np.meshgrid(*([nodes] * dimension), indexing="ij")
    weights = np.meshgrid(*([weights_1d] * dimension), indexing="ij")
    return (
        QuadratureLevelSetData(
            values=np.zeros(order**dimension),
            gradient_magnitudes=np.ones(order**dimension),
            weights=np.prod(np.stack(weights), axis=0).ravel(),
            element_sizes=np.full(order**dimension, 2.0 / order),
        ),
        np.stack(coordinates).reshape(dimension, -1),
    )


@pytest.mark.parametrize(
    ("dimension", "radius", "expected_volume", "expected_density"),
    [
        (2, 0.6, pi * 0.6**2, pi),
        (3, 0.6, 4.0 * pi * 0.6**3 / 3.0, 2.0 * pi * 0.6),
    ],
)
def test_mollified_volume_matches_analytic_circle_and_sphere(
    dimension: int, radius: float, expected_volume: float, expected_density: float
) -> None:
    """(mollified_V) resolves the zero level-set circle/sphere volume."""
    data, coordinates = _tensor_product_data(dimension)
    radial_squared = np.sum(coordinates**2, axis=0)
    values = radius**2 - radial_squared
    gradient_magnitudes = 2.0 * np.sqrt(radial_squared)
    level_data = QuadratureLevelSetData(
        values=values,
        gradient_magnitudes=gradient_magnitudes,
        weights=data.weights,
        element_sizes=data.element_sizes,
    )

    volume_map = MollifiedVolumeMap.build(level_data, spatial_width_cells=1.0, levels=129)

    assert volume_map.volume(0.0) == pytest.approx(expected_volume, rel=3.0e-3)
    assert volume_map.volume(volume_map.minimum_level) == pytest.approx(data.total_volume)
    assert volume_map.volume(volume_map.maximum_level) == pytest.approx(0.0)
    assert volume_map.coarea_density(0.0) == pytest.approx(expected_density, rel=3.0e-2)
    assert -volume_map.volume_derivative(0.0) == pytest.approx(
        volume_map.coarea_density(0.0), rel=3.0e-2
    )
    diagnostics = volume_map.diagnostics()
    assert diagnostics["raw_endpoint_volume_error"] / data.total_volume < 1.0e-3
    assert diagnostics["raw_endpoint_zero_error"] / data.total_volume < 1.0e-3
    assert diagnostics["spline_monotonicity_margin"] > 0.0
    assert diagnostics["coarea_spot_relative_error"] < 3.0e-2
    inverse_probe_levels = np.linspace(0.02, radius**2 - 0.02, 7)
    assert volume_map.inverse_level(volume_map.volume(inverse_probe_levels)) == pytest.approx(
        inverse_probe_levels, abs=4.0e-2
    )


def test_tabulation_is_strictly_monotone_and_uniform_in_volume() -> None:
    """(mollified_V) uses a monotone table whose samples are uniform in volume."""
    data, coordinates = _tensor_product_data(2)
    values = 0.8 - coordinates[0] ** 2 - 0.25 * coordinates[1] ** 2
    gradient_magnitudes = 2.0 * np.sqrt(coordinates[0] ** 2 + 0.0625 * coordinates[1] ** 2)
    volume_map = MollifiedVolumeMap.build(
        QuadratureLevelSetData(
            values=values,
            gradient_magnitudes=gradient_magnitudes,
            weights=data.weights,
            element_sizes=data.element_sizes,
        ),
        spatial_width_cells=1.25,
        levels=65,
    )

    assert np.all(np.diff(volume_map.levels) > 0.0)
    assert np.all(np.diff(volume_map.volumes) < 0.0)
    assert np.diff(volume_map.volumes) == pytest.approx(
        np.full(64, -data.total_volume / 64.0), abs=2.0e-12
    )
    probe_levels = np.linspace(volume_map.minimum_level, volume_map.maximum_level, 401)
    assert np.all(np.diff(volume_map.volume(probe_levels)) <= 0.0)
    probe_volumes = np.linspace(0.0, data.total_volume, 401)
    assert np.all(np.diff(volume_map.inverse_level(probe_volumes)) <= 0.0)


def test_mollified_sphere_volume_has_second_order_resolution_trend() -> None:
    """(mollified_V) has the expected ``O(epsilon**2)`` sphere-volume trend."""
    expected_volume = 4.0 * pi * 0.6**3 / 3.0
    with (_MANUFACTURED_DIRECTORY / "mollified_sphere_volume_rates.csv").open() as stream:
        expected_rows = list(csv.DictReader(stream))

    errors: list[float] = []
    for row in expected_rows:
        data, coordinates = _tensor_product_data(3, order=int(row["quadrature_order"]))
        radial_squared = np.sum(coordinates**2, axis=0)
        volume_map = MollifiedVolumeMap.build(
            QuadratureLevelSetData(
                values=0.6**2 - radial_squared,
                gradient_magnitudes=2.0 * np.sqrt(radial_squared),
                weights=data.weights,
                element_sizes=data.element_sizes,
            ),
            spatial_width_cells=1.0,
            levels=129,
        )
        error = abs(volume_map.volume(0.0) - expected_volume)
        errors.append(error)
        assert error == pytest.approx(float(row["absolute_error"]), rel=1.0e-10)

    rates = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert rates == pytest.approx(
        [float(row["adjacent_rate"]) for row in expected_rows[1:]], rel=1.0e-10
    )
    assert np.all(rates > 1.9)


def test_critical_point_gradient_floor_prevents_a_delta_spike() -> None:
    """(mollified_V) applies its named critical-point width safeguard."""
    with pytest.warns(VolumeMapConsistencyWarning, match="co-area density"):
        volume_map = MollifiedVolumeMap.build(
            QuadratureLevelSetData(
                values=np.concatenate((np.zeros(29), np.ones(71))),
                gradient_magnitudes=np.concatenate((np.zeros(29), np.ones(71))),
                weights=np.full(100, 0.01),
                element_sizes=np.full(100, 0.1),
            ),
            spatial_width_cells=1.0,
            levels=33,
            minimum_gradient_fraction=1.0e-3,
        )

    assert volume_map.diagnostics()["minimum_mollifier_width"] == pytest.approx(1.0e-4)
    assert volume_map.diagnostics()["floored_sample_count"] == 29.0
    assert np.isfinite(volume_map.coarea_density(0.0))
    assert volume_map.coarea_density(0.0) < 3.0e3


def test_build_warns_when_the_mandatory_coarea_check_fails() -> None:
    """§12.3 does not silently accept an inconsistent tabulated derivative."""
    data, coordinates = _tensor_product_data(2)
    radial_squared = np.sum(coordinates**2, axis=0)

    with pytest.warns(VolumeMapConsistencyWarning, match="relative error"):
        MollifiedVolumeMap.build(
            QuadratureLevelSetData(
                values=0.6**2 - radial_squared,
                gradient_magnitudes=2.0 * np.sqrt(radial_squared),
                weights=data.weights,
                element_sizes=data.element_sizes,
            ),
            spatial_width_cells=0.05,
            levels=129,
            coarea_consistency_tolerance=0.1,
        )
