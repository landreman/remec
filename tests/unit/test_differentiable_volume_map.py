"""Finite-difference contracts for the note ``(V_derivatives)`` JVP."""

from __future__ import annotations

import numpy as np
import pytest

from remec.level_set import MollifiedVolumeMap, QuadratureLevelSetData


def test_mollified_volume_jvp_matches_a_directional_finite_difference() -> None:
    """``sum_i H'_epsilon(chi_i-level) w_i delta_chi_i`` is the nonlocal JVP.

    The manufactured one-dimensional level set has spatially uniform gradient and
    element size, so the test isolates the note's ``(V_derivatives)`` term from
    variation of the gradient-scaled mollifier width.
    """
    sample_count = 4096
    values = (np.arange(sample_count, dtype=float) + 0.5) / sample_count
    weights = np.full(sample_count, 1.0 / sample_count)
    perturbation = 0.3 * np.sin(2.0 * np.pi * values) + 0.1 * np.cos(6.0 * np.pi * values)
    volume_map = MollifiedVolumeMap.build(
        QuadratureLevelSetData(
            values=values,
            gradient_magnitudes=np.ones_like(values),
            weights=weights,
            element_sizes=np.full_like(values, 1.0 / sample_count),
        ),
        spatial_width_cells=2.0,
        levels=129,
    )
    probe_levels = np.array([0.23, 0.37, 0.61, 0.74])
    step = 1.0e-6

    finite_difference = (
        MollifiedVolumeMap._mollified_volume(
            values + step * perturbation, volume_map.mollifier_widths, weights, probe_levels
        )
        - MollifiedVolumeMap._mollified_volume(
            values - step * perturbation, volume_map.mollifier_widths, weights, probe_levels
        )
    ) / (2.0 * step)

    assert volume_map.jvp(perturbation, probe_levels) == pytest.approx(
        finite_difference, rel=3.0e-8, abs=1.0e-9
    )

    default_levels_jvp = np.asarray(volume_map.jvp(perturbation), dtype=float)
    assert default_levels_jvp.shape == volume_map.levels.shape
    assert volume_map.jvp(np.ones_like(values), probe_levels) == pytest.approx(
        [volume_map.coarea_density(level) for level in probe_levels], rel=1.0e-13
    )


def test_frozen_width_jvp_has_a_bounded_live_width_discrepancy() -> None:
    """ADR 0003 records the O(epsilon) quasi-Newton difference for live widths.

    This variable-gradient manufactured level set rebuilds ``epsilon_i`` for the
    finite-difference value maps, unlike the frozen-width contract above.  Option 1
    deliberately retains the displayed ``(V_derivatives)`` action, so this checks
    that the omitted live-width contribution is small at the selected resolution
    rather than incorrectly demanding equality with a different functional.
    """
    sample_count = 8192
    coordinates = (np.arange(sample_count, dtype=float) + 0.5) / sample_count
    weights = np.full(sample_count, 1.0 / sample_count)
    sizes = np.full(sample_count, 1.0 / sample_count)
    values = coordinates + 0.08 * np.sin(2.0 * np.pi * coordinates)
    gradients = 1.0 + 0.16 * np.pi * np.cos(2.0 * np.pi * coordinates)
    perturbation = 0.1 * np.sin(4.0 * np.pi * coordinates)
    perturbation_gradient = 0.4 * np.pi * np.cos(4.0 * np.pi * coordinates)
    data = QuadratureLevelSetData(values, gradients, weights, sizes)
    volume_map = MollifiedVolumeMap.build(data, spatial_width_cells=2.0, levels=129)
    probe_levels = np.array([0.23, 0.51, 0.77])
    step = 1.0e-6

    def rebuilt_raw_volume(sign: float) -> np.ndarray:
        rebuilt = MollifiedVolumeMap.build(
            QuadratureLevelSetData(
                values=values + sign * step * perturbation,
                gradient_magnitudes=np.abs(gradients + sign * step * perturbation_gradient),
                weights=weights,
                element_sizes=sizes,
            ),
            spatial_width_cells=2.0,
            levels=129,
        )
        return np.asarray(
            MollifiedVolumeMap._mollified_volume(
                values + sign * step * perturbation,
                rebuilt.mollifier_widths,
                weights,
                probe_levels,
            ),
            dtype=float,
        )

    live_width_finite_difference = (rebuilt_raw_volume(1.0) - rebuilt_raw_volume(-1.0)) / (
        2.0 * step
    )
    relative_discrepancy = np.max(
        np.abs(
            np.asarray(volume_map.jvp(perturbation, probe_levels)) - live_width_finite_difference
        )
        / np.maximum(np.abs(live_width_finite_difference), 1.0e-12)
    )

    assert relative_discrepancy < 2.0e-4
