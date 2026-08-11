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
        MollifiedVolumeMap.mollified_volume(
            values + step * perturbation, volume_map.mollifier_widths, weights, probe_levels
        )
        - MollifiedVolumeMap.mollified_volume(
            values - step * perturbation, volume_map.mollifier_widths, weights, probe_levels
        )
    ) / (2.0 * step)

    assert volume_map.jvp(perturbation, probe_levels) == pytest.approx(
        finite_difference, rel=2.0e-7, abs=2.0e-9
    )
