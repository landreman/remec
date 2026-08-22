"""Manufactured circle/annulus verification of mollified M2 shell-current moments."""

from __future__ import annotations

import csv
from dataclasses import replace
from itertools import pairwise
from math import pi
from pathlib import Path

import numpy as np
import pytest

from remec.current_moments import M2ToroidalCurrentSamples, mollified_shell_current_moments
from remec.level_set import MollifiedVolumeMap, QuadratureLevelSetData

_RATE_TABLE = Path(__file__).with_name("shell_current_moment_rates.csv")


def _circular_toroidal_surrogate(
    radial_cells: int,
    quadrature_order: int,
    *,
    radius: float = 1.0,
    spatial_width_cells: float = 1.0,
) -> tuple[MollifiedVolumeMap, M2ToroidalCurrentSamples]:
    """Return polar-cell quadrature for an axisymmetric circular toroidal surrogate."""
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    radial_edges = np.linspace(0.0, radius, radial_cells + 1)
    radii: list[np.ndarray] = []
    radial_weights: list[np.ndarray] = []
    for left, right in pairwise(radial_edges):
        mapped = 0.5 * ((right - left) * nodes + right + left)
        radii.append(mapped)
        radial_weights.append(0.5 * (right - left) * weights * mapped)
    radius_samples = np.concatenate(radii)
    # The 4*pi**2 factor integrates poloidal angle and toroidal angle. The moment
    # implementation contributes the note-(Itor) factor 1/(2*pi).
    volume_weights = 4.0 * pi**2 * np.concatenate(radial_weights)
    chi = radius**2 - radius_samples**2
    normalized_volume = (radius_samples / radius) ** 2
    inverse_area = radius**-2
    volume_map = MollifiedVolumeMap.build(
        QuadratureLevelSetData(
            values=chi,
            gradient_magnitudes=2.0 * radius_samples,
            weights=volume_weights,
            element_sizes=np.full_like(chi, radius / radial_cells),
        ),
        spatial_width_cells=spatial_width_cells,
        levels=257,
    )
    samples = M2ToroidalCurrentSamples(
        normalized_volume=volume_map.quadrature_normalized_volume,
        parallel=inverse_area * (2.0 + normalized_volume),
        diamagnetic=inverse_area * (-0.5 * normalized_volume),
        regularizing=inverse_area * (0.25 * (1.0 - normalized_volume)),
    )
    return volume_map, samples


def _analytic_cumulative(edges: np.ndarray) -> dict[str, np.ndarray]:
    """Analytic (Itor) components after the radius-independent current scaling."""
    parallel = pi * (2.0 * edges + 0.5 * edges**2)
    diamagnetic = -0.25 * pi * edges**2
    regularizing = 0.25 * pi * (edges - 0.5 * edges**2)
    return {
        "parallel": parallel,
        "diamagnetic": diamagnetic,
        "regularizing": regularizing,
        "total": parallel + diamagnetic + regularizing,
    }


def test_m2_cumulative_and_shellwise_moments_match_circle_annulus_integrals() -> None:
    """(M2)/(M3b) moments include every current component and analytic annuli."""
    volume_map, samples = _circular_toroidal_surrogate(96, 6)
    edges = np.linspace(0.0, 1.0, 9)
    result = mollified_shell_current_moments(volume_map, samples, edges)
    expected = _analytic_cumulative(edges)

    for name in ("parallel", "diamagnetic", "regularizing", "total"):
        assert result.cumulative(name) == pytest.approx(expected[name], abs=2.0e-3)
        assert result.shellwise(name) == pytest.approx(np.diff(expected[name]), abs=2.0e-3)


def test_shell_current_moments_refine_and_match_recorded_rate_table() -> None:
    """Mollified annular (M3b) moments converge under h/quadrature refinement."""
    with _RATE_TABLE.open() as stream:
        recorded = list(csv.DictReader(stream))
    errors: list[float] = []
    for row in recorded:
        cells = int(row["radial_cells"])
        order = int(row["quadrature_order"])
        volume_map, samples = _circular_toroidal_surrogate(cells, order)
        edges = np.linspace(0.0, 1.0, 5)
        result = mollified_shell_current_moments(volume_map, samples, edges)
        error = float(
            np.max(np.abs(result.cumulative("total") - _analytic_cumulative(edges)["total"]))
        )
        errors.append(error)
        assert error == pytest.approx(float(row["maximum_cumulative_error"]), rel=5.0e-10)
    rates = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert rates == pytest.approx(
        [float(row["adjacent_rate"]) for row in recorded[1:]], rel=5.0e-10
    )
    assert np.all(rates > 1.8)


def test_shell_current_moments_converge_with_quadrature_order_at_fixed_mesh() -> None:
    """Fixed-h moment errors decrease until the spatial mollification error dominates."""
    edges = np.linspace(0.0, 1.0, 5)
    expected = _analytic_cumulative(edges)["total"]
    errors = []
    for order in (1, 2, 3):
        volume_map, samples = _circular_toroidal_surrogate(48, order)
        result = mollified_shell_current_moments(volume_map, samples, edges)
        errors.append(float(np.max(np.abs(result.cumulative("total") - expected))))
    assert errors == pytest.approx(
        [5.598889375701965e-4, 1.578133429118722e-4, 6.173052726854422e-5],
        rel=5.0e-10,
    )
    assert np.all(np.diff(errors) < 0.0)


def test_domain_rescaling_preserves_normalized_current_profile_semantics() -> None:
    """Changing V_omega leaves the same I0(s) when density scales by inverse area."""
    edges = np.linspace(0.0, 1.0, 9)
    unit_map, unit_samples = _circular_toroidal_surrogate(96, 6, radius=1.0)
    scaled_map, scaled_samples = _circular_toroidal_surrogate(96, 6, radius=2.75)
    unit = mollified_shell_current_moments(unit_map, unit_samples, edges)
    scaled = mollified_shell_current_moments(scaled_map, scaled_samples, edges)
    assert scaled.cumulative("total") == pytest.approx(unit.cumulative("total"), abs=2.0e-12)
    assert scaled.normalized_volume == pytest.approx(unit.normalized_volume, abs=2.0e-12)


def test_shell_edges_must_be_a_resolved_partition_of_normalized_volume() -> None:
    """M3b shell APIs reject ambiguous endpoints and under-resolved mollified shells."""
    volume_map, samples = _circular_toroidal_surrogate(24, 4)
    with pytest.raises(ValueError, match=r"exactly.*\[0, 1\]"):
        mollified_shell_current_moments(volume_map, samples, [0.0, 0.5, 2.0])
    with pytest.raises(ValueError, match="resolved"):
        mollified_shell_current_moments(volume_map, samples, np.linspace(0.0, 1.0, 4 * 24 + 1))


def test_marginal_local_three_cell_shell_partition_remains_accurate() -> None:
    """The strict local §9.2 guard accepts 15 resolved equal-volume shells."""
    volume_map, samples = _circular_toroidal_surrogate(96, 6)
    edges = np.linspace(0.0, 1.0, 16)
    result = mollified_shell_current_moments(volume_map, samples, edges)
    expected = _analytic_cumulative(edges)["total"]
    assert np.max(np.abs(result.cumulative("total") - expected)) < 3.0e-5


def test_graded_partition_uses_local_not_global_radial_cell_widths() -> None:
    """A resolved graded shell grid is not rejected by an unrelated outer width."""
    volume_map, samples = _circular_toroidal_surrogate(96, 6)
    edges = np.linspace(0.0, 1.0, 19) ** 1.2
    normalized_volume = volume_map.quadrature_normalized_volume
    cell_widths = volume_map.quadrature_normalized_cell_widths
    local_ratios = []
    for left, right in pairwise(edges):
        in_shell = (normalized_volume >= left) & (normalized_volume <= right)
        local_ratios.append((right - left) / float(np.max(cell_widths[in_shell])))
    assert min(local_ratios) > 3.0
    assert float(np.min(np.diff(edges))) / float(np.max(cell_widths)) < 2.0

    result = mollified_shell_current_moments(volume_map, samples, edges)
    expected = _analytic_cumulative(edges)["total"]
    assert np.max(np.abs(result.cumulative("total") - expected)) < 3.0e-5


def test_shell_guard_resolves_cells_and_mollifier_independently() -> None:
    """Neither a small nor a large configured mollifier can bypass the §9.2 guard."""
    narrow_map, narrow_samples = _circular_toroidal_surrogate(96, 6, spatial_width_cells=0.5)
    with pytest.raises(ValueError, match="radial-cell"):
        mollified_shell_current_moments(narrow_map, narrow_samples, np.linspace(0.0, 1.0, 17))

    default_map, default_samples = _circular_toroidal_surrogate(96, 6, spatial_width_cells=1.5)
    edges = np.linspace(0.0, 1.0, 16)
    result = mollified_shell_current_moments(default_map, default_samples, edges)
    expected = _analytic_cumulative(edges)["total"]
    assert np.max(np.abs(result.cumulative("total") - expected)) < 5.0e-5

    wide_map, wide_samples = _circular_toroidal_surrogate(96, 6, spatial_width_cells=2.0)
    with pytest.raises(ValueError, match="mollifier"):
        mollified_shell_current_moments(wide_map, wide_samples, edges)


def test_current_samples_must_retain_the_volume_map_quadrature_order() -> None:
    """A reordered M2 field cannot silently pair currents with the wrong s samples."""
    volume_map, samples = _circular_toroidal_surrogate(48, 6)
    reordered = replace(
        samples,
        normalized_volume=np.asarray(samples.normalized_volume)[::-1],
        parallel=np.asarray(samples.parallel)[::-1],
        diamagnetic=np.asarray(samples.diamagnetic)[::-1],
        regularizing=np.asarray(samples.regularizing)[::-1],
    )
    with pytest.raises(ValueError, match="quadrature ordering"):
        mollified_shell_current_moments(volume_map, reordered, np.linspace(0.0, 1.0, 5))
