"""Manufactured circle/annulus verification of mollified M2 shell-current moments."""

from __future__ import annotations

import csv
from itertools import pairwise
from math import pi
from pathlib import Path

import numpy as np
import pytest

from remec.current_moments import M2ToroidalCurrentSamples, mollified_shell_current_moments
from remec.level_set import MollifiedVolumeMap, QuadratureLevelSetData

_RATE_TABLE = Path(__file__).with_name("shell_current_moment_rates.csv")


def _circular_toroidal_surrogate(
    radial_cells: int, quadrature_order: int, *, radius: float = 1.0
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
    samples = M2ToroidalCurrentSamples(
        parallel=inverse_area * (2.0 + normalized_volume),
        diamagnetic=inverse_area * (-0.5 * normalized_volume),
        regularizing=inverse_area * (0.25 * (1.0 - normalized_volume)),
    )
    volume_map = MollifiedVolumeMap.build(
        QuadratureLevelSetData(
            values=chi,
            gradient_magnitudes=2.0 * radius_samples,
            weights=volume_weights,
            element_sizes=np.full_like(chi, radius / radial_cells),
        ),
        spatial_width_cells=1.0,
        levels=257,
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
        assert result.cumulative(name)[0] == 0.0
        assert np.sum(result.shellwise(name)) == pytest.approx(
            result.cumulative(name)[-1], abs=2.0e-14
        )
    assert result.cumulative("total") == pytest.approx(
        result.cumulative("parallel")
        + result.cumulative("diamagnetic")
        + result.cumulative("regularizing"),
        abs=2.0e-14,
    )


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
