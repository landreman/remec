"""Contracts for note equation (M4b) profile transplantation."""

from __future__ import annotations

import ngsolve as ng
import numpy as np
import pytest
from ngsolve.meshes import MakeStructured2DMesh

from remec.level_set import MollifiedVolumeMap, QuadratureLevelSetData
from remec.profiles import (
    AnalyticVolumeProfile,
    InvalidProfileError,
    TabulatedVolumeProfile,
    TransplantedProfile,
    extract_ngsolve_quadrature,
)


def _unit_interval_volume_map() -> MollifiedVolumeMap:
    """Build the analytic ``V_chi(chi)=1-chi`` map from dense quadrature data."""
    values = (np.arange(4096, dtype=float) + 0.5) / 4096.0
    return MollifiedVolumeMap.build(
        QuadratureLevelSetData(
            values=values,
            gradient_magnitudes=np.ones_like(values),
            weights=np.full_like(values, 1.0 / len(values)),
            element_sizes=np.full_like(values, 1.0 / len(values)),
        ),
        spatial_width_cells=1.0,
        levels=257,
    )


def test_tabulated_profile_rejects_nonmonotone_or_wrong_edge_value() -> None:
    """§12.5 requires non-increasing ``p_0`` and ``p_0(V_omega)=p_b``."""
    with pytest.raises(InvalidProfileError, match="non-increasing"):
        TabulatedVolumeProfile(volumes=[0.0, 0.5, 1.0], pressures=[1.0, 1.1, 0.0])

    profile = TabulatedVolumeProfile(volumes=[0.0, 0.5, 1.0], pressures=[1.0, 0.4, 0.0])
    with pytest.raises(InvalidProfileError, match="edge value"):
        profile.validate(total_volume=1.0, edge_value=0.1)


def test_transplant_realizes_enclosed_volume_and_pressure_bounds() -> None:
    """(M4b) realizes ``p_0(V)`` and confines pressure to the profile range."""
    volume_map = _unit_interval_volume_map()
    profile = AnalyticVolumeProfile(
        value_function=lambda volume: 2.0 - volume,
        derivative_function=lambda volume: -np.ones_like(np.asarray(volume, dtype=float)),
    )
    transplant = TransplantedProfile(volume_map, profile)

    target_volumes = np.linspace(0.05, 0.95, 7)
    target_pressures = profile.value(target_volumes)
    assert transplant.enclosed_volume(target_pressures) == pytest.approx(target_volumes, abs=2.0e-3)
    pressure = transplant.pressure(np.linspace(0.0, 1.0, 101))
    assert np.all(np.diff(pressure) >= 0.0)
    assert np.min(pressure) >= profile.value(1.0)
    assert np.max(pressure) <= profile.value(0.0)


def test_transplant_satisfies_layer_cake_identity_for_smooth_moments() -> None:
    """The (layercake) moments certify the prescribed volume distribution."""
    volume_map = _unit_interval_volume_map()
    profile = AnalyticVolumeProfile(
        value_function=lambda volume: 1.0 - volume**2,
        derivative_function=lambda volume: -2.0 * np.asarray(volume, dtype=float),
    )
    transplant = TransplantedProfile(volume_map, profile)
    levels = np.linspace(volume_map.minimum_level, volume_map.maximum_level, 10001)
    weights = np.full(len(levels), 1.0 / len(levels))
    moments = transplant.layer_cake_moments(
        levels,
        weights,
        test_functions=(lambda pressure: pressure, lambda pressure: pressure**2 + 0.3 * pressure),
        quadrature_order=10001,
    )
    assert np.max(np.abs(moments)) < 3.0e-3


def test_ngsolve_quadrature_extraction_and_bspline_composition() -> None:
    """The FEM pass feeds (M4b) and produces a symbolic NGSolve composition."""
    mesh = MakeStructured2DMesh(quads=False, nx=12, ny=12)
    chi = ng.x * (1.0 - ng.x) * ng.y * (1.0 - ng.y)
    gradient = ng.CoefficientFunction((chi.Diff(ng.x), chi.Diff(ng.y)))
    data = extract_ngsolve_quadrature(mesh, chi, gradient, integration_order=6)

    assert data.total_volume == pytest.approx(1.0)
    assert data.values.size > mesh.ne
    assert np.all(data.gradient_magnitudes >= 0.0)
    volume_map = MollifiedVolumeMap.build(data, levels=129)
    profile = TabulatedVolumeProfile(volumes=[0.0, 0.5, 1.0], pressures=[1.0, 0.5, 0.0])
    transplant = TransplantedProfile(volume_map, profile)
    pressure_cf = transplant.as_ngsolve_coefficient(chi)

    center_pressure = float(pressure_cf(mesh(0.5, 0.5)))
    edge_pressure = float(pressure_cf(mesh(0.0, 0.5)))
    assert center_pressure > edge_pressure
    assert edge_pressure == pytest.approx(0.0, abs=1.0e-12)
    assert float(ng.Integrate(pressure_cf, mesh, order=8)) == pytest.approx(0.5, abs=3.0e-2)


def test_circle_transplant_matches_exact_layer_cake_moment() -> None:
    """A radial (M4b) transplant has the analytic first layer-cake moment."""
    nodes, weights_1d = np.polynomial.legendre.leggauss(128)
    x, y = np.meshgrid(nodes, nodes, indexing="ij")
    weights = np.outer(weights_1d, weights_1d).ravel()
    values = 0.6**2 - (x**2 + y**2).ravel()
    gradients = 2.0 * np.sqrt((x**2 + y**2).ravel())
    volume_map = MollifiedVolumeMap.build(
        QuadratureLevelSetData(
            values=values,
            gradient_magnitudes=gradients,
            weights=weights,
            element_sizes=np.full_like(values, 2.0 / 128.0),
        ),
        levels=129,
    )
    profile = AnalyticVolumeProfile(
        value_function=lambda volume: 1.0 - volume / (4.0),
        derivative_function=lambda volume: -np.ones_like(np.asarray(volume, dtype=float)) / 4.0,
    )
    transplant = TransplantedProfile(volume_map, profile)
    pressure = transplant.pressure(values)
    # Eq. (layercake): int p dA = int_0^4 (1 - V/4) dV = 2.
    assert float(np.dot(weights, pressure)) == pytest.approx(2.0, abs=2.0e-2)
