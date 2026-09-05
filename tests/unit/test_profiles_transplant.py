"""Contracts for note equation (M4b) profile transplantation."""

from __future__ import annotations

import ngsolve as ng
import numpy as np
import pytest
from ngsolve.meshes import MakeStructured2DMesh, MakeStructured3DMesh

from remec import profiles
from remec.level_set import MollifiedVolumeMap, QuadratureLevelSetData
from remec.profiles import (
    AnalyticPressureProfile,
    InvalidProfileError,
    TabulatedPressureProfile,
    TransplantedProfile,
    extract_ngsolve_quadrature,
)


def _quadratic_bspline_family(
    minimum: float, maximum: float, count: int = 8
) -> tuple[callable, ...]:
    """Return an open-knot quadratic B-spline family on a pressure interval."""
    degree = 2
    interior = np.linspace(minimum, maximum, count - degree + 1)[1:-1]
    knots = np.concatenate((np.full(degree + 1, minimum), interior, np.full(degree + 1, maximum)))

    def basis(index: int, order: int, pressure: np.ndarray) -> np.ndarray:
        if order == 0:
            return ((knots[index] <= pressure) & (pressure < knots[index + 1])).astype(float)
        left_width = knots[index + order] - knots[index]
        right_width = knots[index + order + 1] - knots[index + 1]
        left = np.zeros_like(pressure)
        right = np.zeros_like(pressure)
        if left_width > 0.0:
            left = (pressure - knots[index]) / left_width * basis(index, order - 1, pressure)
        if right_width > 0.0:
            right = (
                (knots[index + order + 1] - pressure)
                / right_width
                * basis(index + 1, order - 1, pressure)
            )
        return left + right

    return tuple(
        lambda pressure, index=index: basis(index, degree, np.asarray(pressure, dtype=float))
        for index in range(len(knots) - degree - 1)
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
        TabulatedPressureProfile([0.0, 0.5, 1.0], [1.0, 1.1, 0.0])

    profile = TabulatedPressureProfile([0.0, 0.5, 1.0], [1.0, 0.4, 0.0])
    with pytest.raises(InvalidProfileError, match="edge value"):
        profile.validate(edge_value=0.1)

    with pytest.raises(InvalidProfileError, match="edge value"):
        TransplantedProfile(_unit_interval_volume_map(), profile, edge_pressure=0.1)


def test_analytic_profile_rejects_an_inconsistent_derivative() -> None:
    """§12.6 must not inherit an unchecked analytic ``p_0'`` into its JVP."""
    profile = AnalyticPressureProfile(
        value_function=lambda volume: 1.0 - np.asarray(volume, dtype=float) ** 2,
        derivative_function=lambda volume: -np.ones_like(np.asarray(volume, dtype=float)),
    )
    with pytest.raises(InvalidProfileError, match="derivative disagrees"):
        TransplantedProfile(_unit_interval_volume_map(), profile)


def test_analytic_profile_accepts_a_resolved_sharp_edge_transition() -> None:
    """The derivative guard distinguishes a sharp valid profile from a wrong derivative."""

    def value(volume: float | np.ndarray) -> np.ndarray:
        return 0.5 * (1.0 - np.tanh((np.asarray(volume, dtype=float) - 0.8) / 0.01))

    def derivative(volume: float | np.ndarray) -> np.ndarray:
        return -50.0 / np.cosh((np.asarray(volume, dtype=float) - 0.8) / 0.01) ** 2

    TransplantedProfile(_unit_interval_volume_map(), AnalyticPressureProfile(value, derivative))
    with pytest.raises(InvalidProfileError, match="derivative disagrees"):
        TransplantedProfile(
            _unit_interval_volume_map(),
            AnalyticPressureProfile(value, lambda volume: 2.0 * derivative(volume)),
        )


def test_transplant_realizes_enclosed_volume_and_pressure_bounds() -> None:
    """(M4b) realizes ``p_0(V)`` and confines pressure to the profile range."""
    volume_map = _unit_interval_volume_map()
    profile = AnalyticPressureProfile(
        value_function=lambda volume: 2.0 - volume,
        derivative_function=lambda volume: -np.ones_like(np.asarray(volume, dtype=float)),
    )
    transplant = TransplantedProfile(volume_map, profile)

    chi = (np.arange(4096, dtype=float) + 0.5) / 4096.0
    weights = np.full_like(chi, 1.0 / len(chi))
    target_volumes = np.linspace(0.05, 0.95, 7)
    target_pressures = profile.value(target_volumes)
    assert transplant.enclosed_volume(chi, weights, target_pressures) == pytest.approx(
        target_volumes, abs=2.0e-3
    )
    pressure = transplant.pressure(np.linspace(0.0, 1.0, 101))
    assert np.all(np.diff(pressure) >= 0.0)
    assert np.min(pressure) >= profile.value(1.0)
    assert np.max(pressure) <= profile.value(0.0)


def test_transplant_satisfies_layer_cake_identity_for_smooth_moments() -> None:
    """The (layercake) moments certify the prescribed volume distribution."""
    volume_map = _unit_interval_volume_map()
    profile = AnalyticPressureProfile(
        value_function=lambda volume: 1.0 - volume**2,
        derivative_function=lambda volume: -2.0 * np.asarray(volume, dtype=float),
    )
    transplant = TransplantedProfile(volume_map, profile)
    levels = np.linspace(volume_map.minimum_level, volume_map.maximum_level, 10001)
    weights = np.full(len(levels), 1.0 / len(levels))
    moments = transplant.layer_cake_moments(
        levels,
        weights,
        test_functions=_quadratic_bspline_family(0.0, 1.0),
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
    assert float(np.dot(data.weights, data.values)) == pytest.approx(1.0 / 36.0)
    volume_map = MollifiedVolumeMap.build(data, levels=129)
    profile = TabulatedPressureProfile([0.0, 0.5, 1.0], [1.0, 0.5, 0.0])
    transplant = TransplantedProfile(volume_map, profile)
    pressure_cf = transplant.as_ngsolve_coefficient(chi)

    center_pressure = float(pressure_cf(mesh(0.5, 0.5)))
    edge_pressure = float(pressure_cf(mesh(0.0, 0.5)))
    assert center_pressure > edge_pressure
    assert edge_pressure == pytest.approx(0.0, abs=1.0e-12)
    assert float(ng.Integrate(pressure_cf, mesh, order=8)) == pytest.approx(0.5, abs=3.0e-2)
    probe = np.array([0.01, 0.03, 0.05])
    assert np.asarray([pressure_cf(mesh(x, 0.5)) for x in probe]) == pytest.approx(
        transplant.pressure(probe * (1.0 - probe) * 0.25), abs=1.0e-5
    )


def test_ngsolve_quadrature_extracts_geometry_in_one_batched_evaluation() -> None:
    """The (mollified_V) extraction avoids a Python transformation per sample."""
    mesh = MakeStructured3DMesh(hexes=False, nx=2, ny=2, nz=2)

    class TrafoCountingMesh:
        def __init__(self, wrapped: ng.Mesh) -> None:
            self.wrapped = wrapped
            self.dim = wrapped.dim
            self.get_trafo_calls = 0

        def Elements(self, *args: object) -> object:
            return self.wrapped.Elements(*args)

        def GetTrafo(self, *args: object) -> object:
            self.get_trafo_calls += 1
            return self.wrapped.GetTrafo(*args)

        def MapToAllElements(self, *args: object) -> object:
            return self.wrapped.MapToAllElements(*args)

    counting_mesh = TrafoCountingMesh(mesh)
    data = extract_ngsolve_quadrature(
        counting_mesh,  # type: ignore[arg-type]
        ng.x + 2.0 * ng.y + 3.0 * ng.z,
        ng.CoefficientFunction((1.0, 2.0, 3.0)),
        integration_order=3,
        element_size_mode="level-set-normal",
    )

    assert counting_mesh.get_trafo_calls == 0
    assert data.total_volume == pytest.approx(1.0)
    assert float(np.dot(data.weights, data.values)) == pytest.approx(3.0)
    np.testing.assert_allclose(data.gradient_magnitudes, np.sqrt(14.0))
    assert np.all(data.element_sizes > 0.0)


def test_level_set_normal_metric_width_tracks_only_the_normal_mesh_scale() -> None:
    """ADR 0012 width is rotation invariant and ignores tangential elongation."""
    jacobians = np.asarray(
        [
            np.diag((0.1, 2.0, 5.0)),
            np.diag((0.1, 2.0, 5.0)),
            np.diag((0.1, 2.0, 5.0)),
            2.0 * np.eye(3),
        ]
    )
    gradients = np.asarray(((1.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, -4.0), (1.0, 2.0, 3.0)))
    fallback = np.full(4, 99.0)

    sizes, fallback_count = profiles._level_set_normal_metric_sizes(jacobians, gradients, fallback)

    np.testing.assert_allclose(sizes, (0.1, 2.0, 5.0, 2.0), rtol=2.0e-15)
    assert fallback_count == 0

    angle = 0.61
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    rotated_sizes, _ = profiles._level_set_normal_metric_sizes(
        np.einsum("ij,njk->nik", rotation, jacobians),
        gradients @ rotation.T,
        fallback,
    )
    np.testing.assert_allclose(rotated_sizes, sizes, rtol=2.0e-14)


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_batched_jacobian_measures_match_numpy_determinants(dimension: int) -> None:
    """Batched quadrature geometry retains ``|det J|`` including orientation."""
    generator = np.random.default_rng(20260904 + dimension)
    jacobians = generator.normal(size=(31, dimension, dimension))
    jacobians[::2, 0, :] *= -1.0

    measures = profiles._absolute_jacobian_determinants(jacobians)

    np.testing.assert_allclose(
        measures,
        np.abs(np.linalg.det(jacobians)),
        rtol=8.0 * np.finfo(float).eps,
        atol=8.0 * np.finfo(float).eps,
    )


def test_normal_metric_quadrature_extraction_reports_critical_point_fallback() -> None:
    """ADR 0012 falls back to determinant size only where the level-set normal vanishes."""
    mesh = MakeStructured3DMesh(hexes=False, nx=1, ny=1, nz=1)
    chi = ng.x
    data = extract_ngsolve_quadrature(
        mesh,
        chi,
        ng.CoefficientFunction((1.0, 0.0, 0.0)),
        integration_order=2,
        element_size_mode="level-set-normal",
    )
    assert data.element_size_mode == "level-set-normal"
    assert data.critical_metric_fallback_count == 0
    assert np.all(data.element_sizes > 0.0)

    critical = extract_ngsolve_quadrature(
        mesh,
        ng.CoefficientFunction(1.0),
        ng.CoefficientFunction((0.0, 0.0, 0.0)),
        integration_order=2,
        element_size_mode="level-set-normal",
    )
    assert critical.critical_metric_fallback_count == critical.values.size
    assert np.all(critical.element_sizes > 0.0)


def test_isotropic_quadrature_extraction_retains_embedded_gradient_components() -> None:
    """A 2D mesh may carry the three-component gradients used by the M3 geometry."""
    mesh = MakeStructured2DMesh(quads=False, nx=2, ny=2)
    data = extract_ngsolve_quadrature(
        mesh,
        ng.x + ng.y,
        ng.CoefficientFunction((1.0, 1.0, 0.0)),
        integration_order=2,
    )
    assert data.values.size == data.gradient_magnitudes.size == data.element_sizes.size
    np.testing.assert_allclose(data.gradient_magnitudes, np.sqrt(2.0))


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
    profile = AnalyticPressureProfile(
        value_function=lambda normalized_volume: 1.0 - normalized_volume,
        derivative_function=lambda normalized_volume: (
            -np.ones_like(np.asarray(normalized_volume, dtype=float))
        ),
    )
    transplant = TransplantedProfile(volume_map, profile)
    pressure = transplant.pressure(values)
    # Eq. (layercake): int p dA = V_omega int_0^1 (1 - s) ds = 2.
    assert float(np.dot(weights, pressure)) == pytest.approx(2.0, abs=2.0e-2)
