"""ADR-0009 geometry and periodic-space verification for milestone 6.2."""

from __future__ import annotations

import csv
import sys
from math import log
from pathlib import Path

import ngsolve as ng
import numpy as np
import pytest

from remec.fem.spaces import make_periodic_tetrahedral_de_rham_sequence
from remec.geometry import PeriodicCylinder3D
from remec.geometry.periodic_cylinder import PeriodicCylinderGeometryMetrics

_TABLE_PATH = Path(__file__).with_name("periodic_cylinder_geometry.csv")
_H1_TABLE_PATH = Path(__file__).with_name("periodic_cylinder_h1_rates.csv")
_SCAN_CONFIGURATIONS = ((0, 1), (0, 2), (0, 4), (1, 1), (1, 2), (1, 4))


@pytest.fixture(scope="module")
def geometry_rows() -> dict[tuple[int, int], PeriodicCylinderGeometryMetrics]:
    rows: dict[tuple[int, int], PeriodicCylinderGeometryMetrics] = {}
    for refinements, geometry_order in _SCAN_CONFIGURATIONS:
        cylinder = PeriodicCylinder3D(geometry_order=geometry_order, refinements=refinements)
        bundle = cylinder.build_mesh()
        rows[(refinements, geometry_order)] = cylinder.measure_geometry(bundle)
    return rows


def test_curved_cylinder_geometry_errors_decrease_with_order_and_refinement(
    geometry_rows: dict[tuple[int, int], PeriodicCylinderGeometryMetrics],
) -> None:
    """The checked-in ADR-0009 scan resolves the circular wall systematically."""
    columns = (
        "wall_radius_rms_relative_error",
        "cross_section_area_relative_error",
        "volume_relative_error",
        "boundary_flux_relative_error",
    )
    for refinements in (0, 1):
        for column in columns:
            errors = [getattr(geometry_rows[(refinements, order)], column) for order in (1, 2, 4)]
            assert errors[0] > errors[1] > errors[2], (refinements, column, errors)
    for order in (1, 2, 4):
        coarse = geometry_rows[(0, order)]
        refined = geometry_rows[(1, order)]
        for column in columns:
            assert getattr(refined, column) < getattr(coarse, column), (order, column)
    assert geometry_rows[(0, 4)].maximum_relative_error < 5.0e-5
    assert geometry_rows[(1, 4)].maximum_relative_error < 7.0e-6


def test_mesh_has_named_regions_and_exactly_one_periodic_identification() -> None:
    cylinder = PeriodicCylinder3D(geometry_order=3)
    bundle = cylinder.build_mesh()
    assert bundle.boundary_names == ("wall", "periodic_lower", "periodic_upper")
    assert bundle.periodic_identification_count == 1
    assert bundle._mesh.ngmesh.GetNrIdentifications() == 1
    assert set(bundle._mesh.GetBoundaries()) == set(bundle.boundary_names)
    assert bundle._mesh.ngmesh.GetIdentifications()


def test_geometry_scan_matches_checked_in_platform_table(
    geometry_rows: dict[tuple[int, int], PeriodicCylinderGeometryMetrics],
) -> None:
    """Every measured geometry-order/refinement row has machine-readable provenance."""
    with _TABLE_PATH.open(newline="", encoding="utf-8") as stream:
        recorded = [row for row in csv.DictReader(stream) if row["platform"] == sys.platform]
    indexed = {(int(row["refinements"]), int(row["geometry_order"])): row for row in recorded}
    assert set(indexed) == set(geometry_rows), (
        f"missing {sys.platform} geometry rows; measured values: "
        + repr([(key, geometry_rows[key]) for key in sorted(geometry_rows)])
    )
    for key, actual in geometry_rows.items():
        row = indexed[key]
        assert int(row["elements"]) == actual.elements
        for column in (
            "wall_radius_rms_relative_error",
            "cross_section_area_relative_error",
            "volume_relative_error",
            "boundary_flux_relative_error",
        ):
            assert getattr(actual, column) == pytest.approx(float(row[column]), rel=2.0e-8)


@pytest.fixture(scope="module")
def curved_periodic_mesh() -> object:
    return PeriodicCylinder3D(geometry_order=3).build_mesh()._mesh


def _mass_project(space: object, source: object, *, inverse: str) -> ng.GridFunction:
    trial, test = space.TnT()
    mass = ng.BilinearForm(space)
    mass += ng.InnerProduct(trial, test) * ng.dx
    load = ng.LinearForm(space)
    load += ng.InnerProduct(source, test) * ng.dx
    mass.Assemble()
    load.Assemble()
    result = ng.GridFunction(space)
    result.vec.data = mass.mat.Inverse(space.FreeDofs(), inverse=inverse) * load.vec
    return result


def test_periodic_scalar_manufactured_solution_has_expected_h_rate() -> None:
    """Periodic H1(2) reaches third-order L2 convergence for ``sin(z/R0)``."""
    measured: list[tuple[int, int, float, float]] = []
    for refinements in (0, 1):
        cylinder = PeriodicCylinder3D(geometry_order=2, refinements=refinements)
        mesh = cylinder.build_mesh()._mesh
        space = ng.Periodic(ng.H1(mesh, order=2))
        exact = ng.sin(ng.z / cylinder.major_radius)
        result = _mass_project(space, exact, inverse="sparsecholesky")
        error = float(ng.sqrt(ng.Integrate((result - exact) ** 2, mesh, order=12)))
        volume = float(ng.Integrate(1.0, mesh, order=10))
        h_eff = (volume / mesh.ne) ** (1.0 / 3.0)
        measured.append((refinements, mesh.ne, h_eff, error))
    coarse, refined = measured
    measured_rate = log(coarse[3] / refined[3]) / log(coarse[2] / refined[2])
    assert measured_rate > 2.8

    with _H1_TABLE_PATH.open(newline="", encoding="utf-8") as stream:
        recorded = [row for row in csv.DictReader(stream) if row["platform"] == sys.platform]
    assert len(recorded) == 2, f"missing {sys.platform} periodic H1 rows: {measured!r}"
    actual_rows: tuple[tuple[int, int, float, float, float | None], ...] = (
        (*coarse, None),
        (*refined, measured_rate),
    )
    for row, actual in zip(recorded, actual_rows, strict=True):
        refinements, elements, h_eff, error, rate = actual
        assert int(row["refinements"]) == refinements
        assert int(row["elements"]) == elements
        assert float(row["h_eff"]) == pytest.approx(h_eff, rel=2.0e-8)
        assert float(row["l2_error"]) == pytest.approx(error, rel=2.0e-8)
        if rate is not None:
            assert float(row["finest_pair_rate"]) == pytest.approx(rate, rel=2.0e-8)


@pytest.mark.parametrize("inverse", ["sparsecholesky", "umfpack"])
def test_selected_direct_solvers_support_periodic_h1_wrapper(
    curved_periodic_mesh: object, inverse: str
) -> None:
    """Both selected direct backends solve through NGSolve's periodic H1 wrapper."""
    sequence = make_periodic_tetrahedral_de_rham_sequence(curved_periodic_mesh, order=1)
    source = 1.0 + ng.x + 0.1 * ng.sin(ng.z)
    result = _mass_project(sequence.h1, source, inverse=inverse)
    lower = float(result(curved_periodic_mesh(0.2, -0.1, 0.0)))
    upper = float(result(curved_periodic_mesh(0.2, -0.1, 2.0 * np.pi)))
    assert lower == pytest.approx(upper, abs=2.0e-12)


@pytest.mark.parametrize("space_name", ["hcurl", "hdiv"])
@pytest.mark.parametrize("component", [0, 1, 2])
def test_periodic_vector_spaces_preserve_all_mean_flux_components(
    curved_periodic_mesh: object, space_name: str, component: int
) -> None:
    """Periodic H(curl)/H(div) preserve x, y, and z constant fluxes at high order."""
    # A physical constant is Piola-mapped on curved elements. Geometry order 3
    # therefore needs HCurl(3) and HDiv(4) to contain all three constants exactly;
    # lower HDiv orders are covered by the convergent (M1) scan, not treated as exact.
    if space_name == "hcurl":
        space = ng.Periodic(ng.HCurl(curved_periodic_mesh, order=3))
    else:
        space = ng.Periodic(ng.HDiv(curved_periodic_mesh, order=4))
    values = [0.0, 0.0, 0.0]
    values[component] = 1.0
    source = ng.CoefficientFunction(tuple(values))
    result = _mass_project(space, source, inverse="sparsecholesky")
    error = float(
        ng.sqrt(
            ng.Integrate(
                ng.InnerProduct(result - source, result - source),
                curved_periodic_mesh,
                order=10,
            )
        )
    )
    assert error < 1.0e-11
    lower = np.asarray(result(curved_periodic_mesh(0.2, -0.1, 0.0)), dtype=float)
    upper = np.asarray(result(curved_periodic_mesh(0.2, -0.1, 2.0 * np.pi)), dtype=float)
    np.testing.assert_allclose(lower, upper, atol=2.0e-11)


def test_periodic_cylinder_reuses_normalized_m1_harmonic_flux() -> None:
    """The axial harmonic has unit toroidal flux, wall tangency, and zero divergence."""
    cylinder = PeriodicCylinder3D(geometry_order=4)
    bundle = cylinder.build_mesh()
    mesh = bundle._mesh
    harmonic = cylinder.harmonic_basis(bundle)[0]
    normal = ng.specialcf.normal(3)
    flux = float(
        ng.Integrate(
            harmonic * normal,
            mesh,
            ng.BND,
            definedon=mesh.Boundaries("periodic_upper"),
            order=14,
        )
    )
    wall_normal = float(
        ng.sqrt(
            ng.Integrate(
                (harmonic * normal) ** 2,
                mesh,
                ng.BND,
                definedon=mesh.Boundaries("wall"),
                order=14,
            )
        )
    )
    assert flux == pytest.approx(1.0, rel=5.0e-6)
    # ADR 0009 makes this a measured curved-wall geometry defect, not an exact-zero
    # claim. The geometry-order-4 scan bounds it below the later 6.3 error budget.
    assert wall_normal < 1.0e-4
    divergence = harmonic[0].Diff(ng.x) + harmonic[1].Diff(ng.y) + harmonic[2].Diff(ng.z)
    assert float(ng.Integrate(divergence**2, mesh, order=14)) < 1.0e-24
