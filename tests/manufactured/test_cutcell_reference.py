"""Manufactured cut-cell contracts for the optional sharp ``V_chi`` reference."""

from __future__ import annotations

import csv
from math import pi
from pathlib import Path

import ngsolve as ng
import numpy as np
import pytest
from ngsolve.meshes import MakeStructured2DMesh

pytest.importorskip("xfem", reason="the cut-cell reference is an optional xfem extra")

from remec.fem.cutcell_optional import CutCellVolumeReference
from remec.level_set import MollifiedVolumeMap
from remec.profiles import extract_ngsolve_quadrature

_MANUFACTURED_DIRECTORY = Path(__file__).parent
_RADIUS_SQUARED = 0.6**2


def _circle_level_set() -> ng.CoefficientFunction:
    """Return a circle centered at the origin in the unit-square quadrant."""
    return _RADIUS_SQUARED - ng.x**2 - ng.y**2


def _exact_quarter_circle_volume(level: float) -> float:
    """Return ``|{chi > level}|`` for this manufactured level set."""
    return pi * max(_RADIUS_SQUARED - level, 0.0) / 4.0


def test_cutcell_reference_resolves_sharp_circle_volumes_and_monotonicity() -> None:
    """§12.4 evaluates ``V_chi = int H(chi-level)`` by high-order sub-cell quadrature."""
    mesh = MakeStructured2DMesh(quads=False, nx=16, ny=16)
    reference = CutCellVolumeReference(mesh, _circle_level_set(), geometry_order=3)
    levels = np.array([0.0, 0.1, 0.25, _RADIUS_SQUARED])

    volumes = np.asarray(reference.volume(levels), dtype=float)

    assert volumes == pytest.approx(
        [_exact_quarter_circle_volume(float(level)) for level in levels], abs=4.0e-6
    )
    assert np.all(np.diff(volumes) <= 0.0)
    assert reference.total_volume == pytest.approx(1.0, abs=1.0e-12)


def test_cutcell_circle_converges_at_high_order_and_calibrates_mollified_map() -> None:
    """§12.4 catches a low-order cut geometry and quantifies `(mollified_V)` error."""
    with (_MANUFACTURED_DIRECTORY / "cutcell_circle_rates.csv").open() as stream:
        expected_rows = list(csv.DictReader(stream))

    cutcell_errors: list[float] = []
    mollified_differences: list[float] = []
    for row in expected_rows:
        mesh = MakeStructured2DMesh(
            quads=False, nx=int(row["subdivisions"]), ny=int(row["subdivisions"])
        )
        chi = _circle_level_set()
        cutcell_volume = float(CutCellVolumeReference(mesh, chi, geometry_order=3).volume(0.0))
        gradient = ng.CoefficientFunction((chi.Diff(ng.x), chi.Diff(ng.y)))
        mollified = MollifiedVolumeMap.build(
            extract_ngsolve_quadrature(mesh, chi, gradient, integration_order=6),
            spatial_width_cells=1.0,
            levels=129,
        )
        cutcell_error = abs(cutcell_volume - _exact_quarter_circle_volume(0.0))
        mollified_difference = abs(float(mollified.volume(0.0)) - cutcell_volume)
        cutcell_errors.append(cutcell_error)
        mollified_differences.append(mollified_difference)

        assert cutcell_volume == pytest.approx(float(row["cutcell_volume"]), rel=2.0e-10)
        assert cutcell_error == pytest.approx(float(row["cutcell_absolute_error"]), rel=2.0e-10)
        assert mollified_difference == pytest.approx(
            float(row["mollified_reference_difference"]), rel=2.0e-10
        )

    cutcell_rates = np.log2(np.asarray(cutcell_errors[:-1]) / np.asarray(cutcell_errors[1:]))
    mollified_rates = np.log2(
        np.asarray(mollified_differences[:-1]) / np.asarray(mollified_differences[1:])
    )
    # The table records the local measurement; rates derived from near-roundoff
    # finest-grid errors vary in their final bits across supported wheel builds.
    # The cross-platform physics contract is the order threshold below.
    assert np.all(cutcell_rates > 3.5)
    assert np.all(mollified_rates > 1.9)
