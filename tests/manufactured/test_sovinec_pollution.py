"""Sovinec numerical-pollution regression for note equation (M4a).

"Sovinec" refers to the anisotropic-conduction verification in section 4.2 of C. R. Sovinec
et al., *Journal of Computational Physics* 195 (2004) 355–386,
https://doi.org/10.1016/j.jcp.2003.10.004.
"""

from __future__ import annotations

import csv
from itertools import pairwise
from math import isfinite, pi
from pathlib import Path

import ngsolve as ng
import pytest

from remec.fem._anisotropic_diffusion import (
    measure_sovinec_pollution,
    solve_frozen_field_anisotropic_diffusion,
)
from remec.geometry.slab import Slab2D

_MESH_SIZES = (0.25, 0.125, 0.0625)
_POLYNOMIAL_ORDERS = (1, 2, 3)
_POLLUTION_TABLE = Path(__file__).with_name("sovinec_pollution.csv")


def _recorded_pollution() -> dict[tuple[int, float], dict[str, float]]:
    """Read the checked-in machine-readable pollution measurements."""
    with _POLLUTION_TABLE.open(newline="") as table_file:
        return {
            (int(row["polynomial_order"]), float(row["maxh"])): {
                "elements": float(row["elements"]),
                "central_amplitude": float(row["central_amplitude"]),
                "numerical_to_parallel_ratio": float(row["numerical_to_parallel_ratio"]),
            }
            for row in csv.DictReader(table_file)
        }


def test_sovinec_pollution_decreases_with_order_and_refinement() -> None:
    """(M4a) pollution decreases strictly under every p- and h-refinement."""
    measured: dict[tuple[int, float], float] = {}
    diagnostics = {}
    for polynomial_order in _POLYNOMIAL_ORDERS:
        for maxh in _MESH_SIZES:
            diagnostic = measure_sovinec_pollution(
                Slab2D.unit_square(maxh=maxh),
                polynomial_order=polynomial_order,
            )
            diagnostics[polynomial_order, maxh] = diagnostic
            measured[polynomial_order, maxh] = diagnostic.numerical_to_parallel_ratio

    recorded = _recorded_pollution()
    assert set(recorded) == set(measured)
    for key, diagnostic in diagnostics.items():
        expected = recorded[key]
        assert diagnostic.elements == expected["elements"]
        assert diagnostic.unit_direction_defect_l2_squared < 1.0e-12
        assert diagnostic.source_tangency_l2_squared < 1.0e-12
        assert diagnostic.source_laplacian_eigenvalue == pytest.approx(2.0 * pi**2)
        assert diagnostic.numerical_to_parallel_ratio < 0.2
        assert diagnostic.central_amplitude == pytest.approx(
            expected["central_amplitude"], rel=1.0e-5
        )
        assert diagnostic.numerical_to_parallel_ratio == pytest.approx(
            expected["numerical_to_parallel_ratio"], rel=1.0e-5
        )
        assert diagnostic.central_amplitude > 0.0
        assert diagnostic.free_dof_relative_residual_norm <= 1.0e-6

    for polynomial_order in _POLYNOMIAL_ORDERS:
        by_refinement = [measured[polynomial_order, maxh] for maxh in _MESH_SIZES]
        assert all(fine < coarse for coarse, fine in pairwise(by_refinement))
    for maxh in _MESH_SIZES:
        by_order = [measured[polynomial_order, maxh] for polynomial_order in _POLYNOMIAL_ORDERS]
        assert all(higher < lower for lower, higher in pairwise(by_order))

    base = diagnostics[1, 0.25]
    scaled = measure_sovinec_pollution(
        Slab2D.unit_square(maxh=0.25),
        polynomial_order=1,
        parallel_conductivity=10.0,
        source_amplitude=3.0,
    )
    assert scaled.central_amplitude == pytest.approx(0.3 * base.central_amplitude)
    assert scaled.numerical_perpendicular_diffusivity == pytest.approx(
        10.0 * base.numerical_perpendicular_diffusivity
    )
    assert scaled.numerical_to_parallel_ratio == pytest.approx(base.numerical_to_parallel_ratio)


def test_sovinec_extended_scan_reports_finite_residual() -> None:
    """A permanent pollution scan reports finite solver degradation instead of aborting."""
    diagnostic = measure_sovinec_pollution(
        Slab2D.unit_square(maxh=0.0625),
        polynomial_order=4,
    )

    assert isfinite(diagnostic.free_dof_relative_residual_norm)


def test_sovinec_common_path_preserves_the_bit_exact_reference_solution() -> None:
    """(M4a) the rank-one route must not re-normalize its unit tangent field."""
    slab = Slab2D.unit_square(maxh=0.0625)
    psi = ng.sin(ng.pi * ng.x) * ng.sin(ng.pi * ng.y)
    tangent = ng.CoefficientFunction(
        (
            psi.Diff(ng.y) / ng.sqrt(psi.Diff(ng.x) ** 2 + psi.Diff(ng.y) ** 2),
            -psi.Diff(ng.x) / ng.sqrt(psi.Diff(ng.x) ** 2 + psi.Diff(ng.y) ** 2),
        )
    )
    reference = solve_frozen_field_anisotropic_diffusion(
        slab,
        polynomial_order=3,
        source=psi,
        raw_field=tangent,
        parallel_conductivity=1.0,
        perpendicular_conductivity=0.0,
        field_floor=0.0,
        quadrature_bonus_intorder=6,
        residual_tolerance=None,
        direction_is_normalized=True,
    )
    observed = measure_sovinec_pollution(slab, polynomial_order=3)

    assert observed.central_amplitude == reference.center_value()
