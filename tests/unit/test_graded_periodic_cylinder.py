"""ADR 0011 contracts for parameterized radial annulus packing."""

from __future__ import annotations

import pytest

from remec.geometry import GradedAnnulus, PeriodicCylinder3D


def test_graded_annulus_contract_rejects_nonphysical_widths() -> None:
    """Radial packing never creates an unresolved or exterior target silently."""
    with pytest.raises(ValueError):
        GradedAnnulus(radius=0.7, half_width=0.1, maximum_radial_width=0.0)
    with pytest.raises(ValueError):
        GradedAnnulus(radius=0.0, half_width=0.1, maximum_radial_width=0.01)


def test_target_annulus_caps_true_intersecting_cell_radial_projection() -> None:
    """The ADR-0011 mesh realizes the literal six projected-cell-width gate."""
    critical_width = 0.094074
    target = GradedAnnulus(
        radius=0.74339,
        half_width=0.6 * critical_width,
        maximum_radial_width=critical_width / 6.0,
        measurement_half_width=0.5 * critical_width,
    )
    cylinder = PeriodicCylinder3D(geometry_order=4)
    bundle = cylinder.build_graded_mesh(
        (target,), angular_cells=24, axial_cells=8, background_radial_width=0.125
    )

    assert bundle.maximum_target_radial_projection <= target.maximum_radial_width * (1.0 + 1.0e-12)
