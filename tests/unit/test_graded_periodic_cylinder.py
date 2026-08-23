"""ADR 0011 contracts for parameterized radial annulus packing."""

from __future__ import annotations

from itertools import pairwise

import pytest

from remec.geometry import GradedAnnulus, PeriodicCylinder3D


def test_graded_annulus_contract_rejects_nonphysical_widths() -> None:
    """Radial packing never creates an unresolved or exterior target silently."""
    with pytest.raises(ValueError):
        GradedAnnulus(radius=0.7, half_width=0.1, maximum_radial_width=0.0)
    with pytest.raises(ValueError):
        GradedAnnulus(radius=0.0, half_width=0.1, maximum_radial_width=0.01)


def test_target_annulus_is_packed_by_at_least_six_radial_intervals() -> None:
    """The ADR-0011 mesh realizes the literal six-element-width w_c gate."""
    critical_width = 0.094074
    target = GradedAnnulus(
        radius=0.74339,
        half_width=0.5 * critical_width,
        maximum_radial_width=critical_width / 6.0,
    )
    cylinder = PeriodicCylinder3D(geometry_order=4)
    bundle = cylinder.build_graded_mesh(
        (target,), angular_cells=24, axial_cells=8, background_radial_width=0.125
    )

    radial_coordinates = bundle.radial_coordinates
    inside = [
        width
        for lower, upper in pairwise(radial_coordinates)
        if lower < target.radius + target.half_width and upper > target.radius - target.half_width
        for width in (upper - lower,)
    ]
    assert len(inside) >= 6
    assert max(inside) <= target.maximum_radial_width * (1.0 + 1.0e-12)
