"""Public contracts for the ADR-0009 periodic circular cylinder."""

from __future__ import annotations

from math import pi

import pytest

from remec.geometry import PeriodicCylinder3D


@pytest.mark.parametrize("name", ["radius", "major_radius", "max_element_size"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("inf")])
def test_periodic_cylinder_rejects_invalid_positive_parameters(name: str, value: float) -> None:
    arguments = {name: value}
    with pytest.raises(ValueError, match=name):
        PeriodicCylinder3D(**arguments)


@pytest.mark.parametrize("geometry_order", [0, 5, 1.5, True])
def test_periodic_cylinder_rejects_unverified_geometry_order(geometry_order: object) -> None:
    with pytest.raises((TypeError, ValueError), match="geometry_order"):
        PeriodicCylinder3D(geometry_order=geometry_order)  # type: ignore[arg-type]


@pytest.mark.parametrize("refinements", [-1, 1.5, True])
def test_periodic_cylinder_rejects_invalid_refinements(refinements: object) -> None:
    with pytest.raises((TypeError, ValueError), match="refinements"):
        PeriodicCylinder3D(refinements=refinements)  # type: ignore[arg-type]


def test_periodic_cylinder_protocol_records_period_and_named_faces() -> None:
    cylinder = PeriodicCylinder3D(
        radius=0.8,
        major_radius=1.25,
        max_element_size=1.1,
        geometry_order=4,
        refinements=1,
    )
    assert cylinder.periodic_length == pytest.approx(2.5 * pi)
    assert cylinder.boundary_regions() == {
        "wall": "wall",
        "periodic_lower": "periodic_lower",
        "periodic_upper": "periodic_upper",
    }
    assert cylinder.characteristic_length() == pytest.approx(2.5 * pi)
    assert cylinder.metadata() == {
        "geometry": "PeriodicCylinder3D",
        "radius": 0.8,
        "major_radius": 1.25,
        "periodic_length": 2.5 * pi,
        "max_element_size": 1.1,
        "geometry_order": 4,
        "refinements": 1,
        "boundary_regions": cylinder.boundary_regions(),
    }
    assert not hasattr(cylinder, "mesh")
