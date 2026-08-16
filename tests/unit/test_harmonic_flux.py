"""Input contracts for the analytic-torus harmonic (M1) field."""

from __future__ import annotations

import pytest

from remec.geometry import AnalyticSolidTorus


@pytest.mark.parametrize(
    ("major_radius", "minor_radius"),
    [(0.0, 0.5), (2.0, 0.0), (1.0, 1.0), (1.0, 1.1), (float("inf"), 0.5)],
)
def test_analytic_torus_rejects_invalid_radii(
    major_radius: float,
    minor_radius: float,
) -> None:
    with pytest.raises(ValueError, match="radius"):
        AnalyticSolidTorus(major_radius=major_radius, minor_radius=minor_radius)


@pytest.mark.parametrize("geometry_order", [0, 5, 1.5, True])
def test_analytic_torus_rejects_unverified_geometry_order(geometry_order: object) -> None:
    with pytest.raises((TypeError, ValueError), match="geometry_order"):
        AnalyticSolidTorus(
            major_radius=2.0,
            minor_radius=0.5,
            geometry_order=geometry_order,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("max_element_size", [0.0, -1.0, float("inf")])
def test_analytic_torus_rejects_invalid_element_size(max_element_size: float) -> None:
    with pytest.raises(ValueError, match="max_element_size"):
        AnalyticSolidTorus(
            major_radius=2.0,
            minor_radius=0.5,
            max_element_size=max_element_size,
        )
