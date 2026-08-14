"""Structured slab geometry contracts used by frozen-kernel verification."""

from __future__ import annotations

import pytest

from remec.geometry.slab import Slab2D


def test_periodic_y_slab_supports_layer_aligned_resolution() -> None:
    """A resonant harmonic can be periodic while x retains 64 normal elements."""
    slab = Slab2D(maxh=1.0 / 16.0, subdivisions=(64, 16), periodic_y=True)
    mesh_bundle = slab.build_mesh()

    assert mesh_bundle.boundary_names == ("right", "left")
    assert mesh_bundle._mesh.ne == 2 * 64 * 16
    assert slab.boundary_regions() == {"right": "right", "left": "left"}
    assert slab.metadata()["subdivisions"] == (64, 16)
    assert slab.metadata()["periodic_y"] is True


@pytest.mark.parametrize("subdivisions", [(0, 4), (4, 0), (-1, 4), (2.5, 4)])
def test_slab_rejects_invalid_explicit_subdivisions(subdivisions: tuple[float, int]) -> None:
    """Explicit structured counts are positive integers in both directions."""
    with pytest.raises(ValueError, match="subdivisions"):
        Slab2D(
            maxh=0.1,
            subdivisions=subdivisions,  # type: ignore[arg-type]
        )
