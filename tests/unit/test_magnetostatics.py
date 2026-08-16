"""Input-contract tests for the fixed-boundary (M1) magnetic kernel."""

from __future__ import annotations

import ngsolve as ng
import pytest
from ngsolve.meshes import MakeStructured2DMesh, MakeStructured3DMesh

from remec.fem._magnetostatics import solve_gauge_fixed_curl_curl


@pytest.fixture
def tetrahedral_mesh() -> object:
    return MakeStructured3DMesh(hexes=False, nx=1, ny=1, nz=1)


@pytest.mark.parametrize("vacuum_permeability", [0.0, -1.0, float("inf")])
def test_magnetostatic_solve_rejects_invalid_permeability(
    tetrahedral_mesh: object,
    vacuum_permeability: float,
) -> None:
    with pytest.raises(ValueError, match="vacuum_permeability"):
        solve_gauge_fixed_curl_curl(
            tetrahedral_mesh,
            ng.CoefficientFunction((1.0, 0.0, 0.0)),
            base_order=1,
            vacuum_permeability=vacuum_permeability,
        )


@pytest.mark.parametrize("bonus_order", [-1, 1.5, True])
def test_magnetostatic_solve_rejects_invalid_bonus_order(
    tetrahedral_mesh: object,
    bonus_order: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="bonus_integration_order"):
        solve_gauge_fixed_curl_curl(
            tetrahedral_mesh,
            ng.CoefficientFunction((1.0, 0.0, 0.0)),
            base_order=1,
            bonus_integration_order=bonus_order,  # type: ignore[arg-type]
        )


def test_magnetostatic_solve_requires_full_fixed_boundary(tetrahedral_mesh: object) -> None:
    with pytest.raises(ValueError, match="full boundary"):
        solve_gauge_fixed_curl_curl(
            tetrahedral_mesh,
            ng.CoefficientFunction((1.0, 0.0, 0.0)),
            base_order=1,
            boundary="left",
        )


def test_magnetostatic_solve_rejects_scalar_current(tetrahedral_mesh: object) -> None:
    with pytest.raises(ValueError, match="three-component"):
        solve_gauge_fixed_curl_curl(tetrahedral_mesh, ng.x, base_order=1)


def test_magnetostatic_solve_rejects_non_tetrahedral_or_non_3d_mesh() -> None:
    current = ng.CoefficientFunction((1.0, 0.0, 0.0))
    hexahedral_mesh = MakeStructured3DMesh(hexes=True, nx=1, ny=1, nz=1)
    with pytest.raises(ValueError, match="tetrahedral"):
        solve_gauge_fixed_curl_curl(hexahedral_mesh, current, base_order=1)
    with pytest.raises(ValueError, match="three-dimensional"):
        solve_gauge_fixed_curl_curl(MakeStructured2DMesh(nx=1, ny=1), current, base_order=1)
