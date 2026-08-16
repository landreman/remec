"""Manufactured discrete de Rham-sequence tests supporting note equation (M1)."""

from __future__ import annotations

from dataclasses import dataclass

import ngsolve as ng
import numpy as np
import pytest
from ngsolve.meshes import MakeStructured3DMesh

from remec.fem.spaces import make_tetrahedral_de_rham_sequence

_BASE_ORDERS = (0, 1, 2, 3)
_SUBDIVISIONS = (1, 2)
_ROUNDOFF_GATE = 1.0e-12


@dataclass(frozen=True, slots=True)
class _Projection:
    field: ng.GridFunction
    relative_defect: float
    source_norm: float


def _random_field(space: object, *, seed: int) -> ng.GridFunction:
    field = ng.GridFunction(space)
    field.vec.FV().NumPy()[:] = np.random.default_rng(seed).standard_normal(space.ndof)
    return field


def _mass_project(
    source: object,
    target: object,
    mesh: object,
    *,
    integration_order: int,
) -> _Projection:
    trial, test = target.TnT()
    mass = ng.BilinearForm(target)
    mass += ng.InnerProduct(trial, test) * ng.dx
    rhs = ng.LinearForm(target)
    rhs += ng.InnerProduct(source, test) * ng.dx
    mass.Assemble()
    rhs.Assemble()

    projected = ng.GridFunction(target)
    projected.vec.data = mass.mat.Inverse(target.FreeDofs(), inverse="sparsecholesky") * rhs.vec
    source_norm = float(
        ng.sqrt(
            ng.Integrate(
                ng.InnerProduct(source, source),
                mesh,
                order=integration_order,
            )
        )
    )
    defect = float(
        ng.sqrt(
            ng.Integrate(
                ng.InnerProduct(source - projected, source - projected),
                mesh,
                order=integration_order,
            )
        )
        / source_norm
    )
    return _Projection(projected, defect, source_norm)


def _relative_norm(field: object, mesh: object, *, scale: float, order: int) -> float:
    return float(ng.sqrt(ng.Integrate(ng.InnerProduct(field, field), mesh, order=order)) / scale)


@pytest.mark.parametrize("subdivisions", _SUBDIVISIONS)
@pytest.mark.parametrize("base_order", _BASE_ORDERS)
def test_tetrahedral_de_rham_sequence_maps_and_composes_at_roundoff(
    subdivisions: int,
    base_order: int,
) -> None:
    """The chosen spaces form the exact affine-tetrahedral chain behind (M1).

    Random coefficients excite every source-space basis direction on 6 and 48
    structured tetrahedra.  L2 projection then verifies the three mappings

    ``H1 --grad--> HCurl --curl--> HDiv --div--> L2``

    independently of the factory internals.  The projected differential is also
    differentiated again to check ``curl(grad(q_h)) = 0`` and
    ``div(curl(A_h)) = 0`` at roundoff, the latter being DESIGN section 5's
    magnetic-divergence invariant for ``B_h = curl(A_h)``.
    """
    mesh = MakeStructured3DMesh(
        hexes=False,
        nx=subdivisions,
        ny=subdivisions,
        nz=subdivisions,
    )
    sequence = make_tetrahedral_de_rham_sequence(mesh, order=base_order)
    expected_orders = (
        base_order + 1,
        base_order,
        max(base_order - 1, 0),
        max(base_order - 2, 0),
    )
    assert (
        sequence.h1_order,
        sequence.hcurl_order,
        sequence.hdiv_order,
        sequence.l2_order,
    ) == expected_orders

    dimensions = tuple(
        space.ndof for space in (sequence.h1, sequence.hcurl, sequence.hdiv, sequence.l2)
    )
    assert dimensions[0] - dimensions[1] + dimensions[2] - dimensions[3] == 1

    seed = 4100 + 100 * subdivisions + base_order
    scalar = _random_field(sequence.h1, seed=seed)
    vector_potential = _random_field(sequence.hcurl, seed=seed + 10)
    flux = _random_field(sequence.hdiv, seed=seed + 20)
    integration_order = 2 * base_order + 6

    gradient = _mass_project(
        ng.grad(scalar),
        sequence.hcurl,
        mesh,
        integration_order=integration_order,
    )
    curl = _mass_project(
        ng.curl(vector_potential),
        sequence.hdiv,
        mesh,
        integration_order=integration_order,
    )
    divergence = _mass_project(
        ng.div(flux),
        sequence.l2,
        mesh,
        integration_order=integration_order,
    )

    assert gradient.relative_defect < _ROUNDOFF_GATE
    assert curl.relative_defect < _ROUNDOFF_GATE
    assert divergence.relative_defect < _ROUNDOFF_GATE
    assert (
        _relative_norm(
            ng.curl(gradient.field),
            mesh,
            scale=gradient.source_norm,
            order=integration_order,
        )
        < _ROUNDOFF_GATE
    )
    assert (
        _relative_norm(
            ng.div(curl.field),
            mesh,
            scale=curl.source_norm,
            order=integration_order,
        )
        < _ROUNDOFF_GATE
    )


@pytest.mark.parametrize("bad_order", [-1, True, 1.5])
def test_tetrahedral_de_rham_sequence_rejects_invalid_orders(bad_order: object) -> None:
    """The base-order contract rejects values NGSolve could reinterpret ambiguously."""
    mesh = MakeStructured3DMesh(hexes=False, nx=1, ny=1, nz=1)
    with pytest.raises((TypeError, ValueError), match="order"):
        make_tetrahedral_de_rham_sequence(mesh, order=bad_order)  # type: ignore[arg-type]


def test_tetrahedral_de_rham_sequence_rejects_tensor_product_elements() -> None:
    """A tetrahedral pairing is not silently reused for a different element family."""
    mesh = MakeStructured3DMesh(hexes=True, nx=1, ny=1, nz=1)
    with pytest.raises(ValueError, match="tetrahedral"):
        make_tetrahedral_de_rham_sequence(mesh, order=2)
