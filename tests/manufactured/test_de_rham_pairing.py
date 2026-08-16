"""Manufactured discrete de Rham-sequence tests supporting note equation (M1)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import ngsolve as ng
import numpy as np
import pytest
from netgen.occ import OCCGeometry, Pnt, Sphere
from ngsolve.meshes import MakeStructured3DMesh

from remec.fem.spaces import make_tetrahedral_de_rham_sequence

_BASE_ORDERS = (0, 1, 2, 3, 4, 5)
_SUBDIVISIONS = (1, 2)
_TABLE_PATH = Path(__file__).with_name("de_rham_pairing.csv")
_CURVED_TABLE_PATH = Path(__file__).with_name("de_rham_curved.csv")
_DEFECT_COLUMNS = (
    "grad_mapping_defect",
    "curl_mapping_defect",
    "div_mapping_defect",
    "curl_grad_defect",
    "div_curl_defect",
)


@dataclass(frozen=True, slots=True)
class _Projection:
    field: ng.GridFunction
    relative_defect: float
    source_norm: float


def _recorded_rows() -> dict[tuple[int, int], dict[str, str]]:
    with _TABLE_PATH.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    indexed = {(int(row["subdivisions"]), int(row["base_order"])): row for row in rows}
    assert len(indexed) == len(rows), "verification table contains duplicate mesh/order rows"
    return indexed


def _recorded_curved_row() -> dict[str, str]:
    with _CURVED_TABLE_PATH.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1, "curved verification table must contain exactly one row"
    return rows[0]


def _roundoff_gate(base_order: int) -> float:
    """Scale roundoff allowance with polynomial degree and basis conditioning."""
    return float(32.0 * np.finfo(float).eps * (base_order + 2) ** 3)


def _recorded_defect_gate(recorded: float) -> float:
    """Allow backend-level roundoff drift while retaining row-specific regression data."""
    return max(8.0 * recorded, 64.0 * np.finfo(float).eps)


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
    structured tetrahedra at the validated base orders 0--5.  L2 projection verifies
    the three mappings

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
    recorded = _recorded_rows()[(subdivisions, base_order)]
    assert recorded["element_family"] == "tetrahedron"
    expected_orders = tuple(
        int(recorded[column]) for column in ("h1_order", "hcurl_order", "hdiv_order", "l2_order")
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
    recorded_dimensions = tuple(
        int(recorded[column]) for column in ("h1_dofs", "hcurl_dofs", "hdiv_dofs", "l2_dofs")
    )
    assert dimensions == recorded_dimensions
    euler_characteristic = dimensions[0] - dimensions[1] + dimensions[2] - dimensions[3]
    assert euler_characteristic == int(recorded["euler_characteristic"]) == 1
    assert mesh.ne == int(recorded["elements"])

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

    defects = {
        "grad_mapping_defect": gradient.relative_defect,
        "curl_mapping_defect": curl.relative_defect,
        "div_mapping_defect": divergence.relative_defect,
        "curl_grad_defect": _relative_norm(
            ng.curl(gradient.field),
            mesh,
            scale=gradient.source_norm,
            order=integration_order,
        ),
        "div_curl_defect": _relative_norm(
            ng.div(curl.field),
            mesh,
            scale=curl.source_norm,
            order=integration_order,
        ),
    }
    for column in _DEFECT_COLUMNS:
        actual = defects[column]
        table_value = float(recorded[column])
        assert actual < _roundoff_gate(base_order), (column, actual)
        assert actual <= _recorded_defect_gate(table_value), (
            column,
            actual,
            table_value,
        )


def test_curved_tetrahedral_magnetic_subcomplex_composes_at_roundoff() -> None:
    """The magnetic half of (M1) remains exact on curved tetrahedra.

    ADR 0005 distinguishes the exact HCurl-to-HDiv composition from ordinary-L2
    containment of a general curved HDiv divergence. This test verifies the magnetic
    mapping and then applies ``ng.div`` to the projected HDiv GridFunction. A random
    HDiv negative control proves the diagnostic is not a vacuous symbolic derivative.
    """
    recorded = _recorded_curved_row()
    assert recorded["geometry"] == "occ_ball"
    geometry = OCCGeometry(Sphere(Pnt(0.0, 0.0, 0.0), 1.0))
    mesh = ng.Mesh(geometry.GenerateMesh(maxh=0.9))
    geometry_order = int(recorded["geometry_order"])
    mesh.Curve(geometry_order)

    base_order = int(recorded["base_order"])
    sequence = make_tetrahedral_de_rham_sequence(mesh, order=base_order)
    assert mesh.ne == int(recorded["elements"])
    assert sequence.hcurl_order == int(recorded["hcurl_order"])
    assert sequence.hdiv_order == int(recorded["hdiv_order"])
    assert sequence.hcurl.ndof == int(recorded["hcurl_dofs"])
    assert sequence.hdiv.ndof == int(recorded["hdiv_dofs"])
    vector_potential = _random_field(sequence.hcurl, seed=4199)
    magnetic_field = ng.curl(vector_potential)
    integration_order = 2 * max(base_order, geometry_order) + 6
    curl = _mass_project(
        magnetic_field,
        sequence.hdiv,
        mesh,
        integration_order=integration_order,
    )

    divergence_defect = _relative_norm(
        ng.div(curl.field),
        mesh,
        scale=curl.source_norm,
        order=integration_order,
    )
    divergent_control = _random_field(sequence.hdiv, seed=4200)
    control_norm = float(
        ng.sqrt(
            ng.Integrate(
                ng.InnerProduct(divergent_control, divergent_control),
                mesh,
                order=integration_order,
            )
        )
    )
    control_divergence_defect = _relative_norm(
        ng.div(divergent_control),
        mesh,
        scale=control_norm,
        order=integration_order,
    )
    recorded_curl_defect = float(recorded["curl_mapping_defect"])
    recorded_divergence_defect = float(recorded["div_curl_defect"])
    recorded_control = float(recorded["divergent_control"])
    assert curl.relative_defect < 1.0e-12
    assert curl.relative_defect <= _recorded_defect_gate(recorded_curl_defect)
    assert divergence_defect < 1.0e-12
    assert divergence_defect <= _recorded_defect_gate(recorded_divergence_defect)
    assert control_divergence_defect > 1.0e-2
    assert recorded_control / 8.0 <= control_divergence_defect <= 8.0 * recorded_control


def test_de_rham_pairing_table_covers_the_validated_sweep_exactly() -> None:
    """The checked-in verification table cannot silently omit a tested order or mesh."""
    rows = _recorded_rows()
    assert set(rows) == {
        (subdivisions, base_order) for subdivisions in _SUBDIVISIONS for base_order in _BASE_ORDERS
    }


@pytest.mark.parametrize("bad_order", [-1, 6, True, 1.5])
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


def test_tetrahedral_de_rham_sequence_rejects_non_mesh_objects() -> None:
    """The factory reports a type error before trying to construct NGSolve spaces."""
    with pytest.raises(TypeError, match="NGSolve mesh"):
        make_tetrahedral_de_rham_sequence(object(), order=2)
