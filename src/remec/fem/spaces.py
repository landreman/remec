"""Compatible finite-element space construction for the magnetic complex."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_MAX_VALIDATED_BASE_ORDER = 5


@dataclass(frozen=True, slots=True)
class DeRhamSequence:
    """NGSolve spaces for the discrete complex supporting note equation (M1).

    On affine tetrahedra the exact polynomial sequence is
    ``H1(p+1) --grad--> HCurl(p) --curl--> HDiv(p-1) --div--> L2(p-2)``,
    with the two downstream orders floored at zero.  It makes
    ``div(curl(A_h)) = 0`` algebraic on affine and curved tetrahedra, as required
    by (M1). On curved elements ordinary NGSolve ``L2`` is not the density-mapped
    strong image of a general HDiv divergence, but its paired weak constraint remains
    pointwise coercive by ADR 0005. Verify the magnetic invariant by projecting
    ``curl(A_h)`` into the paired HDiv space and applying ``ng.div`` to that HDiv
    GridFunction.
    """

    base_order: int
    h1_order: int
    hcurl_order: int
    hdiv_order: int
    l2_order: int
    h1: Any
    hcurl: Any
    hdiv: Any
    l2: Any


def make_tetrahedral_de_rham_sequence(mesh: Any, *, order: int) -> DeRhamSequence:
    """Build the tetrahedral discrete de Rham complex used to preserve (M1).

    ``order`` is the HCurl/base order ``p``, not the independent H1 order used for
    ``chi`` and ``utilde`` elsewhere in DESIGN section 7.1. Base orders 0--5 are
    covered by the affine and curved manufactured verification tables.

    Implements the space pairing
    ``H1(p+1) --grad--> HCurl(p) --curl--> HDiv(max(p-1, 0))
    --div--> L2(max(p-2, 0))`` on affine tetrahedra.  The mapped HCurl/HDiv
    composition also gives ``div(curl(A_h)) = 0`` on curved tetrahedra.
    """
    if isinstance(order, bool) or not isinstance(order, int):
        raise TypeError("order must be an integer")
    if order < 0:
        raise ValueError("order must be non-negative")
    if order > _MAX_VALIDATED_BASE_ORDER:
        raise ValueError(
            f"order must not exceed the validated base order {_MAX_VALIDATED_BASE_ORDER}"
        )

    import ngsolve as ng  # type: ignore[import-untyped]

    try:
        mesh_dimension = mesh.dim
        volume_elements = iter(mesh.Elements(ng.VOL))
    except AttributeError as error:
        raise TypeError("mesh must be an NGSolve mesh") from error
    if mesh_dimension != 3:
        raise ValueError("the tetrahedral de Rham sequence requires a three-dimensional mesh")

    has_volume_elements = False
    for element in volume_elements:
        has_volume_elements = True
        if element.type != ng.ET.TET:
            raise ValueError("the tetrahedral de Rham sequence requires only tetrahedral elements")
    if not has_volume_elements:
        raise ValueError("the tetrahedral de Rham sequence requires only tetrahedral elements")

    h1_order = order + 1
    hcurl_order = order
    hdiv_order = max(order - 1, 0)
    l2_order = max(order - 2, 0)
    return DeRhamSequence(
        base_order=order,
        h1_order=h1_order,
        hcurl_order=hcurl_order,
        hdiv_order=hdiv_order,
        l2_order=l2_order,
        h1=ng.H1(mesh, order=h1_order),
        hcurl=ng.HCurl(mesh, order=hcurl_order),
        hdiv=ng.HDiv(mesh, order=hdiv_order),
        l2=ng.L2(mesh, order=l2_order),
    )
