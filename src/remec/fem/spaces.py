"""Compatible finite-element space construction for the magnetic complex."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DeRhamSequence:
    """NGSolve spaces for the discrete complex supporting note equation (M1).

    On affine tetrahedra the exact polynomial sequence is
    ``H1(p+1) --grad--> HCurl(p) --curl--> HDiv(p-1) --div--> L2(p-2)``,
    with the two downstream orders floored at zero.  It makes
    ``div(curl(A_h)) = 0`` algebraic on affine and curved tetrahedra, as required
    by (M1).  On curved elements ordinary NGSolve ``L2`` is not the density-mapped
    terminal space, so use the direct composition for the magnetic invariant.
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

    Implements the space pairing
    ``H1(p+1) --grad--> HCurl(p) --curl--> HDiv(max(p-1, 0))
    --div--> L2(max(p-2, 0))`` on affine tetrahedra.  The mapped HCurl/HDiv
    composition also gives ``div(curl(A_h)) = 0`` on curved tetrahedra.
    """
    if isinstance(order, bool) or not isinstance(order, int):
        raise TypeError("order must be an integer")
    if order < 0:
        raise ValueError("order must be non-negative")

    import ngsolve as ng  # type: ignore[import-untyped]

    if getattr(mesh, "dim", None) != 3:
        raise ValueError("the tetrahedral de Rham sequence requires a three-dimensional mesh")
    try:
        volume_elements = tuple(mesh.Elements(ng.VOL))
    except AttributeError as error:
        raise TypeError("mesh must be an NGSolve mesh") from error
    if not volume_elements or any(element.type != ng.ET.TET for element in volume_elements):
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
