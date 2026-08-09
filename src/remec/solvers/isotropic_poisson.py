"""Isotropic reference-potential solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from remec.geometry.slab import Slab2D


@dataclass(frozen=True, slots=True)
class IsotropicPoissonSolution:
    """Discrete solution and mesh for a verified isotropic slab solve."""

    mesh: Any
    field: Any
    polynomial_order: int


def solve_isotropic_poisson(
    slab: Slab2D, *, polynomial_order: int, source: Any
) -> IsotropicPoissonSolution:
    """Solve the isotropic weak form of note equation (M4a).

    For ``κ_parallel = κ_perp = 1``, (M4a) is ``-Δχ = S_ref``. This
    assembles ``∫_Ω ∇v·∇χ dV = ∫_Ω v S_ref dV`` on ``H¹_0(Ω)`` with
    homogeneous Dirichlet data on the boundary of the `Slab2D` unit square.
    """
    if polynomial_order < 1:
        raise ValueError("polynomial_order must be at least one")

    import ngsolve as ng  # type: ignore[import-untyped]
    from netgen.geom2d import SplineGeometry  # type: ignore[import-untyped]

    geometry = SplineGeometry()
    geometry.AddRectangle((0.0, 0.0), (1.0, 1.0), bcs=("left", "right", "bottom", "top"))
    mesh = ng.Mesh(geometry.GenerateMesh(maxh=slab.maxh))
    space = ng.H1(mesh, order=polynomial_order, dirichlet="left|right|bottom|top")
    trial, test = space.TnT()
    bilinear_form = ng.BilinearForm(space)
    bilinear_form += ng.grad(trial) * ng.grad(test) * ng.dx
    linear_form = ng.LinearForm(space)
    linear_form += source * test * ng.dx

    with ng.TaskManager():
        bilinear_form.Assemble()
        linear_form.Assemble()
        field = ng.GridFunction(space)
        field.vec.data = (
            bilinear_form.mat.Inverse(space.FreeDofs(), inverse="sparsecholesky") * linear_form.vec
        )

    return IsotropicPoissonSolution(
        mesh=mesh,
        field=field,
        polynomial_order=polynomial_order,
    )
