"""Manufactured convergence test for the isotropic reduction of (M4a)."""

from __future__ import annotations

from math import log2

import ngsolve as ng
import pytest

from remec.geometry.slab import Slab2D
from remec.solvers.isotropic_poisson import solve_isotropic_poisson


@pytest.mark.parametrize("polynomial_order", [1, 2])
def test_isotropic_poisson_manufactured_convergence(polynomial_order: int) -> None:
    """M4a has L² order p+1 and energy order p for a smooth slab solution.

    The manufactured solution is ``χ = sin(πx) sin(πy)`` on ``[0, 1]²`` with
    zero Dirichlet boundary data and ``S_ref = 2π²χ``.  For isotropic unit
    conductivity, (M4a) reduces to ``-Δχ = S_ref``.
    """
    exact = ng.sin(ng.pi * ng.x) * ng.sin(ng.pi * ng.y)
    exact_gradient = ng.CoefficientFunction(
        (
            ng.pi * ng.cos(ng.pi * ng.x) * ng.sin(ng.pi * ng.y),
            ng.pi * ng.sin(ng.pi * ng.x) * ng.cos(ng.pi * ng.y),
        )
    )
    source = 2.0 * ng.pi**2 * exact
    errors: list[tuple[float, float]] = []

    for maxh in (0.35, 0.175):
        solution = solve_isotropic_poisson(
            Slab2D.unit_square(maxh=maxh),
            polynomial_order=polynomial_order,
            source=source,
        )
        l2_error = ng.sqrt(ng.Integrate((solution.field - exact) ** 2, solution.mesh, order=8))
        energy_error = ng.sqrt(
            ng.Integrate(
                ng.InnerProduct(
                    ng.grad(solution.field) - exact_gradient,
                    ng.grad(solution.field) - exact_gradient,
                ),
                solution.mesh,
                order=8,
            )
        )
        errors.append((float(l2_error), float(energy_error)))

    l2_rate = log2(errors[0][0] / errors[1][0])
    energy_rate = log2(errors[0][1] / errors[1][1])
    assert l2_rate >= polynomial_order + 0.8
    assert energy_rate >= polynomial_order - 0.2
