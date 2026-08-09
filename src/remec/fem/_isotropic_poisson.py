"""Internal verification kernel for the isotropic reference-potential equation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from remec.common.threads import configure_threads
from remec.geometry.slab import Slab2D
from remec.options import RuntimeOptions


@dataclass(frozen=True, slots=True)
class _IsotropicPoissonSolution:
    """Internal discrete result with a free-DOF algebraic residual diagnostic."""

    _mesh: Any
    _field: Any
    polynomial_order: int
    free_dof_residual_norm: float
    free_dof_relative_residual_norm: float


def solve_isotropic_poisson(
    slab: Slab2D,
    *,
    polynomial_order: int,
    source: Any,
    runtime: RuntimeOptions | None = None,
) -> _IsotropicPoissonSolution:
    """Solve the verification-only isotropic weak form of note equation (M4a).

    For ``κ_parallel = κ_perp = 1``, (M4a) is ``-Δχ = S_ref``. This
    assembles ``∫_Ω ∇v·∇χ dV = ∫_Ω v S_ref dV`` on ``H¹_0(Ω)`` with
    homogeneous Dirichlet data. It is not the future production anisotropic-M4a
    solver; that path starts with milestone 1.2.
    """
    if polynomial_order < 1:
        raise ValueError("polynomial_order must be at least one")
    if slab.lower != (0.0, 0.0) or slab.upper != (1.0, 1.0):
        raise ValueError("the verification kernel currently supports the unit square only")

    resolved_runtime = RuntimeOptions() if runtime is None else runtime

    import ngsolve as ng  # type: ignore[import-untyped]

    mesh = slab.build_mesh()._mesh
    space = ng.H1(mesh, order=polynomial_order, dirichlet="bottom|right|top|left")
    trial, test = space.TnT()
    quadrature = ng.dx(bonus_intorder=4)
    bilinear_form = ng.BilinearForm(space)
    bilinear_form += ng.grad(trial) * ng.grad(test) * quadrature
    linear_form = ng.LinearForm(space)
    linear_form += source * test * quadrature
    free_dofs = space.FreeDofs()

    configure_threads(resolved_runtime.threads)
    with ng.TaskManager():
        bilinear_form.Assemble()
        linear_form.Assemble()
        field = ng.GridFunction(space)
        field.vec.data = (
            bilinear_form.mat.Inverse(free_dofs, inverse="sparsecholesky") * linear_form.vec
        )
        residual = linear_form.vec.CreateVector()
        residual.data = bilinear_form.mat * field.vec - linear_form.vec
        free_residual = ng.Projector(free_dofs, True) * residual
        source_on_free_dofs = ng.Projector(free_dofs, True) * linear_form.vec
        free_dof_residual_norm = float(ng.Norm(free_residual))
        free_dof_relative_residual_norm = free_dof_residual_norm / max(
            1.0, float(ng.Norm(source_on_free_dofs))
        )

    if free_dof_relative_residual_norm > 1.0e-11:
        raise RuntimeError(
            "isotropic Poisson direct solve failed: free-DOF relative residual "
            f"{free_dof_relative_residual_norm:.3e} exceeds 1e-11"
        )
    return _IsotropicPoissonSolution(
        _mesh=mesh,
        _field=field,
        polynomial_order=polynomial_order,
        free_dof_residual_norm=free_dof_residual_norm,
        free_dof_relative_residual_norm=free_dof_relative_residual_norm,
    )
