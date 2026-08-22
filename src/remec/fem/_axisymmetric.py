"""Internal NGSolve kernel for the axisymmetric Grad-Shafranov equation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from remec.common.threads import configure_threads
from remec.geometry.axisymmetric import AxisymmetricFluxContourDomain, AxisymmetricRZDomain
from remec.options import RuntimeOptions


@dataclass(frozen=True, slots=True)
class AxisymmetricGradShafranovCoefficients:
    """Frozen right-hand-side coefficients in note ``(M1)``/``GS_recovered``.

    ``pressure_flux_derivative`` is ``d p_0 / d psi`` and
    ``toroidal_field_drive`` is ``I dI/dpsi``.  The reduced equation is
    ``-Delta*psi = mu0 R^2 dp_0/dpsi + I dI/dpsi``.
    """

    pressure_flux_derivative: Any
    toroidal_field_drive: Any
    mu0: float = 1.0

    def __post_init__(self) -> None:
        if not isfinite(self.mu0) or self.mu0 <= 0.0:
            raise ValueError("mu0 must be finite and positive")


@dataclass(frozen=True, slots=True)
class _AxisymmetricGradShafranovSolution:
    """Internal finite-element result for note ``(M1)``/``GS_recovered``."""

    _mesh: Any
    _flux: Any
    _geometry_owner: Any
    polynomial_order: int
    elements: int
    free_dof_residual_norm: float
    free_dof_relative_residual_norm: float
    weighted_magnetic_energy: float


def solve_axisymmetric_grad_shafranov(
    domain: AxisymmetricRZDomain | AxisymmetricFluxContourDomain,
    *,
    polynomial_order: int,
    coefficients: AxisymmetricGradShafranovCoefficients,
    runtime: RuntimeOptions | None = None,
) -> _AxisymmetricGradShafranovSolution:
    r"""Solve the true R-Z weak form of note ``(M1)``/``GS_recovered``.

    For ``Delta*psi = -mu0 R^2 p'(psi) - I I'(psi)``, division by ``R^2``
    before using the axisymmetric volume measure gives, after cancelling the
    common ``2*pi`` factor,

    ``integral (grad(psi).grad(v))/R dR dZ
       = integral (mu0 R p'(psi) + I I'(psi)/R) v dR dZ``.

    Both cylindrical metric factors are therefore assembled explicitly; this
    is a two-dimensional reduced solve and contains no toroidal wedge cells.
    """
    if polynomial_order < 1:
        raise ValueError("polynomial_order must be at least one")
    resolved_runtime = RuntimeOptions() if runtime is None else runtime

    import ngsolve as ng  # type: ignore[import-untyped]

    mesh_bundle = domain.build_mesh()
    mesh = mesh_bundle._mesh
    space = ng.H1(mesh, order=polynomial_order, dirichlet=".*")
    trial, test = space.TnT()
    quadrature = ng.dx(bonus_intorder=6)
    bilinear_form = ng.BilinearForm(space)
    bilinear_form += (ng.InnerProduct(ng.grad(trial), ng.grad(test)) / ng.x).Compile() * quadrature
    linear_form = ng.LinearForm(space)
    source = (
        coefficients.mu0 * ng.x * coefficients.pressure_flux_derivative
        + coefficients.toroidal_field_drive / ng.x
    )
    linear_form += (source * test).Compile() * quadrature
    free_dofs = space.FreeDofs()

    configure_threads(resolved_runtime.threads)
    with ng.TaskManager():
        bilinear_form.Assemble()
        linear_form.Assemble()
        flux = ng.GridFunction(space)
        inverse = bilinear_form.mat.Inverse(free_dofs, inverse="umfpack")
        flux.vec.data = inverse * linear_form.vec
        residual = linear_form.vec.CreateVector()
        residual.data = linear_form.vec - bilinear_form.mat * flux.vec
        free_residual = ng.Projector(free_dofs, True) * residual
        free_load = ng.Projector(free_dofs, True) * linear_form.vec
        residual_norm = float(ng.Norm(free_residual))
        relative_residual_norm = residual_norm / max(1.0e-300, float(ng.Norm(free_load)))
        weighted_magnetic_energy = float(
            ng.Integrate(
                ng.InnerProduct(ng.grad(flux), ng.grad(flux)) / ng.x,
                mesh,
                order=2 * polynomial_order + 4,
            )
        )

    return _AxisymmetricGradShafranovSolution(
        mesh,
        flux,
        mesh_bundle._geometry_owner,
        polynomial_order,
        mesh.ne,
        residual_norm,
        relative_residual_norm,
        weighted_magnetic_energy,
    )
