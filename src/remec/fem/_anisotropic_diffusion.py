"""Internal verification kernel for the anisotropic reference-potential equation."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite, pi
from typing import Any

from remec.common.threads import configure_threads
from remec.geometry.slab import Slab2D
from remec.options import RuntimeOptions


@dataclass(frozen=True, slots=True)
class _AnisotropicDiffusionSolution:
    """Internal discrete result with a free-DOF algebraic residual diagnostic."""

    _mesh: Any
    _field: Any
    polynomial_order: int
    free_dof_residual_norm: float
    free_dof_relative_residual_norm: float
    energy_diagnostics: _EnergyDiagnostics


@dataclass(frozen=True, slots=True)
class _EnergyDiagnostics:
    """Separate M4a weak-form energies and their positive total."""

    parallel: float
    perpendicular: float
    total: float


@dataclass(frozen=True, slots=True)
class PollutionDiagnostic:
    """Measured effective perpendicular diffusion in the Sovinec M4a test."""

    polynomial_order: int
    maxh: float
    elements: int
    central_amplitude: float
    numerical_perpendicular_diffusivity: float
    numerical_to_parallel_ratio: float
    free_dof_relative_residual_norm: float


@dataclass(frozen=True, slots=True)
class DirectionalConductivity:
    """Constant positive-definite 2D tensor K for the M4a verification kernel.

    The tensor is ``K = κ_perp I + (κ_parallel - κ_perp) b⊗b``. It has
    eigenvalue ``κ_parallel`` along the unit direction ``b`` and
    ``κ_perp`` in its transverse direction.
    """

    parallel: float
    perpendicular: float
    direction: tuple[float, float]

    def __post_init__(self) -> None:
        if not all(
            isfinite(value) and value > 0.0 for value in (self.parallel, self.perpendicular)
        ):
            raise ValueError("conductivities must be finite and positive")
        direction_norm = hypot(*self.direction)
        if not isfinite(direction_norm) or direction_norm == 0.0:
            raise ValueError("direction must be finite and nonzero")
        object.__setattr__(
            self,
            "direction",
            (self.direction[0] / direction_norm, self.direction[1] / direction_norm),
        )

    @property
    def components(self) -> tuple[float, float, float]:
        """Return the symmetric ``(K_xx, K_xy, K_yy)`` tensor components."""
        bx, by = self.direction
        contrast = self.parallel - self.perpendicular
        return (
            self.perpendicular + contrast * bx * bx,
            contrast * bx * by,
            self.perpendicular + contrast * by * by,
        )

    def apply(self, vector: tuple[float, float]) -> tuple[float, float]:
        """Return the tensor action on a plain two-component vector."""
        k_xx, k_xy, k_yy = self.components
        return (k_xx * vector[0] + k_xy * vector[1], k_xy * vector[0] + k_yy * vector[1])

    def quadratic_form(self, gradient: Any) -> Any:
        """Return ``∇χ·K∇χ`` for an NGSolve vector coefficient function."""
        import ngsolve as ng  # type: ignore[import-untyped]

        direction = ng.CoefficientFunction(self.direction)
        parallel_gradient = ng.InnerProduct(direction, gradient)
        transverse_gradient = gradient - direction * parallel_gradient
        return self.parallel * parallel_gradient**2 + self.perpendicular * ng.InnerProduct(
            transverse_gradient, transverse_gradient
        )


def solve_anisotropic_diffusion(
    slab: Slab2D,
    *,
    polynomial_order: int,
    source: Any,
    conductivity: DirectionalConductivity,
    runtime: RuntimeOptions | None = None,
) -> _AnisotropicDiffusionSolution:
    """Solve the verification weak form of note equation (M4a).

    It assembles the M4a form
    ``∫_Ω κ_parallel(b·∇χ)(b·∇v) + κ_perp∇_perpχ·∇_perpv dV = ∫_Ω vS_ref dV``
    on ``H¹_0(Ω)`` with homogeneous Dirichlet data. This remains a
    verification-only kernel, not the future production solver interface.
    """
    if polynomial_order < 1:
        raise ValueError("polynomial_order must be at least one")
    if slab.lower != (0.0, 0.0) or slab.upper != (1.0, 1.0):
        raise ValueError("the verification kernel currently supports the unit square only")

    resolved_runtime = RuntimeOptions() if runtime is None else runtime

    import ngsolve as ng

    mesh = slab.build_mesh()._mesh
    space = ng.H1(mesh, order=polynomial_order, dirichlet="bottom|right|top|left")
    trial, test = space.TnT()
    quadrature = ng.dx(bonus_intorder=4)
    bilinear_form = ng.BilinearForm(space)
    direction = ng.CoefficientFunction(conductivity.direction)
    parallel_trial = ng.InnerProduct(direction, ng.grad(trial))
    parallel_test = ng.InnerProduct(direction, ng.grad(test))
    perpendicular_trial = ng.grad(trial) - direction * parallel_trial
    perpendicular_test = ng.grad(test) - direction * parallel_test
    bilinear_form += (
        conductivity.parallel * parallel_trial * parallel_test
        + conductivity.perpendicular * ng.InnerProduct(perpendicular_trial, perpendicular_test)
    ) * quadrature
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
        gradient = ng.grad(field)
        direction = ng.CoefficientFunction(conductivity.direction)
        parallel_gradient = ng.InnerProduct(direction, gradient)
        perpendicular_gradient = gradient - direction * parallel_gradient
        parallel_energy = float(
            ng.Integrate(conductivity.parallel * parallel_gradient**2, mesh, order=8)
        )
        perpendicular_energy = float(
            ng.Integrate(
                conductivity.perpendicular
                * ng.InnerProduct(perpendicular_gradient, perpendicular_gradient),
                mesh,
                order=8,
            )
        )
        energy_diagnostics = _EnergyDiagnostics(
            parallel=parallel_energy,
            perpendicular=perpendicular_energy,
            total=parallel_energy + perpendicular_energy,
        )

    if free_dof_relative_residual_norm > 1.0e-11:
        raise RuntimeError(
            "anisotropic diffusion direct solve failed: free-DOF relative residual "
            f"{free_dof_relative_residual_norm:.3e} exceeds 1e-11"
        )
    return _AnisotropicDiffusionSolution(
        _mesh=mesh,
        _field=field,
        polynomial_order=polynomial_order,
        free_dof_residual_norm=free_dof_residual_norm,
        free_dof_relative_residual_norm=free_dof_relative_residual_norm,
        energy_diagnostics=energy_diagnostics,
    )


def measure_sovinec_pollution(
    slab: Slab2D,
    *,
    polynomial_order: int,
    parallel_conductivity: float = 1.0,
    source_amplitude: float = 1.0,
    runtime: RuntimeOptions | None = None,
) -> PollutionDiagnostic:
    """Measure numerical perpendicular diffusion for note equation (M4a).

    The benchmark uses ``kappa_perp = 0`` and the field tangent to
    ``psi = sin(pi*x) sin(pi*y)``.  For source ``Q = Q0*psi``, the central
    response defines ``kappa_perp,num = Q0 / (2*pi**2*chi(1/2, 1/2))``.
    """
    if polynomial_order < 1:
        raise ValueError("polynomial_order must be at least one")
    if not isfinite(parallel_conductivity) or parallel_conductivity <= 0.0:
        raise ValueError("parallel_conductivity must be finite and positive")
    if not isfinite(source_amplitude) or source_amplitude <= 0.0:
        raise ValueError("source_amplitude must be finite and positive")
    if slab.lower != (0.0, 0.0) or slab.upper != (1.0, 1.0):
        raise ValueError("the Sovinec benchmark currently supports the unit square only")

    resolved_runtime = RuntimeOptions() if runtime is None else runtime

    import ngsolve as ng

    mesh = slab.build_mesh()._mesh
    space = ng.H1(mesh, order=polynomial_order, dirichlet="bottom|right|top|left")
    trial, test = space.TnT()
    quadrature = ng.dx(bonus_intorder=6)

    psi = ng.sin(ng.pi * ng.x) * ng.sin(ng.pi * ng.y)
    dpsi_dx = ng.pi * ng.cos(ng.pi * ng.x) * ng.sin(ng.pi * ng.y)
    dpsi_dy = ng.pi * ng.sin(ng.pi * ng.x) * ng.cos(ng.pi * ng.y)
    tangent_norm = ng.sqrt(dpsi_dx**2 + dpsi_dy**2)
    tangent = ng.CoefficientFunction((dpsi_dy / tangent_norm, -dpsi_dx / tangent_norm))

    parallel_trial = ng.InnerProduct(tangent, ng.grad(trial))
    parallel_test = ng.InnerProduct(tangent, ng.grad(test))
    bilinear_form = ng.BilinearForm(space)
    bilinear_form += parallel_conductivity * parallel_trial * parallel_test * quadrature
    linear_form = ng.LinearForm(space)
    linear_form += source_amplitude * psi * test * quadrature
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
        relative_residual = float(ng.Norm(free_residual)) / max(
            1.0, float(ng.Norm(source_on_free_dofs))
        )
        central_amplitude = float(field(mesh(0.5, 0.5)))

    if relative_residual > 1.0e-6:
        raise RuntimeError(
            "Sovinec pollution solve failed: free-DOF relative residual "
            f"{relative_residual:.3e} exceeds 1e-6"
        )
    if not isfinite(central_amplitude) or central_amplitude <= 0.0:
        raise RuntimeError(
            "Sovinec pollution solve produced a non-positive or non-finite central amplitude"
        )

    numerical_perpendicular_diffusivity = source_amplitude / (2.0 * pi**2 * central_amplitude)
    return PollutionDiagnostic(
        polynomial_order=polynomial_order,
        maxh=slab.maxh,
        elements=int(mesh.ne),
        central_amplitude=central_amplitude,
        numerical_perpendicular_diffusivity=numerical_perpendicular_diffusivity,
        numerical_to_parallel_ratio=(numerical_perpendicular_diffusivity / parallel_conductivity),
        free_dof_relative_residual_norm=relative_residual,
    )
