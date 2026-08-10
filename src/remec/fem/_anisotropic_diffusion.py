"""Internal verification kernel for the anisotropic reference-potential equation."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
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
class _FieldDirectionDiagnostics:
    """Diagnostics for smooth small-B protection in a frozen M4a field."""

    floor: float
    floor_activity_l2_squared: float


@dataclass(frozen=True, slots=True)
class _FrozenFieldDiffusionSolution:
    """Internal result for a spatially varying frozen-field M4a solve."""

    _mesh: Any
    _field: Any
    polynomial_order: int
    free_dof_residual_norm: float
    free_dof_relative_residual_norm: float
    energy_diagnostics: _EnergyDiagnostics
    field_direction_diagnostics: _FieldDirectionDiagnostics


@dataclass(frozen=True, slots=True)
class PollutionDiagnostic:
    """Measured effective perpendicular diffusion in the Sovinec M4a test.

    The name refers to the anisotropic-conduction verification in section 4.2 of
    Carl R. Sovinec et al., "Nonlinear magnetohydrodynamics simulation using high-order
    finite elements," J. Comput. Phys. 195 (2004) 355–386,
    https://doi.org/10.1016/j.jcp.2003.10.004.
    """

    polynomial_order: int
    maxh: float
    elements: int
    parallel_conductivity: float
    physical_perpendicular_conductivity: float
    central_amplitude: float
    numerical_perpendicular_diffusivity: float
    numerical_to_parallel_ratio: float
    free_dof_relative_residual_norm: float
    unit_direction_defect_l2_squared: float
    source_tangency_l2_squared: float
    source_laplacian_eigenvalue: float


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


def solve_frozen_field_anisotropic_diffusion(
    slab: Slab2D,
    *,
    polynomial_order: int,
    source: Any,
    raw_field: Any,
    parallel_conductivity: float,
    perpendicular_conductivity: float,
    field_floor: float,
    runtime: RuntimeOptions | None = None,
) -> _FrozenFieldDiffusionSolution:
    """Solve note equation (M4a) for a spatially varying frozen field.

    The intended weak form is
    ``integral kappa_perp grad(chi).grad(v) + (kappa_parallel-kappa_perp)
    (b_safe.grad(chi))(b_safe.grad(v)) = integral v S_ref``, where
    ``b_safe = B / sqrt(B.B + B_floor**2)``.  The smooth floor makes the
    tensor finite at analytic island O- and X-point nulls. This is the direct
    strong-form M4a tensor extension. When ``|b_safe| < 1`` it differs from the
    doubly projected perpendicular-gradient form used by the constant-direction
    helper; milestone 1.5 must reconcile that distinction while extracting the
    public solver interface.
    """
    if polynomial_order < 1:
        raise ValueError("polynomial_order must be at least one")
    if not all(
        isfinite(value) and value > 0.0
        for value in (parallel_conductivity, perpendicular_conductivity)
    ):
        raise ValueError("conductivities must be finite and positive")
    if parallel_conductivity < perpendicular_conductivity:
        raise ValueError("parallel_conductivity must not be below perpendicular_conductivity")
    if not isfinite(field_floor) or field_floor <= 0.0:
        raise ValueError("field_floor must be finite and positive")
    if slab.lower != (0.0, 0.0) or slab.upper != (1.0, 1.0):
        raise ValueError("the frozen-field verification kernel supports the unit square only")

    resolved_runtime = RuntimeOptions() if runtime is None else runtime

    import ngsolve as ng

    mesh = slab.build_mesh()._mesh
    space = ng.H1(mesh, order=polynomial_order, dirichlet="bottom|right|top|left")
    trial, test = space.TnT()
    quadrature = ng.dx(bonus_intorder=20)
    safe_norm = ng.sqrt(ng.InnerProduct(raw_field, raw_field) + field_floor**2)
    direction = raw_field / safe_norm
    parallel_trial = ng.InnerProduct(direction, ng.grad(trial))
    parallel_test = ng.InnerProduct(direction, ng.grad(test))
    conductivity_contrast = parallel_conductivity - perpendicular_conductivity

    bilinear_form = ng.BilinearForm(space)
    bilinear_form += (
        perpendicular_conductivity * ng.InnerProduct(ng.grad(trial), ng.grad(test))
        + conductivity_contrast * parallel_trial * parallel_test
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
        parallel_gradient = ng.InnerProduct(direction, gradient)
        gradient_norm_squared = ng.InnerProduct(gradient, gradient)
        parallel_energy = float(
            ng.Integrate(parallel_conductivity * parallel_gradient**2, mesh, order=20)
        )
        perpendicular_energy = float(
            ng.Integrate(
                perpendicular_conductivity * (gradient_norm_squared - parallel_gradient**2),
                mesh,
                order=20,
            )
        )
        energy_diagnostics = _EnergyDiagnostics(
            parallel=parallel_energy,
            perpendicular=perpendicular_energy,
            total=parallel_energy + perpendicular_energy,
        )
        direction_norm_defect = 1.0 - ng.InnerProduct(direction, direction)
        floor_activity_l2_squared = float(ng.Integrate(direction_norm_defect**2, mesh, order=40))

    if not isfinite(free_dof_relative_residual_norm):
        raise RuntimeError("frozen-field solve produced a non-finite algebraic residual")
    if free_dof_relative_residual_norm > 1.0e-11:
        raise RuntimeError(
            "frozen-field direct solve failed: free-DOF relative residual "
            f"{free_dof_relative_residual_norm:.3e} exceeds 1e-11"
        )
    if not all(
        isfinite(value) and value >= 0.0
        for value in (parallel_energy, perpendicular_energy, floor_activity_l2_squared)
    ):
        raise RuntimeError("frozen-field solve produced a non-finite or negative diagnostic")

    return _FrozenFieldDiffusionSolution(
        _mesh=mesh,
        _field=field,
        polynomial_order=polynomial_order,
        free_dof_residual_norm=free_dof_residual_norm,
        free_dof_relative_residual_norm=free_dof_relative_residual_norm,
        energy_diagnostics=energy_diagnostics,
        field_direction_diagnostics=_FieldDirectionDiagnostics(
            floor=field_floor,
            floor_activity_l2_squared=floor_activity_l2_squared,
        ),
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

    "Sovinec" refers to the anisotropic-conduction test in section 4.2 of Carl R. Sovinec
    et al., "Nonlinear magnetohydrodynamics simulation using high-order finite
    elements," J. Comput. Phys. 195 (2004) 355–386,
    https://doi.org/10.1016/j.jcp.2003.10.004. The test exposes artificial
    cross-field transport from a non-field-aligned discretization: ``b`` is
    tangent to the closed contours of ``psi = sin(pi*x) sin(pi*y)``, so
    ``b·grad(psi) = 0``. With physical ``kappa_perp = 0``, any finite effective
    perpendicular diffusivity is therefore numerical pollution. For source
    ``Q = Q0*psi``, the discrete central response defines
    ``kappa_perp,num = Q0 / (2*pi**2*chi(1/2, 1/2))``.

    This benchmark currently has a dedicated spatially varying, rank-one M4a
    assembly because ``DirectionalConductivity`` and
    ``solve_anisotropic_diffusion`` support only constant directions and
    strictly positive ``kappa_perp``. The perpendicular form is identically
    zero here, so this path reports pollution, residual, unit-direction, and
    source-tangency diagnostics rather than the two energy contributions.
    Milestone 1.5 must unify both assemblies without changing their results.
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
    # Differentiate the same coefficient function used in the linear form, so
    # the tangency diagnostic cannot agree with an independently mistyped source.
    dpsi_dx = psi.Diff(ng.x)
    dpsi_dy = psi.Diff(ng.y)
    source_laplacian = dpsi_dx.Diff(ng.x) + dpsi_dy.Diff(ng.y)
    tangent_norm = ng.sqrt(dpsi_dx**2 + dpsi_dy**2)
    # Rotate and normalize grad(psi): the resulting field follows the closed
    # source contours, so the exact parallel operator cannot transport psi
    # across them. The measured cross-contour response is discretization error.
    tangent = ng.CoefficientFunction((dpsi_dy / tangent_norm, -dpsi_dx / tangent_norm))
    source_gradient = ng.CoefficientFunction((psi.Diff(ng.x), psi.Diff(ng.y)))

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
        unit_direction_defect_l2_squared = float(
            ng.Integrate((ng.InnerProduct(tangent, tangent) - 1.0) ** 2, mesh, order=8)
        )
        source_tangency_l2_squared = float(
            ng.Integrate(ng.InnerProduct(tangent, source_gradient) ** 2, mesh, order=8)
        )
        source_l2_squared = float(ng.Integrate(psi**2, mesh, order=8))
        source_laplacian_eigenvalue = (
            -float(ng.Integrate(source_laplacian * psi, mesh, order=8)) / source_l2_squared
        )

    if not isfinite(relative_residual):
        raise RuntimeError("Sovinec pollution solve produced a non-finite algebraic residual")
    if not isfinite(central_amplitude) or central_amplitude <= 0.0:
        raise RuntimeError(
            "Sovinec pollution solve produced a non-positive or non-finite central amplitude"
        )
    if not isfinite(source_laplacian_eigenvalue) or source_laplacian_eigenvalue <= 0.0:
        raise RuntimeError("Sovinec source has no finite positive Laplacian eigenvalue")

    # Match the center response to an isotropic perpendicular operator. Deriving
    # the Laplacian eigenvalue from psi keeps the Sovinec 2*k**2 factor tied to
    # the actual source rather than to a second hard-coded wavenumber.
    numerical_perpendicular_diffusivity = source_amplitude / (
        source_laplacian_eigenvalue * central_amplitude
    )
    return PollutionDiagnostic(
        polynomial_order=polynomial_order,
        maxh=slab.maxh,
        elements=int(mesh.ne),
        parallel_conductivity=parallel_conductivity,
        physical_perpendicular_conductivity=0.0,
        central_amplitude=central_amplitude,
        numerical_perpendicular_diffusivity=numerical_perpendicular_diffusivity,
        numerical_to_parallel_ratio=(numerical_perpendicular_diffusivity / parallel_conductivity),
        free_dof_relative_residual_norm=relative_residual,
        unit_direction_defect_l2_squared=unit_direction_defect_l2_squared,
        source_tangency_l2_squared=source_tangency_l2_squared,
        source_laplacian_eigenvalue=source_laplacian_eigenvalue,
    )
