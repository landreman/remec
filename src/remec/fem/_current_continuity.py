"""Internal NGSolve implementation of the direct-u current-continuity kernel."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any, Protocol

from remec.common.threads import configure_threads
from remec.geometry.slab import Slab2D
from remec.options import RegularizationGradient, RuntimeOptions


class _FrozenCoefficients(Protocol):
    @property
    def magnetic_field(self) -> Any: ...

    @property
    def pressure_gradient(self) -> Any: ...

    @property
    def magnetic_magnitude_gradient(self) -> Any: ...

    @property
    def current_diffusivity(self) -> float: ...

    @property
    def magnetic_floor(self) -> float: ...

    @property
    def vacuum_permeability(self) -> float: ...


@dataclass(frozen=True, slots=True)
class _CurrentContinuitySolution:
    """Internal direct-u result for note equations (M2)--(M3)."""

    _mesh: Any
    _field: Any
    _gradient: Any
    _current: Any
    _parallel_current_over_field: Any
    polynomial_order: int
    regularization_gradient: RegularizationGradient
    free_dof_residual_norm: float
    free_dof_relative_residual_norm: float
    diagnostics: dict[str, float]

    def mesh(self) -> Any:
        """Return the internal mesh for same-kernel verification integrals."""
        return self._mesh

    def grid_function(self) -> Any:
        """Return the internal direct-u GridFunction for weak-form verification."""
        return self._field

    def solution_at(self, x_coordinate: float, y_coordinate: float) -> float:
        """Evaluate u at a physical point in the slab."""
        return float(self._field(self._mesh(x_coordinate, y_coordinate)))

    def solution_gradient_at(self, x_coordinate: float, y_coordinate: float) -> tuple[float, float]:
        """Evaluate the two in-plane components of grad(u) at a physical point."""
        value = self._gradient(self._mesh(x_coordinate, y_coordinate))
        return float(value[0]), float(value[1])

    def current_at(self, x_coordinate: float, y_coordinate: float) -> tuple[float, float, float]:
        """Evaluate the reconstructed (M2) current at a physical point."""
        value = self._current(self._mesh(x_coordinate, y_coordinate))
        return float(value[0]), float(value[1]), float(value[2])

    def parallel_current_over_field_at(self, x_coordinate: float, y_coordinate: float) -> float:
        """Evaluate the physical J_parallel/B diagnostic at a physical point."""
        return float(self._parallel_current_over_field(self._mesh(x_coordinate, y_coordinate)))


def _embedded_gradient(gradient: Any) -> Any:
    """Embed a two-dimensional NGSolve gradient in the three-vector M2/M3 algebra."""
    import ngsolve as ng  # type: ignore[import-untyped]

    return ng.CoefficientFunction((gradient[0], gradient[1], 0.0))


def _regularized_gradient(gradient: Any, direction: Any, variant: RegularizationGradient) -> Any:
    """Return the selected M2/M3 gradient: grad_perp or the full gradient."""
    if variant == "full":
        return gradient

    import ngsolve as ng

    return gradient - direction * ng.InnerProduct(direction, gradient)


def solve_frozen_current_continuity(
    slab: Slab2D,
    *,
    polynomial_order: int,
    coefficients: _FrozenCoefficients,
    runtime: RuntimeOptions | None = None,
    boundary: str = "bottom|right|top|left",
    boundary_value: Any = 0.0,
    quadrature_bonus_intorder: int = 12,
    residual_tolerance: float = 1.0e-11,
) -> _CurrentContinuitySolution:
    r"""Solve the unstabilized direct-u weak form of note equation (M3).

    With ``grad_r`` selected at runtime, this assembles

    ``integral v B.grad(u) + D_u grad_r(v).grad_r(u)
    + mu0 v u (B.grad(p))/B_safe^2
    - mu0 D_u v grad_r(u).grad(p)/B_safe^2
    = integral 2 v B.(grad(p) x grad(B))/B_safe^3``.

    The same ``grad_r`` reconstructs note equation (M2),
    ``J = u B + B x grad(p)/B_safe^2 - D_u grad_r(u)``. For the full-gradient
    variant, the reported physical parallel current is
    ``J_parallel/B = u - (D_u/B_safe) b_safe.grad(u)``.
    """
    if polynomial_order < 1:
        raise ValueError("polynomial_order must be at least one")
    if quadrature_bonus_intorder < 0:
        raise ValueError("quadrature_bonus_intorder must be non-negative")
    if not isfinite(residual_tolerance) or residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be finite and positive")
    if slab.lower != (0.0, 0.0) or slab.upper != (1.0, 1.0):
        raise ValueError("the frozen M3 verification kernel supports the unit square only")
    if boundary != "bottom|right|top|left":
        raise ValueError("only the unit-square named Dirichlet boundary is supported")

    resolved_runtime = RuntimeOptions() if runtime is None else runtime

    import ngsolve as ng

    for name in ("magnetic_field", "pressure_gradient", "magnetic_magnitude_gradient"):
        coefficient = getattr(coefficients, name)
        if getattr(coefficient, "dim", None) != 3:
            raise ValueError(f"{name} must be a three-component coefficient function")

    mesh = slab.build_mesh()._mesh
    space = ng.H1(mesh, order=polynomial_order, dirichlet=boundary)
    trial, test = space.TnT()
    trial_gradient = _embedded_gradient(ng.grad(trial))
    test_gradient = _embedded_gradient(ng.grad(test))
    magnetic_field = coefficients.magnetic_field
    pressure_gradient = coefficients.pressure_gradient
    safe_magnitude = ng.sqrt(
        ng.InnerProduct(magnetic_field, magnetic_field) + coefficients.magnetic_floor**2
    )
    direction = magnetic_field / safe_magnitude
    regularized_trial = _regularized_gradient(
        trial_gradient, direction, resolved_runtime.regularization_gradient
    )
    regularized_test = _regularized_gradient(
        test_gradient, direction, resolved_runtime.regularization_gradient
    )
    b_dot_grad_p = ng.InnerProduct(magnetic_field, pressure_gradient)
    drive = (
        2.0
        * ng.InnerProduct(
            magnetic_field,
            ng.Cross(pressure_gradient, coefficients.magnetic_magnitude_gradient),
        )
        / safe_magnitude**3
    )
    final_correction = (
        coefficients.vacuum_permeability
        * coefficients.current_diffusivity
        * ng.InnerProduct(regularized_trial, pressure_gradient)
        / safe_magnitude**2
    )
    quadrature = ng.dx(bonus_intorder=quadrature_bonus_intorder)

    bilinear_form = ng.BilinearForm(space)
    bilinear_form += (
        test * ng.InnerProduct(magnetic_field, trial_gradient)
        + coefficients.current_diffusivity * ng.InnerProduct(regularized_test, regularized_trial)
        + coefficients.vacuum_permeability * b_dot_grad_p * test * trial / safe_magnitude**2
        - final_correction * test
    ) * quadrature
    linear_form = ng.LinearForm(space)
    linear_form += drive * test * quadrature
    free_dofs = space.FreeDofs()

    configure_threads(resolved_runtime.threads)
    with ng.TaskManager():
        bilinear_form.Assemble()
        linear_form.Assemble()
        field = ng.GridFunction(space)
        field.Set(boundary_value, definedon=mesh.Boundaries(boundary))
        inverse = bilinear_form.mat.Inverse(free_dofs, inverse="umfpack")
        correction = field.vec.CreateVector()
        correction.data = inverse * (linear_form.vec - bilinear_form.mat * field.vec)
        field.vec.data += correction

        residual = linear_form.vec.CreateVector()
        residual.data = bilinear_form.mat * field.vec - linear_form.vec
        free_residual = ng.Projector(free_dofs, True) * residual
        free_source = ng.Projector(free_dofs, True) * linear_form.vec
        free_dof_residual_norm = float(ng.Norm(free_residual))
        free_dof_relative_residual_norm = free_dof_residual_norm / max(
            1.0, float(ng.Norm(free_source))
        )

        in_plane_gradient = ng.grad(field)
        field_gradient = _embedded_gradient(in_plane_gradient)
        regularized_field_gradient = _regularized_gradient(
            field_gradient, direction, resolved_runtime.regularization_gradient
        )
        diamagnetic_current = ng.Cross(magnetic_field, pressure_gradient) / safe_magnitude**2
        current = (
            field * magnetic_field
            + diamagnetic_current
            - coefficients.current_diffusivity * regularized_field_gradient
        )
        if resolved_runtime.regularization_gradient == "full":
            parallel_current_over_field = (
                field
                - coefficients.current_diffusivity
                * ng.InnerProduct(direction, field_gradient)
                / safe_magnitude
            )
        else:
            parallel_current_over_field = field

        solution_l2_squared = float(ng.Integrate(field**2, mesh, order=20))
        current_l2_squared = float(ng.Integrate(ng.InnerProduct(current, current), mesh, order=20))
        parallel_current_l2_squared = float(
            ng.Integrate(parallel_current_over_field**2, mesh, order=20)
        )
        drive_l2_squared = float(ng.Integrate(drive**2, mesh, order=20))
        final_term = (
            coefficients.vacuum_permeability
            * coefficients.current_diffusivity
            * ng.InnerProduct(regularized_field_gradient, pressure_gradient)
            / safe_magnitude**2
        )
        final_term_l2_squared = float(ng.Integrate(final_term**2, mesh, order=20))
        reaction_term = coefficients.vacuum_permeability * field * b_dot_grad_p / safe_magnitude**2
        reaction_term_l2_squared = float(ng.Integrate(reaction_term**2, mesh, order=20))

    diagnostics = {
        "solution_l2": sqrt(max(0.0, solution_l2_squared)),
        "current_l2": sqrt(max(0.0, current_l2_squared)),
        "parallel_current_over_field_l2": sqrt(max(0.0, parallel_current_l2_squared)),
        "m3_drive_l2": sqrt(max(0.0, drive_l2_squared)),
        "m3_final_correction_l2": sqrt(max(0.0, final_term_l2_squared)),
        "m3_reaction_l2": sqrt(max(0.0, reaction_term_l2_squared)),
    }
    if not isfinite(free_dof_relative_residual_norm):
        raise RuntimeError("direct-u M3 solve produced a non-finite algebraic residual")
    if free_dof_relative_residual_norm > residual_tolerance:
        raise RuntimeError(
            "direct-u M3 solve failed: free-DOF relative residual "
            f"{free_dof_relative_residual_norm:.3e} exceeds {residual_tolerance:.0e}"
        )
    if not all(isfinite(value) and value >= 0.0 for value in diagnostics.values()):
        raise RuntimeError("direct-u M3 solve produced a non-finite diagnostic")

    return _CurrentContinuitySolution(
        _mesh=mesh,
        _field=field,
        _gradient=in_plane_gradient,
        _current=current,
        _parallel_current_over_field=parallel_current_over_field,
        polynomial_order=polynomial_order,
        regularization_gradient=resolved_runtime.regularization_gradient,
        free_dof_residual_norm=free_dof_residual_norm,
        free_dof_relative_residual_norm=free_dof_relative_residual_norm,
        diagnostics=diagnostics,
    )
