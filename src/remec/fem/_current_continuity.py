"""Internal NGSolve implementation of the direct-u current-continuity kernel."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any, Protocol

from remec.common.threads import configure_threads
from remec.geometry.slab import Slab2D
from remec.options import (
    CurrentContinuityStabilization,
    RegularizationGradient,
    RuntimeOptions,
)


class _FrozenCoefficients(Protocol):
    @property
    def magnetic_field(self) -> Any: ...

    @property
    def magnetic_field_gradient(self) -> Any | None: ...

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
    stabilization: CurrentContinuityStabilization
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


def _supg_parameter_expression(
    element_size_along_field: Any,
    diffusive_element_size: Any,
    magnetic_magnitude: Any,
    transverse_diffusion: float,
    polynomial_order: int,
) -> Any:
    r"""Return the centralized DESIGN §9.1 M3 SUPG parameter expression.

    The effective high-order scales are ``h_stream = h_parallel / p`` and
    ``h_diff = h_K / p`` and

    ``tau = ((2 |B| / h_stream)^2 + (4 D_u / h_diff^2)^2)^(-1/2)``.

    The separate diffusive scale keeps tau bounded when the field is nearly normal to
    the modeled slab. Inputs may be scalar numbers or NGSolve coefficient functions;
    validation of scalar public inputs lives in :func:`supg_stabilization_parameter`.
    """
    effective_streamline_size = element_size_along_field / polynomial_order
    effective_diffusive_size = diffusive_element_size / polynomial_order
    advective_rate = 2.0 * magnetic_magnitude / effective_streamline_size
    diffusive_rate = 4.0 * transverse_diffusion / effective_diffusive_size**2
    return 1.0 / (advective_rate**2 + diffusive_rate**2) ** 0.5


def supg_stabilization_parameter(
    *,
    element_size_along_field: float,
    magnetic_magnitude: float,
    transverse_diffusion: float,
    polynomial_order: int,
    diffusive_element_size: float | None = None,
) -> float:
    r"""Return the DESIGN §9.1 stabilization parameter for note equation (M3).

    ``tau = ((2 |B| p / h_parallel)^2
    + (4 D_u p^2 / h_K^2)^2)^(-1/2)``. ``h_K`` defaults to ``h_parallel`` for
    scalar use; the FEM assembly supplies the bounded physical element size.
    """
    if not isfinite(element_size_along_field) or element_size_along_field <= 0.0:
        raise ValueError("element_size_along_field must be finite and positive")
    if not isfinite(magnetic_magnitude) or magnetic_magnitude < 0.0:
        raise ValueError("magnetic_magnitude must be finite and non-negative")
    if not isfinite(transverse_diffusion) or transverse_diffusion < 0.0:
        raise ValueError("transverse_diffusion must be finite and non-negative")
    if polynomial_order < 1:
        raise ValueError("polynomial_order must be at least one")
    resolved_diffusive_size = (
        element_size_along_field if diffusive_element_size is None else diffusive_element_size
    )
    if not isfinite(resolved_diffusive_size) or resolved_diffusive_size <= 0.0:
        raise ValueError("diffusive_element_size must be finite and positive")
    if magnetic_magnitude == 0.0 and transverse_diffusion == 0.0:
        raise ValueError("magnetic_magnitude and transverse_diffusion cannot both be zero")
    value = _supg_parameter_expression(
        element_size_along_field,
        resolved_diffusive_size,
        magnetic_magnitude,
        transverse_diffusion,
        polynomial_order,
    )
    return float(value)


def _resolve_magnetic_field_gradient(
    mesh: Any,
    magnetic_field: Any,
    supplied_gradient: Any | None,
) -> Any:
    r"""Return ``∂_j B_i`` for the complete strong (M3) projector divergence.

    NGSolve 6.2.2606 silently returns zero from ``GridFunction.Diff(x)``. Therefore a
    varying GridFunction-backed magnetic field must supply its native 2-by-3 gradient;
    analytic coefficient functions retain the coordinate-differentiation fallback.
    """
    import ngsolve as ng

    if supplied_gradient is not None:
        supplied_dimensions = tuple(getattr(supplied_gradient, "dims", ()))
        if supplied_dimensions == (3, 2):
            return supplied_gradient
        if supplied_dimensions == (2, 3):
            return ng.CoefficientFunction(
                tuple(
                    supplied_gradient[coordinate, component]
                    for component in range(3)
                    for coordinate in range(2)
                ),
                dims=(3, 2),
            )
        raise ValueError("magnetic_field_gradient must have dimensions (3, 2) or (2, 3)")

    coordinates = (ng.x, ng.y)
    derived = ng.CoefficientFunction(
        tuple(
            magnetic_field[component].Diff(coordinate)
            for component in range(3)
            for coordinate in coordinates
        ),
        dims=(3, 2),
    )
    variation = 0.0
    scale = 1.0
    for component in range(3):
        minimum, maximum = _quadrature_extrema(mesh, magnetic_field[component], integration_order=8)
        variation = max(variation, maximum - minimum)
        scale = max(scale, abs(minimum), abs(maximum))
    derivative_l2_squared = float(
        ng.Integrate(
            sum(
                derived[component, coordinate] ** 2
                for component in range(3)
                for coordinate in range(2)
            ),
            mesh,
            order=8,
        )
    )
    if variation > 1.0e-10 * scale and derivative_l2_squared < 1.0e-28:
        raise ValueError(
            "magnetic_field_gradient is required for a varying GridFunction-backed magnetic_field"
        )
    return derived


def _normalized_direction_gradient(
    magnetic_field: Any,
    magnetic_field_gradient: Any,
    safe_magnitude: Any,
) -> Any:
    r"""Return ``∂_j(B_i/B_safe)`` for the strong note-(M3) diffusion operator."""
    import ngsolve as ng

    entries: list[Any] = []
    for component in range(3):
        for coordinate in range(2):
            magnitude_numerator = sum(
                magnetic_field[index] * magnetic_field_gradient[index, coordinate]
                for index in range(3)
            )
            entries.append(
                magnetic_field_gradient[component, coordinate] / safe_magnitude
                - magnetic_field[component] * magnitude_numerator / safe_magnitude**3
            )
    return ng.CoefficientFunction(tuple(entries), dims=(3, 2))


def _regularized_gradient_divergence(
    scalar: Any,
    direction: Any,
    variant: RegularizationGradient,
    direction_gradient: Any | None = None,
) -> Any:
    r"""Return the complete strong M3 diffusion divergence ``div(grad_r u)``.

    For note equation (M3), ``grad_r = grad`` in the isotropic variant and
    ``grad_r = (I - b_safe b_safe^T) grad`` in the perpendicular variant. The
    latter expansion includes derivatives of the spatially varying projector:
    ``P_ij d_ij u + (d_i P_ij) d_j u`` for in-plane ``i,j``.
    """
    hessian = scalar.Operator("hesse")
    if variant == "full":
        return hessian[0, 0] + hessian[1, 1]

    import ngsolve as ng

    if direction_gradient is None:
        raise ValueError("direction_gradient is required for perpendicular strong diffusion")

    gradient = ng.grad(scalar)
    divergence: Any = 0.0
    for row in range(2):
        for column in range(2):
            projector = (1.0 if row == column else 0.0) - direction[row] * direction[column]
            divergence += projector * hessian[row, column]
            projector_derivative = -(
                direction_gradient[row, row] * direction[column]
                + direction[row] * direction_gradient[column, row]
            )
            divergence += projector_derivative * gradient[column]
    return divergence


def _quadrature_extrema(mesh: Any, coefficient: Any, integration_order: int) -> tuple[float, float]:
    """Return deterministic volume-quadrature extrema for an elementwise coefficient."""
    import ngsolve as ng
    import numpy as np

    element_types = {element.type for element in mesh.Elements(ng.VOL)}
    rules = {
        element_type: ng.IntegrationRule(element_type, integration_order)
        for element_type in element_types
    }
    mapped_points = mesh.MapToAllElements(rules, ng.VOL)
    values = np.asarray(coefficient(mapped_points), dtype=float).reshape(-1)
    return float(np.min(values)), float(np.max(values))


def _minimum_sampled_value(mesh: Any, coefficient: Any, integration_order: int) -> float:
    """Return the minimum over mesh vertices and deterministic volume quadrature points."""
    import ngsolve as ng
    import numpy as np

    vertex_values = [float(coefficient(mesh(*vertex.point))) for vertex in mesh.vertices]
    element_types = {element.type for element in mesh.Elements(ng.VOL)}
    rules = {
        element_type: ng.IntegrationRule(element_type, integration_order)
        for element_type in element_types
    }
    mapped_points = mesh.MapToAllElements(rules, ng.VOL)
    quadrature_values = np.asarray(coefficient(mapped_points), dtype=float).reshape(-1)
    return min(*vertex_values, float(np.min(quadrature_values)))


def solve_frozen_current_continuity(
    slab: Slab2D,
    *,
    polynomial_order: int,
    coefficients: _FrozenCoefficients,
    runtime: RuntimeOptions | None = None,
    boundary: str = "bottom|right|top|left",
    boundary_value: Any = 0.0,
    stabilization: CurrentContinuityStabilization = "none",
    quadrature_bonus_intorder: int = 12,
    residual_tolerance: float = 1.0e-11,
) -> _CurrentContinuitySolution:
    r"""Solve the direct-u weak form of note equation (M3), optionally with SUPG.

    With ``grad_r`` selected at runtime, this assembles

    ``integral v B.grad(u) + D_u grad_r(v).grad_r(u)
    + mu0 v u (B.grad(p))/B_safe^2
    - mu0 D_u v grad_r(u).grad(p)/B_safe^2
    = integral 2 v B.(grad(p) x grad(B))/B_safe^3``.

    With ``stabilization="supg"``, DESIGN §9.1 adds
    ``integral tau (B.grad(v)) (L_M3(u) - drive)``. The strong operator
    ``L_M3`` contains parallel advection, the complete strong divergence
    ``-div(D_u grad_r(u))`` including projector derivatives, the reaction, and the
    final ``-mu0 D_u grad_r(u).grad(p)/B_safe^2`` correction. The streamline
    derivative is ``B.grad(v) = |B| b.grad(v)`` and ``tau`` is supplied by
    :func:`supg_stabilization_parameter`.

    The same ``grad_r`` reconstructs note equation (M2),
    ``J = u B + B x grad(p)/B_safe^2 - D_u grad_r(u)``. For the full-gradient
    variant, the reported physical parallel current is
    ``J_parallel/B = u - (D_u/B_safe) b_safe.grad(u)``.

    For the perpendicular variant, the symmetric DESIGN §9.1 form
    ``grad_perp(v).grad_perp(u)`` is used rather than the note-literal single
    projection ``grad(v).grad_perp(u)``. They differ only by
    ``O(B_floor**2/B**2)`` because ``b_safe`` is not exactly unit, while the
    symmetric convention preserves a positive diffusion block for later solvers. The
    SUPG strong residual retains the note-literal single projection ``div(P grad(u))``;
    consequently Galerkin and SUPG diffusion differ by the same declared floor-order
    amount when the smooth floor is active.

    ``magnetic_magnitude_gradient`` normally supplies ``grad(|B|)`` from the same
    frozen field. It remains an explicit input so a strong-form manufactured test can
    prescribe the M3 drive; production callers own that consistency. This numerator
    is the gradient of the true magnitude, while all inverse powers use ``B_safe`` as
    required by DESIGN §6.

    ``minimum_field_magnitude`` is the minimum over mesh vertices and deterministic
    quadrature samples. It monitors DESIGN §5 invariant 4 but is a sampled upper bound
    on the true domain minimum, not a certified global bound.
    """
    if polynomial_order < 1:
        raise ValueError("polynomial_order must be at least one")
    if quadrature_bonus_intorder < 0:
        raise ValueError("quadrature_bonus_intorder must be non-negative")
    if not isfinite(residual_tolerance) or residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be finite and positive")
    if stabilization not in ("none", "supg"):
        raise ValueError("stabilization must be 'none' or 'supg'")
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
    galerkin_operator = (
        test * ng.InnerProduct(magnetic_field, trial_gradient)
        + coefficients.current_diffusivity * ng.InnerProduct(regularized_test, regularized_trial)
        + coefficients.vacuum_permeability * b_dot_grad_p * test * trial / safe_magnitude**2
        - final_correction * test
    )

    bilinear_form = ng.BilinearForm(space)
    bilinear_form += galerkin_operator * quadrature
    linear_form = ng.LinearForm(space)
    linear_form += drive * test * quadrature
    supg_bilinear_form: Any | None = None
    supg_linear_form: Any | None = None
    tau: Any | None = None
    direction_gradient: Any | None = None
    if stabilization == "supg":
        in_plane_direction_norm = ng.sqrt(direction[0] ** 2 + direction[1] ** 2 + 1.0e-30)
        element_size_along_field = ng.specialcf.mesh_size / in_plane_direction_norm
        tau = _supg_parameter_expression(
            element_size_along_field,
            ng.specialcf.mesh_size,
            safe_magnitude,
            coefficients.current_diffusivity,
            polynomial_order,
        )
        if resolved_runtime.regularization_gradient == "perpendicular":
            magnetic_field_gradient = _resolve_magnetic_field_gradient(
                mesh,
                magnetic_field,
                supplied_gradient=getattr(coefficients, "magnetic_field_gradient", None),
            )
            direction_gradient = _normalized_direction_gradient(
                magnetic_field,
                magnetic_field_gradient,
                safe_magnitude,
            )
        strong_diffusion_divergence = _regularized_gradient_divergence(
            trial,
            direction,
            resolved_runtime.regularization_gradient,
            direction_gradient,
        )
        strong_operator = (
            ng.InnerProduct(magnetic_field, trial_gradient)
            - coefficients.current_diffusivity * strong_diffusion_divergence
            + coefficients.vacuum_permeability * b_dot_grad_p * trial / safe_magnitude**2
            - final_correction
        )
        streamline_test = ng.InnerProduct(magnetic_field, test_gradient)
        supg_bilinear_integrand = tau * streamline_test * strong_operator
        supg_linear_integrand = tau * streamline_test * drive
        bilinear_form += supg_bilinear_integrand * quadrature
        linear_form += supg_linear_integrand * quadrature
        supg_bilinear_form = ng.BilinearForm(space)
        supg_bilinear_form += supg_bilinear_integrand * quadrature
        supg_linear_form = ng.LinearForm(space)
        supg_linear_form += supg_linear_integrand * quadrature
    free_dofs = space.FreeDofs()

    configure_threads(resolved_runtime.threads)
    with ng.TaskManager():
        bilinear_form.Assemble()
        linear_form.Assemble()
        if supg_bilinear_form is not None and supg_linear_form is not None:
            supg_bilinear_form.Assemble()
            supg_linear_form.Assemble()
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
        direction_norm_defect = 1.0 - ng.InnerProduct(direction, direction)
        floor_activity_l2_squared = float(ng.Integrate(direction_norm_defect**2, mesh, order=20))
        physical_magnitude = ng.sqrt(ng.InnerProduct(magnetic_field, magnetic_field))
        minimum_field_magnitude = _minimum_sampled_value(
            mesh, physical_magnitude, integration_order=20
        )
        if supg_bilinear_form is not None and supg_linear_form is not None and tau is not None:
            stabilization_residual = supg_linear_form.vec.CreateVector()
            stabilization_residual.data = supg_bilinear_form.mat * field.vec - supg_linear_form.vec
            free_stabilization_residual = ng.Projector(free_dofs, True) * stabilization_residual
            supg_stabilization_norm = float(ng.Norm(free_stabilization_residual))
            supg_stabilization_relative_norm = supg_stabilization_norm / max(
                1.0, float(ng.Norm(free_source))
            )
            field_diffusion_divergence = _regularized_gradient_divergence(
                field,
                direction,
                resolved_runtime.regularization_gradient,
                direction_gradient,
            )
            strong_residual = (
                ng.InnerProduct(magnetic_field, field_gradient)
                - coefficients.current_diffusivity * field_diffusion_divergence
                + coefficients.vacuum_permeability * b_dot_grad_p * field / safe_magnitude**2
                - final_term
                - drive
            )
            supg_strong_residual_l2 = sqrt(
                max(0.0, float(ng.Integrate(strong_residual**2, mesh, order=20)))
            )
            supg_tau_min, supg_tau_max = _quadrature_extrema(
                mesh,
                tau,
                integration_order=20,
            )
        else:
            supg_stabilization_norm = 0.0
            supg_stabilization_relative_norm = 0.0
            supg_strong_residual_l2 = 0.0
            supg_tau_min = 0.0
            supg_tau_max = 0.0

    diagnostics = {
        "solution_l2": sqrt(max(0.0, solution_l2_squared)),
        "current_l2": sqrt(max(0.0, current_l2_squared)),
        "parallel_current_over_field_l2": sqrt(max(0.0, parallel_current_l2_squared)),
        "m3_drive_l2": sqrt(max(0.0, drive_l2_squared)),
        "m3_final_correction_l2": sqrt(max(0.0, final_term_l2_squared)),
        "m3_reaction_l2": sqrt(max(0.0, reaction_term_l2_squared)),
        "m3_supg_stabilization_norm": supg_stabilization_norm,
        "m3_supg_stabilization_relative_norm": supg_stabilization_relative_norm,
        "m3_supg_strong_residual_l2": supg_strong_residual_l2,
        "m3_supg_tau_min": supg_tau_min,
        "m3_supg_tau_max": supg_tau_max,
        "floor_activity_l2": sqrt(max(0.0, floor_activity_l2_squared)),
        "floor_activity_l2_squared": floor_activity_l2_squared,
        "minimum_field_magnitude": minimum_field_magnitude,
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
        stabilization=stabilization,
        free_dof_residual_norm=free_dof_residual_norm,
        free_dof_relative_residual_norm=free_dof_relative_residual_norm,
        diagnostics=diagnostics,
    )
