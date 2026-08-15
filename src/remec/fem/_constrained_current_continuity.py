"""Internal bordered finite-element solve for note equations ``(M3)``--``(M3b)``."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import isfinite, sqrt
from typing import Any, Protocol

import numpy as np

from remec.common.threads import configure_threads
from remec.current_moments import (
    M2ToroidalCurrentSamples,
    ShellCurrentMoments,
    mollified_shell_current_moments,
)
from remec.fem._current_continuity import (
    _compiled,
    _embedded_gradient,
    _normalized_direction_gradient,
    _quadrature_extrema,
    _regularized_gradient,
    _regularized_gradient_divergence,
    _resolve_magnetic_field_gradient,
    _supg_parameter_expression,
)
from remec.geometry.slab import Slab2D
from remec.level_set import MollifiedVolumeMap
from remec.options import (
    CurrentContinuityStabilization,
    RegularizationGradient,
    RuntimeOptions,
)
from remec.profiles import ToroidalCurrentProfile, extract_ngsolve_quadrature


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


class _ConstraintGeometry(Protocol):
    @property
    def level_set(self) -> Any: ...

    @property
    def level_set_gradient(self) -> Any: ...

    @property
    def toroidal_angle_gradient(self) -> Any: ...


@dataclass(frozen=True, slots=True)
class _ConstrainedCurrentContinuitySolution:
    """Internal physical fields and diagnostics from the bordered ``(M3)``--``(M3b)`` solve."""

    _mesh: Any
    _utilde: Any
    _physical_u: Any
    _g_profile: Any
    _g_gradient: Any
    _current: Any
    _moments: ShellCurrentMoments
    polynomial_order: int
    regularization_gradient: RegularizationGradient
    stabilization: CurrentContinuityStabilization
    shell_edges: tuple[float, ...]
    g_coefficients: tuple[float, ...]
    m3_relative_residual_norm: float
    constraint_relative_residual_norm: float
    schur_condition_number: float
    target_cumulative_current: tuple[float, ...]
    independent_cumulative_current: tuple[float, ...]
    shell_constraint_residuals: tuple[float, ...]
    diagnostics: dict[str, float]

    def mesh(self) -> Any:
        """Return the private NGSolve mesh for manufactured verification integrals."""
        return self._mesh

    def utilde_grid_function(self) -> Any:
        """Return the homogeneous finite-element unknown ``utilde``."""
        return self._utilde

    def grid_function(self) -> Any:
        """Return the reconstructed physical coefficient ``u=G(s)+utilde``."""
        return self._physical_u

    def solution_at(self, x_coordinate: float, y_coordinate: float) -> float:
        """Evaluate reconstructed physical ``u`` at one slab point."""
        return float(self._physical_u(self._mesh(x_coordinate, y_coordinate)))

    def utilde_at(self, x_coordinate: float, y_coordinate: float) -> float:
        """Evaluate the homogeneous solved variable at one slab point."""
        return float(self._utilde(self._mesh(x_coordinate, y_coordinate)))

    def g_at(self, x_coordinate: float, y_coordinate: float) -> float:
        """Evaluate the solved one-dimensional multiplier ``G(s)``."""
        return float(self._g_profile(self._mesh(x_coordinate, y_coordinate)))

    def current_at(self, x_coordinate: float, y_coordinate: float) -> tuple[float, float, float]:
        """Evaluate the reconstructed physical note-``(M2)`` current."""
        value = self._current(self._mesh(x_coordinate, y_coordinate))
        return float(value[0]), float(value[1]), float(value[2])


def _pchip_volume_coordinate(
    volume_map: MollifiedVolumeMap,
    level_set: Any,
    level_set_gradient: Any,
) -> tuple[Any, Any]:
    r"""Return the exact tabulated ``s=V_chi/V_omega`` coefficient and ``grad(s)``.

    The interval polynomials transcribe the same monotone cubic Hermite interpolant
    used by :class:`MollifiedVolumeMap`; this prevents the ``G`` basis and the shell
    moments from quietly using two different normalized-volume fields.
    """
    import ngsolve as ng  # type: ignore[import-untyped]

    levels = np.asarray(volume_map.levels, dtype=float)
    total_volume = float(volume_map.volumes[0])
    values = np.asarray(volume_map.volumes, dtype=float) / total_volume
    slopes = np.asarray(
        [volume_map.volume_derivative(float(level)) / total_volume for level in levels],
        dtype=float,
    )
    interval_values: list[Any] = []
    interval_derivatives: list[Any] = []
    for index, width in enumerate(np.diff(levels)):
        coordinate = (level_set - float(levels[index])) / float(width)
        h00 = 2.0 * coordinate**3 - 3.0 * coordinate**2 + 1.0
        h10 = coordinate**3 - 2.0 * coordinate**2 + coordinate
        h01 = -2.0 * coordinate**3 + 3.0 * coordinate**2
        h11 = coordinate**3 - coordinate**2
        interval_values.append(
            h00 * float(values[index])
            + h10 * float(width * slopes[index])
            + h01 * float(values[index + 1])
            + h11 * float(width * slopes[index + 1])
        )
        dh00 = (6.0 * coordinate**2 - 6.0 * coordinate) / float(width)
        dh10 = 3.0 * coordinate**2 - 4.0 * coordinate + 1.0
        dh01 = (-6.0 * coordinate**2 + 6.0 * coordinate) / float(width)
        dh11 = 3.0 * coordinate**2 - 2.0 * coordinate
        interval_derivatives.append(
            dh00 * float(values[index])
            + dh10 * float(slopes[index])
            + dh01 * float(values[index + 1])
            + dh11 * float(slopes[index + 1])
        )

    value: Any = interval_values[-1]
    derivative: Any = interval_derivatives[-1]
    for index in range(len(interval_values) - 2, -1, -1):
        value = ng.IfPos(float(levels[index + 1]) - level_set, interval_values[index], value)
        derivative = ng.IfPos(
            float(levels[index + 1]) - level_set,
            interval_derivatives[index],
            derivative,
        )
    value = ng.IfPos(
        level_set - float(levels[0]),
        ng.IfPos(float(levels[-1]) - level_set, value, float(values[-1])),
        float(values[0]),
    )
    derivative = ng.IfPos(
        level_set - float(levels[0]),
        ng.IfPos(float(levels[-1]) - level_set, derivative, 0.0),
        0.0,
    )
    return value, derivative * level_set_gradient


def _linear_g_basis(
    normalized_volume: Any,
    normalized_volume_gradient: Any,
    shell_edges: np.ndarray,
    basis_index: int,
) -> tuple[Any, Any]:
    r"""Return one piecewise-linear ``G(s)`` basis function and its physical gradient."""
    import ngsolve as ng

    values = np.zeros(len(shell_edges), dtype=float)
    values[basis_index] = 1.0
    spline = ng.BSpline(
        2,
        [float(shell_edges[0]), *map(float, shell_edges), float(shell_edges[-1])],
        values.tolist(),
    )(normalized_volume)
    value = ng.IfPos(
        normalized_volume - float(shell_edges[0]),
        ng.IfPos(float(shell_edges[-1]) - normalized_volume, spline, float(values[-1])),
        float(values[0]),
    )
    interval_slopes = np.diff(values) / np.diff(shell_edges)
    derivative: Any = float(interval_slopes[-1])
    for index in range(len(interval_slopes) - 2, -1, -1):
        derivative = ng.IfPos(
            float(shell_edges[index + 1]) - normalized_volume,
            float(interval_slopes[index]),
            derivative,
        )
    return value, derivative * normalized_volume_gradient


def _mapped_quadrature(mesh: Any, integration_order: int) -> Any:
    """Return mapped points in the ordering used by ``extract_ngsolve_quadrature``."""
    import ngsolve as ng

    element_types = {element.type for element in mesh.Elements(ng.VOL)}
    rules = {
        element_type: ng.IntegrationRule(element_type, integration_order)
        for element_type in element_types
    }
    return mesh.MapToAllElements(rules, ng.VOL)


def _sample_scalar(coefficient: Any, mapped_points: Any) -> np.ndarray:
    """Evaluate one scalar coefficient in deterministic mapped-quadrature order."""
    return np.asarray(coefficient(mapped_points), dtype=float).reshape(-1)


def _moments(
    volume_map: MollifiedVolumeMap,
    shell_edges: np.ndarray,
    *,
    parallel: np.ndarray,
    diamagnetic: np.ndarray,
    regularizing: np.ndarray,
) -> ShellCurrentMoments:
    """Apply the shared mollified ``(M3b)`` functional to separate ``(M2)`` terms."""
    return mollified_shell_current_moments(
        volume_map,
        M2ToroidalCurrentSamples(
            normalized_volume=volume_map.quadrature_normalized_volume,
            parallel=parallel,
            diamagnetic=diamagnetic,
            regularizing=regularizing,
        ),
        shell_edges,
    )


def _component_moments(
    volume_map: MollifiedVolumeMap,
    shell_edges: np.ndarray,
    mapped_points: Any,
    *,
    parallel: Any = 0.0,
    diamagnetic: Any = 0.0,
    regularizing: Any = 0.0,
) -> ShellCurrentMoments:
    """Evaluate coefficient expressions, then integrate their independent shell rows."""
    sample_count = len(volume_map.quadrature_weights)

    def samples(value: Any) -> np.ndarray:
        if isinstance(value, (int, float)):
            return np.full(sample_count, float(value), dtype=float)
        return _sample_scalar(value, mapped_points)

    return _moments(
        volume_map,
        shell_edges,
        parallel=samples(parallel),
        diamagnetic=samples(diamagnetic),
        regularizing=samples(regularizing),
    )


def _solve_with_inverse(inverse: Any, right_hand_side: Any, space: Any) -> Any:
    """Apply the reusable M3 ``A`` inverse to one Schur-complement response."""
    import ngsolve as ng

    field = ng.GridFunction(space)
    field.vec.data = inverse * right_hand_side
    return field


def solve_constrained_current_continuity(
    slab: Slab2D,
    *,
    polynomial_order: int,
    coefficients: _FrozenCoefficients,
    geometry: _ConstraintGeometry,
    current_profile: ToroidalCurrentProfile,
    shell_edges: Sequence[float],
    edge_value: float,
    runtime: RuntimeOptions | None = None,
    boundary: str = "left|right",
    stabilization: CurrentContinuityStabilization = "supg",
    quadrature_order: int = 8,
    quadrature_bonus_intorder: int = 12,
    volume_levels: int = 65,
    spatial_width_cells: float = 1.0,
    residual_tolerance: float = 1.0e-10,
) -> _ConstrainedCurrentContinuitySolution:
    r"""Solve the square bordered system for note equations ``(M3)``--``(M3b)``.

    With ``u=G(s)+utilde``, homogeneous ``utilde`` boundary data, and the selected
    ``grad_r``, the M3 block is

    ``A utilde + P G = drive``,

    where ``P G = B.grad(G) + mu0 G B.grad(p)/B_safe**2`` carries both terms that
    appear with a minus sign on the right of note Eq. ``utilde_equation``.  The shell
    rows implement

    ``C_u utilde + C_G G = Delta I_0 - I_diamagnetic``

    from ``(M2)``--``(M3b)``, with ``C_u`` containing
    ``utilde B.grad(phi) - D_u grad_r(utilde).grad(phi)`` and ``C_G`` containing only
    ``G B.grad(phi)``.  Thus diffusion acts on ``utilde``, never on full ``u``, and no
    separate multiplier-current term is double counted in ``C_G``.  A single sparse
    factorization of ``A`` supplies all response columns; the dense solve is only the
    one-dimensional Schur complement.
    """
    if polynomial_order < 1:
        raise ValueError("polynomial_order must be at least one")
    if stabilization not in ("none", "supg"):
        raise ValueError("stabilization must be 'none' or 'supg'")
    if quadrature_order < 2 or quadrature_bonus_intorder < 0:
        raise ValueError("quadrature orders must be positive and diagnostic order at least two")
    if volume_levels < 3:
        raise ValueError("volume_levels must be at least three")
    if not isfinite(spatial_width_cells) or spatial_width_cells <= 0.0:
        raise ValueError("spatial_width_cells must be finite and positive")
    if not isfinite(edge_value):
        raise ValueError("edge_value must be finite")
    if not isfinite(residual_tolerance) or residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be finite and positive")
    if slab.lower != (0.0, 0.0) or slab.upper != (1.0, 1.0):
        raise ValueError("the constrained M3-M3b verification kernel supports the unit square")
    boundary_names = tuple(boundary.split("|"))
    available_boundaries = slab.boundary_regions()
    if (
        not boundary
        or any(not name for name in boundary_names)
        or any(name not in available_boundaries for name in boundary_names)
    ):
        available = "|".join(available_boundaries)
        raise ValueError(f"boundary must name only available slab regions: {available}")
    edges = np.asarray(shell_edges, dtype=float)
    if (
        edges.ndim != 1
        or len(edges) < 2
        or not np.all(np.isfinite(edges))
        or edges[0] != 0.0
        or edges[-1] != 1.0
        or np.any(np.diff(edges) <= 0.0)
    ):
        raise ValueError("shell_edges must be a strictly increasing partition of exactly [0, 1]")
    current_profile.validate()
    target_cumulative = np.asarray(current_profile.enclosed_current(edges), dtype=float)
    if target_cumulative.shape != edges.shape or not np.all(np.isfinite(target_cumulative)):
        raise ValueError("current_profile must return one finite cumulative target per shell edge")

    import ngsolve as ng

    for name in ("magnetic_field", "pressure_gradient", "magnetic_magnitude_gradient"):
        if getattr(getattr(coefficients, name), "dim", None) != 3:
            raise ValueError(f"{name} must be a three-component coefficient function")
    resolved_runtime = RuntimeOptions() if runtime is None else runtime
    mesh = slab.build_mesh()._mesh
    base_space = ng.H1(mesh, order=polynomial_order, dirichlet=boundary)
    space = ng.Periodic(base_space) if slab.periodic_y else base_space
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
    reaction_coefficient = coefficients.vacuum_permeability * b_dot_grad_p / safe_magnitude**2
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
        + reaction_coefficient * test * trial
        - final_correction * test
    )
    bilinear_form = ng.BilinearForm(space)
    bilinear_form += _compiled(galerkin_operator) * quadrature
    drive_form = ng.LinearForm(space)
    drive_form += _compiled(drive * test) * quadrature

    tau: Any | None = None
    supg_bilinear_form: Any | None = None
    supg_drive_form: Any | None = None
    streamline_test: Any = 0.0
    if stabilization == "supg":
        in_plane_direction_norm = ng.sqrt(direction[0] ** 2 + direction[1] ** 2 + 1.0e-30)
        tau = _supg_parameter_expression(
            ng.specialcf.mesh_size / in_plane_direction_norm,
            ng.specialcf.mesh_size,
            safe_magnitude,
            coefficients.current_diffusivity,
            polynomial_order,
        )
        direction_gradient: Any | None = None
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
        diffusion_divergence = _regularized_gradient_divergence(
            trial,
            direction,
            resolved_runtime.regularization_gradient,
            direction_gradient,
        )
        strong_operator = (
            ng.InnerProduct(magnetic_field, trial_gradient)
            - coefficients.current_diffusivity * diffusion_divergence
            + reaction_coefficient * trial
            - final_correction
        )
        streamline_test = ng.InnerProduct(magnetic_field, test_gradient)
        supg_operator = _compiled(tau * streamline_test * strong_operator)
        supg_drive = _compiled(tau * streamline_test * drive)
        bilinear_form += supg_operator * quadrature
        drive_form += supg_drive * quadrature
        supg_bilinear_form = ng.BilinearForm(space)
        supg_bilinear_form += supg_operator * quadrature
        supg_drive_form = ng.LinearForm(space)
        supg_drive_form += supg_drive * quadrature

    quadrature_data = extract_ngsolve_quadrature(
        mesh,
        geometry.level_set,
        geometry.level_set_gradient,
        integration_order=quadrature_order,
    )
    volume_map = MollifiedVolumeMap.build(
        quadrature_data,
        spatial_width_cells=spatial_width_cells,
        levels=volume_levels,
    )
    normalized_volume, normalized_volume_gradient = _pchip_volume_coordinate(
        volume_map,
        geometry.level_set,
        geometry.level_set_gradient,
    )
    g_basis = [
        _linear_g_basis(normalized_volume, normalized_volume_gradient, edges, index)
        for index in range(len(edges))
    ]
    p_forms: list[Any] = []
    supg_p_forms: list[Any] = []
    for basis, basis_gradient in g_basis:
        coupling = ng.InnerProduct(magnetic_field, basis_gradient) + reaction_coefficient * basis
        column = ng.LinearForm(space)
        column += _compiled(test * coupling) * quadrature
        if tau is not None:
            column += _compiled(tau * streamline_test * coupling) * quadrature
        p_forms.append(column)
        if tau is not None:
            supg_column = ng.LinearForm(space)
            supg_column += _compiled(tau * streamline_test * coupling) * quadrature
            supg_p_forms.append(supg_column)

    free_dofs = space.FreeDofs()
    configure_threads(resolved_runtime.threads)
    with ng.TaskManager():
        bilinear_form.Assemble()
        drive_form.Assemble()
        for column in p_forms:
            column.Assemble()
        if supg_bilinear_form is not None and supg_drive_form is not None:
            supg_bilinear_form.Assemble()
            supg_drive_form.Assemble()
            for column in supg_p_forms:
                column.Assemble()
        inverse = bilinear_form.mat.Inverse(free_dofs, inverse="umfpack")
        base_rhs = drive_form.vec.CreateVector()
        base_rhs.data = drive_form.vec - edge_value * p_forms[-1].vec
        base_utilde = _solve_with_inverse(inverse, base_rhs, space)
        responses = [_solve_with_inverse(inverse, column.vec, space) for column in p_forms[:-1]]

    mapped_points = _mapped_quadrature(mesh, quadrature_order)
    sampled_normalized_volume = _sample_scalar(normalized_volume, mapped_points)
    if not np.allclose(
        sampled_normalized_volume,
        volume_map.quadrature_normalized_volume,
        rtol=2.0e-12,
        atol=2.0e-12,
    ):
        raise RuntimeError("the G basis and M3b shell rows do not share the same s field")
    b_dot_grad_phi = ng.InnerProduct(magnetic_field, geometry.toroidal_angle_gradient)
    diamagnetic_dot_phi = ng.InnerProduct(
        ng.Cross(magnetic_field, pressure_gradient) / safe_magnitude**2,
        geometry.toroidal_angle_gradient,
    )

    def utilde_moments(field: Any) -> ShellCurrentMoments:
        gradient = _embedded_gradient(ng.grad(field))
        regularized = _regularized_gradient(
            gradient,
            direction,
            resolved_runtime.regularization_gradient,
        )
        return _component_moments(
            volume_map,
            edges,
            mapped_points,
            parallel=field * b_dot_grad_phi,
            regularizing=(
                -coefficients.current_diffusivity
                * ng.InnerProduct(regularized, geometry.toroidal_angle_gradient)
            ),
        )

    base_moments = utilde_moments(base_utilde)
    response_rows = np.column_stack(
        [utilde_moments(response).shellwise("total") for response in responses]
    )
    g_rows = np.column_stack(
        [
            _component_moments(
                volume_map,
                edges,
                mapped_points,
                parallel=basis * b_dot_grad_phi,
            ).shellwise("total")
            for basis, _ in g_basis[:-1]
        ]
    )
    edge_g_rows = _component_moments(
        volume_map,
        edges,
        mapped_points,
        parallel=g_basis[-1][0] * b_dot_grad_phi,
    ).shellwise("total")
    diamagnetic_moments = _component_moments(
        volume_map,
        edges,
        mapped_points,
        diamagnetic=diamagnetic_dot_phi,
    )
    target_shellwise = np.diff(target_cumulative)
    schur = g_rows - response_rows
    schur_rhs = (
        target_shellwise
        - diamagnetic_moments.shellwise("total")
        - edge_value * edge_g_rows
        - base_moments.shellwise("total")
    )
    schur_condition_number = float(np.linalg.cond(schur))
    if not isfinite(schur_condition_number):
        raise RuntimeError("the M3-M3b Schur complement is singular")
    try:
        free_g_coefficients = np.linalg.solve(schur, schur_rhs)
    except np.linalg.LinAlgError as error:
        raise RuntimeError("the M3-M3b Schur complement solve failed") from error

    utilde = ng.GridFunction(space)
    utilde.vec.data = base_utilde.vec
    for coefficient, response in zip(free_g_coefficients, responses, strict=True):
        utilde.vec.data -= float(coefficient) * response.vec
    g_profile: Any = edge_value * g_basis[-1][0]
    g_gradient: Any = edge_value * g_basis[-1][1]
    for coefficient, (basis, basis_gradient) in zip(
        free_g_coefficients,
        g_basis[:-1],
        strict=True,
    ):
        g_profile += float(coefficient) * basis
        g_gradient += float(coefficient) * basis_gradient
    physical_u = g_profile + utilde
    utilde_gradient = _embedded_gradient(ng.grad(utilde))
    regularized_utilde_gradient = _regularized_gradient(
        utilde_gradient,
        direction,
        resolved_runtime.regularization_gradient,
    )
    parallel_current = physical_u * magnetic_field
    diamagnetic_current = ng.Cross(magnetic_field, pressure_gradient) / safe_magnitude**2
    regularizing_current = -coefficients.current_diffusivity * regularized_utilde_gradient
    physical_current = parallel_current + diamagnetic_current + regularizing_current
    independent_moments = _component_moments(
        volume_map,
        edges,
        mapped_points,
        parallel=ng.InnerProduct(parallel_current, geometry.toroidal_angle_gradient),
        diamagnetic=ng.InnerProduct(diamagnetic_current, geometry.toroidal_angle_gradient),
        regularizing=ng.InnerProduct(regularizing_current, geometry.toroidal_angle_gradient),
    )

    m3_residual = drive_form.vec.CreateVector()
    m3_residual.data = bilinear_form.mat * utilde.vec - drive_form.vec
    for coefficient, column in zip(free_g_coefficients, p_forms[:-1], strict=True):
        m3_residual.data += float(coefficient) * column.vec
    m3_residual.data += edge_value * p_forms[-1].vec
    free_m3_residual = ng.Projector(free_dofs, True) * m3_residual
    projected_rhs = ng.Projector(free_dofs, True) * base_rhs
    m3_residual_norm = float(ng.Norm(free_m3_residual))
    m3_relative_residual_norm = m3_residual_norm / max(1.0, float(ng.Norm(projected_rhs)))
    constraint_residual = independent_moments.shellwise("total") - target_shellwise
    constraint_residual_norm = float(np.linalg.norm(constraint_residual))
    constraint_relative_residual_norm = constraint_residual_norm / max(
        1.0,
        float(np.linalg.norm(target_shellwise)),
    )
    if m3_relative_residual_norm > residual_tolerance:
        raise RuntimeError(
            "constrained M3 solve failed: relative residual "
            f"{m3_relative_residual_norm:.3e} exceeds {residual_tolerance:.0e}"
        )
    if constraint_relative_residual_norm > residual_tolerance:
        raise RuntimeError(
            "constrained M3b solve failed: relative residual "
            f"{constraint_relative_residual_norm:.3e} exceeds {residual_tolerance:.0e}"
        )

    if supg_bilinear_form is not None and supg_drive_form is not None and tau is not None:
        supg_residual = supg_drive_form.vec.CreateVector()
        supg_residual.data = supg_bilinear_form.mat * utilde.vec - supg_drive_form.vec
        for coefficient, column in zip(free_g_coefficients, supg_p_forms[:-1], strict=True):
            supg_residual.data += float(coefficient) * column.vec
        supg_residual.data += edge_value * supg_p_forms[-1].vec
        free_supg_residual = ng.Projector(free_dofs, True) * supg_residual
        supg_stabilization_norm = float(ng.Norm(free_supg_residual))
        supg_tau_min, supg_tau_max = _quadrature_extrema(mesh, _compiled(tau), 20)
    else:
        supg_stabilization_norm = 0.0
        supg_tau_min = 0.0
        supg_tau_max = 0.0

    multiplier_current = coefficients.current_diffusivity * _regularized_gradient(
        g_gradient,
        direction,
        resolved_runtime.regularization_gradient,
    )
    g_advection_coupling = ng.InnerProduct(magnetic_field, g_gradient)
    g_reaction_coupling = reaction_coefficient * g_profile
    weights = volume_map.quadrature_weights
    parallel_samples = _sample_scalar(
        ng.InnerProduct(parallel_current, geometry.toroidal_angle_gradient), mapped_points
    )
    diamagnetic_samples = _sample_scalar(
        ng.InnerProduct(diamagnetic_current, geometry.toroidal_angle_gradient), mapped_points
    )
    regularizing_samples = _sample_scalar(
        ng.InnerProduct(regularizing_current, geometry.toroidal_angle_gradient), mapped_points
    )
    scalar_utilde_moments = _component_moments(
        volume_map,
        edges,
        mapped_points,
        parallel=utilde,
    )
    shell_volume_moments = _component_moments(
        volume_map,
        edges,
        mapped_points,
        parallel=1.0,
    )
    shell_means = scalar_utilde_moments.shellwise("total") / shell_volume_moments.shellwise("total")

    def l2(expression: Any) -> float:
        return sqrt(max(0.0, float(ng.Integrate(_compiled(expression), mesh, order=20))))

    diagnostics = {
        "m3_residual_norm": m3_residual_norm,
        "m3b_constraint_residual_norm": constraint_residual_norm,
        "m3_supg_stabilization_norm": supg_stabilization_norm,
        "m3_supg_tau_min": supg_tau_min,
        "m3_supg_tau_max": supg_tau_max,
        "schur_condition_number": schur_condition_number,
        "a_factorizations": 1.0,
        "a_response_solves": float(len(edges)),
        "maximum_shell_mean_utilde": float(np.max(np.abs(shell_means))),
        "multiplier_current_l2": l2(ng.InnerProduct(multiplier_current, multiplier_current)),
        "g_advection_coupling_l2": l2(g_advection_coupling**2),
        "g_reaction_coupling_l2": l2(g_reaction_coupling**2),
        "parallel_toroidal_current_l2": sqrt(float(np.dot(weights, parallel_samples**2))),
        "diamagnetic_toroidal_current_l2": sqrt(float(np.dot(weights, diamagnetic_samples**2))),
        "regularizing_toroidal_current_l2": sqrt(float(np.dot(weights, regularizing_samples**2))),
        "minimum_shell_width": float(np.min(np.diff(edges))),
        "minimum_shell_radial_cells": float(
            min(
                (right - left)
                / float(
                    np.max(
                        volume_map.quadrature_normalized_cell_widths[
                            (volume_map.quadrature_normalized_volume >= left)
                            & (volume_map.quadrature_normalized_volume <= right)
                        ]
                    )
                )
                for left, right in pairwise(edges)
            )
        ),
        "minimum_shell_mollifier_widths": float(
            min(
                (right - left)
                / float(
                    np.max(
                        volume_map.quadrature_normalized_mollifier_widths[
                            (volume_map.quadrature_normalized_volume >= left)
                            & (volume_map.quadrature_normalized_volume <= right)
                        ]
                    )
                )
                for left, right in pairwise(edges)
            )
        ),
    }
    if not all(isfinite(value) and value >= 0.0 for value in diagnostics.values()):
        raise RuntimeError("constrained M3-M3b solve produced a non-finite diagnostic")

    all_g_coefficients = tuple(float(value) for value in (*free_g_coefficients, edge_value))
    return _ConstrainedCurrentContinuitySolution(
        _mesh=mesh,
        _utilde=utilde,
        _physical_u=physical_u,
        _g_profile=g_profile,
        _g_gradient=g_gradient,
        _current=physical_current,
        _moments=independent_moments,
        polynomial_order=polynomial_order,
        regularization_gradient=resolved_runtime.regularization_gradient,
        stabilization=stabilization,
        shell_edges=tuple(float(value) for value in edges),
        g_coefficients=all_g_coefficients,
        m3_relative_residual_norm=m3_relative_residual_norm,
        constraint_relative_residual_norm=constraint_relative_residual_norm,
        schur_condition_number=schur_condition_number,
        target_cumulative_current=tuple(float(value) for value in target_cumulative),
        independent_cumulative_current=tuple(
            float(value) for value in independent_moments.cumulative("total")
        ),
        shell_constraint_residuals=tuple(float(value) for value in constraint_residual),
        diagnostics=diagnostics,
    )
