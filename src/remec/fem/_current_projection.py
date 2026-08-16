"""Internal compatible projection for note equations ``(M1)``--``(M3b)``."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, pi
from typing import Any

import numpy as np

from remec.fem.spaces import make_tetrahedral_de_rham_sequence


@dataclass(frozen=True, slots=True)
class CurrentMomentConstraint:
    r"""One linear shell-current row ``int W_j dot J dV = Delta I_0,j``."""

    weight: Any
    target: float
    name: str


@dataclass(frozen=True, slots=True)
class ConstraintRankDiagnostics:
    """Numerical-rank evidence for one HDiv-to-L2 divergence block."""

    rows: int
    columns: int
    rank: int
    retained_singular_value_ratio: float
    first_discarded_singular_value_ratio: float


@dataclass(frozen=True, slots=True)
class CurrentProjectionSolution:
    """Projected current, continuity multiplier, and mandatory diagnostics."""

    current_density: Any
    continuity_multiplier: Any
    moment_multipliers: tuple[float, ...]
    base_order: int
    hdiv_order: int
    terminal_order: int
    raw_current_l2_norm: float
    projected_current_l2_norm: float
    pre_projection_divergence_relative_norm: float
    post_projection_divergence_relative_norm: float
    boundary_normal_relative_norm: float
    projection_correction_relative_norm: float
    continuity_multiplier_l2_norm: float
    continuity_multiplier_relative_norm: float
    free_dof_relative_residual: float
    ampere_compatibility_relative_residual: float
    raw_moments: tuple[float, ...]
    target_moments: tuple[float, ...]
    projected_moments: tuple[float, ...]
    raw_cumulative_moments: tuple[float, ...]
    target_cumulative_moments: tuple[float, ...]
    projected_cumulative_moments: tuple[float, ...]
    raw_moment_relative_residuals: tuple[float, ...]
    moment_relative_residuals: tuple[float, ...]
    raw_cumulative_moment_relative_residuals: tuple[float, ...]
    cumulative_moment_relative_residuals: tuple[float, ...]


def verification_mollified_shell_moment_weights(
    normalized_volume: Any,
    toroidal_angle_gradient: Any,
    shell_edges: Sequence[float],
    *,
    mollifier_width: float,
) -> tuple[Any, ...]:
    r"""Build fixed-in-``s`` note-``(M3b)`` weights for manufactured verification.

    For shell edges ``s_j``, this returns

    ``W_j = [H_eps(s_j-s)-H_eps(s_{j-1}-s)] grad(phi)/(2*pi)``,

    with exact zero/one endpoint memberships and the compact moment-matched
    ``H_eps`` from note equation ``(mollified_V)``.  The scalar width is deliberately
    fixed in normalized-volume space, so this helper is only for analytic manufactured
    tests.  Production ``(M3b)`` rows must be supplied through
    :class:`CurrentMomentConstraint` using the shared gradient-scaled volume-map
    mollifier and its resolution guards from milestone 3.6.
    """
    if getattr(normalized_volume, "dim", None) != 1:
        raise ValueError("normalized_volume must be a scalar coefficient function")
    if getattr(toroidal_angle_gradient, "dim", None) != 3:
        raise ValueError("toroidal_angle_gradient must have three components")
    if not isfinite(mollifier_width) or mollifier_width <= 0.0:
        raise ValueError("mollifier_width must be finite and positive")
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

    import ngsolve as ng  # type: ignore[import-untyped]

    memberships: list[Any] = [0.0]
    for edge in edges[1:-1]:
        argument = (float(edge) - normalized_volume) / mollifier_width
        transition = 0.5 * (1.0 + argument + ng.sin(pi * argument) / pi)
        memberships.append(ng.IfPos(argument + 1.0, ng.IfPos(1.0 - argument, transition, 1.0), 0.0))
    memberships.append(1.0)
    return tuple(
        (memberships[index + 1] - memberships[index]) * toroidal_angle_gradient / (2.0 * pi)
        for index in range(len(edges) - 1)
    )


def _validated_order(value: int, name: str) -> int:
    """Return one non-negative integer finite-element order."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _constraint_matrix(
    mesh: Any,
    *,
    hdiv_order: int,
    terminal_order: int,
) -> tuple[Any, Any, Any]:
    r"""Assemble ``B[q,v]=(q, div(v))`` for the ``(M3)`` continuity row."""
    import ngsolve as ng

    current_space = ng.HDiv(mesh, order=hdiv_order)
    terminal_space = ng.L2(mesh, order=terminal_order)
    mixed_space = ng.FESpace([current_space, terminal_space])
    current_trial, _ = mixed_space.TrialFunction()
    _, terminal_test = mixed_space.TestFunction()
    form = ng.BilinearForm(mixed_space, check_unused=False)
    form += ng.div(current_trial) * terminal_test * ng.dx
    form.Assemble()
    return current_space, terminal_space, form.mat


def analyze_divergence_constraint_rank(
    mesh: Any,
    *,
    hdiv_order: int,
    terminal_order: int,
    relative_tolerance: float = 1.0e-10,
    element_index: int | None = None,
) -> ConstraintRankDiagnostics:
    """Measure the numerical rank of the paired divergence constraint block.

    ``element_index=None`` analyzes the global block.  Selecting one element analyzes
    all of that element's discontinuous terminal rows against the global HDiv columns;
    this cheap local test proves redundant oversized rows on a large curved mesh
    without making singular-factorization behavior an acceptance observable.
    """
    _validated_order(hdiv_order, "hdiv_order")
    _validated_order(terminal_order, "terminal_order")
    if not isfinite(relative_tolerance) or not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must be finite and lie in (0, 1)")
    current_space, terminal_space, matrix = _constraint_matrix(
        mesh,
        hdiv_order=hdiv_order,
        terminal_order=terminal_order,
    )
    row_indices: np.ndarray[Any, np.dtype[np.int64]]
    if element_index is None:
        row_indices = np.arange(terminal_space.ndof, dtype=np.int64)
    else:
        if isinstance(element_index, bool) or not isinstance(element_index, int):
            raise TypeError("element_index must be an integer or None")
        volume_elements = list(mesh.Elements())
        if not 0 <= element_index < len(volume_elements):
            raise ValueError("element_index is outside the volume-element range")
        row_indices = np.asarray(
            [int(dof) for dof in terminal_space.GetDofNrs(volume_elements[element_index])],
            dtype=np.int64,
        )
        row_indices = row_indices[row_indices >= 0]

    dense = np.zeros((len(row_indices), current_space.ndof), dtype=float)
    local_rows = {int(row): index for index, row in enumerate(row_indices)}
    coo_rows, coo_columns, coo_values = matrix.COO()
    current_dofs = current_space.ndof
    for row, column, value in zip(coo_rows, coo_columns, coo_values, strict=True):
        terminal_row = int(row) - current_dofs
        local_row = local_rows.get(terminal_row)
        if local_row is not None and int(column) < current_dofs:
            dense[local_row, int(column)] += float(value)
    singular_values = np.linalg.svd(dense, compute_uv=False)
    maximum = float(singular_values[0]) if len(singular_values) else 0.0
    if maximum == 0.0:
        rank = 0
        retained_ratio = 0.0
        discarded_ratio = 0.0
    else:
        ratios = singular_values / maximum
        rank = int(np.count_nonzero(ratios > relative_tolerance))
        retained_ratio = float(ratios[rank - 1]) if rank else 0.0
        discarded_ratio = float(ratios[rank]) if rank < len(ratios) else 0.0
    return ConstraintRankDiagnostics(
        rows=dense.shape[0],
        columns=dense.shape[1],
        rank=rank,
        retained_singular_value_ratio=retained_ratio,
        first_discarded_singular_value_ratio=discarded_ratio,
    )


def _l2_norm(mesh: Any, coefficient: Any, *, integration_order: int) -> float:
    """Return a scalar or vector physical L2 norm."""
    import ngsolve as ng

    integrand = (
        coefficient**2
        if getattr(coefficient, "dim", None) == 1
        else ng.InnerProduct(coefficient, coefficient)
    )
    return float(ng.sqrt(ng.Integrate(integrand, mesh, order=integration_order)))


def _ampere_compatibility_residual(
    mesh: Any,
    current_density: Any,
    *,
    test_order: int,
    integration_order: int,
) -> float:
    r"""Return the H1-dual residual of ``(J_h, grad(q))=0`` from ``(M1)``."""
    import ngsolve as ng

    test_space = ng.H1(mesh, order=test_order, dirichlet=".*")
    trial, test = test_space.TnT()
    riesz = ng.BilinearForm(test_space)
    riesz += (ng.InnerProduct(ng.grad(trial), ng.grad(test)) + trial * test) * ng.dx
    residual = ng.LinearForm(test_space)
    residual += ng.InnerProduct(current_density, ng.grad(test)) * ng.dx
    riesz.Assemble()
    residual.Assemble()
    free = test_space.FreeDofs()
    representer = ng.GridFunction(test_space)
    representer.vec.data = riesz.mat.Inverse(free, inverse="sparsecholesky") * residual.vec
    dual_product = float(abs(residual.vec.InnerProduct(representer.vec)))
    current_norm = _l2_norm(mesh, current_density, integration_order=integration_order)
    return float(np.sqrt(dual_product) / max(current_norm, np.finfo(float).tiny))


def solve_constrained_current_projection(
    mesh: Any,
    raw_current: Any,
    *,
    base_order: int,
    raw_divergence: Any | None = None,
    terminal_order: int | None = None,
    moment_constraints: Sequence[CurrentMomentConstraint] = (),
    bonus_integration_order: int = 8,
    moment_integration_order: int | None = None,
    ampere_test_order: int | None = None,
) -> CurrentProjectionSolution:
    r"""Solve the mixed projection required by ``(M1)``--``(M3b)`` and Section 10.

    The nontrivial weak form is

    ``(J_h,v) + (lambda_h,div(v)) + (div(J_h),q)``
    ``+ sum_j alpha_j M_j(v) + sum_j beta_j M_j(J_h)``
    ``= (J_raw,v) + sum_j beta_j Delta I_0,j``,

    where ``J_raw = u B + B cross grad(p)/B_safe**2 - D_u grad_r(utilde)`` is note
    equation ``(M2)`` and ``M_j(J)=int W_j dot J dV`` are the mollified shell rows of
    ``(M3b)``.  The paired L2 constraint makes ``div(J_h)=0`` pointwise on affine and
    curved tetrahedra (ADR 0005), which is the compatibility condition for Ampere's
    law ``(M1)``.  The HDiv space deliberately uses the design's natural-trace form;
    essential normal-trace data require a separately derived lifting and are not
    exposed by this solver.
    """
    if getattr(raw_current, "dim", None) != 3:
        raise ValueError("raw_current must be a three-component coefficient function")
    if raw_divergence is not None and getattr(raw_divergence, "dim", None) != 1:
        raise ValueError("raw_divergence must be a scalar coefficient function or None")
    if isinstance(bonus_integration_order, bool) or not isinstance(bonus_integration_order, int):
        raise TypeError("bonus_integration_order must be an integer")
    if bonus_integration_order < 0:
        raise ValueError("bonus_integration_order must be non-negative")
    sequence = make_tetrahedral_de_rham_sequence(mesh, order=base_order)
    resolved_terminal_order = (
        sequence.l2_order
        if terminal_order is None
        else _validated_order(terminal_order, "terminal_order")
    )
    resolved_test_order = (
        sequence.h1_order
        if ampere_test_order is None
        else _validated_order(ampere_test_order, "ampere_test_order")
    )
    resolved_moment_order = (
        2 * max(sequence.hdiv_order, 1) + bonus_integration_order
        if moment_integration_order is None
        else _validated_order(moment_integration_order, "moment_integration_order")
    )
    if resolved_moment_order < 1:
        raise ValueError("moment_integration_order must be at least one")
    constraints = tuple(moment_constraints)
    names: set[str] = set()
    for constraint in constraints:
        if not isinstance(constraint, CurrentMomentConstraint):
            raise TypeError("moment_constraints must contain CurrentMomentConstraint values")
        if getattr(constraint.weight, "dim", None) != 3:
            raise ValueError("every moment weight must have three components")
        if not isfinite(constraint.target):
            raise ValueError("every moment target must be finite")
        if not constraint.name or constraint.name in names:
            raise ValueError("moment constraint names must be non-empty and unique")
        names.add(constraint.name)

    import ngsolve as ng

    current_space = ng.HDiv(mesh, order=sequence.hdiv_order)
    terminal_space = ng.L2(mesh, order=resolved_terminal_order)
    number_spaces = [ng.NumberSpace(mesh) for _ in constraints]
    mixed_space = ng.FESpace([current_space, terminal_space, *number_spaces])
    trials = mixed_space.TrialFunction()
    tests = mixed_space.TestFunction()
    current_trial, multiplier_trial = trials[:2]
    current_test, multiplier_test = tests[:2]
    dx = ng.dx(bonus_intorder=bonus_integration_order)
    moment_dx = ng.dx(intrules={ng.ET.TET: ng.IntegrationRule(ng.ET.TET, resolved_moment_order)})
    operator = ng.BilinearForm(mixed_space, symmetric=True)
    operator += (
        ng.InnerProduct(current_trial, current_test)
        + multiplier_trial * ng.div(current_test)
        + ng.div(current_trial) * multiplier_test
    ) * dx
    right_hand_side = ng.LinearForm(mixed_space)
    right_hand_side += ng.InnerProduct(raw_current, current_test) * dx
    domain_volume = float(ng.Integrate(1.0, mesh, order=resolved_moment_order))
    for index, constraint in enumerate(constraints):
        moment_trial = trials[index + 2]
        moment_test = tests[index + 2]
        operator += (
            moment_trial * ng.InnerProduct(constraint.weight, current_test)
            + moment_test * ng.InnerProduct(constraint.weight, current_trial)
        ) * moment_dx
        right_hand_side += constraint.target * moment_test / domain_volume * moment_dx
    operator.Assemble()
    right_hand_side.Assemble()

    mixed_field = ng.GridFunction(mixed_space)
    mixed_field.vec.data = (
        operator.mat.Inverse(mixed_space.FreeDofs(), inverse="umfpack") * right_hand_side.vec
    )
    components = mixed_field.components
    current_density = components[0]
    continuity_multiplier = components[1]
    moment_multipliers = tuple(
        float(components[index + 2].vec.FV().NumPy()[0]) for index in range(len(constraints))
    )

    free_projector = ng.Projector(mixed_space.FreeDofs(), True)
    residual = free_projector * (operator.mat * mixed_field.vec - right_hand_side.vec)
    free_rhs = free_projector * right_hand_side.vec
    free_dof_relative_residual = float(ng.Norm(residual)) / max(
        float(ng.Norm(free_rhs)), np.finfo(float).tiny
    )
    integration_order = (
        2
        * max(
            sequence.hdiv_order,
            resolved_terminal_order,
            resolved_test_order,
            1,
        )
        + bonus_integration_order
        + 4
    )
    raw_current_l2_norm = _l2_norm(mesh, raw_current, integration_order=integration_order)
    projected_current_l2_norm = _l2_norm(mesh, current_density, integration_order=integration_order)
    if raw_divergence is None:
        try:
            resolved_raw_divergence = ng.div(raw_current)
        except (AttributeError, TypeError) as error:
            raise ValueError(
                "raw_divergence is required when raw_current has no native divergence operator"
            ) from error
    else:
        resolved_raw_divergence = raw_divergence
    pre_projection_divergence_relative_norm = _l2_norm(
        mesh, resolved_raw_divergence, integration_order=integration_order
    ) / max(raw_current_l2_norm, np.finfo(float).tiny)
    post_projection_divergence_relative_norm = _l2_norm(
        mesh, ng.div(current_density), integration_order=integration_order
    ) / max(projected_current_l2_norm, np.finfo(float).tiny)
    correction = current_density - raw_current
    projection_correction_relative_norm = _l2_norm(
        mesh, correction, integration_order=integration_order
    ) / max(raw_current_l2_norm, np.finfo(float).tiny)
    continuity_multiplier_l2_norm = _l2_norm(
        mesh, continuity_multiplier, integration_order=integration_order
    )
    continuity_multiplier_relative_norm = continuity_multiplier_l2_norm / max(
        projected_current_l2_norm, np.finfo(float).tiny
    )
    normal = ng.specialcf.normal(3)
    boundary_normal_relative_norm = float(
        ng.sqrt(
            ng.Integrate(
                (current_density * normal) ** 2,
                mesh,
                ng.BND,
                order=integration_order,
            )
        )
        / max(projected_current_l2_norm, np.finfo(float).tiny)
    )
    raw_moments = tuple(
        float(
            ng.Integrate(
                ng.InnerProduct(raw_current, constraint.weight),
                mesh,
                order=resolved_moment_order,
            )
        )
        for constraint in constraints
    )
    projected_moments = tuple(
        float(
            ng.Integrate(
                ng.InnerProduct(current_density, constraint.weight),
                mesh,
                order=resolved_moment_order,
            )
        )
        for constraint in constraints
    )
    target_moments = tuple(constraint.target for constraint in constraints)
    raw_moment_relative_residuals = tuple(
        abs(actual - target) / max(abs(actual), abs(target), 1.0)
        for actual, target in zip(raw_moments, target_moments, strict=True)
    )
    moment_relative_residuals = tuple(
        abs(actual - target) / max(abs(actual), abs(target), 1.0)
        for actual, target in zip(projected_moments, target_moments, strict=True)
    )
    raw_cumulative_moments = (0.0, *np.cumsum(raw_moments).tolist())
    target_cumulative_moments = (0.0, *np.cumsum(target_moments).tolist())
    projected_cumulative_moments = (0.0, *np.cumsum(projected_moments).tolist())
    raw_cumulative_moment_relative_residuals = tuple(
        abs(actual - target) / max(abs(actual), abs(target), 1.0)
        for actual, target in zip(raw_cumulative_moments, target_cumulative_moments, strict=True)
    )
    cumulative_moment_relative_residuals = tuple(
        abs(actual - target) / max(abs(actual), abs(target), 1.0)
        for actual, target in zip(
            projected_cumulative_moments, target_cumulative_moments, strict=True
        )
    )
    ampere_compatibility_relative_residual = _ampere_compatibility_residual(
        mesh,
        current_density,
        test_order=resolved_test_order,
        integration_order=integration_order,
    )
    return CurrentProjectionSolution(
        current_density=current_density,
        continuity_multiplier=continuity_multiplier,
        moment_multipliers=moment_multipliers,
        base_order=base_order,
        hdiv_order=sequence.hdiv_order,
        terminal_order=resolved_terminal_order,
        raw_current_l2_norm=raw_current_l2_norm,
        projected_current_l2_norm=projected_current_l2_norm,
        pre_projection_divergence_relative_norm=pre_projection_divergence_relative_norm,
        post_projection_divergence_relative_norm=post_projection_divergence_relative_norm,
        boundary_normal_relative_norm=boundary_normal_relative_norm,
        projection_correction_relative_norm=projection_correction_relative_norm,
        continuity_multiplier_l2_norm=continuity_multiplier_l2_norm,
        continuity_multiplier_relative_norm=continuity_multiplier_relative_norm,
        free_dof_relative_residual=free_dof_relative_residual,
        ampere_compatibility_relative_residual=ampere_compatibility_relative_residual,
        raw_moments=raw_moments,
        target_moments=target_moments,
        projected_moments=projected_moments,
        raw_cumulative_moments=raw_cumulative_moments,
        target_cumulative_moments=target_cumulative_moments,
        projected_cumulative_moments=projected_cumulative_moments,
        raw_moment_relative_residuals=raw_moment_relative_residuals,
        moment_relative_residuals=moment_relative_residuals,
        raw_cumulative_moment_relative_residuals=raw_cumulative_moment_relative_residuals,
        cumulative_moment_relative_residuals=cumulative_moment_relative_residuals,
    )
