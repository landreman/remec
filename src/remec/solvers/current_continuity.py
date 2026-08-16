"""Public interface for the regularized current-continuity equation (M3)."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from remec.common.logging import JsonEventLogger
from remec.common.serialization import configuration_digest
from remec.fem._constrained_current_continuity import solve_constrained_current_continuity
from remec.fem._current_continuity import solve_frozen_current_continuity
from remec.geometry.slab import Slab2D
from remec.options import (
    CurrentContinuityStabilization,
    RegularizationGradient,
    RuntimeOptions,
)
from remec.profiles import ToroidalCurrentProfile

if TYPE_CHECKING:
    from remec.common.checkpoint import ConstrainedCurrentCheckpoint

CurrentContinuityFormulation = Literal["direct-u", "utilde"]


class UnresolvedCurrentLayerWarning(RuntimeWarning):
    """Warning that the DESIGN §5 current-layer resolution gate failed."""


class UnresolvedCurrentLayerError(RuntimeError):
    """Strict-mode failure for an unresolved note-``layer_width`` current layer."""


@dataclass(frozen=True, slots=True)
class CurrentLayerResolutionDiagnostic:
    r"""Resolution report for the (M3) layer ``delta proportional D_u**(1/3)``."""

    layer_width: float
    normal_element_width: float
    cells_across_layer: float
    minimum_cells: int
    resolved: bool


@dataclass(frozen=True, slots=True)
class FrozenCurrentContinuityCoefficients:
    """Frozen coefficients required by the direct-u form of note equation (M3).

    ``magnetic_magnitude_gradient`` is normally ``grad(|B|)`` derived from
    ``magnetic_field``. It is explicit so verification can inject a prescribed
    strong-form M3 drive; production callers are responsible for their consistency.
    ``magnetic_field_gradient``, when supplied, is either NGSolve's native 2-by-3
    vector gradient or the transposed 3-by-2 matrix ``(∂_x B_i, ∂_y B_i)`` required by
    the perpendicular SUPG strong divergence. It is mandatory for a varying
    GridFunction-backed magnetic field because NGSolve coordinate differentiation of
    such a field silently returns zero.
    """

    magnetic_field: Any
    pressure_gradient: Any
    magnetic_magnitude_gradient: Any
    current_diffusivity: float
    magnetic_field_gradient: Any | None = None
    magnetic_floor: float = 1.0e-12
    vacuum_permeability: float = 1.0

    def __post_init__(self) -> None:
        if not isfinite(self.current_diffusivity) or self.current_diffusivity <= 0.0:
            raise ValueError("current_diffusivity must be finite and positive")
        if not isfinite(self.magnetic_floor) or self.magnetic_floor <= 0.0:
            raise ValueError("magnetic_floor must be finite and positive")
        if not isfinite(self.vacuum_permeability) or self.vacuum_permeability <= 0.0:
            raise ValueError("vacuum_permeability must be finite and positive")


@dataclass(frozen=True, slots=True)
class FrozenCurrentConstraintGeometry:
    r"""Frozen normalized-volume and toroidal geometry for note equation ``(M3b)``.

    ``level_set`` and ``level_set_gradient`` define the one shared
    ``s=V_chi(level_set)/V_omega`` field used by both the unknown ``G(s)`` and the
    mollified shell constraints. ``toroidal_angle_gradient`` is ``grad(phi)`` in
    ``I_tor=(2*pi)**-1 integral J.grad(phi) dV``.
    """

    level_set: Any
    level_set_gradient: Any
    toroidal_angle_gradient: Any

    def __post_init__(self) -> None:
        if getattr(self.level_set, "dim", 1) != 1:
            raise ValueError("level_set must be scalar")
        for name in ("level_set_gradient", "toroidal_angle_gradient"):
            if getattr(getattr(self, name), "dim", None) != 3:
                raise ValueError(f"{name} must be a three-component coefficient function")


@dataclass(frozen=True, slots=True)
class ConstrainedCurrentContinuityResult:
    r"""Public result of the bordered note-``(M3)``--``(M3b)`` solve."""

    polynomial_order: int
    regularization_gradient: RegularizationGradient
    stabilization: CurrentContinuityStabilization
    shell_edges: tuple[float, ...]
    g_coefficients: tuple[float, ...]
    configuration_digest: str
    m3_relative_residual_norm: float
    constraint_relative_residual_norm: float
    schur_condition_number: float
    target_cumulative_current: tuple[float, ...]
    independent_cumulative_current: tuple[float, ...]
    shell_constraint_residuals: tuple[float, ...]
    diagnostics: dict[str, float]

    def checkpoint_state(self) -> ConstrainedCurrentCheckpoint:
        r"""Return restart state for the solved ``G`` border and ``(M3b)`` rows."""
        from remec.common.checkpoint import ConstrainedCurrentCheckpoint

        return ConstrainedCurrentCheckpoint(
            shell_edges=self.shell_edges,
            g_coefficients=self.g_coefficients,
            edge_value=self.g_coefficients[-1],
            shell_constraint_residuals=self.shell_constraint_residuals,
            m3_relative_residual_norm=self.m3_relative_residual_norm,
            m3b_relative_residual_norm=self.constraint_relative_residual_norm,
        )


class ConstrainedCurrentContinuitySolver:
    r"""Joint unknown-``G`` Schur solve for note equations ``(M3)``--``(M3b)``.

    The solver represents ``G`` on the supplied normalized-volume shell grid, fixes
    ``G(1)=edge_value``, applies current diffusion only to homogeneous ``utilde``, and
    uses independently reconstructed physical ``(M2)`` current samples for its public
    ``I_tor`` diagnostics.
    """

    def __init__(
        self,
        *,
        polynomial_order: int = 3,
        runtime: RuntimeOptions | None = None,
        stabilization: CurrentContinuityStabilization = "supg",
        quadrature_order: int = 8,
        volume_levels: int = 17,
        spatial_width_cells: float = 1.0,
        diagnostic_detail: Literal["full", "core"] = "full",
        logger: JsonEventLogger | None = None,
    ) -> None:
        if polynomial_order < 1:
            raise ValueError("polynomial_order must be at least one")
        if stabilization not in ("none", "supg"):
            raise ValueError("stabilization must be 'none' or 'supg'")
        if quadrature_order < 2:
            raise ValueError("quadrature_order must be at least two")
        if volume_levels < 3:
            raise ValueError("volume_levels must be at least three")
        if not isfinite(spatial_width_cells) or spatial_width_cells <= 0.0:
            raise ValueError("spatial_width_cells must be finite and positive")
        if diagnostic_detail not in ("full", "core"):
            raise ValueError("diagnostic_detail must be 'full' or 'core'")
        self.polynomial_order = polynomial_order
        self.runtime = RuntimeOptions() if runtime is None else runtime
        self.stabilization = stabilization
        self.quadrature_order = quadrature_order
        self.volume_levels = volume_levels
        self.spatial_width_cells = spatial_width_cells
        self.diagnostic_detail = diagnostic_detail
        self.logger = logger
        self._internal_solution: Any = None

    def solve(
        self,
        field: Slab2D,
        coefficients: FrozenCurrentContinuityCoefficients,
        geometry: FrozenCurrentConstraintGeometry,
        current_profile: ToroidalCurrentProfile,
        *,
        shell_edges: Sequence[float],
        edge_value: float = 0.0,
        boundary: str = "left|right",
    ) -> ConstrainedCurrentContinuityResult:
        r"""Solve the square bordered ``[A P; C_u C_G]`` system from ``(M3b)``."""
        if not isinstance(field, Slab2D):
            raise TypeError("the constrained M3-M3b kernel requires a Slab2D mesh")
        if not isinstance(coefficients, FrozenCurrentContinuityCoefficients):
            raise TypeError("coefficients must be FrozenCurrentContinuityCoefficients")
        if not isinstance(geometry, FrozenCurrentConstraintGeometry):
            raise TypeError("geometry must be FrozenCurrentConstraintGeometry")
        edges = tuple(float(value) for value in shell_edges)
        target = tuple(
            float(value)
            for value in np.asarray(current_profile.enclosed_current(edges), dtype=float).reshape(
                -1
            )
        )
        configuration = {
            "formulation": "constrained-unknown-g",
            "polynomial_order": self.polynomial_order,
            "runtime": self.runtime,
            "current_diffusivity": coefficients.current_diffusivity,
            "magnetic_floor": coefficients.magnetic_floor,
            "vacuum_permeability": coefficients.vacuum_permeability,
            "stabilization": self.stabilization,
            "quadrature_order": self.quadrature_order,
            "volume_levels": self.volume_levels,
            "spatial_width_cells": self.spatial_width_cells,
            "diagnostic_detail": self.diagnostic_detail,
            "shell_edges": edges,
            "target_cumulative_current": target,
            "edge_value": edge_value,
        }
        digest = configuration_digest(configuration)
        log_fields = {
            "configuration_digest": digest,
            "formulation": "constrained-unknown-g",
            "polynomial_order": self.polynomial_order,
            "regularization_gradient": self.runtime.regularization_gradient,
            "stabilization": self.stabilization,
            "shell_count": len(edges) - 1,
            "diagnostic_detail": self.diagnostic_detail,
        }
        if self.logger is not None:
            self.logger.info("m3_m3b_solve_started", **log_fields)
        internal = solve_constrained_current_continuity(
            field,
            polynomial_order=self.polynomial_order,
            coefficients=coefficients,
            geometry=geometry,
            current_profile=current_profile,
            shell_edges=edges,
            edge_value=edge_value,
            runtime=self.runtime,
            boundary=boundary,
            stabilization=self.stabilization,
            quadrature_order=self.quadrature_order,
            volume_levels=self.volume_levels,
            spatial_width_cells=self.spatial_width_cells,
            diagnostic_detail=self.diagnostic_detail,
        )
        self._internal_solution = internal
        if self.logger is not None:
            self.logger.info(
                "m3_m3b_solve_completed",
                **log_fields,
                m3_relative_residual_norm=internal.m3_relative_residual_norm,
                m3b_constraint_relative_residual_norm=(internal.constraint_relative_residual_norm),
                schur_condition_number=internal.schur_condition_number,
            )
        return ConstrainedCurrentContinuityResult(
            polynomial_order=internal.polynomial_order,
            regularization_gradient=internal.regularization_gradient,
            stabilization=internal.stabilization,
            shell_edges=internal.shell_edges,
            g_coefficients=internal.g_coefficients,
            configuration_digest=digest,
            m3_relative_residual_norm=internal.m3_relative_residual_norm,
            constraint_relative_residual_norm=internal.constraint_relative_residual_norm,
            schur_condition_number=internal.schur_condition_number,
            target_cumulative_current=internal.target_cumulative_current,
            independent_cumulative_current=internal.independent_cumulative_current,
            shell_constraint_residuals=internal.shell_constraint_residuals,
            diagnostics={
                **internal.diagnostics,
                "m3_relative_residual_norm": internal.m3_relative_residual_norm,
                "m3b_constraint_relative_residual_norm": (
                    internal.constraint_relative_residual_norm
                ),
            },
        )

    def _solution(self) -> Any:
        if self._internal_solution is None:
            raise RuntimeError("solve must be called before evaluating the constrained result")
        return self._internal_solution

    def solution_at(self, x_coordinate: float, y_coordinate: float) -> float:
        """Return reconstructed physical ``u=G(s)+utilde`` at one point."""
        return float(self._solution().solution_at(x_coordinate, y_coordinate))

    def utilde_at(self, x_coordinate: float, y_coordinate: float) -> float:
        """Return the homogeneous finite-element unknown at one point."""
        return float(self._solution().utilde_at(x_coordinate, y_coordinate))

    def g_at(self, x_coordinate: float, y_coordinate: float) -> float:
        """Return the solved one-dimensional multiplier at one point."""
        return float(self._solution().g_at(x_coordinate, y_coordinate))

    def current_at(self, x_coordinate: float, y_coordinate: float) -> tuple[float, float, float]:
        """Return the independently reconstructible physical note-``(M2)`` current."""
        value: tuple[float, float, float] = self._solution().current_at(x_coordinate, y_coordinate)
        return value

    def parallel_current_over_field_at(
        self,
        x_coordinate: float,
        y_coordinate: float,
    ) -> float:
        r"""Return physical ``J_parallel/B``; full grad includes the ``utilde`` correction."""
        return float(self._solution().parallel_current_over_field_at(x_coordinate, y_coordinate))


@dataclass(frozen=True, slots=True)
class PrescribedCurrentProfile:
    r"""Legacy verification-only ``F(p)`` data for ``u = F(p) + utilde``.

    This superseded parameterization cannot impose a physical current profile and is
    intentionally absent from :mod:`remec.solvers` exports and checkpoint profile
    records. It remains here only for milestone 3.6's required two-F cancellation
    negative control. ``identifier`` is a stable caller-owned provenance key included in the solve
    configuration digest and structured records. ``value`` is ``F(p)`` and
    ``pressure_derivative`` is ``F'(p)``. Together with
    ``FrozenCurrentContinuityCoefficients.pressure_gradient`` they define
    ``grad(F(p)) = F'(p) grad(p)``. The two explicit divergence coefficients are
    ``div(F'(p) grad_perp(p))`` and ``div(F'(p) grad(p))``, where
    ``grad_perp = (I - b_safe b_safe^T) grad`` is the note-literal single projection;
    the solver selects the runtime variant for the SUPG strong source. The Galerkin
    load instead moves its symmetric ``grad_r(v).grad_r(F)`` term directly, so it
    remains algebraically identical to direct-u when the B floor is active. The
    divergence coefficients are explicit because NGSolve coordinate differentiation
    can silently return zero for a GridFunction-backed pressure gradient.
    """

    identifier: str
    value: Any
    pressure_derivative: Any
    perpendicular_gradient_divergence: Any
    full_gradient_divergence: Any

    def __post_init__(self) -> None:
        if not isinstance(self.identifier, str):
            raise TypeError("identifier must be a string")
        if not self.identifier or self.identifier.strip() != self.identifier:
            raise ValueError("identifier must be non-empty without surrounding whitespace")
        for name in (
            "value",
            "pressure_derivative",
            "perpendicular_gradient_divergence",
            "full_gradient_divergence",
        ):
            if getattr(getattr(self, name), "dim", 1) != 1:
                raise ValueError(f"{name} must be scalar")


@dataclass(frozen=True, slots=True)
class CurrentContinuityResult:
    """Scalar public result for a frozen-field direct-u or utilde solve of (M3)."""

    formulation: CurrentContinuityFormulation
    profile_identifier: str | None
    polynomial_order: int
    regularization_gradient: RegularizationGradient
    stabilization: CurrentContinuityStabilization
    configuration_digest: str
    free_dof_residual_norm: float
    free_dof_relative_residual_norm: float
    diagnostics: dict[str, float]


class CurrentContinuitySolver:
    r"""Direct-u note-``(M2)``--``(M3)`` kernel with a legacy F-shift oracle.

    The selected ``grad_r`` is used consistently in
    ``J = uB + B x grad(p)/B_safe^2 - D_u grad_r(u)`` and in the M3 weak form
    documented by :func:`remec.fem._current_continuity.solve_frozen_current_continuity`.
    The DESIGN §9.1 baseline is ``stabilization="supg"``; ``"none"`` retains the
    unstabilized Galerkin form for cross-verification.
    """

    def __init__(
        self,
        *,
        polynomial_order: int = 3,
        runtime: RuntimeOptions | None = None,
        stabilization: CurrentContinuityStabilization = "supg",
        logger: JsonEventLogger | None = None,
    ) -> None:
        if polynomial_order < 1:
            raise ValueError("polynomial_order must be at least one")
        if stabilization not in ("none", "supg"):
            raise ValueError("stabilization must be 'none' or 'supg'")
        self.polynomial_order = polynomial_order
        self.runtime = RuntimeOptions() if runtime is None else runtime
        self.stabilization = stabilization
        self.logger = logger
        self._internal_solution: Any = None

    def solve(
        self,
        field: Slab2D,
        coefficients: FrozenCurrentContinuityCoefficients,
        *,
        boundary: str = "bottom|right|top|left",
        boundary_value: Any = 0.0,
    ) -> CurrentContinuityResult:
        """Solve the frozen-field direct-u weak form of note equation (M3)."""
        return self._solve(
            field,
            coefficients,
            formulation="direct-u",
            profile=None,
            boundary=boundary,
            boundary_value=boundary_value,
        )

    def _solve(
        self,
        field: Slab2D,
        coefficients: FrozenCurrentContinuityCoefficients,
        *,
        formulation: CurrentContinuityFormulation,
        profile: PrescribedCurrentProfile | None,
        boundary: str,
        boundary_value: Any,
    ) -> CurrentContinuityResult:
        """Run one validated direct-u or transformed frozen-field solve."""
        if not isinstance(field, Slab2D):
            raise TypeError("the M3 kernel requires a Slab2D mesh")
        if not isinstance(coefficients, FrozenCurrentContinuityCoefficients):
            raise TypeError("coefficients must be FrozenCurrentContinuityCoefficients")
        if formulation == "utilde" and not isinstance(profile, PrescribedCurrentProfile):
            raise TypeError("profile must be PrescribedCurrentProfile")
        configuration = {
            "formulation": formulation,
            "profile_identifier": None if profile is None else profile.identifier,
            "polynomial_order": self.polynomial_order,
            "runtime": self.runtime,
            "current_diffusivity": coefficients.current_diffusivity,
            "magnetic_floor": coefficients.magnetic_floor,
            "vacuum_permeability": coefficients.vacuum_permeability,
            "stabilization": self.stabilization,
        }
        digest = configuration_digest(configuration)
        log_fields = {
            "configuration_digest": digest,
            "formulation": formulation,
            "profile_identifier": None if profile is None else profile.identifier,
            "polynomial_order": self.polynomial_order,
            "regularization_gradient": self.runtime.regularization_gradient,
            "stabilization": self.stabilization,
        }
        if self.logger is not None:
            self.logger.info("m3_solve_started", **log_fields)

        internal = solve_frozen_current_continuity(
            field,
            polynomial_order=self.polynomial_order,
            coefficients=coefficients,
            runtime=self.runtime,
            prescribed_current_profile=profile,
            boundary=boundary,
            boundary_value=boundary_value,
            stabilization=self.stabilization,
        )
        self._internal_solution = internal
        diagnostics = {
            **internal.diagnostics,
            "free_dof_residual_norm": internal.free_dof_residual_norm,
            "free_dof_relative_residual_norm": internal.free_dof_relative_residual_norm,
        }
        if self.logger is not None:
            self.logger.info(
                "m3_solve_completed",
                **log_fields,
                free_dof_relative_residual_norm=internal.free_dof_relative_residual_norm,
                m3_supg_stabilization_norm=internal.diagnostics["m3_supg_stabilization_norm"],
            )
        return CurrentContinuityResult(
            formulation=formulation,
            profile_identifier=None if profile is None else profile.identifier,
            polynomial_order=self.polynomial_order,
            regularization_gradient=self.runtime.regularization_gradient,
            stabilization=self.stabilization,
            configuration_digest=digest,
            free_dof_residual_norm=internal.free_dof_residual_norm,
            free_dof_relative_residual_norm=internal.free_dof_relative_residual_norm,
            diagnostics=diagnostics,
        )

    def solve_utilde(
        self,
        field: Slab2D,
        coefficients: FrozenCurrentContinuityCoefficients,
        profile: PrescribedCurrentProfile,
        *,
        boundary: str = "bottom|right|top|left",
    ) -> CurrentContinuityResult:
        r"""Run the deprecated algebraic F-shift negative-control solve.

        This is not the constrained unknown-G ``(M3)``--``(M3b)`` formulation. It
        solves for ``utilde`` on the same
        Galerkin/SUPG operator as direct ``u``, then reconstructs
        ``u = F(p) + utilde`` for note-(M2) current and physical diagnostics.
        """
        warnings.warn(
            "solve_utilde is a legacy F(p)-shift verification path and does not impose "
            "a production current profile; use normalized I_0(s) with the milestone-3.6 "
            "constrained solver",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._solve(
            field,
            coefficients,
            formulation="utilde",
            profile=profile,
            boundary=boundary,
            boundary_value=0.0,
        )

    def _solution(self) -> Any:
        if self._internal_solution is None:
            raise RuntimeError("solve must be called before evaluating the M3 result")
        return self._internal_solution

    def assess_layer_resolution(
        self,
        *,
        layer_width: float,
        normal_element_width: float,
        strict: bool = False,
    ) -> CurrentLayerResolutionDiagnostic:
        r"""Apply the DESIGN §5 gate to a measured note-(M3) layer width.

        For Eq. ``layer_width``, callers measure or estimate ``delta`` and supply the
        local element width normal to the layer. Production callers warn below
        ``RuntimeOptions.min_layer_cells`` by default and fail when ``strict=True``.
        """
        if not isfinite(layer_width) or layer_width <= 0.0:
            raise ValueError("layer_width must be finite and positive")
        if not isfinite(normal_element_width) or normal_element_width <= 0.0:
            raise ValueError("normal_element_width must be finite and positive")
        cells_across_layer = layer_width / normal_element_width
        diagnostic = CurrentLayerResolutionDiagnostic(
            layer_width=layer_width,
            normal_element_width=normal_element_width,
            cells_across_layer=cells_across_layer,
            minimum_cells=self.runtime.min_layer_cells,
            resolved=cells_across_layer >= self.runtime.min_layer_cells,
        )
        if not diagnostic.resolved:
            message = (
                "M3 current layer is unresolved: "
                f"{cells_across_layer:.3f} normal element widths across the layer, "
                f"required at least {self.runtime.min_layer_cells}"
            )
            if strict:
                raise UnresolvedCurrentLayerError(message)
            warnings.warn(message, UnresolvedCurrentLayerWarning, stacklevel=2)
        return diagnostic

    def solution_at(self, x_coordinate: float, y_coordinate: float) -> float:
        """Return u at a physical point after a solve."""
        return float(self._solution().solution_at(x_coordinate, y_coordinate))

    def solution_gradient_at(self, x_coordinate: float, y_coordinate: float) -> tuple[float, float]:
        """Return the in-plane gradient of u at a physical point after a solve."""
        value: tuple[float, float] = self._solution().solution_gradient_at(
            x_coordinate, y_coordinate
        )
        return value

    def utilde_at(self, x_coordinate: float, y_coordinate: float) -> float:
        """Return the homogeneous utilde unknown after :meth:`solve_utilde`."""
        return float(self._solution().utilde_at(x_coordinate, y_coordinate))

    def current_at(self, x_coordinate: float, y_coordinate: float) -> tuple[float, float, float]:
        """Return the reconstructed note-(M2) current at a physical point."""
        value: tuple[float, float, float] = self._solution().current_at(x_coordinate, y_coordinate)
        return value

    def parallel_current_over_field_at(self, x_coordinate: float, y_coordinate: float) -> float:
        r"""Return J_parallel/B; full grad uses ``u-(D_u/B)b.grad(u)``."""
        return float(self._solution().parallel_current_over_field_at(x_coordinate, y_coordinate))

    def diagnostics(self) -> dict[str, float]:
        """Return scalar diagnostics from the latest direct-u M3 solve."""
        solution = self._solution()
        diagnostics: dict[str, float] = {
            **solution.diagnostics,
            "free_dof_residual_norm": solution.free_dof_residual_norm,
            "free_dof_relative_residual_norm": solution.free_dof_relative_residual_norm,
        }
        return diagnostics
