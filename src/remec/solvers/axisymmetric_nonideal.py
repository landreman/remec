r"""Shaped axisymmetric non-ideal benchmark for ``(M1)``--``(M4b)``.

This module is deliberately an R--Z reduction rather than a Cartesian surrogate.  It
solves note ``axi_M4`` for the reference potential, uses the shared mollified
``s=V_chi/V_omega`` field in ``(M4b)`` and ``(M3b)``, solves the bordered
``axi_M3`` block for ``(utilde,G)``, reconstructs every ``(M2)`` current component,
and applies the two scalar axisymmetric Ampere updates for ``psi`` and ``I=R B_phi``.
The smooth Zheng equilibrium is used only as an independent ideal reference.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import pi, sqrt
from typing import Any

import numpy as np
from numpy.typing import NDArray

from remec.analytic_equilibria import ZhengEquilibrium, ZhengShape, solve_zheng_equilibrium
from remec.common.threads import configure_threads
from remec.fem._constrained_current_continuity import (
    _linear_g_basis,
    _mapped_quadrature,
    _pchip_volume_coordinate,
)
from remec.geometry.axisymmetric import AxisymmetricFluxContourDomain
from remec.level_set import (
    MollifiedVolumeMap,
    QuadratureLevelSetData,
    compact_moment_matched_heaviside,
)
from remec.profiles import (
    TabulatedPressureProfile,
    TabulatedToroidalCurrentProfile,
    extract_ngsolve_quadrature,
)
from remec.solvers._anderson import AndersonAccelerator
from remec.solvers.continuation import (
    ContinuationStage,
    ContinuationStageResult,
    StagedContinuationOptions,
    StagedContinuationResult,
    StagedContinuationSolver,
)

FloatArray = NDArray[np.float64]
# The benchmark is nondimensionalized with mu0=1 (DESIGN §6).  The analytic
# coefficient scale still distinguishes the 0.8 MA and 1.0 MA Zheng inputs.
_MU0 = 1.0


@dataclass(frozen=True, slots=True)
class _MomentRows:
    cumulative: FloatArray
    shellwise: FloatArray


@dataclass(frozen=True, slots=True)
class _CurrentSolution:
    utilde: Any
    physical_u: Any
    physical_current: Any
    independent_current: Any
    target_current: FloatArray
    measured_current: FloatArray
    pressure_profile_error: float
    relative_m3_residual: float
    relative_m3b_residual: float
    independent_m2_relative_error: float


def _sample_scalar(coefficient: Any, mapped_points: Any) -> FloatArray:
    """Evaluate one scalar coefficient in deterministic mapped-quadrature order."""
    return np.asarray(coefficient(mapped_points), dtype=float).reshape(-1)


def _axisymmetric_volume_map(
    mesh: Any,
    level_set: Any,
    level_set_gradient: Any,
    *,
    integration_order: int,
    levels: int,
) -> tuple[MollifiedVolumeMap, Any]:
    r"""Build ``V_chi`` with the physical ``2*pi*R dR dZ`` volume measure."""
    import ngsolve as ng  # type: ignore[import-untyped]

    data = extract_ngsolve_quadrature(
        mesh,
        level_set,
        level_set_gradient,
        integration_order=integration_order,
    )
    mapped_points = _mapped_quadrature(mesh, integration_order)
    radii = _sample_scalar(ng.x, mapped_points)
    axisymmetric_data = QuadratureLevelSetData(
        values=data.values,
        gradient_magnitudes=data.gradient_magnitudes,
        weights=data.weights * (2.0 * pi * radii),
        element_sizes=data.element_sizes,
    )
    return (
        MollifiedVolumeMap.build(
            axisymmetric_data,
            spatial_width_cells=0.35,
            levels=levels,
            coarea_consistency_tolerance=0.1,
        ),
        mapped_points,
    )


def _moment_rows(
    volume_map: MollifiedVolumeMap,
    shell_edges: FloatArray,
    samples: FloatArray,
) -> _MomentRows:
    r"""Apply the exact ``(M3b)`` volume functional without a second s field."""
    normalized_volume = volume_map.quadrature_normalized_volume
    widths = volume_map.quadrature_normalized_mollifier_widths
    weights = volume_map.quadrature_weights
    membership = np.empty((len(shell_edges), len(weights)), dtype=float)
    membership[0] = 0.0
    membership[-1] = 1.0
    for index, edge in enumerate(shell_edges[1:-1], start=1):
        membership[index] = compact_moment_matched_heaviside(
            (float(edge) - normalized_volume) / widths
        )
    cumulative = membership @ (weights * samples) / (2.0 * pi)
    return _MomentRows(cumulative, np.diff(cumulative))


def _profile_coefficient(
    normalized_volume: Any,
    normalized_volume_gradient: Any,
    coordinates: Sequence[float],
    values: Sequence[float],
) -> tuple[Any, Any]:
    """Return a piecewise-linear normalized profile and its physical gradient."""
    import ngsolve as ng

    nodes = np.asarray(coordinates, dtype=float)
    ordinates = np.asarray(values, dtype=float)
    spline = ng.BSpline(
        2,
        [float(nodes[0]), *map(float, nodes), float(nodes[-1])],
        ordinates.tolist(),
    )(normalized_volume)
    value = ng.IfPos(
        normalized_volume - float(nodes[0]),
        ng.IfPos(float(nodes[-1]) - normalized_volume, spline, float(ordinates[-1])),
        float(ordinates[0]),
    )
    slopes = np.diff(ordinates) / np.diff(nodes)
    derivative: Any = float(slopes[-1])
    for index in range(len(slopes) - 2, -1, -1):
        derivative = ng.IfPos(
            float(nodes[index + 1]) - normalized_volume,
            float(slopes[index]),
            derivative,
        )
    return value, derivative * normalized_volume_gradient


def _vector_state(psi: Any, toroidal_field: Any) -> FloatArray:
    """Flatten the two scalar potentials that determine axisymmetric ``B``."""
    psi_values: FloatArray = np.asarray(psi.vec.FV().NumPy(), dtype=np.float64).reshape(-1)
    toroidal_values: FloatArray = np.asarray(
        toroidal_field.vec.FV().NumPy(), dtype=np.float64
    ).reshape(-1)
    return np.concatenate((psi_values, toroidal_values))


class _ZhengContinuationContext:
    """Shared mesh, ideal profiles, and stage-local reduced solves."""

    def __init__(
        self,
        equilibrium: ZhengEquilibrium,
        *,
        maxh: float,
        polynomial_order: int,
        toroidal_flux: float | None = None,
        magnetic_floor: float = 1.0e-12,
    ) -> None:
        import ngsolve as ng

        if not np.isfinite(magnetic_floor) or magnetic_floor <= 0.0:
            raise ValueError("magnetic_floor must be finite and positive")
        if toroidal_flux is not None and (not np.isfinite(toroidal_flux) or toroidal_flux == 0.0):
            raise ValueError("toroidal_flux must be finite and nonzero")
        self.equilibrium = equilibrium
        self.polynomial_order = polynomial_order
        self.magnetic_floor = magnetic_floor
        self.domain = AxisymmetricFluxContourDomain(
            equilibrium.boundary_contour(samples=257),
            maxh=maxh,
            geometry_order=polynomial_order + 1,
        )
        bundle = self.domain.build_mesh()
        self.mesh = bundle._mesh
        self.geometry_owner = bundle._geometry_owner
        self.scalar_space = ng.H1(self.mesh, order=polynomial_order, dirichlet=".*")
        self.unconstrained_space = ng.H1(self.mesh, order=polynomial_order)
        self.toroidal_space = self.unconstrained_space
        self.ndof = self.scalar_space.ndof
        self.toroidal_ndof = self.toroidal_space.ndof
        self._ampere_cache: tuple[Any, Any, Any, Any, Any, Any] | None = None
        # Explicit annotations keep the NumPy 2.2/Python 3.10 stubs from widening
        # these deterministic grids to ``floating[Any]``.
        self.shell_edges: FloatArray = np.linspace(0.0, 1.0, 5, dtype=np.float64)
        self.pressure_nodes: FloatArray = np.linspace(0.0, 1.0, 9, dtype=np.float64)
        self.edge_toroidal_field = 1.0
        self.reference_flux = equilibrium.flux(ng.x, ng.y)
        self.reference_flux_gradient = ng.CoefficientFunction(
            (
                equilibrium.radial_derivative(ng.x, ng.y),
                equilibrium.vertical_derivative(ng.x, ng.y),
            )
        )
        full_ideal_i = ng.sqrt(
            self.edge_toroidal_field**2 + 2.0 * equilibrium.a2 * self.reference_flux
        )
        self.toroidal_flux = (
            float(toroidal_flux)
            if toroidal_flux is not None
            else float(ng.Integrate(full_ideal_i / ng.x, self.mesh, order=10))
        )
        self.reference_volume_map, self.reference_mapped_points = _axisymmetric_volume_map(
            self.mesh,
            self.reference_flux,
            self.reference_flux_gradient,
            integration_order=8,
            levels=65,
        )
        reference_s = self.reference_volume_map.quadrature_normalized_volume
        reference_j_phi = (
            -equilibrium.a1 * _sample_scalar(ng.x, self.reference_mapped_points) ** 2
            + equilibrium.a2
        ) / (_MU0 * _sample_scalar(ng.x, self.reference_mapped_points))
        j_dot_grad_phi = reference_j_phi / _sample_scalar(ng.x, self.reference_mapped_points)
        self.full_current_profile = _moment_rows(
            self.reference_volume_map,
            self.shell_edges,
            j_dot_grad_phi,
        ).cumulative
        self.reference_s = reference_s
        self.reference_level_nodes = np.asarray(
            self.reference_volume_map.inverse_level(
                self.pressure_nodes * float(self.reference_volume_map.volumes[0])
            ),
            dtype=float,
        )

    def profiles(
        self, stage: ContinuationStage
    ) -> tuple[TabulatedPressureProfile, TabulatedToroidalCurrentProfile]:
        """Return stage-scaled ideal ``p_0(s)`` and cumulative ``I_0(s)``."""
        amplitude = stage.pressure_amplitude
        pressure_flux_derivative = amplitude * (-self.equilibrium.a1 / _MU0)
        pressure = 1.0e3 + pressure_flux_derivative * amplitude * self.reference_level_nodes
        pressure_profile = TabulatedPressureProfile(
            tuple(float(value) for value in self.pressure_nodes),
            tuple(float(value) for value in pressure),
        )
        current_profile = TabulatedToroidalCurrentProfile(
            tuple(float(value) for value in self.shell_edges),
            tuple(float(value) for value in amplitude * self.full_current_profile),
        )
        return pressure_profile, current_profile

    def initial_state(self, first_stage: ContinuationStage) -> FloatArray:
        """Project the first stage's independent ideal Zheng field into the shared spaces."""
        import ngsolve as ng

        amplitude = first_stage.pressure_amplitude
        psi = ng.GridFunction(self.scalar_space)
        psi.Set(amplitude * self.reference_flux)
        toroidal_field = ng.GridFunction(self.toroidal_space)
        ideal_i = ng.sqrt(
            self.edge_toroidal_field**2
            + 2.0 * amplitude**2 * self.equilibrium.a2 * self.reference_flux
        )
        toroidal_field.Set(ideal_i)
        self._enforce_toroidal_flux(toroidal_field)
        return _vector_state(psi, toroidal_field)

    def _enforce_toroidal_flux(self, toroidal_field: Any) -> float:
        r"""Fix the ``(Igrad)`` constant mode by ``Psi_t=int_Omega I/R dR dZ``."""
        import ngsolve as ng

        flux_weight = float(ng.Integrate(1.0 / ng.x, self.mesh, order=10))
        realized = float(ng.Integrate(toroidal_field / ng.x, self.mesh, order=10))
        shift = (self.toroidal_flux - realized) / flux_weight
        constant = ng.GridFunction(self.toroidal_space)
        constant.Set(1.0)
        toroidal_field.vec.data += shift * constant.vec
        return self._toroidal_flux_relative_error(toroidal_field)

    def _toroidal_flux_relative_error(self, toroidal_field: Any) -> float:
        """Measure the accepted state's prescribed toroidal-flux invariant."""
        import ngsolve as ng

        realized = float(ng.Integrate(toroidal_field / ng.x, self.mesh, order=10))
        return abs(realized - self.toroidal_flux) / abs(self.toroidal_flux)

    def fields_from_state(self, state: FloatArray) -> tuple[Any, Any]:
        """Restore the two full finite-element coefficient vectors."""
        import ngsolve as ng

        if state.shape != (self.ndof + self.toroidal_ndof,):
            raise ValueError("axisymmetric continuation state has the wrong size")
        psi = ng.GridFunction(self.scalar_space)
        toroidal_field = ng.GridFunction(self.toroidal_space)
        psi.vec.FV().NumPy()[:] = state[: self.ndof]
        toroidal_field.vec.FV().NumPy()[:] = state[self.ndof :]
        return psi, toroidal_field

    def magnetic_field(self, psi: Any, toroidal_field: Any) -> Any:
        r"""Return ``B=(-psi_Z/R,psi_R/R,-I/R)`` in a right-handed R,Z,-phi basis."""
        import ngsolve as ng

        gradient = ng.grad(psi)
        return ng.CoefficientFunction(
            (-gradient[1] / ng.x, gradient[0] / ng.x, -toroidal_field / ng.x)
        )

    def solve_reference_potential(
        self,
        magnetic_field: Any,
        perpendicular_ratio: float,
    ) -> tuple[Any, MollifiedVolumeMap, Any, float]:
        r"""Solve axisymmetric ``(M4a)`` with ``K_2`` and uniform ``S_ref=1``.

        The assembled R--Z form is
        ``int_Omega R grad(v).K_2 grad(chi) = int_Omega R v`` with ``chi=0`` on
        the wall. The same solved ``chi`` owns the mollified ``s=V_chi/V_Omega``
        composition used by ``(M4b)`` and ``(M3b)``.
        """
        import ngsolve as ng

        trial, test = self.scalar_space.TnT()
        trial_gradient = ng.grad(trial)
        test_gradient = ng.grad(test)
        magnitude = ng.sqrt(
            ng.InnerProduct(magnetic_field, magnetic_field) + self.magnetic_floor**2
        )
        poloidal_direction = ng.CoefficientFunction(
            (magnetic_field[0] / magnitude, magnetic_field[1] / magnitude)
        )
        tensor_action = perpendicular_ratio * trial_gradient + (
            1.0 - perpendicular_ratio
        ) * poloidal_direction * ng.InnerProduct(poloidal_direction, trial_gradient)
        quadrature = ng.dx(bonus_intorder=5)
        bilinear = ng.BilinearForm(self.scalar_space)
        bilinear += (ng.x * ng.InnerProduct(test_gradient, tensor_action)).Compile() * quadrature
        linear = ng.LinearForm(self.scalar_space)
        linear += (ng.x * test).Compile() * quadrature
        free = self.scalar_space.FreeDofs()
        with ng.TaskManager():
            bilinear.Assemble()
            linear.Assemble()
            chi = ng.GridFunction(self.scalar_space)
            inverse = bilinear.mat.Inverse(free, inverse="umfpack")
            chi.vec.data = inverse * linear.vec
            residual = linear.vec.CreateVector()
            residual.data = linear.vec - bilinear.mat * chi.vec
            residual.data = ng.Projector(free, True) * residual
        relative_residual = float(ng.Norm(residual)) / max(1.0, float(ng.Norm(linear.vec)))
        volume_map, mapped_points = _axisymmetric_volume_map(
            self.mesh,
            chi,
            ng.grad(chi),
            integration_order=7,
            levels=33,
        )
        return chi, volume_map, mapped_points, relative_residual

    def solve_current(
        self,
        magnetic_field: Any,
        pressure_profile: TabulatedPressureProfile,
        current_profile: TabulatedToroidalCurrentProfile,
        volume_map: MollifiedVolumeMap,
        mapped_points: Any,
        *,
        stage: ContinuationStage,
        current_diffusivity: float,
    ) -> _CurrentSolution:
        r"""Solve bordered ``axi_M3``--``(M3b)`` and reconstruct physical ``(M2)``.

        The weak operator contains ``v B.grad(utilde)``,
        ``D_u grad_perp(v).grad_perp(utilde)``, the ``(B.grad(p))/B_safe^2`` reaction,
        and ``-mu0 D_u grad_perp(utilde).grad(p) v/B_safe^2``. Its right-hand side is
        ``2 B.(grad(p) x grad(B_safe))/B_safe^3``. Shell rows impose ``(M3b)`` and
        the returned current is
        ``J = u B + B x grad(p)/B_safe^2 - D_u grad_perp(utilde)`` from ``(M2)``.
        """
        import ngsolve as ng

        normalized_volume, normalized_gradient = _pchip_volume_coordinate(
            volume_map,
            self._last_chi,
            ng.grad(self._last_chi),
        )
        pressure, pressure_gradient_2d = _profile_coefficient(
            normalized_volume,
            normalized_gradient,
            pressure_profile.normalized_volumes,
            pressure_profile.pressures,
        )
        realized_pressure = _sample_scalar(pressure, mapped_points)
        expected_levels = np.interp(
            volume_map.quadrature_normalized_volume,
            self.pressure_nodes,
            self.reference_level_nodes,
        )
        expected_pressure = (
            1.0e3 - stage.pressure_amplitude**2 * self.equilibrium.a1 * expected_levels / _MU0
        )
        pressure_scale = max(1.0e-300, float(np.ptp(expected_pressure)))
        pressure_profile_error = float(
            np.max(np.abs(realized_pressure - expected_pressure)) / pressure_scale
        )
        pressure_gradient = ng.CoefficientFunction(
            (pressure_gradient_2d[0], pressure_gradient_2d[1], 0.0)
        )
        trial, test = self.scalar_space.TnT()
        trial_gradient = ng.CoefficientFunction((ng.grad(trial)[0], ng.grad(trial)[1], 0.0))
        test_gradient = ng.CoefficientFunction((ng.grad(test)[0], ng.grad(test)[1], 0.0))
        magnitude = ng.sqrt(
            ng.InnerProduct(magnetic_field, magnetic_field) + self.magnetic_floor**2
        )
        direction = magnetic_field / magnitude

        def perpendicular(gradient: Any) -> Any:
            return gradient - direction * ng.InnerProduct(direction, gradient)

        b_dot_grad_p = ng.InnerProduct(magnetic_field, pressure_gradient)
        reaction = _MU0 * b_dot_grad_p / magnitude**2
        magnitude_space_field = ng.GridFunction(self.unconstrained_space)
        magnitude_space_field.Set(magnitude)
        magnitude_gradient_2d = ng.grad(magnitude_space_field)
        magnitude_gradient = ng.CoefficientFunction(
            (magnitude_gradient_2d[0], magnitude_gradient_2d[1], 0.0)
        )
        drive = (
            2.0
            * ng.InnerProduct(
                magnetic_field,
                ng.Cross(pressure_gradient, magnitude_gradient),
            )
            / magnitude**3
        )
        operator = ng.x * (
            test * ng.InnerProduct(magnetic_field, trial_gradient)
            + current_diffusivity
            * ng.InnerProduct(perpendicular(test_gradient), perpendicular(trial_gradient))
            + reaction * test * trial
            - _MU0
            * current_diffusivity
            * ng.InnerProduct(perpendicular(trial_gradient), pressure_gradient)
            * test
            / magnitude**2
        )
        quadrature = ng.dx(bonus_intorder=6)
        bilinear = ng.BilinearForm(self.scalar_space)
        bilinear += operator.Compile() * quadrature
        drive_form = ng.LinearForm(self.scalar_space)
        drive_form += (ng.x * drive * test).Compile() * quadrature
        g_basis = [
            _linear_g_basis(normalized_volume, normalized_gradient, self.shell_edges, index)
            for index in range(len(self.shell_edges))
        ]
        p_forms: list[Any] = []
        for basis, basis_gradient_2d in g_basis:
            basis_gradient = ng.CoefficientFunction(
                (basis_gradient_2d[0], basis_gradient_2d[1], 0.0)
            )
            coupling = ng.InnerProduct(magnetic_field, basis_gradient) + reaction * basis
            column = ng.LinearForm(self.scalar_space)
            column += (ng.x * test * coupling).Compile() * quadrature
            p_forms.append(column)
        free = self.scalar_space.FreeDofs()
        with ng.TaskManager():
            bilinear.Assemble()
            drive_form.Assemble()
            for column in p_forms:
                column.Assemble()
            inverse = bilinear.mat.Inverse(free, inverse="umfpack")
            base = ng.GridFunction(self.scalar_space)
            base.vec.data = inverse * drive_form.vec
            responses: list[Any] = []
            for column in p_forms[:-1]:
                response = ng.GridFunction(self.scalar_space)
                response.vec.data = inverse * column.vec
                responses.append(response)
        toroidal_gradient = ng.CoefficientFunction((0.0, 0.0, -1.0 / ng.x))
        b_dot_phi = ng.InnerProduct(magnetic_field, toroidal_gradient)
        diamagnetic = ng.Cross(magnetic_field, pressure_gradient) / magnitude**2

        def moments(expression: Any) -> _MomentRows:
            return _moment_rows(
                volume_map,
                self.shell_edges,
                _sample_scalar(expression, mapped_points),
            )

        def utilde_rows(field: Any) -> _MomentRows:
            gradient_2d = ng.grad(field)
            gradient = ng.CoefficientFunction((gradient_2d[0], gradient_2d[1], 0.0))
            return moments(
                field * b_dot_phi
                - current_diffusivity * ng.InnerProduct(perpendicular(gradient), toroidal_gradient)
            )

        base_rows = utilde_rows(base).shellwise
        response_rows = np.column_stack([utilde_rows(response).shellwise for response in responses])
        g_rows = np.column_stack(
            [moments(basis * b_dot_phi).shellwise for basis, _ in g_basis[:-1]]
        )
        diamagnetic_rows = moments(ng.InnerProduct(diamagnetic, toroidal_gradient)).shellwise
        target = np.asarray(current_profile.enclosed_current(self.shell_edges), dtype=float)
        schur = g_rows - response_rows
        free_g = np.linalg.solve(
            schur,
            np.diff(target) - diamagnetic_rows - base_rows,
        )
        utilde = ng.GridFunction(self.scalar_space)
        utilde.vec.data = base.vec
        for coefficient, response in zip(free_g, responses, strict=True):
            utilde.vec.data -= float(coefficient) * response.vec
        g_profile: Any = 0.0
        for coefficient, (basis, _) in zip(free_g, g_basis[:-1], strict=True):
            g_profile += float(coefficient) * basis
        physical_u = g_profile + utilde
        utilde_gradient_2d = ng.grad(utilde)
        utilde_gradient = ng.CoefficientFunction(
            (utilde_gradient_2d[0], utilde_gradient_2d[1], 0.0)
        )
        regularizing = -current_diffusivity * perpendicular(utilde_gradient)
        physical_current = physical_u * magnetic_field + diamagnetic + regularizing
        # Rebuild (M2) independently of the component expressions used by the
        # bordered rows above.  This is intentionally redundant: a coordinated
        # mutation to (say) the diamagnetic term must not also mutate the public
        # current-profile oracle.  A separate order-8 test integral additionally
        # compares the assembled current with this oracle away from the constraint
        # quadrature rule.
        independent_current = (
            physical_u * magnetic_field
            + ng.Cross(magnetic_field, pressure_gradient) / magnitude**2
            - current_diffusivity * perpendicular(utilde_gradient)
        )
        independent = moments(ng.InnerProduct(independent_current, toroidal_gradient)).cumulative
        current_difference = physical_current - independent_current
        weights = volume_map.quadrature_weights
        oracle_difference_squared = float(
            np.dot(
                weights,
                _sample_scalar(
                    ng.InnerProduct(current_difference, current_difference), mapped_points
                ),
            )
        )
        oracle_norm_squared = float(
            np.dot(
                weights,
                _sample_scalar(
                    ng.InnerProduct(independent_current, independent_current), mapped_points
                ),
            )
        )
        independent_m2_relative_error = sqrt(
            max(0.0, oracle_difference_squared) / max(1.0e-300, oracle_norm_squared)
        )
        m3_residual = drive_form.vec.CreateVector()
        m3_residual.data = bilinear.mat * utilde.vec - drive_form.vec
        for coefficient, column in zip(free_g, p_forms[:-1], strict=True):
            m3_residual.data += float(coefficient) * column.vec
        m3_residual.data = ng.Projector(free, True) * m3_residual
        relative_m3 = float(ng.Norm(m3_residual)) / max(1.0, float(ng.Norm(drive_form.vec)))
        relative_m3b = max(
            float(np.linalg.norm(independent - target)) / max(1.0, float(np.linalg.norm(target))),
            independent_m2_relative_error,
        )
        return _CurrentSolution(
            utilde,
            physical_u,
            physical_current,
            independent_current,
            target,
            independent,
            pressure_profile_error,
            relative_m3,
            relative_m3b,
            independent_m2_relative_error,
        )

    def solve_ampere_candidates(
        self,
        current: Any,
        volume_map: MollifiedVolumeMap,
        mapped_points: Any,
        target_current: FloatArray,
    ) -> tuple[Any, Any, float, FloatArray, float]:
        r"""Apply compatible axisymmetric ``(M1)`` and ``(Igrad)`` updates.

        The raw ``(M2)`` toroidal current is first mapped through the scalar
        Grad--Shafranov operator.  Four shell-local response columns then correct the
        strong discrete curl so its independently re-integrated ``(M3b)`` moments equal
        ``I_0``.  The poloidal current is projected through the companion ``I=R B_phi``
        potential, so the returned current is a curl representation and therefore
        divergence-free by construction. The weak forms are
        ``int grad(psi).grad(v)/R = int mu0 J_phi v + shell corrections`` and
        ``int grad(I).grad(q)/R = int (mu0 R J_Z,-mu0 R J_R).grad(q)/R``.
        The ``(Igrad)`` constant mode is fixed by the prescribed note-§11.2 condition
        ``Psi_t=int_Omega I/R dR dZ``. The reported M1 residual is the maximum of the
        corrected Grad--Shafranov and anchored ``Igrad`` residuals.
        """
        import ngsolve as ng

        _, test = self.scalar_space.TnT()
        _, i_test = self.toroidal_space.TnT()
        quadrature = ng.dx(bonus_intorder=6)
        (
            psi_form,
            psi_inverse,
            free,
            i_form,
            i_inverse,
            anchored_free,
        ) = self._ampere_operators()
        psi_rhs = ng.LinearForm(self.scalar_space)
        physical_j_phi = -current[2]
        psi_rhs += (_MU0 * physical_j_phi * test).Compile() * quadrature
        normalized_volume, normalized_gradient = _pchip_volume_coordinate(
            volume_map,
            self._last_chi,
            ng.grad(self._last_chi),
        )
        correction_bases = [
            _linear_g_basis(normalized_volume, normalized_gradient, self.shell_edges, index)[0]
            for index in range(len(self.shell_edges) - 1)
        ]
        correction_forms: list[Any] = []
        for basis in correction_bases:
            form = ng.LinearForm(self.scalar_space)
            form += (_MU0 * basis * test).Compile() * quadrature
            correction_forms.append(form)
        target_i_gradient = ng.CoefficientFunction(
            (_MU0 * ng.x * current[1], -_MU0 * ng.x * current[0])
        )
        i_rhs = ng.LinearForm(self.toroidal_space)
        i_rhs += (ng.InnerProduct(target_i_gradient, ng.grad(i_test)) / ng.x).Compile() * quadrature
        with ng.TaskManager():
            psi_rhs.Assemble()
            for form in correction_forms:
                form.Assemble()
            i_rhs.Assemble()
            psi = ng.GridFunction(self.scalar_space)
            psi.vec.data = psi_inverse * psi_rhs.vec
            responses: list[Any] = []
            for form in correction_forms:
                response = ng.GridFunction(self.scalar_space)
                response.vec.data = psi_inverse * form.vec
                responses.append(response)
            toroidal_field = ng.GridFunction(self.toroidal_space)
            toroidal_field.vec.data = i_inverse * i_rhs.vec
        self._enforce_toroidal_flux(toroidal_field)
        i_residual = i_rhs.vec.CreateVector()
        i_residual.data = i_rhs.vec - i_form.mat * toroidal_field.vec
        i_residual.data = ng.Projector(anchored_free, True) * i_residual
        relative_i_residual = float(ng.Norm(i_residual)) / max(1.0, float(ng.Norm(i_rhs.vec)))

        def projected_moments(field: Any) -> _MomentRows:
            gradient = ng.grad(field)
            hessian = field.Operator("hesse")
            delta_star = hessian[0, 0] - gradient[0] / ng.x + hessian[1, 1]
            projected_j_dot_phi = -delta_star / (_MU0 * ng.x**2)
            return _moment_rows(
                volume_map,
                self.shell_edges,
                _sample_scalar(projected_j_dot_phi, mapped_points),
            )

        base_moments = projected_moments(psi)
        response_matrix = np.column_stack(
            [projected_moments(response).shellwise for response in responses]
        )
        correction_coefficients = np.linalg.solve(
            response_matrix,
            np.diff(target_current) - base_moments.shellwise,
        )
        for coefficient, response in zip(correction_coefficients, responses, strict=True):
            psi.vec.data += float(coefficient) * response.vec
        projected_current = projected_moments(psi).cumulative
        corrected_rhs = psi_rhs.vec.CreateVector()
        corrected_rhs.data = psi_rhs.vec
        for coefficient, form in zip(correction_coefficients, correction_forms, strict=True):
            corrected_rhs.data += float(coefficient) * form.vec
        psi_residual = corrected_rhs.CreateVector()
        psi_residual.data = corrected_rhs - psi_form.mat * psi.vec
        psi_residual.data = ng.Projector(free, True) * psi_residual
        residual = max(
            float(ng.Norm(psi_residual)) / max(1.0, float(ng.Norm(corrected_rhs))),
            relative_i_residual,
        )
        correction_samples = np.zeros_like(volume_map.quadrature_weights)
        for coefficient, basis in zip(correction_coefficients, correction_bases, strict=True):
            correction_samples += float(coefficient) * _sample_scalar(basis, mapped_points)
        raw_samples = _sample_scalar(physical_j_phi, mapped_points)
        weights = volume_map.quadrature_weights
        correction_norm = sqrt(
            float(np.dot(weights, correction_samples**2))
            / max(1.0e-300, float(np.dot(weights, raw_samples**2)))
        )
        return psi, toroidal_field, residual, projected_current, correction_norm

    def _ampere_operators(self) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Assemble and factor the two mesh-constant ``(M1)``/``(Igrad)`` blocks once."""
        import ngsolve as ng

        if self._ampere_cache is None:
            psi_trial, psi_test = self.scalar_space.TnT()
            i_trial, i_test = self.toroidal_space.TnT()
            quadrature = ng.dx(bonus_intorder=6)
            psi_form = ng.BilinearForm(self.scalar_space)
            psi_form += (
                ng.InnerProduct(ng.grad(psi_trial), ng.grad(psi_test)) / ng.x
            ).Compile() * quadrature
            i_form = ng.BilinearForm(self.toroidal_space)
            i_form += (
                ng.InnerProduct(ng.grad(i_trial), ng.grad(i_test)) / ng.x
            ).Compile() * quadrature
            free = self.scalar_space.FreeDofs()
            anchored_free = ng.BitArray(self.toroidal_space.FreeDofs())
            anchored_free.Clear(0)
            with ng.TaskManager():
                psi_form.Assemble()
                i_form.Assemble()
                psi_inverse = psi_form.mat.Inverse(free, inverse="umfpack")
                i_inverse = i_form.mat.Inverse(anchored_free, inverse="umfpack")
            self._ampere_cache = (
                psi_form,
                psi_inverse,
                free,
                i_form,
                i_inverse,
                anchored_free,
            )
        return self._ampere_cache

    def magnetic_floor_diagnostics(self, psi: Any, toroidal_field: Any) -> tuple[float, float]:
        """Monitor the physical field minimum and regularized-direction floor activity."""
        import ngsolve as ng

        magnetic_field = self.magnetic_field(psi, toroidal_field)
        magnitude_squared = _sample_scalar(
            ng.InnerProduct(magnetic_field, magnetic_field), self.reference_mapped_points
        )
        physical_magnitude = np.sqrt(np.maximum(0.0, magnitude_squared))
        activity = self.magnetic_floor**2 / (magnitude_squared + self.magnetic_floor**2)
        weights = self.reference_volume_map.quadrature_weights
        activity_l2 = sqrt(
            float(np.dot(weights, activity**2)) / max(1.0e-300, float(np.sum(weights)))
        )
        return float(np.min(physical_magnitude)), activity_l2

    def ideal_fem(self, stage: ContinuationStage) -> Any:
        """Solve the same-mesh ideal GS problem used to separate discretization bias."""
        import ngsolve as ng

        trial, test = self.scalar_space.TnT()
        bilinear = ng.BilinearForm(self.scalar_space)
        bilinear += (ng.InnerProduct(ng.grad(trial), ng.grad(test)) / ng.x).Compile() * ng.dx(
            bonus_intorder=6
        )
        linear = ng.LinearForm(self.scalar_space)
        source = stage.pressure_amplitude * (
            -self.equilibrium.a1 * ng.x + self.equilibrium.a2 / ng.x
        )
        linear += (source * test).Compile() * ng.dx(bonus_intorder=6)
        free = self.scalar_space.FreeDofs()
        with ng.TaskManager():
            bilinear.Assemble()
            linear.Assemble()
            field = ng.GridFunction(self.scalar_space)
            field.vec.data = bilinear.mat.Inverse(free, inverse="umfpack") * linear.vec
        return field

    def relative_l2(self, first: Any, second: Any) -> float:
        """Return the physical axisymmetric relative L2 difference on the shared mesh."""
        import ngsolve as ng

        numerator = float(ng.Integrate(ng.x * (first - second) ** 2, self.mesh, order=10))
        denominator = float(ng.Integrate(ng.x * second**2, self.mesh, order=10))
        return sqrt(max(0.0, numerator) / max(1.0e-300, denominator))

    def solve_stage(
        self,
        stage: ContinuationStage,
        initial_state: FloatArray,
    ) -> ContinuationStageResult:
        """Run one damped reduced Picard stage from the previous converged state."""
        pressure_profile, current_profile = self.profiles(stage)
        psi, toroidal_field = self.fields_from_state(initial_state)
        damping = 0.55
        accelerator = AndersonAccelerator(
            depth=3,
            damping=damping,
            regularization=1.0e-12,
            condition_limit=1.0e5,
        )
        rejected_acceleration_attempts = 0
        fixed_residual = float("inf")
        current_solution: _CurrentSolution | None = None
        m1_residual = float("inf")
        m4a_residual = float("inf")
        projected_current = np.asarray((0.0,), dtype=float)
        projection_correction = float("inf")
        for iteration in range(1, 41):
            magnetic_field = self.magnetic_field(psi, toroidal_field)
            chi, volume_map, mapped_points, m4a_residual = self.solve_reference_potential(
                magnetic_field,
                stage.perpendicular_ratio,
            )
            self._last_chi = chi
            current_solution = self.solve_current(
                magnetic_field,
                pressure_profile,
                current_profile,
                volume_map,
                mapped_points,
                stage=stage,
                current_diffusivity=stage.current_diffusivity,
            )
            (
                candidate_psi,
                candidate_i,
                m1_residual,
                projected_current,
                projection_correction,
            ) = self.solve_ampere_candidates(
                current_solution.physical_current,
                volume_map,
                mapped_points,
                current_solution.target_current,
            )
            old_state = _vector_state(psi, toroidal_field)
            candidate_state = _vector_state(candidate_psi, candidate_i)
            scale = max(1.0e-12, float(np.linalg.norm(candidate_state)))
            fixed_residual = float(np.linalg.norm(candidate_state - old_state)) / scale
            acceleration = accelerator.update(old_state, candidate_state)
            if acceleration.rejection_reason is not None:
                rejected_acceleration_attempts += 1
            accepted_state = acceleration.state
            psi.vec.FV().NumPy()[:] = accepted_state[: self.ndof]
            toroidal_field.vec.FV().NumPy()[:] = accepted_state[self.ndof :]
            if fixed_residual < 5.0e-9:
                break
        else:
            raise RuntimeError("axisymmetric non-ideal Picard stage did not converge")
        assert current_solution is not None
        ideal_fem = self.ideal_fem(stage)
        analytic = stage.pressure_amplitude * self.reference_flux
        current_error = float(
            np.max(np.abs(current_solution.measured_current - current_solution.target_current))
        )
        projected_current_error = float(
            np.max(np.abs(projected_current - current_solution.target_current))
        )
        toroidal_flux_error = self._toroidal_flux_relative_error(toroidal_field)
        minimum_magnetic_magnitude, floor_activity_l2 = self.magnetic_floor_diagnostics(
            psi, toroidal_field
        )
        return ContinuationStageResult(
            stage=stage,
            state=tuple(float(value) for value in _vector_state(psi, toroidal_field)),
            nonlinear_iterations=iteration,
            m1_relative_residual=m1_residual,
            m3_relative_residual=current_solution.relative_m3_residual,
            m3b_relative_residual=current_solution.relative_m3b_residual,
            m4a_relative_residual=m4a_residual,
            fixed_point_residual_norm=max(fixed_residual, m1_residual),
            pressure_profile_error=current_solution.pressure_profile_error,
            current_profile_error=current_error,
            projected_current_profile_error=projected_current_error,
            target_toroidal_flux=self.toroidal_flux,
            toroidal_flux_relative_error=toroidal_flux_error,
            target_total_current=float(current_solution.target_current[-1]),
            projection_correction_relative_norm=projection_correction,
            nonideal_to_analytic_relative_l2_error=self.relative_l2(psi, analytic),
            ideal_fem_to_analytic_relative_l2_error=self.relative_l2(ideal_fem, analytic),
            nonideal_to_ideal_fem_relative_l2_difference=self.relative_l2(psi, ideal_fem),
            # This smooth nested-surface case contains neither a resonant M3 layer
            # nor an island-flattening M4 layer.  A domain-width count is not a layer
            # diagnostic, so both fields are explicitly not applicable here.
            minimum_current_layer_cells=None,
            minimum_pressure_layer_cells=None,
            minimum_magnetic_magnitude=minimum_magnetic_magnitude,
            magnetic_floor_activity_l2=floor_activity_l2,
            rejected_acceleration_attempts=rejected_acceleration_attempts,
        )


class _StageSolver:
    """Fresh stage-local wrapper; constructing it clears all acceleration history."""

    def __init__(self, context: _ZhengContinuationContext, stage: ContinuationStage) -> None:
        self.context = context
        self.stage = stage

    def solve(self, initial_state: FloatArray) -> ContinuationStageResult:
        return self.context.solve_stage(self.stage, initial_state)


def run_zheng_nonideal_continuation(
    *,
    plasma_current: float,
    stages: Sequence[ContinuationStage],
    maxh: float = 0.18,
    polynomial_order: int = 2,
    require_decreasing_projection_correction: bool = True,
    toroidal_flux: float | None = None,
    magnetic_floor: float = 1.0e-12,
) -> StagedContinuationResult:
    r"""Run milestone 5.5 on one smooth Zheng ``psi=0`` shaped boundary.

    Two calls with distinct ``plasma_current`` values produce distinct analytic
    cumulative ``I_0(s)`` targets.  Each stage compares its non-ideal field with both
    the independent analytic Zheng field and an ideal FEM solve on the identical mesh.
    """
    if polynomial_order < 1:
        raise ValueError("polynomial_order must be at least one")
    stage_tuple = tuple(stages)
    if not stage_tuple:
        raise ValueError("stages must contain at least one continuation point")
    equilibrium = solve_zheng_equilibrium(
        shape=ZhengShape(0.70, 0.49, 1.7, 0.125),
        poloidal_beta=0.40,
        plasma_current=plasma_current,
    )
    configure_threads(1)
    context = _ZhengContinuationContext(
        equilibrium,
        maxh=maxh,
        polynomial_order=polynomial_order,
        toroidal_flux=toroidal_flux,
        magnetic_floor=magnetic_floor,
    )
    solver = StagedContinuationSolver(
        lambda stage: _StageSolver(context, stage),
        options=StagedContinuationOptions(
            stage_tuple,
            residual_tolerance=1.0e-8,
            profile_tolerance=1.0e-10,
            minimum_layer_cells=6.0,
            require_decreasing_projection_correction=require_decreasing_projection_correction,
        ),
    )
    return solver.solve(context.initial_state(stage_tuple[0]))
