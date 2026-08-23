"""NGSolve kernel for the Section-8.6 frozen (M4a)--(M4b) island benchmark."""

from __future__ import annotations

import resource
import sys
from dataclasses import dataclass
from math import isfinite, pi, sqrt
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from remec.common.threads import configure_threads
from remec.fem._reiman_greenside import reiman_greenside_coefficient_functions
from remec.fem.spaces import make_periodic_tetrahedral_de_rham_sequence
from remec.geometry import GradedAnnulus, PeriodicCylinder3D
from remec.level_set import MollifiedVolumeMap
from remec.profiles import (
    PressureProfile,
    TabulatedPressureProfile,
    extract_ngsolve_quadrature,
)
from remec.reiman_greenside import ReimanGreensideField
from remec.solvers.frozen_field_island import (
    FrozenFieldIslandConfig,
    FrozenFieldIslandResult,
    critical_layer_width,
    exact_island_width,
)


@dataclass(frozen=True, slots=True)
class _M4Row:
    """Internal state and timings for one note-(M4a) solve and (M4b) transplant."""

    mesh: Any
    field: Any
    volume_map: MollifiedVolumeMap
    pressure_profile: PressureProfile
    elements: int
    dofs: int
    assembly_seconds: float
    solve_seconds: float
    free_dof_relative_residual: float
    total_power_relative_error: float
    periodic_boundary_power_relative_error: float
    divergence_theorem_relative_error: float
    conservative_flux_relative_correction: float
    conservative_flux_relative_residual: float
    conservative_flux_divergence_relative_error: float
    floor_relative_activity: float
    local_resonant_element_width: float
    pressure_minimum: float
    pressure_maximum: float
    linear_solver_path: str
    iteration_count: int


def _peak_memory_megabytes() -> float:
    """Return process peak RSS in MiB using the platform's ``getrusage`` units."""
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0


def _recover_conservative_flux(
    mesh: Any, raw_flux: Any, *, order: int
) -> tuple[Any, float, float, float]:
    r"""Recover an H(div) heat flux satisfying ``div(q_h)=S_ref=1`` from (M4a).

    Essential H1 traces make ``grad(chi_h)`` evaluate as zero directly on the wall in
    NGSolve 6.2.2606.  The mixed projection
    ``(q_h,v)+(lambda,div(v))+(div(q_h),mu)=(q_raw,v)+(1,mu)`` supplies the independent
    conservative normal trace used for the global power balance.
    """
    import ngsolve as ng  # type: ignore[import-untyped]

    # A lowest-order conservative diagnostic is sufficient for the integral balance
    # and avoids a second high-order direct factorization on exhaustive meshes.
    sequence = make_periodic_tetrahedral_de_rham_sequence(mesh, order=1)
    mixed = ng.FESpace([sequence.hdiv, sequence.l2])
    (flux_trial, multiplier_trial), (flux_test, multiplier_test) = mixed.TnT()
    quadrature = ng.dx(bonus_intorder=6)
    operator = ng.BilinearForm(mixed, symmetric=True)
    operator += (
        ng.InnerProduct(flux_trial, flux_test)
        + multiplier_trial * ng.div(flux_test)
        + ng.div(flux_trial) * multiplier_test
    ) * quadrature
    right_hand_side = ng.LinearForm(mixed)
    right_hand_side += (ng.InnerProduct(raw_flux, flux_test) + multiplier_test) * quadrature
    operator.Assemble()
    right_hand_side.Assemble()
    solution = ng.GridFunction(mixed)
    free_dofs = mixed.FreeDofs()
    inverse = operator.mat.Inverse(free_dofs, inverse="umfpack")
    solution.vec.data = inverse * right_hand_side.vec
    projector = ng.Projector(free_dofs, True)
    for _ in range(3):
        residual = right_hand_side.vec - operator.mat * solution.vec
        free_residual = projector * residual
        relative_residual = float(ng.Norm(free_residual)) / max(
            1.0, float(ng.Norm(right_hand_side.vec))
        )
        if relative_residual < 1.0e-11:
            break
        solution.vec.data += inverse * residual
    conservative_flux = solution.components[0]
    integration_order = 2 * order + 10
    raw_norm = float(ng.sqrt(ng.Integrate(raw_flux * raw_flux, mesh, order=integration_order)))
    correction = float(
        ng.sqrt(
            ng.Integrate(
                (conservative_flux - raw_flux) * (conservative_flux - raw_flux),
                mesh,
                order=integration_order,
            )
        )
        / max(raw_norm, np.finfo(float).tiny)
    )
    divergence_error = float(
        ng.sqrt(
            ng.Integrate(
                (ng.div(conservative_flux) - 1.0) ** 2,
                mesh,
                order=integration_order,
            )
        )
        / ng.sqrt(ng.Integrate(1.0, mesh, order=integration_order))
    )
    return conservative_flux, correction, relative_residual, divergence_error


def _solve_m4(
    mesh: Any,
    model: ReimanGreensideField,
    config: FrozenFieldIslandConfig,
    pressure_profile: PressureProfile,
    *,
    perpendicular_conductivity: float,
    isotropic: bool = False,
    recover_flux: bool = False,
) -> _M4Row:
    r"""Solve ``(M4a)`` and transplant ``p=p0(V_chi(chi)/V_omega)`` from ``(M4b)``.

    The assembled form is
    ``int [epsilon_kappa grad(chi).grad(v) + (1-epsilon_kappa)
    (b_safe.grad(chi))(b_safe.grad(v))] = int v S_ref`` with ``S_ref=1``.
    For the isotropic falsifiability control it becomes ``int grad(chi).grad(v)``.
    """
    import ngsolve as ng

    _, magnetic_field = reiman_greenside_coefficient_functions(model)
    magnetic_magnitude = ng.sqrt(ng.InnerProduct(magnetic_field, magnetic_field))
    safe_magnitude = ng.sqrt(magnetic_magnitude**2 + config.b_floor**2)
    direction = magnetic_field / safe_magnitude
    base = ng.H1(mesh, order=config.polynomial_order, dirichlet="wall")
    space = ng.Periodic(base)
    trial, test = space.TnT()
    gradient_trial = ng.grad(trial)
    gradient_test = ng.grad(test)
    quadrature = ng.dx(bonus_intorder=4)
    if isotropic:
        integrand = ng.InnerProduct(gradient_trial, gradient_test)
    else:
        contrast = 1.0 - perpendicular_conductivity
        integrand = perpendicular_conductivity * ng.InnerProduct(
            gradient_trial, gradient_test
        ) + contrast * ng.InnerProduct(direction, gradient_trial) * ng.InnerProduct(
            direction, gradient_test
        )
    bilinear = ng.BilinearForm(space)
    bilinear += integrand.Compile() * quadrature
    linear = ng.LinearForm(space)
    linear += test * quadrature

    configure_threads(1)
    iterative = space.ndof > config.direct_dof_threshold
    preconditioner = ng.Preconditioner(bilinear, "h1amg") if iterative else None
    assembly_start = perf_counter()
    with ng.TaskManager():
        bilinear.Assemble()
        linear.Assemble()
    assembly_seconds = perf_counter() - assembly_start
    free_dofs = space.FreeDofs()
    solution_start = perf_counter()
    with ng.TaskManager():
        field = ng.GridFunction(space)
        iteration_count = 1
        solver_path = "direct:sparsecholesky"
        if iterative:
            iterations: list[float] = []
            assert preconditioner is not None
            field.vec.data = ng.solvers.CG(
                mat=bilinear.mat,
                rhs=linear.vec,
                pre=preconditioner.mat,
                tol=1.0e-11,
                maxsteps=2_000,
                printrates=False,
                callback=lambda _iteration, error: iterations.append(float(error)),
            )
            iteration_count = len(iterations)
            solver_path = "iterative:cg-h1amg"
        else:
            inverse = bilinear.mat.Inverse(free_dofs, inverse="sparsecholesky")
            field.vec.data = inverse * linear.vec
    solve_seconds = perf_counter() - solution_start

    residual = bilinear.mat * field.vec - linear.vec
    free_residual = ng.Projector(free_dofs, True) * residual
    free_rhs = ng.Projector(free_dofs, True) * linear.vec
    relative_residual = float(ng.Norm(free_residual)) / max(1.0, float(ng.Norm(free_rhs)))
    if not isfinite(relative_residual) or relative_residual > 1.0e-9:
        raise RuntimeError(f"frozen island M4a solve residual {relative_residual:.3e} exceeds 1e-9")

    gradient = ng.grad(field)
    parallel_gradient = ng.InnerProduct(direction, gradient)
    if isotropic:
        heat_flux = -gradient
    else:
        heat_flux = -(
            perpendicular_conductivity * gradient
            + (1.0 - perpendicular_conductivity) * direction * parallel_gradient
        )
    total_power = float(ng.Integrate(1.0, mesh, order=2 * config.polynomial_order + 8))
    total_power_relative_error = 0.0
    periodic_boundary_power_relative_error = 0.0
    divergence_theorem_relative_error = 0.0
    flux_correction = 0.0
    flux_residual = 0.0
    flux_divergence_error = 0.0
    if recover_flux:
        (
            conservative_flux,
            flux_correction,
            flux_residual,
            flux_divergence_error,
        ) = _recover_conservative_flux(mesh, heat_flux, order=config.polynomial_order)
        wall = mesh.Boundaries("wall")
        normal = ng.specialcf.normal(3)
        boundary_power = float(
            ng.Integrate(
                conservative_flux * normal,
                mesh,
                ng.BND,
                definedon=wall,
                order=2 * config.polynomial_order + 8,
            )
        )
        total_power_relative_error = abs(boundary_power - total_power) / total_power
        periodic_ends = mesh.Boundaries("periodic_lower|periodic_upper")
        periodic_power = float(
            ng.Integrate(
                conservative_flux * normal,
                mesh,
                ng.BND,
                definedon=periodic_ends,
                order=2 * config.polynomial_order + 8,
            )
        )
        periodic_boundary_power_relative_error = abs(periodic_power) / total_power
        divergence_power = float(
            ng.Integrate(ng.div(conservative_flux), mesh, order=2 * config.polynomial_order + 8)
        )
        divergence_theorem_relative_error = abs(boundary_power - divergence_power) / total_power
    floor_activity = float(
        ng.Integrate(
            ((safe_magnitude - magnetic_magnitude) / magnetic_magnitude) ** 2,
            mesh,
            order=2 * config.polynomial_order + 8,
        )
    )

    integration_order = 2 * config.polynomial_order + 4
    data = extract_ngsolve_quadrature(
        mesh,
        field,
        gradient,
        integration_order=integration_order,
        element_size_mode="level-set-normal",
    )
    volume_map = MollifiedVolumeMap.build(
        data,
        spatial_width_cells=0.5,
        levels=config.volume_levels,
        coarea_consistency_tolerance=0.2,
    )
    pressure_profile.validate(edge_value=0.0)
    pressure_values = np.asarray(
        pressure_profile.value(volume_map.quadrature_normalized_volume), dtype=float
    )

    element_types = {element.type for element in mesh.Elements(ng.VOL)}
    rules = {
        element_type: ng.IntegrationRule(element_type, integration_order)
        for element_type in element_types
    }
    mapped = mesh.MapToAllElements(rules, ng.VOL)
    radii = np.sqrt(
        np.asarray(ng.x(mapped), dtype=float).reshape(-1) ** 2
        + np.asarray(ng.y(mapped), dtype=float).reshape(-1) ** 2
    )
    resonance = model.resonance_radius(0.5)
    shell_distance = np.abs(radii - resonance)
    selected = shell_distance <= max(np.quantile(shell_distance, 0.05), np.finfo(float).eps)
    local_width = float(np.median(data.element_sizes[selected]))
    return _M4Row(
        mesh=mesh,
        field=field,
        volume_map=volume_map,
        pressure_profile=pressure_profile,
        elements=int(mesh.ne),
        dofs=int(space.ndof),
        assembly_seconds=assembly_seconds,
        solve_seconds=solve_seconds,
        free_dof_relative_residual=relative_residual,
        total_power_relative_error=total_power_relative_error,
        periodic_boundary_power_relative_error=periodic_boundary_power_relative_error,
        divergence_theorem_relative_error=divergence_theorem_relative_error,
        conservative_flux_relative_correction=flux_correction,
        conservative_flux_relative_residual=flux_residual,
        conservative_flux_divergence_relative_error=flux_divergence_error,
        floor_relative_activity=sqrt(max(0.0, floor_activity / total_power)),
        local_resonant_element_width=local_width,
        pressure_minimum=float(np.min(pressure_values)),
        pressure_maximum=float(np.max(pressure_values)),
        linear_solver_path=solver_path,
        iteration_count=iteration_count,
    )


def _ray_pressure(
    row: _M4Row, config: FrozenFieldIslandConfig, *, angle: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sample transplanted ``(M4b)`` pressure along one ``Phi=0`` radial ray."""
    radii = np.linspace(0.0, 0.995, config.ray_samples)
    x = radii * np.cos(angle)
    y = radii * np.sin(angle)
    chi = np.asarray(row.field(row.mesh(x, y, np.zeros_like(radii))), dtype=float).reshape(-1)
    pressure = np.asarray(
        row.pressure_profile.value(row.volume_map.evaluate_volume_coordinate(chi)), dtype=float
    )
    return radii, pressure


def _flattening_width(
    radii: NDArray[np.float64],
    pressure: NDArray[np.float64],
    control_pressure: NDArray[np.float64],
    *,
    resonance_radius: float,
    search_width: float,
    gradient_fraction: float,
) -> float:
    """Measure the widest contiguous low-``|dp/dr|`` interval near the resonance."""
    gradient = np.abs(np.gradient(pressure, radii))
    control_gradient = np.abs(np.gradient(control_pressure, radii))
    ratio = gradient / np.maximum(control_gradient, 1.0e-12 * np.max(control_gradient))
    eligible = (ratio < gradient_fraction) & (np.abs(radii - resonance_radius) < search_width)
    widest = 0.0
    lower = 0
    while lower < len(radii):
        if not eligible[lower]:
            lower += 1
            continue
        upper = lower
        while upper + 1 < len(radii) and eligible[upper + 1]:
            upper += 1
        widest = max(widest, float(radii[upper] - radii[lower]))
        lower = upper + 1
    return widest


def _pressure_drop(
    radii: NDArray[np.float64],
    pressure: NDArray[np.float64],
    resonance_radius: float,
    island_width: float,
) -> float:
    """Return the (M4b) pressure drop across the exact island separatrix extrema."""
    lower = max(0.0, resonance_radius - 0.5 * island_width)
    upper = min(float(radii[-1]), resonance_radius + 0.5 * island_width)
    return abs(float(np.interp(lower, radii, pressure) - np.interp(upper, radii, pressure)))


def run_frozen_field_island(
    config: FrozenFieldIslandConfig, pressure_profile: PressureProfile | None
) -> FrozenFieldIslandResult:
    """Run the live Section-8.6 benchmark and its integrable/isotropic controls."""
    profile: PressureProfile = (
        TabulatedPressureProfile((0.0, 1.0), (1.0, 0.0))
        if pressure_profile is None
        else pressure_profile
    )
    profile.validate(edge_value=0.0)
    cylinder = PeriodicCylinder3D(
        max_element_size=config.max_element_size,
        geometry_order=config.geometry_order,
        refinements=config.refinements,
    )
    model = ReimanGreensideField(epsilon_1=config.epsilon_1, epsilon_2=0.0)
    resonance = model.resonance_radius(0.5)
    island_width = exact_island_width(epsilon_1=config.epsilon_1, t1=model.t1)
    critical_width = critical_layer_width(
        epsilon_kappa=config.epsilon_kappa,
        resonance_radius=resonance,
        t1=model.t1,
        major_radius=model.major_radius,
    )
    bundle = cylinder.build_graded_mesh(
        (
            GradedAnnulus(
                radius=resonance,
                half_width=0.6 * max(island_width, critical_width),
                maximum_radial_width=critical_width / config.min_layer_cells,
            ),
        ),
        angular_cells=config.angular_cells,
        axial_cells=config.axial_cells,
        background_radial_width=config.background_radial_width,
    )
    geometry = cylinder.measure_geometry(bundle)
    if geometry.minimum_mapped_jacobian_ratio <= 0.02:
        raise RuntimeError("periodic-cylinder mapped-Jacobian ratio does not clear 0.02")
    if geometry.maximum_relative_error > 5.0e-3:
        raise RuntimeError("geometry error exceeds 0.1 of the 5% physics tolerance")

    integrable_model = ReimanGreensideField(epsilon_1=0.0, epsilon_2=0.0)
    subcritical_width = 0.5 * critical_width
    subcritical_epsilon_1 = subcritical_width**2 * model.t1 / 8.0
    subcritical_model = ReimanGreensideField(epsilon_1=subcritical_epsilon_1, epsilon_2=0.0)
    main = _solve_m4(
        bundle._mesh,
        model,
        config,
        profile,
        perpendicular_conductivity=config.epsilon_kappa,
        recover_flux=True,
    )
    integrable = _solve_m4(
        bundle._mesh,
        integrable_model,
        config,
        profile,
        perpendicular_conductivity=config.epsilon_kappa,
    )
    isotropic = _solve_m4(
        bundle._mesh,
        model,
        config,
        profile,
        perpendicular_conductivity=1.0,
        isotropic=True,
    )
    subcritical = _solve_m4(
        bundle._mesh,
        subcritical_model,
        config,
        profile,
        perpendicular_conductivity=config.epsilon_kappa,
    )

    radii, o_pressure = _ray_pressure(main, config, angle=0.0)
    _, x_pressure = _ray_pressure(main, config, angle=0.5 * pi)
    _, integrable_pressure = _ray_pressure(integrable, config, angle=0.0)
    _, isotropic_pressure = _ray_pressure(isotropic, config, angle=0.0)
    _, subcritical_pressure = _ray_pressure(subcritical, config, angle=0.0)
    search_width = 1.5 * max(island_width, critical_width)
    island_flattening = _flattening_width(
        radii,
        o_pressure,
        integrable_pressure,
        resonance_radius=resonance,
        search_width=search_width,
        gradient_fraction=config.flattening_gradient_fraction,
    )
    integrable_flattening = _flattening_width(
        radii,
        integrable_pressure,
        integrable_pressure,
        resonance_radius=resonance,
        search_width=search_width,
        gradient_fraction=config.flattening_gradient_fraction,
    )
    isotropic_flattening = _flattening_width(
        radii,
        isotropic_pressure,
        integrable_pressure,
        resonance_radius=resonance,
        search_width=search_width,
        gradient_fraction=config.flattening_gradient_fraction,
    )
    subcritical_flattening = _flattening_width(
        radii,
        subcritical_pressure,
        integrable_pressure,
        resonance_radius=resonance,
        search_width=search_width,
        gradient_fraction=config.flattening_gradient_fraction,
    )
    densities = np.asarray(
        [main.volume_map.coarea_density(float(level)) for level in main.volume_map.levels[1:-1]]
    )
    positive_densities = densities[densities > 0.0]
    coarea_spike_ratio = float(
        np.max(positive_densities) / max(np.median(positive_densities), np.finfo(float).tiny)
    )
    volume_diagnostics = main.volume_map.diagnostics()
    level_spacings = np.diff(main.volume_map.levels)
    volume_plateau_ratio = float(np.max(level_spacings) / np.median(level_spacings))
    numerical_perpendicular = _measure_rank_one_pollution(
        bundle._mesh, integrable_model, config, profile
    )
    pollution_ratio = numerical_perpendicular / config.epsilon_kappa
    critical_lower = resonance - 0.5 * critical_width
    critical_upper = resonance + 0.5 * critical_width
    resonant_radial_width = max(
        upper - lower
        for lower, upper in zip(
            bundle.radial_coordinates[:-1], bundle.radial_coordinates[1:], strict=True
        )
        if lower < critical_upper and upper > critical_lower
    )

    diagnostics: dict[str, float | int | str | bool] = {
        "equations": "M4a-M4b",
        "elements": main.elements,
        "h1_dofs": main.dofs,
        "polynomial_order": config.polynomial_order,
        "assembly_seconds": main.assembly_seconds,
        "factorization_solve_seconds": main.solve_seconds,
        "peak_memory_megabytes": _peak_memory_megabytes(),
        "linear_solver_path": main.linear_solver_path,
        "iteration_count": main.iteration_count,
        "free_dof_relative_residual": main.free_dof_relative_residual,
        "geometry_maximum_relative_error": geometry.maximum_relative_error,
        "minimum_mapped_jacobian_ratio": geometry.minimum_mapped_jacobian_ratio,
        "b_floor_relative_activity": main.floor_relative_activity,
        "total_power_relative_error": main.total_power_relative_error,
        "periodic_boundary_power_relative_error": (main.periodic_boundary_power_relative_error),
        "divergence_theorem_relative_error": main.divergence_theorem_relative_error,
        "conservative_flux_relative_correction": main.conservative_flux_relative_correction,
        "conservative_flux_relative_residual": main.conservative_flux_relative_residual,
        "conservative_flux_divergence_relative_error": (
            main.conservative_flux_divergence_relative_error
        ),
        "pressure_minimum": main.pressure_minimum,
        "pressure_maximum": main.pressure_maximum,
        "local_resonant_element_width": resonant_radial_width,
        "layer_cells": critical_width / resonant_radial_width,
        "layer_is_resolved": critical_width / resonant_radial_width >= config.min_layer_cells,
        "numerical_perpendicular_diffusivity": numerical_perpendicular,
        "pollution_ratio": pollution_ratio,
        "island_flattening_width": island_flattening,
        "integrable_flattening_width": integrable_flattening,
        "isotropic_flattening_width": isotropic_flattening,
        "subcritical_flattening_width": subcritical_flattening,
        "subcritical_island_width": subcritical_width,
        "flattening_gradient_fraction": config.flattening_gradient_fraction,
        "island_pressure_drop": _pressure_drop(radii, o_pressure, resonance, island_width),
        "x_ray_pressure_drop": _pressure_drop(radii, x_pressure, resonance, island_width),
        "integrable_pressure_drop": _pressure_drop(
            radii, integrable_pressure, resonance, island_width
        ),
        "isotropic_pressure_drop": _pressure_drop(
            radii, isotropic_pressure, resonance, island_width
        ),
        "coarea_spike_ratio": coarea_spike_ratio,
        "volume_plateau_ratio": volume_plateau_ratio,
        "critical_safeguard_samples": int(volume_diagnostics["floored_sample_count"]),
        "minimum_mollifier_width": volume_diagnostics["minimum_mollifier_width"],
        "maximum_mollifier_width": volume_diagnostics["maximum_mollifier_width"],
        "mollifier_element_size_mode": main.volume_map.element_size_mode,
        "normal_metric_critical_fallback_samples": (main.volume_map.critical_metric_fallback_count),
        "mesh_family": "graded-extruded-tetrahedral",
        "maximum_element_aspect_ratio": bundle.maximum_aspect_ratio,
        "maximum_target_aspect_ratio": bundle.maximum_target_aspect_ratio,
    }
    return FrozenFieldIslandResult(
        epsilon_kappa=config.epsilon_kappa,
        island_width=island_width,
        critical_width=critical_width,
        diagnostics=diagnostics,
    )


def _measure_rank_one_pollution(
    mesh: Any,
    model: ReimanGreensideField,
    config: FrozenFieldIslandConfig,
    pressure_profile: PressureProfile,
) -> float:
    r"""Measure ``kappa_perp,num`` by the Section-8.6 radial power-balance amplitude.

    With ``S_ref=1`` in a radius-one cylinder, the radial solution obeys
    ``chi(0)=1/(4 kappa_perp)``.  The rank-one ``kappa_perp=0`` solve therefore gives
    ``kappa_perp,num=1/(4 chi_h(0))`` from the current production operator.
    """
    del pressure_profile
    import ngsolve as ng

    _, magnetic_field = reiman_greenside_coefficient_functions(model)
    direction = magnetic_field / ng.sqrt(
        ng.InnerProduct(magnetic_field, magnetic_field) + config.b_floor**2
    )
    space = ng.Periodic(ng.H1(mesh, order=config.polynomial_order, dirichlet="wall"))
    trial, test = space.TnT()
    operator = ng.BilinearForm(space)
    operator += (
        ng.InnerProduct(direction, ng.grad(trial)) * ng.InnerProduct(direction, ng.grad(test))
    ).Compile() * ng.dx(bonus_intorder=4)
    right_hand_side = ng.LinearForm(space)
    right_hand_side += test * ng.dx(bonus_intorder=4)
    iterative = space.ndof > config.pollution_direct_dof_threshold
    # The rank-one operator is globally invertible only through discrete cross-field
    # pollution; H1-AMG's element blocks see its exact local nullspace and are singular.
    preconditioner = ng.Preconditioner(operator, "local") if iterative else None
    operator.Assemble()
    right_hand_side.Assemble()
    field = ng.GridFunction(space)
    if iterative:
        assert preconditioner is not None
        field.vec.data = ng.solvers.CG(
            mat=operator.mat,
            rhs=right_hand_side.vec,
            pre=preconditioner.mat,
            tol=1.0e-10,
            maxsteps=4_000,
            printrates=False,
        )
    else:
        field.vec.data = (
            operator.mat.Inverse(space.FreeDofs(), inverse="umfpack") * right_hand_side.vec
        )
    residual = operator.mat * field.vec - right_hand_side.vec
    free_residual = ng.Projector(space.FreeDofs(), True) * residual
    relative_residual = float(ng.Norm(free_residual)) / max(
        1.0, float(ng.Norm(right_hand_side.vec))
    )
    if not isfinite(relative_residual) or relative_residual > 1.0e-8:
        raise RuntimeError(
            f"rank-one pollution solve residual {relative_residual:.3e} exceeds 1e-8"
        )
    axial_values = np.asarray(
        field(
            mesh(
                np.zeros(9),
                np.zeros(9),
                np.linspace(0.0, 2.0 * pi, 9, endpoint=False),
            )
        ),
        dtype=float,
    )
    amplitude = float(np.mean(axial_values))
    if not isfinite(amplitude) or amplitude <= 0.0:
        raise RuntimeError("rank-one M4a pollution solve produced no positive core amplitude")
    return 1.0 / (4.0 * amplitude)
