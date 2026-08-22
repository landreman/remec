r"""Plot the shaped-boundary Zheng Grad-Shafranov verification test.

This is the plotting companion to ``test_shaped_zheng_smooth_boundary_sentinel``
and ``test_shaped_zheng_full_order_scan`` in
``tests/verification/test_shaped_grad_shafranov.py``.  Every number it computes is
produced by the same calls the test makes, so a panel here and an assertion there
are measuring the same thing.

Reference: S. B. Zheng, A. J. Wootton and E. R. Solano, "Analytical tokamak
equilibrium for shaped plasmas", Phys. Plasmas **3**, 1176 (1996)
(``docs/Zheng_1996_PoP_Analytic_Grad_Shafranov_solutions.pdf``).

**How this differs from ``scratch/zheng_grad_shafranov_benchmark.py``.**  The older
script poses the benchmark on a *rectangle* enclosing the plasma and supplies
inhomogeneous Dirichlet data from Zheng Eq. (14), which the shipped solver cannot
do, so it assembles its own system.  The verification test plotted here instead
meshes the exact ``Psi=0`` flux contour itself with
``AxisymmetricFluxContourDomain`` and runs the *shipped*
``solve_axisymmetric_grad_shafranov`` with its homogeneous Dirichlet condition --
which is exact on that wall, because the wall is the ``Psi=0`` surface.  Nothing
in this file re-implements the solver.

Measured, exactly as the test measures it:

* relative ``L2`` error and relative weighted-energy error
  ``||grad(Psi_h-Psi)||_{1/R}/||grad Psi||_{1/R}``;
* ``boundary_geometry_error``, the RMS of the analytic ``Psi`` over the *discrete*
  wall normalized by ``|Psi_axis|`` -- i.e. how far the meshed boundary sits from
  the true ``Psi=0`` contour, which is a pure geometry-approximation error;
* the axis radius and the four Zheng shape parameters recovered from an interior
  level set of ``Psi_h`` by ``recover_smooth_flux_observables``, each differenced
  against the same recovery applied to the analytic ``Psi``.

Rates use the unstructured-mesh convention of the test,
``h_eff proportional n_e^{-1/2}``, so the slope drawn in the convergence panels is
the number the test asserts on.

Usage::

    python scratch/shaped_zheng_grad_shafranov_benchmark.py
    python scratch/shaped_zheng_grad_shafranov_benchmark.py --orders 2 --maxh 0.24 0.12
    python scratch/shaped_zheng_grad_shafranov_benchmark.py --show
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from itertools import pairwise
from math import log, pi, sqrt
from pathlib import Path
from typing import Any

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import ngsolve as ng
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from remec.analytic_equilibria import (
    FluxContour,
    ZhengEquilibrium,
    ZhengShape,
    recover_smooth_flux_observables,
    solve_zheng_equilibrium,
)
from remec.fem._axisymmetric import (
    AxisymmetricGradShafranovCoefficients,
    solve_axisymmetric_grad_shafranov,
)
from remec.geometry.axisymmetric import AxisymmetricFluxContourDomain

_MU0 = 4.0e-7 * pi

# The Fig. 1 spherical-tokamak point design, identical to the test's `_ZHENG`.
_SHAPE = ZhengShape(0.70, 0.49, 1.7, 0.125)
_POLOIDAL_BETA = 0.40
_PLASMA_CURRENT = 1.0e6
_CONTOUR_SAMPLES = 257


# --------------------------------------------------------------------------------
# Solve and measure -- mirrors `_zheng_row` in the verification test
# --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One ``(polynomial_order, maxh)`` measurement of the shaped Zheng benchmark."""

    polynomial_order: int
    maxh: float
    elements: int
    free_dof_relative_residual_norm: float
    relative_l2_error: float
    relative_weighted_energy_error: float
    boundary_geometry_error: float
    axis_radius_error: float
    major_radius_error: float
    minor_radius_error: float
    elongation_error: float
    triangularity_error: float


@dataclass(frozen=True)
class Case:
    """A solved case plus the objects the figure needs to draw it."""

    row: Row
    mesh: Any
    flux: Any

    def flux_on(
        self, radius: NDArray[np.float64], height: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Evaluate ``Psi_h`` on points, returning NaN outside the meshed domain."""
        return _evaluate_on(self.flux, self.mesh, radius, height)


def _evaluate_on(
    coefficient: Any,
    mesh: Any,
    radius: NDArray[np.float64],
    height: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate an NGSolve coefficient at scattered points, NaN outside the mesh.

    ``Mesh.Contains`` is the authoritative inside test -- the meshed region is the
    *discrete* domain, which differs from the analytic ``Psi=0`` interior by
    exactly the geometry error this benchmark measures.
    """
    radius, height = np.broadcast_arrays(
        np.asarray(radius, dtype=float), np.asarray(height, dtype=float)
    )
    flat_radius, flat_height = radius.ravel(), height.ravel()
    inside = np.asarray(
        [mesh.Contains(float(r), float(z)) for r, z in zip(flat_radius, flat_height)]
    )
    values = np.full(flat_radius.shape, np.nan)
    if np.any(inside):
        values[inside] = np.asarray(
            coefficient(mesh(flat_radius[inside], flat_height[inside])), dtype=float
        ).ravel()
    return values.reshape(radius.shape)


def solve_case(
    equilibrium: ZhengEquilibrium,
    contour: FluxContour,
    *,
    polynomial_order: int,
    maxh: float,
) -> Case:
    """Run one shaped-boundary solve and measure it exactly as ``_zheng_row`` does."""
    domain = AxisymmetricFluxContourDomain(
        contour,
        maxh=maxh,
        geometry_order=polynomial_order + 1,
    )
    solution = solve_axisymmetric_grad_shafranov(
        domain,
        polynomial_order=polynomial_order,
        coefficients=AxisymmetricGradShafranovCoefficients(
            pressure_flux_derivative=-equilibrium.a1 / _MU0,
            toroidal_field_drive=equilibrium.a2,
            mu0=_MU0,
        ),
    )
    mesh, computed = solution._mesh, solution._flux

    exact = equilibrium.flux(ng.x, ng.y)
    exact_gradient = ng.CoefficientFunction(
        (
            equilibrium.radial_derivative(ng.x, ng.y),
            equilibrium.vertical_derivative(ng.x, ng.y),
        )
    )
    order = 2 * polynomial_order + 8
    l2_error = float(ng.sqrt(ng.Integrate((computed - exact) ** 2, mesh, order=order)))
    l2_norm = float(ng.sqrt(ng.Integrate(exact**2, mesh, order=order)))
    gradient_error = ng.grad(computed) - exact_gradient
    energy_error = float(
        ng.sqrt(
            ng.Integrate(ng.InnerProduct(gradient_error, gradient_error) / ng.x, mesh, order=order)
        )
    )
    energy_norm = float(
        ng.sqrt(
            ng.Integrate(ng.InnerProduct(exact_gradient, exact_gradient) / ng.x, mesh, order=order)
        )
    )
    boundary_length = float(ng.Integrate(1.0, mesh, ng.BND, order=order))
    boundary_geometry_error = float(
        ng.sqrt(ng.Integrate(exact**2, mesh, ng.BND, order=order) / boundary_length)
        / abs(equilibrium.magnetic_axis()[1])
    )

    observables = recover_smooth_flux_observables(mesh=mesh, flux=computed, search_contour=contour)
    exact_observables = recover_smooth_flux_observables(
        mesh=mesh, flux=exact, search_contour=contour, validate_boundary=False
    )
    exact_axis_radius, _ = equilibrium.magnetic_axis()

    row = Row(
        polynomial_order,
        maxh,
        solution.elements,
        solution.free_dof_relative_residual_norm,
        l2_error / l2_norm,
        energy_error / energy_norm,
        boundary_geometry_error,
        abs(observables.axis_radius - exact_axis_radius),
        abs(observables.major_radius - exact_observables.major_radius),
        abs(observables.minor_radius - exact_observables.minor_radius),
        abs(observables.elongation - exact_observables.elongation),
        abs(observables.triangularity - exact_observables.triangularity),
    )
    return Case(row, mesh, computed)


def effective_mesh_size(elements: int) -> float:
    """Return ``h_eff = n_e^{-1/2}``, the test's unstructured-mesh size surrogate."""
    return 1.0 / sqrt(elements)


def observed_rate(coarse: Row, fine: Row, attribute: str) -> float:
    """Return the test's rate ``log(e_c/e_f)/log(sqrt(n_f/n_c))`` for one quantity."""
    coarse_error = getattr(coarse, attribute)
    fine_error = getattr(fine, attribute)
    if coarse_error <= 0.0 or fine_error <= 0.0:
        return float("nan")
    return log(coarse_error / fine_error) / log(sqrt(fine.elements / coarse.elements))


# --------------------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------------------


def mesh_triangulation(mesh: Any) -> mtri.Triangulation:
    """Convert the NGSolve mesh topology into a Matplotlib triangulation.

    The triangles are the *straight-sided* elements; the curved geometry of order
    ``p+1`` is not represented, so the drawn wall cuts the true contour slightly.
    That gap is the ``boundary_geometry_error`` panel's subject and is shown
    explicitly by overlaying the analytic ``Psi=0`` contour.
    """
    points = np.asarray([point.p[:2] for point in mesh.ngmesh.Points()])
    triangles = np.asarray(list(mesh.ngmesh._get2dElementsAsTriangles()), dtype=int).reshape(-1, 3)
    return mtri.Triangulation(points[:, 0], points[:, 1], triangles)


def sample_grid(
    contour: FluxContour, resolution: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a regular ``(R,Z)`` grid with a small margin around the contour."""
    radial = contour.radial_bounds
    vertical = contour.vertical_bounds
    radial_margin = 0.03 * (radial[1] - radial[0])
    vertical_margin = 0.03 * (vertical[1] - vertical[0])
    return np.meshgrid(
        np.linspace(radial[0] - radial_margin, radial[1] + radial_margin, resolution),
        np.linspace(vertical[0] - vertical_margin, vertical[1] + vertical_margin, resolution),
    )


def draw_boundary(axis: Axes, contour: FluxContour, *, colour: str, width: float = 1.4) -> None:
    """Overlay the closed analytic ``Psi=0`` contour that defines the wall."""
    axis.plot(
        np.append(contour.radius, contour.radius[0]),
        np.append(contour.height, contour.height[0]),
        "-",
        color=colour,
        lw=width,
    )


def draw_shape_points(axis: Axes, shape: ZhengShape, *, colour: str) -> None:
    """Mark Zheng's three defining shape points, Eqs. (15)-(17)."""
    axis.plot(
        (shape.inner_radius, shape.outer_radius, shape.top_radius),
        (0.0, 0.0, shape.top_height),
        "o",
        ms=4.5,
        color=colour,
        mfc="none",
    )


def add_field_panel(
    figure: Figure,
    axis: Axes,
    grid_radius: NDArray[np.float64],
    grid_height: NDArray[np.float64],
    values: NDArray[np.float64],
    title: str,
    *,
    contour: FluxContour,
    shape: ZhengShape,
    cmap: str = "viridis",
    symmetric: bool = False,
    levels: NDArray[np.float64] | None = None,
) -> None:
    """Filled-contour one masked field with the analytic plasma boundary on top."""
    limit = float(np.nanmax(np.abs(values))) or 1.0
    filled = axis.contourf(
        grid_radius,
        grid_height,
        values,
        levels=40,
        cmap=cmap,
        **({"vmin": -limit, "vmax": limit} if symmetric else {}),
    )
    if levels is not None:
        axis.contour(
            grid_radius, grid_height, values, levels=levels, colors="white", linewidths=0.6
        )
    outline = "0.25" if symmetric else "white"
    draw_boundary(axis, contour, colour=outline)
    draw_shape_points(axis, shape, colour=outline)
    figure.colorbar(filled, ax=axis, shrink=0.9, label="Wb/rad")
    axis.set(title=title, xlabel="R [m]", ylabel="Z [m]", aspect="equal")


def add_convergence_panel(
    axis: Axes,
    rows: list[Row],
    attribute: str,
    title: str,
    *,
    reference_offset: int | None,
) -> None:
    """Log-log one error quantity against ``h_eff``, one curve per polynomial order."""
    for order in sorted({row.polynomial_order for row in rows}):
        ordered = sorted(
            (row for row in rows if row.polynomial_order == order), key=lambda r: -r.maxh
        )
        sizes = np.asarray([effective_mesh_size(row.elements) for row in ordered])
        errors = np.asarray([getattr(row, attribute) for row in ordered])
        line = axis.loglog(sizes, errors, "o-", label=f"order {order}")[0]
        if reference_offset is not None:
            exponent = order + reference_offset
            axis.loglog(
                sizes,
                errors[0] * (sizes / sizes[0]) ** exponent,
                "--",
                color=line.get_color(),
                lw=0.9,
                label=f"$h_{{\\rm eff}}^{{{exponent}}}$",
            )
    axis.set(title=title, xlabel="$h_{\\rm eff}=n_e^{-1/2}$", ylabel="error")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=7, ncol=2)


def add_observable_panel(
    axis: Axes,
    rows: list[Row],
    attributes: tuple[tuple[str, str, str], ...],
    title: str,
) -> None:
    """Log-log recovered-shape errors: colour is the order, line style the quantity."""
    orders = sorted({row.polynomial_order for row in rows})
    colours = {order: f"C{index}" for index, order in enumerate(orders)}
    for order in orders:
        ordered = sorted(
            (row for row in rows if row.polynomial_order == order), key=lambda r: -r.maxh
        )
        sizes = np.asarray([effective_mesh_size(row.elements) for row in ordered])
        for attribute, style, label in attributes:
            errors = np.asarray([max(getattr(row, attribute), 1.0e-16) for row in ordered])
            axis.loglog(
                sizes,
                errors,
                style,
                color=colours[order],
                ms=4.0,
                lw=1.2,
                label=f"{label}, order {order}",
            )
    axis.set(title=title, xlabel="$h_{\\rm eff}=n_e^{-1/2}$", ylabel="absolute error")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=6, ncol=len(orders))


def make_figure(
    equilibrium: ZhengEquilibrium,
    contour: FluxContour,
    cases: dict[tuple[int, float], Case],
    *,
    orders: tuple[int, ...],
    mesh_sizes: tuple[float, ...],
    resolution: int,
) -> Figure:
    """Build the mesh strip plus the field, profile, and convergence panels."""
    rows = [case.row for case in cases.values()]
    finest = cases[max(orders), mesh_sizes[-1]]
    _, axis_flux = equilibrium.magnetic_axis()

    figure = plt.figure(figsize=(17.0, 15.5), layout="constrained")
    mesh_figure, panel_figure = figure.subfigures(2, 1, height_ratios=(0.9, 3.0))

    # -- one mesh per resolution -------------------------------------------------
    mesh_axes = mesh_figure.subplots(1, len(mesh_sizes), squeeze=False)[0]
    for axis, maxh in zip(mesh_axes, mesh_sizes):
        case = cases[max(orders), maxh]
        axis.triplot(mesh_triangulation(case.mesh), color="0.25", linewidth=0.45)
        draw_boundary(axis, contour, colour="tab:red", width=1.1)
        axis.set(
            title=f"maxh {maxh:g}: {case.row.elements} elements",
            xlabel="R [m]",
            ylabel="Z [m]",
            aspect="equal",
        )
    mesh_figure.suptitle(
        f"Meshed $\\Psi=0$ flux-contour domain (geometry order {max(orders) + 1}); "
        "red is the exact analytic contour",
        fontsize=11,
    )

    axes = panel_figure.subplots(3, 3)

    grid_radius, grid_height = sample_grid(contour, resolution)
    computed = finest.flux_on(grid_radius, grid_height)
    exact = np.where(
        np.isnan(computed),
        np.nan,
        np.asarray(equilibrium.flux(grid_radius, grid_height), dtype=float),
    )
    levels = np.linspace(0.0, axis_flux, 13)[1:-1]

    add_field_panel(
        panel_figure,
        axes[0, 0],
        grid_radius,
        grid_height,
        exact,
        "Analytic $\\Psi$ (Zheng Eq. 14) on the meshed domain",
        contour=contour,
        shape=equilibrium.shape,
        levels=levels,
    )
    add_field_panel(
        panel_figure,
        axes[0, 1],
        grid_radius,
        grid_height,
        computed,
        f"FEM $\\Psi_h$ (order {finest.row.polynomial_order}, {finest.row.elements} elements)",
        contour=contour,
        shape=equilibrium.shape,
        levels=levels,
    )
    difference = computed - exact
    add_field_panel(
        panel_figure,
        axes[0, 2],
        grid_radius,
        grid_height,
        difference,
        f"$\\Psi_h-\\Psi$ (max $|\\cdot|$ = {float(np.nanmax(np.abs(difference))):.2e})",
        contour=contour,
        shape=equilibrium.shape,
        cmap="RdBu_r",
        symmetric=True,
    )

    # -- midplane profile --------------------------------------------------------
    panel = axes[1, 0]
    inner, outer = contour.radial_bounds
    midplane = np.linspace(inner, outer, 401)
    panel.plot(midplane, equilibrium.flux(midplane, 0.0), "k-", lw=2.0, label="analytic")
    sampled = finest.flux_on(midplane[::8], np.zeros_like(midplane[::8]))
    panel.plot(
        midplane[::8], sampled, "o", ms=4.0, mfc="none", color="tab:red", label="FEM $\\Psi_h$"
    )
    panel.axhline(0.0, color="0.7", lw=0.8)
    for marker in (equilibrium.shape.inner_radius, equilibrium.shape.outer_radius):
        panel.axvline(marker, color="0.7", lw=0.8, ls=":")
    panel.set(title="Midplane $\\Psi(R,0)$", xlabel="R [m]", ylabel="$\\Psi$ [Wb/rad]")
    panel.legend(fontsize=8)

    add_convergence_panel(
        axes[1, 1], rows, "relative_l2_error", "Relative $L^2$ error", reference_offset=1
    )
    add_convergence_panel(
        axes[1, 2],
        rows,
        "relative_weighted_energy_error",
        "Relative weighted-energy error $\\|\\nabla(\\Psi_h-\\Psi)\\|_{1/R}$",
        reference_offset=0,
    )
    add_convergence_panel(
        axes[2, 0],
        rows,
        "boundary_geometry_error",
        "Boundary geometry error: RMS $\\Psi$ on the discrete wall / $|\\Psi_{axis}|$",
        reference_offset=None,
    )
    add_observable_panel(
        axes[2, 1],
        rows,
        (
            ("axis_radius_error", "o-", "$|\\Delta R_{axis}|$"),
            ("major_radius_error", "s--", "$|\\Delta R_0|$"),
            ("minor_radius_error", "^:", "$|\\Delta a|$"),
        ),
        "Recovered radii (level set of $\\Psi_h$)",
    )
    add_observable_panel(
        axes[2, 2],
        rows,
        (
            ("elongation_error", "o-", "$|\\Delta\\kappa|$"),
            ("triangularity_error", "s--", "$|\\Delta\\delta|$"),
        ),
        "Recovered elongation and triangularity",
    )

    figure.suptitle(
        "remec shaped-boundary Grad-Shafranov verification vs. Zheng (1996): "
        "$R_0$=0.70 m, $a$=0.49 m, $\\kappa$=1.7, $\\delta$=0.125, "
        "$\\beta_{pol}$=0.40, $I_p$=1 MA "
        "(tests/verification/test_shaped_grad_shafranov.py)",
        fontsize=12,
    )
    return figure


# --------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------


def write_table(rows: list[Row], path: Path) -> None:
    """Write every measured row next to the figure."""
    fields = list(asdict(rows[0]))
    with path.open("w", newline="") as table_file:
        writer = csv.writer(table_file)
        writer.writerow(fields)
        for row in rows:
            record = asdict(row)
            writer.writerow(
                [
                    record[name] if isinstance(record[name], int) else repr(record[name])
                    for name in fields
                ]
            )


def parse_arguments() -> argparse.Namespace:
    """Command-line settings; the defaults reproduce the full-order-scan test."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--orders", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--maxh", type=float, nargs="+", default=(0.20, 0.12, 0.07))
    parser.add_argument("--output-dir", type=Path, default=Path("scratch"))
    parser.add_argument("--resolution", type=int, default=181)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Solve the shaped Zheng benchmark on every (order, maxh) pair and plot it."""
    arguments = parse_arguments()
    orders = tuple(arguments.orders)
    mesh_sizes = tuple(sorted(arguments.maxh, reverse=True))

    print("Zheng (1996) Fig. 1 equilibrium -- solving Eqs. (15)-(20)")
    equilibrium = solve_zheng_equilibrium(
        shape=_SHAPE, poloidal_beta=_POLOIDAL_BETA, plasma_current=_PLASMA_CURRENT
    )
    integrals = equilibrium.figure_of_merit_integrals()
    axis_radius, axis_flux = equilibrium.magnetic_axis()
    print(f"  A1 {equilibrium.a1:+.9e}   A2 {equilibrium.a2:+.9e}")
    print(f"  Ip {integrals.plasma_current:.9e} A   beta_pol {integrals.poloidal_beta:.9f}")
    print(f"  magnetic axis R {axis_radius:.9f} m, Psi {axis_flux:.9e} Wb/rad")

    contour = equilibrium.boundary_contour(samples=_CONTOUR_SAMPLES)
    print(f"\nShaped domain: {_CONTOUR_SAMPLES}-point Psi=0 contour, homogeneous Dirichlet wall")

    cases: dict[tuple[int, float], Case] = {}
    for order in orders:
        for maxh in mesh_sizes:
            case = solve_case(equilibrium, contour, polynomial_order=order, maxh=maxh)
            cases[order, maxh] = case
            row = case.row
            print(
                f"  order {order}  maxh {maxh:<6.4g} elements {row.elements:>6d} "
                f"L2 {row.relative_l2_error:.4e}  energy "
                f"{row.relative_weighted_energy_error:.4e}  "
                f"wall {row.boundary_geometry_error:.3e}  "
                f"residual {row.free_dof_relative_residual_norm:.2e}"
            )

    rows = [case.row for case in cases.values()]
    print("\nObserved rates (h_eff = n_e^{-1/2}, the convention of the verification test)")
    for order in orders:
        ordered = sorted(
            (row for row in rows if row.polynomial_order == order), key=lambda r: -r.maxh
        )
        l2 = ", ".join(
            f"{observed_rate(a, b, 'relative_l2_error'):.2f}" for a, b in pairwise(ordered)
        )
        energy = ", ".join(
            f"{observed_rate(a, b, 'relative_weighted_energy_error'):.2f}"
            for a, b in pairwise(ordered)
        )
        print(
            f"  order {order}:  L2 [{l2}] (expect {order + 1})   energy [{energy}] (expect {order})"
        )

    print(f"\nRecovered observables on the finest mesh (maxh {mesh_sizes[-1]:g})")
    print(
        f"  {'order':>5s} {'|dR_axis|':>11s} {'|dR0|':>11s} {'|da|':>11s} "
        f"{'|dkappa|':>11s} {'|ddelta|':>11s}"
    )
    for order in orders:
        row = cases[order, mesh_sizes[-1]].row
        print(
            f"  {order:>5d} {row.axis_radius_error:>11.3e} {row.major_radius_error:>11.3e} "
            f"{row.minor_radius_error:>11.3e} {row.elongation_error:>11.3e} "
            f"{row.triangularity_error:>11.3e}"
        )

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = arguments.output_dir / "shaped_zheng_grad_shafranov_rates.csv"
    write_table(rows, table_path)
    print(f"\nWrote {table_path}")

    figure = make_figure(
        equilibrium,
        contour,
        cases,
        orders=orders,
        mesh_sizes=mesh_sizes,
        resolution=arguments.resolution,
    )
    if arguments.show:
        plt.show()
    else:
        figure_path = arguments.output_dir / "shaped_zheng_grad_shafranov_benchmark.png"
        figure.savefig(figure_path, dpi=130)
        print(f"Wrote {figure_path}")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
