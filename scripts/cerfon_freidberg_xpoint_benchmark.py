r"""Plot the Cerfon--Freidberg double-null (X-point) Grad-Shafranov verification test.

This is the plotting companion to ``test_cerfon_freidberg_xpoint_boundary_converges``
in ``tests/verification/test_shaped_grad_shafranov.py``.  Every number it computes
comes from the same calls the test makes, so a panel here and an assertion there
are measuring the same thing.

Reference: A. J. Cerfon and J. P. Freidberg, "'One size fits all' analytic
solutions to the Grad-Shafranov equation", Phys. Plasmas **17**, 032502 (2010)
(``docs/CerfonFreidberg_2010_PoP_v17_p032502.pdf``).

Cerfon--Freidberg close the Grad-Shafranov equation with the Solov'ev choice that
reduces it to ``Delta* psi = (1-A) R^2 + A`` in the normalized units
``R -> R/R0``, ``psi -> psi/(psi0)``.  The exact solution is Eq. (8): the two
particular terms ``(1-A) R^4/8`` and ``A R^2 log(R)/2`` plus the seven
``Delta*``-homogeneous polynomials of Eq. (7).  For the double-null case the seven
coefficients are fixed by Eq. (12) -- ``psi=0`` at the inboard and outboard
equatorial points and at the X-point, ``psi_R = psi_Z = 0`` at the X-point (that
is what makes it an X-point), and the two equatorial curvature conditions of
Eq. (10) with ``N1, N2`` from Eq. (11).

This script benchmarks the *shipped* solver: the ``psi=0`` separatrix is meshed
directly by ``AxisymmetricFluxContourDomain`` and
``solve_axisymmetric_grad_shafranov`` runs on it with its homogeneous Dirichlet
condition, which is exact on that wall.  Nothing here re-implements the solver.

**Why this case is harder than the smooth Zheng one.**  The separatrix is not
smooth: it has two genuine corners, at the upper and lower X-points, where the
boundary tangent jumps.  The contour therefore carries explicit
``corner_indices``, the geometry is split into spline chains at those corners, and
the solution has a corner singularity there.  The test consequently asserts only
that both errors *fall monotonically* under refinement, and does not claim a rate:
the measured behaviour is neither a clean ``h^{p+1}`` nor a fixed reduced order.
The two error panels below plot the measured values and the successive-refinement
rates, so the actual behaviour is visible rather than asserted.

Measured, exactly as the test measures it:

* ``relative_l2_error``, ``||psi_h-psi||_2 / ||psi||_2`` over the meshed domain;
* ``boundary_geometry_error``, the RMS of the analytic ``psi`` over the *discrete*
  wall -- how far the meshed separatrix sits from the true ``psi=0`` contour,
  which is a pure geometry-approximation error and the quantity the corners
  degrade;
* ``free_dof_relative_residual_norm``, asserted below ``1e-11``.

Usage::

    python scratch/cerfon_freidberg_xpoint_benchmark.py
    python scratch/cerfon_freidberg_xpoint_benchmark.py --maxh 0.38 0.18 --order 2
    python scratch/cerfon_freidberg_xpoint_benchmark.py --show
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from itertools import pairwise
from math import log, sqrt
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
    CerfonFreidbergBoundary,
    CerfonFreidbergEquilibrium,
    CerfonFreidbergShape,
    FluxContour,
    solve_cerfon_freidberg,
)
from remec.fem._axisymmetric import (
    AxisymmetricGradShafranovCoefficients,
    solve_axisymmetric_grad_shafranov,
)
from remec.geometry.axisymmetric import AxisymmetricFluxContourDomain

# The double-null point design used by the verification test: a spherical-tokamak
# aspect ratio with strong shaping.  `source_parameter = A = 0` is Cerfon-Freidberg's
# pure-pressure-drive limit, `Delta* psi = R^2`.
_SHAPE = CerfonFreidbergShape(0.78, 2.0, 0.35)
_SOURCE_PARAMETER = 0.0
_BOUNDARY = CerfonFreidbergBoundary.DOUBLE_NULL
_CONTOUR_SAMPLES = 385
_GEOMETRY_ORDER = 2


# --------------------------------------------------------------------------------
# Solve and measure -- mirrors `_cerfon_xpoint_rows` in the verification test
# --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One ``maxh`` measurement of the double-null separatrix benchmark."""

    maxh: float
    elements: int
    relative_l2_error: float
    boundary_geometry_error: float
    free_dof_relative_residual_norm: float


@dataclass(frozen=True)
class Case:
    """A solved case plus the objects the figure needs to draw it."""

    row: Row
    mesh: Any
    flux: Any

    def flux_on(
        self, radius: NDArray[np.float64], height: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Evaluate ``psi_h`` on points, returning NaN outside the meshed domain."""
        return _evaluate_on(self.flux, self.mesh, radius, height)


def _evaluate_on(
    coefficient: Any,
    mesh: Any,
    radius: NDArray[np.float64],
    height: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate an NGSolve coefficient at scattered points, NaN outside the mesh.

    ``Mesh.Contains`` is the authoritative inside test -- the meshed region is the
    *discrete* domain, which differs from the analytic ``psi=0`` interior by
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
    equilibrium: CerfonFreidbergEquilibrium,
    contour: FluxContour,
    *,
    polynomial_order: int,
    maxh: float,
) -> Case:
    """Run one separatrix solve and measure it exactly as the test's loop does."""
    domain = AxisymmetricFluxContourDomain(contour, maxh=maxh, geometry_order=_GEOMETRY_ORDER)
    solution = solve_axisymmetric_grad_shafranov(
        domain,
        polynomial_order=polynomial_order,
        coefficients=AxisymmetricGradShafranovCoefficients(
            pressure_flux_derivative=-1.0,
            toroidal_field_drive=0.0,
            mu0=1.0,
        ),
    )
    mesh, computed = solution._mesh, solution._flux
    exact = equilibrium.flux(ng.x, ng.y)
    l2_error = float(ng.sqrt(ng.Integrate((computed - exact) ** 2, mesh, order=12)))
    l2_norm = float(ng.sqrt(ng.Integrate(exact**2, mesh, order=12)))
    length = float(ng.Integrate(1.0, mesh, ng.BND, order=12))
    geometry_error = float(ng.sqrt(ng.Integrate(exact**2, mesh, ng.BND, order=12) / length))
    row = Row(
        maxh,
        solution.elements,
        l2_error / l2_norm,
        geometry_error,
        solution.free_dof_relative_residual_norm,
    )
    return Case(row, mesh, computed)


def effective_mesh_size(elements: int) -> float:
    """Return ``h_eff = n_e^{-1/2}``, the unstructured-mesh size surrogate."""
    return 1.0 / sqrt(elements)


def observed_rate(coarse: Row, fine: Row, attribute: str) -> float:
    """Return ``log(e_c/e_f)/log(sqrt(n_f/n_c))``, the test module's rate formula."""
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

    The triangles are the *straight-sided* elements; the ``Curve(2)`` geometry is
    not represented, so the drawn wall cuts the true separatrix slightly.  That gap
    is the ``boundary_geometry_error`` panel's subject and is made visible by
    overlaying the analytic ``psi=0`` contour.
    """
    points = np.asarray([point.p[:2] for point in mesh.ngmesh.Points()])
    triangles = np.asarray(list(mesh.ngmesh._get2dElementsAsTriangles()), dtype=int).reshape(-1, 3)
    return mtri.Triangulation(points[:, 0], points[:, 1], triangles)


def sample_grid(
    contour: FluxContour, resolution: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a regular ``(R,Z)`` grid with a small margin around the separatrix."""
    radial = contour.radial_bounds
    vertical = contour.vertical_bounds
    radial_margin = 0.03 * (radial[1] - radial[0])
    vertical_margin = 0.03 * (vertical[1] - vertical[0])
    return np.meshgrid(
        np.linspace(radial[0] - radial_margin, radial[1] + radial_margin, resolution),
        np.linspace(vertical[0] - vertical_margin, vertical[1] + vertical_margin, resolution),
    )


def draw_separatrix(
    axis: Axes,
    contour: FluxContour,
    equilibrium: CerfonFreidbergEquilibrium,
    *,
    colour: str,
    width: float = 1.4,
    mark_xpoints: bool = True,
) -> None:
    """Overlay the closed analytic ``psi=0`` separatrix and both X-points."""
    axis.plot(
        np.append(contour.radius, contour.radius[0]),
        np.append(contour.height, contour.height[0]),
        "-",
        color=colour,
        lw=width,
    )
    upper = equilibrium.upper_xpoint
    if mark_xpoints and upper is not None:
        axis.plot(
            (upper[0], upper[0]),
            (upper[1], -upper[1]),
            "x",
            ms=7.0,
            mew=1.8,
            color=colour,
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
    equilibrium: CerfonFreidbergEquilibrium,
    cmap: str = "viridis",
    symmetric: bool = False,
    levels: NDArray[np.float64] | None = None,
) -> None:
    """Filled-contour one masked field with the separatrix and X-points on top."""
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
    draw_separatrix(axis, contour, equilibrium, colour=outline)
    figure.colorbar(filled, ax=axis, shrink=0.9, label="$\\psi$")
    axis.set(title=title, xlabel="$R/R_0$", ylabel="$Z/R_0$", aspect="equal")


def add_convergence_panel(
    axis: Axes,
    rows: list[Row],
    attribute: str,
    title: str,
    *,
    colour: str,
    reference_exponents: tuple[int, ...] = (),
) -> None:
    """Log-log one error quantity against ``h_eff`` with optional reference slopes."""
    ordered = sorted(rows, key=lambda row: -row.maxh)
    sizes = np.asarray([effective_mesh_size(row.elements) for row in ordered])
    errors = np.asarray([getattr(row, attribute) for row in ordered])
    axis.loglog(sizes, errors, "o-", color=colour, label="measured")
    for exponent in reference_exponents:
        axis.loglog(
            sizes,
            errors[0] * (sizes / sizes[0]) ** exponent,
            "--",
            lw=0.9,
            label=f"$h_{{\\rm eff}}^{{{exponent}}}$",
        )
    for coarse, fine, size, error in zip(ordered, ordered[1:], sizes[1:], errors[1:]):
        axis.annotate(
            f"{observed_rate(coarse, fine, attribute):.2f}",
            (size, error),
            textcoords="offset points",
            xytext=(4, -11),
            fontsize=7,
            color=colour,
        )
    axis.set(title=title, xlabel="$h_{\\rm eff}=n_e^{-1/2}$", ylabel="error")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8)


def make_figure(
    equilibrium: CerfonFreidbergEquilibrium,
    contour: FluxContour,
    cases: dict[float, Case],
    *,
    mesh_sizes: tuple[float, ...],
    polynomial_order: int,
    resolution: int,
) -> Figure:
    """Build the mesh strip plus the field, profile, X-point, and error panels."""
    rows = [case.row for case in cases.values()]
    finest = cases[mesh_sizes[-1]]
    axis_radius, axis_flux = equilibrium.magnetic_axis()
    upper_xpoint = equilibrium.upper_xpoint
    assert upper_xpoint is not None, "this script plots the double-null boundary"

    figure = plt.figure(figsize=(18.0, 15.5), layout="constrained")
    mesh_figure, panel_figure = figure.subfigures(2, 1, height_ratios=(0.9, 3.0))

    # -- one mesh per resolution -------------------------------------------------
    mesh_axes = mesh_figure.subplots(1, len(mesh_sizes), squeeze=False)[0]
    for axis, maxh in zip(mesh_axes, mesh_sizes):
        case = cases[maxh]
        axis.triplot(mesh_triangulation(case.mesh), color="0.25", linewidth=0.4)
        draw_separatrix(axis, contour, equilibrium, colour="tab:red", width=1.0)
        axis.set(
            title=f"maxh {maxh:g}: {case.row.elements} elements",
            xlabel="$R/R_0$",
            ylabel="$Z/R_0$",
            aspect="equal",
        )
    mesh_figure.suptitle(
        f"Meshed double-null $\\psi=0$ separatrix (geometry order {_GEOMETRY_ORDER}); "
        "red is the exact analytic contour, crosses are the X-points",
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
    levels = np.sort(np.linspace(axis_flux, 0.0, 13)[1:-1])

    add_field_panel(
        panel_figure,
        axes[0, 0],
        grid_radius,
        grid_height,
        exact,
        "Analytic $\\psi$ (Cerfon--Freidberg Eq. 8) on the meshed domain",
        contour=contour,
        equilibrium=equilibrium,
        levels=levels,
    )
    add_field_panel(
        panel_figure,
        axes[0, 1],
        grid_radius,
        grid_height,
        computed,
        f"FEM $\\psi_h$ (order {polynomial_order}, {finest.row.elements} elements)",
        contour=contour,
        equilibrium=equilibrium,
        levels=levels,
    )
    difference = computed - exact
    add_field_panel(
        panel_figure,
        axes[0, 2],
        grid_radius,
        grid_height,
        difference,
        f"$\\psi_h-\\psi$ (max $|\\cdot|$ = {float(np.nanmax(np.abs(difference))):.2e})",
        contour=contour,
        equilibrium=equilibrium,
        cmap="RdBu_r",
        symmetric=True,
    )

    # -- midplane profile --------------------------------------------------------
    panel = axes[1, 0]
    inner, outer = contour.radial_bounds
    midplane = np.linspace(inner, outer, 401)
    panel.plot(midplane, equilibrium.flux(midplane, 0.0), "k-", lw=2.0, label="analytic")
    panel.plot(
        midplane[::8],
        finest.flux_on(midplane[::8], np.zeros_like(midplane[::8])),
        "o",
        ms=4.0,
        mfc="none",
        color="tab:red",
        label="FEM $\\psi_h$",
    )
    panel.axhline(0.0, color="0.7", lw=0.8)
    panel.axvline(axis_radius, color="0.7", lw=0.8, ls=":")
    panel.set(title="Midplane $\\psi(R,0)$", xlabel="$R/R_0$", ylabel="$\\psi$")
    panel.legend(fontsize=8)

    # -- vertical cut through the two X-points -----------------------------------
    panel = axes[1, 1]
    lower, upper = contour.vertical_bounds
    vertical = np.linspace(lower, upper, 401)
    panel.plot(
        vertical,
        equilibrium.flux(np.full_like(vertical, upper_xpoint[0]), vertical),
        "k-",
        lw=2.0,
        label="analytic",
    )
    panel.plot(
        vertical[::8],
        finest.flux_on(np.full_like(vertical[::8], upper_xpoint[0]), vertical[::8]),
        "o",
        ms=4.0,
        mfc="none",
        color="tab:red",
        label="FEM $\\psi_h$",
    )
    panel.axhline(0.0, color="0.7", lw=0.8)
    for marker in (upper_xpoint[1], -upper_xpoint[1]):
        panel.axvline(marker, color="0.7", lw=0.8, ls=":")
    panel.set(
        title=f"Vertical cut $\\psi(R={upper_xpoint[0]:.3f},Z)$ through both X-points",
        xlabel="$Z/R_0$",
        ylabel="$\\psi$",
    )
    panel.legend(fontsize=8)

    # -- X-point zoom on the finest mesh -----------------------------------------
    panel = axes[1, 2]
    half_width = 0.32 * _SHAPE.inverse_aspect_ratio
    # Separatrix first, mesh on top: the corner element's two boundary edges lie
    # almost exactly on the analytic contour, and drawing the red line last would
    # hide them and make the tip look unmeshed.
    draw_separatrix(panel, contour, equilibrium, colour="tab:red", width=2.6)
    panel.triplot(mesh_triangulation(finest.mesh), color="0.15", linewidth=0.7)
    panel.set(
        title=f"Upper X-point, finest mesh (maxh {mesh_sizes[-1]:g})",
        xlabel="$R/R_0$",
        ylabel="$Z/R_0$",
        aspect="equal",
        xlim=(upper_xpoint[0] - half_width, upper_xpoint[0] + half_width),
        ylim=(upper_xpoint[1] - 1.5 * half_width, upper_xpoint[1] + 0.2 * half_width),
    )

    add_convergence_panel(
        axes[2, 0],
        rows,
        "relative_l2_error",
        "Relative $L^2$ error (labels: successive-refinement rate)",
        colour="tab:blue",
        reference_exponents=(polynomial_order + 1,),
    )
    add_convergence_panel(
        axes[2, 1],
        rows,
        "boundary_geometry_error",
        "Boundary geometry error: RMS analytic $\\psi$ on the discrete wall",
        colour="tab:orange",
        reference_exponents=(2, 3),
    )

    panel = axes[2, 2]
    ordered = sorted(rows, key=lambda row: -row.maxh)
    sizes = [effective_mesh_size(row.elements) for row in ordered]
    panel.loglog(
        sizes,
        [row.free_dof_relative_residual_norm for row in ordered],
        "o-",
        color="tab:green",
        label="measured",
    )
    panel.axhline(1.0e-11, color="tab:red", ls="--", lw=1.0, label="test threshold $10^{-11}$")
    panel.set(
        title="Free-dof relative residual norm (direct solve)",
        xlabel="$h_{\\rm eff}=n_e^{-1/2}$",
        ylabel="relative residual",
    )
    panel.grid(True, which="both", alpha=0.3)
    panel.legend(fontsize=8)

    figure.suptitle(
        "remec double-null Grad-Shafranov verification vs. Cerfon--Freidberg (2010): "
        f"$\\epsilon$={_SHAPE.inverse_aspect_ratio}, $\\kappa$={_SHAPE.elongation}, "
        f"$\\delta$={_SHAPE.triangularity}, $A$={_SOURCE_PARAMETER} "
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
    """Command-line settings; the defaults reproduce the verification test."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--maxh", type=float, nargs="+", default=(0.38, 0.26, 0.18, 0.12, 0.08))
    parser.add_argument("--output-dir", type=Path, default=Path("scratch"))
    parser.add_argument("--resolution", type=int, default=181)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Solve the double-null benchmark on every mesh size and plot it."""
    arguments = parse_arguments()
    mesh_sizes = tuple(sorted(arguments.maxh, reverse=True))
    polynomial_order = int(arguments.order)

    print("Cerfon-Freidberg (2010) double-null equilibrium -- solving Eq. (12)")
    equilibrium = solve_cerfon_freidberg(
        shape=_SHAPE, source_parameter=_SOURCE_PARAMETER, boundary=_BOUNDARY
    )
    axis_radius, axis_flux = equilibrium.magnetic_axis()
    upper_xpoint = equilibrium.upper_xpoint
    assert upper_xpoint is not None
    print(
        "  coefficients c1..c7 " + ", ".join(f"{value:+.6e}" for value in equilibrium.coefficients)
    )
    print(f"  max Eq. (12) constraint residual  {equilibrium.maximum_constraint_residual():.3e}")
    print(f"  magnetic axis  R {axis_radius:.9f}, psi {axis_flux:.9e}")
    print(f"  upper X-point  R {upper_xpoint[0]:.9f}, Z {upper_xpoint[1]:.9f}")

    contour = equilibrium.boundary_contour(samples=_CONTOUR_SAMPLES)
    print(
        f"\nShaped domain: {contour.radius.size}-point separatrix with "
        f"{len(contour.corner_indices)} corners (the X-points), homogeneous Dirichlet wall"
    )

    cases: dict[float, Case] = {}
    for maxh in mesh_sizes:
        case = solve_case(equilibrium, contour, polynomial_order=polynomial_order, maxh=maxh)
        cases[maxh] = case
        row = case.row
        print(
            f"  maxh {maxh:<6.4g} elements {row.elements:>6d} "
            f"L2 {row.relative_l2_error:.4e}  wall {row.boundary_geometry_error:.4e}  "
            f"residual {row.free_dof_relative_residual_norm:.2e}"
        )

    rows = [case.row for case in cases.values()]
    ordered = sorted(rows, key=lambda row: -row.maxh)
    print("\nSuccessive-refinement rates (h_eff = n_e^{-1/2}); the test asserts only monotonicity")
    l2 = ", ".join(f"{observed_rate(a, b, 'relative_l2_error'):.2f}" for a, b in pairwise(ordered))
    wall = ", ".join(
        f"{observed_rate(a, b, 'boundary_geometry_error'):.2f}" for a, b in pairwise(ordered)
    )
    print(f"  relative L2            [{l2}]")
    print(f"  boundary geometry      [{wall}]")

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = arguments.output_dir / "cerfon_freidberg_xpoint_rates.csv"
    write_table(rows, table_path)
    print(f"\nWrote {table_path}")

    figure = make_figure(
        equilibrium,
        contour,
        cases,
        mesh_sizes=mesh_sizes,
        polynomial_order=polynomial_order,
        resolution=arguments.resolution,
    )
    if arguments.show:
        plt.show()
    else:
        figure_path = arguments.output_dir / "cerfon_freidberg_xpoint_benchmark.png"
        figure.savefig(figure_path, dpi=130)
        print(f"Wrote {figure_path}")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
