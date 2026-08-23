"""Poincare-section diagnostics for the magnetic field used by (M4a)--(M4b)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import cos, isfinite, pi, sin
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

MagneticField = Callable[[float, float, float], Sequence[float] | NDArray[np.float64]]
Invariant = Callable[[float, float], float]
_SCHEMA_VERSION = 1


class PoincareTraceVersionError(ValueError):
    """Raised for an invalid or unsupported persisted Poincare record."""


@dataclass(frozen=True, slots=True)
class PoincareTrace:
    """Versioned section samples from field lines associated with (M4a)--(M4b)."""

    section_phis: NDArray[np.float64]
    radii: NDArray[np.float64]
    theta_unwrapped: NDArray[np.float64]
    major_radius: float
    relative_tolerance: float
    absolute_tolerance: float

    def __post_init__(self) -> None:
        section_phis = np.asarray(self.section_phis, dtype=float)
        radii = np.asarray(self.radii, dtype=float)
        theta_unwrapped = np.asarray(self.theta_unwrapped, dtype=float)
        if section_phis.ndim != 1 or section_phis.size < 2:
            raise ValueError("section_phis must contain at least two samples")
        if np.any(np.diff(section_phis) <= 0.0):
            raise ValueError("section_phis must be strictly increasing")
        if radii.ndim != 2 or radii.shape[1] != section_phis.size:
            raise ValueError("radii must have shape (field_lines, sections)")
        if theta_unwrapped.shape != radii.shape:
            raise ValueError("theta_unwrapped must have the same shape as radii")
        if not (
            np.all(np.isfinite(section_phis))
            and np.all(np.isfinite(radii))
            and np.all(np.isfinite(theta_unwrapped))
        ):
            raise ValueError("Poincare trace arrays must be finite")
        if np.any(radii <= 0.0):
            raise ValueError("Poincare trace radii must be positive")
        for name, value in (
            ("major_radius", self.major_radius),
            ("relative_tolerance", self.relative_tolerance),
            ("absolute_tolerance", self.absolute_tolerance),
        ):
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        section_phis = np.array(section_phis, copy=True)
        radii = np.array(radii, copy=True)
        theta_unwrapped = np.array(theta_unwrapped, copy=True)
        section_phis.setflags(write=False)
        radii.setflags(write=False)
        theta_unwrapped.setflags(write=False)
        object.__setattr__(self, "section_phis", section_phis)
        object.__setattr__(self, "radii", radii)
        object.__setattr__(self, "theta_unwrapped", theta_unwrapped)


@dataclass(frozen=True, slots=True)
class PeriodicPoint:
    """A numerically located periodic point of a magnetic-field Poincare map."""

    radius: float
    theta: float
    kind: str
    residual_norm: float
    monodromy: NDArray[np.float64]


def trace_poincare(
    field: MagneticField,
    seeds: Sequence[Sequence[float]] | NDArray[np.float64],
    *,
    turns: int,
    major_radius: float = 1.0,
    section_phi: float = 0.0,
    relative_tolerance: float = 1.0e-10,
    absolute_tolerance: float = 1.0e-12,
    maximum_step: float = 0.2,
) -> PoincareTrace:
    r"""Trace field lines associated with (M4a)--(M4b) at fixed ``Phi``.

    With ``z = R0*Phi``, cylindrical components of the supplied Cartesian field
    give ``dr/dPhi = R0*B_r/B_z`` and
    ``dTheta/dPhi = R0*B_Theta/(r*B_z)``. SciPy's adaptive DOP853 integrator samples
    the solution exactly at successive copies of the requested Poincare plane.
    """
    seed_array = np.asarray(seeds, dtype=float)
    if seed_array.ndim != 2 or seed_array.shape[1] != 2 or seed_array.shape[0] == 0:
        raise ValueError("seeds must have shape (field_lines, 2) for (radius, theta)")
    if not np.all(np.isfinite(seed_array)) or np.any(seed_array[:, 0] <= 0.0):
        raise ValueError("seed radii must be positive and all seed values finite")
    _validate_integration_options(
        turns=turns,
        major_radius=major_radius,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_step=maximum_step,
    )
    if not isfinite(section_phi):
        raise ValueError("section_phi must be finite")
    section_phis = section_phi + 2.0 * pi * np.arange(turns + 1, dtype=float)
    radii = np.empty((seed_array.shape[0], section_phis.size))
    theta_unwrapped = np.empty_like(radii)
    for line, seed in enumerate(seed_array):
        solution = _integrate_field_line(
            field,
            seed,
            section_phis,
            major_radius=major_radius,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            maximum_step=maximum_step,
        )
        radii[line] = solution[0]
        theta_unwrapped[line] = solution[1]
    return PoincareTrace(
        section_phis=section_phis,
        radii=radii,
        theta_unwrapped=theta_unwrapped,
        major_radius=major_radius,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )


def recover_rotational_transform(trace: PoincareTrace) -> NDArray[np.float64]:
    """Recover iota = Delta Theta / Delta Phi from unwrapped section angles."""
    delta_phi = trace.section_phis[-1] - trace.section_phis[0]
    return np.asarray(
        (trace.theta_unwrapped[:, -1] - trace.theta_unwrapped[:, 0]) / delta_phi,
        dtype=float,
    )


def radial_excursion(trace: PoincareTrace) -> NDArray[np.float64]:
    """Return each field line's apparent radial island width on the section."""
    return np.asarray(np.ptp(trace.radii, axis=1), dtype=float)


def find_periodic_point(
    field: MagneticField,
    guess: Sequence[float],
    *,
    period_turns: int,
    poloidal_winding: int,
    major_radius: float = 1.0,
    section_phi: float = 0.0,
    relative_tolerance: float = 1.0e-11,
    absolute_tolerance: float = 1.0e-13,
    maximum_step: float = 0.1,
) -> PeriodicPoint:
    r"""Locate and classify an O- or X-point from a field-line return map.

    The root condition is ``P^q(r,Theta) = (r, Theta + 2*pi*p)`` for
    ``q=period_turns`` and ``p=poloidal_winding``. Classification uses the trace of
    the numerically differentiated monodromy: ``|tr(M)| < 2`` is elliptic (O), while
    ``|tr(M)| > 2`` is hyperbolic (X).
    """
    guess_array = np.asarray(guess, dtype=float)
    if guess_array.shape != (2,) or not np.all(np.isfinite(guess_array)) or guess_array[0] <= 0.0:
        raise ValueError("guess must contain one finite pair with positive radius")
    _validate_integration_options(
        turns=period_turns,
        major_radius=major_radius,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_step=maximum_step,
    )
    if not isfinite(section_phi):
        raise ValueError("section_phi must be finite")
    if isinstance(poloidal_winding, bool) or not isinstance(poloidal_winding, int):
        raise TypeError("poloidal_winding must be an integer")

    from scipy.optimize import root  # type: ignore[import-untyped]

    phi_end = section_phi + 2.0 * pi * period_turns
    target_angle_advance = 2.0 * pi * poloidal_winding

    def return_map(state: NDArray[np.float64]) -> NDArray[np.float64]:
        samples = _integrate_field_line(
            field,
            state,
            np.array([section_phi, phi_end]),
            major_radius=major_radius,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            maximum_step=maximum_step,
        )
        return samples[:, -1]

    def residual(state: NDArray[np.float64]) -> NDArray[np.float64]:
        returned = return_map(state)
        return np.array([returned[0] - state[0], returned[1] - state[1] - target_angle_advance])

    solved = root(residual, guess_array, method="hybr", options={"xtol": 1.0e-10})
    residual_at_root = residual(np.asarray(solved.x, dtype=float))
    residual_norm = float(np.linalg.norm(residual_at_root))
    if not solved.success and residual_norm > 10.0 * max(absolute_tolerance, relative_tolerance):
        raise RuntimeError(f"periodic-point solve failed: {solved.message}")

    state = np.asarray(solved.x, dtype=float)
    monodromy = np.empty((2, 2))
    for column in range(2):
        step = 1.0e-5 * max(1.0, abs(float(state[column])))
        perturbation = np.zeros(2)
        perturbation[column] = step
        monodromy[:, column] = (
            return_map(state + perturbation) - return_map(state - perturbation)
        ) / (2.0 * step)
    trace_value = float(np.trace(monodromy))
    if abs(trace_value) < 2.0:
        kind = "O"
    elif abs(trace_value) > 2.0:
        kind = "X"
    else:
        kind = "parabolic"
    normalized_theta = float((state[1] + pi) % (2.0 * pi) - pi)
    return PeriodicPoint(
        radius=float(state[0]),
        theta=normalized_theta,
        kind=kind,
        residual_norm=residual_norm,
        monodromy=monodromy,
    )


def separatrix_width_from_invariant(
    invariant: Invariant,
    *,
    level: float,
    cut_theta: float,
    inner_bracket: tuple[float, float],
    outer_bracket: tuple[float, float],
) -> float:
    r"""Derive a full radial separatrix width from ``K(r,Theta_cut)=K_X``.

    This deliberately solves the two conserved-Hamiltonian root equations rather than
    reusing the reduced Reiman--Greenside width formula that verifies the result.
    """
    from scipy.optimize import brentq

    def residual(radius: float) -> float:
        value = float(invariant(radius, cut_theta) - level)
        if not isfinite(value):
            raise ValueError("invariant must return finite values")
        return value

    inner = float(brentq(residual, *inner_bracket, xtol=1.0e-13, rtol=1.0e-14))
    outer = float(brentq(residual, *outer_bracket, xtol=1.0e-13, rtol=1.0e-14))
    if outer <= inner:
        raise ValueError("separatrix brackets must identify ordered radial roots")
    return outer - inner


def save_poincare_trace(trace: PoincareTrace, path: str | Path) -> None:
    """Persist a Poincare trace as an explicit versioned NumPy record."""
    with Path(path).open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.array(_SCHEMA_VERSION),
            section_phis=trace.section_phis,
            radii=trace.radii,
            theta_unwrapped=trace.theta_unwrapped,
            major_radius=np.array(trace.major_radius),
            relative_tolerance=np.array(trace.relative_tolerance),
            absolute_tolerance=np.array(trace.absolute_tolerance),
        )


def load_poincare_trace(path: str | Path) -> PoincareTrace:
    """Load and validate a supported versioned Poincare trace record."""
    expected_keys = {
        "schema_version",
        "section_phis",
        "radii",
        "theta_unwrapped",
        "major_radius",
        "relative_tolerance",
        "absolute_tolerance",
    }
    try:
        with np.load(Path(path), allow_pickle=False) as record:
            if set(record.files) != expected_keys:
                raise PoincareTraceVersionError("invalid Poincare trace schema fields")
            raw_schema = record["schema_version"]
            if raw_schema.shape != () or not np.issubdtype(raw_schema.dtype, np.integer):
                raise PoincareTraceVersionError("invalid Poincare trace schema version")
            schema_version = int(raw_schema.item())
            if schema_version != _SCHEMA_VERSION:
                raise PoincareTraceVersionError(
                    f"unsupported Poincare trace schema version {schema_version}"
                )
            return PoincareTrace(
                section_phis=record["section_phis"],
                radii=record["radii"],
                theta_unwrapped=record["theta_unwrapped"],
                major_radius=float(record["major_radius"].item()),
                relative_tolerance=float(record["relative_tolerance"].item()),
                absolute_tolerance=float(record["absolute_tolerance"].item()),
            )
    except PoincareTraceVersionError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise PoincareTraceVersionError("invalid Poincare trace record") from error


def plot_poincare(trace: PoincareTrace, axes: Any, **scatter_options: Any) -> Any:
    """Plot section points on a caller-supplied Matplotlib-compatible axes."""
    x = trace.radii * np.cos(trace.theta_unwrapped)
    y = trace.radii * np.sin(trace.theta_unwrapped)
    plotted = axes.scatter(x.ravel(), y.ravel(), **scatter_options)
    axes.set_aspect("equal")
    return plotted


def plot_isobar_overlay(
    trace: PoincareTrace,
    axes: Any,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    pressure: NDArray[np.float64],
    *,
    levels: int | Sequence[float] = 12,
    **contour_options: Any,
) -> tuple[Any, Any]:
    """Overlay Poincare points on caller-supplied cross-section pressure isobars."""
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    pressure_array = np.asarray(pressure, dtype=float)
    if not (x_array.shape == y_array.shape == pressure_array.shape):
        raise ValueError("x, y, and pressure must have identical shapes")
    isobars = axes.tricontour(
        x_array.ravel(),
        y_array.ravel(),
        pressure_array.ravel(),
        levels=levels,
        **contour_options,
    )
    points = plot_poincare(trace, axes)
    return isobars, points


def _validate_integration_options(
    *,
    turns: int,
    major_radius: float,
    relative_tolerance: float,
    absolute_tolerance: float,
    maximum_step: float,
) -> None:
    if isinstance(turns, bool) or not isinstance(turns, int):
        raise TypeError("turns must be an integer")
    if turns < 1:
        raise ValueError("turns must be positive")
    for name, value in (
        ("major_radius", major_radius),
        ("relative_tolerance", relative_tolerance),
        ("absolute_tolerance", absolute_tolerance),
        ("maximum_step", maximum_step),
    ):
        if not isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")


def _integrate_field_line(
    field: MagneticField,
    seed: NDArray[np.float64],
    phis: NDArray[np.float64],
    *,
    major_radius: float,
    relative_tolerance: float,
    absolute_tolerance: float,
    maximum_step: float,
) -> NDArray[np.float64]:
    from scipy.integrate import solve_ivp  # type: ignore[import-untyped]

    def rhs(phi: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        radius, theta = (float(state[0]), float(state[1]))
        if radius <= 0.0:
            raise RuntimeError("field-line trace reached the cylindrical axis")
        x = radius * cos(theta)
        y = radius * sin(theta)
        magnetic = np.asarray(field(x, y, major_radius * phi), dtype=float)
        if magnetic.shape != (3,) or not np.all(np.isfinite(magnetic)):
            raise ValueError("magnetic field evaluator must return three finite components")
        if abs(float(magnetic[2])) <= np.finfo(float).eps:
            raise RuntimeError("B_z vanished; Phi cannot parameterize this field line")
        radial_field = magnetic[0] * cos(theta) + magnetic[1] * sin(theta)
        poloidal_field = -magnetic[0] * sin(theta) + magnetic[1] * cos(theta)
        scale = major_radius / magnetic[2]
        return np.array([scale * radial_field, scale * poloidal_field / radius])

    result = solve_ivp(
        rhs,
        (float(phis[0]), float(phis[-1])),
        np.asarray(seed, dtype=float),
        method="DOP853",
        t_eval=phis,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        max_step=maximum_step,
    )
    if not result.success or result.y.shape != (2, phis.size):
        raise RuntimeError(f"field-line integration failed: {result.message}")
    return np.asarray(result.y, dtype=float)
