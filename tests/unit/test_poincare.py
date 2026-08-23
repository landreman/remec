"""Unit contracts for versioned and plottable Poincare diagnostics."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from remec.diagnostics.poincare import (
    PoincareTrace,
    PoincareTraceVersionError,
    find_periodic_point,
    load_poincare_trace,
    plot_isobar_overlay,
    plot_poincare,
    save_poincare_trace,
)


def _cylindrical_field(
    x: float,
    y: float,
    *,
    radial: float,
    poloidal_rate: float,
) -> tuple[float, float, float]:
    radius = float(np.hypot(x, y))
    cosine = x / radius
    sine = y / radius
    poloidal = poloidal_rate * radius
    return (
        radial * cosine - poloidal * sine,
        radial * sine + poloidal * cosine,
        1.0,
    )


def _trace() -> PoincareTrace:
    return PoincareTrace(
        section_phis=np.array([0.0, 2.0 * np.pi, 4.0 * np.pi]),
        radii=np.array([[0.2, 0.2, 0.2], [0.6, 0.61, 0.59]]),
        theta_unwrapped=np.array([[0.1, 2.1, 4.1], [0.3, 3.0, 5.7]]),
        major_radius=1.0,
        relative_tolerance=1.0e-10,
        absolute_tolerance=1.0e-12,
    )


class _Axes:
    def __init__(self) -> None:
        self.scatter_call: tuple[tuple[Any, ...], dict[str, Any]] | None = None
        self.contour_call: tuple[tuple[Any, ...], dict[str, Any]] | None = None

    def scatter(self, *args: Any, **kwargs: Any) -> str:
        self.scatter_call = (args, kwargs)
        return "points"

    def tricontour(self, *args: Any, **kwargs: Any) -> str:
        self.contour_call = (args, kwargs)
        return "isobars"

    def set_aspect(self, aspect: str) -> None:
        assert aspect == "equal"


def test_poincare_trace_round_trip_is_explicit_and_versioned(tmp_path: Path) -> None:
    path = tmp_path / "trace"
    save_poincare_trace(_trace(), path)
    restored = load_poincare_trace(path)

    assert restored.major_radius == 1.0
    assert restored.relative_tolerance == 1.0e-10
    assert restored.absolute_tolerance == 1.0e-12
    np.testing.assert_array_equal(restored.section_phis, _trace().section_phis)
    np.testing.assert_array_equal(restored.radii, _trace().radii)
    np.testing.assert_array_equal(restored.theta_unwrapped, _trace().theta_unwrapped)


def test_poincare_trace_rejects_an_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "future.npz"
    np.savez(
        path,
        schema_version=np.array(99),
        section_phis=np.array([0.0, 2.0 * np.pi]),
        radii=np.array([[0.2, 0.2]]),
        theta_unwrapped=np.array([[0.0, 1.0]]),
        major_radius=np.array(1.0),
        relative_tolerance=np.array(1.0e-10),
        absolute_tolerance=np.array(1.0e-12),
    )
    with pytest.raises(PoincareTraceVersionError, match="schema"):
        load_poincare_trace(path)


def test_plotting_entry_points_use_caller_supplied_axes() -> None:
    trace = _trace()
    axes = _Axes()

    points = plot_poincare(trace, axes, s=3.0, color="black")
    assert axes.scatter_call is not None
    assert axes.scatter_call[1] == {"s": 3.0, "color": "black"}

    overlay_axes = _Axes()
    isobars, overlay = plot_isobar_overlay(
        trace,
        overlay_axes,
        np.array([-1.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 1.0]),
        levels=4,
        colors="blue",
    )

    assert points == overlay == "points"
    assert isobars == "isobars"
    assert overlay_axes.scatter_call is not None
    assert overlay_axes.contour_call is not None
    assert overlay_axes.contour_call[1] == {"levels": 4, "colors": "blue"}


def test_periodic_point_rejects_non_area_preserving_return_map() -> None:
    def field(x: float, y: float, _z: float) -> tuple[float, float, float]:
        radius = float(np.hypot(x, y))
        return _cylindrical_field(
            x,
            y,
            radial=0.02 * (radius - 0.4),
            poloidal_rate=0.5,
        )

    with pytest.raises(RuntimeError, match=r"det\(M\)"):
        find_periodic_point(
            field,
            (0.4, 0.0),
            period_turns=2,
            poloidal_winding=1,
        )


def test_periodic_point_rejects_nonconverged_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def field(x: float, y: float, _z: float) -> tuple[float, float, float]:
        return _cylindrical_field(x, y, radial=0.01, poloidal_rate=0.5)

    monkeypatch.setattr(
        "scipy.optimize.root",
        lambda *_args, **_kwargs: SimpleNamespace(
            x=np.array([0.4, 0.0]),
            message="stagnated",
        ),
    )

    with pytest.raises(RuntimeError, match="periodic-point residual"):
        find_periodic_point(
            field,
            (0.4, 0.0),
            period_turns=2,
            poloidal_winding=1,
        )
