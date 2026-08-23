"""Frozen-field three-dimensional island benchmark for note equations (M4a)--(M4b)."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from remec.profiles import PressureProfile


@dataclass(frozen=True, slots=True)
class FrozenFieldIslandConfig:
    """Configuration for the Section-8.6 Reiman--Greenside benchmark."""

    epsilon_kappa: float
    epsilon_1: float = 1.0e-3
    polynomial_order: int = 1
    max_element_size: float = 0.45
    geometry_order: int = 4
    refinements: int = 0
    min_layer_cells: int = 6
    b_floor: float = 1.0e-8
    volume_levels: int = 65
    ray_samples: int = 257
    angular_cells: int = 12
    axial_cells: int = 2
    background_radial_width: float = 0.125
    direct_dof_threshold: int = 20_000
    pollution_direct_dof_threshold: int = 100_000
    flattening_gradient_fraction: float = 0.97

    def __post_init__(self) -> None:
        for name, value in (
            ("epsilon_kappa", self.epsilon_kappa),
            ("max_element_size", self.max_element_size),
            ("b_floor", self.b_floor),
            ("background_radial_width", self.background_radial_width),
        ):
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not isfinite(self.flattening_gradient_fraction)
            or not 0.0 < self.flattening_gradient_fraction < 1.0
        ):
            raise ValueError("flattening_gradient_fraction must be finite and in (0, 1)")
        if not isfinite(self.epsilon_1) or self.epsilon_1 < 0.0:
            raise ValueError("epsilon_1 must be finite and non-negative")
        if self.epsilon_1 >= (1.0 - 2.0 * 0.29) / 4.0:
            raise ValueError("epsilon_1 lies outside the exact single-island-width regime")
        for name, value, minimum in (
            ("polynomial_order", self.polynomial_order, 1),
            ("geometry_order", self.geometry_order, 1),
            ("refinements", self.refinements, 0),
            ("min_layer_cells", self.min_layer_cells, 1),
            ("volume_levels", self.volume_levels, 17),
            ("ray_samples", self.ray_samples, 65),
            ("angular_cells", self.angular_cells, 8),
            ("axial_cells", self.axial_cells, 2),
            ("direct_dof_threshold", self.direct_dof_threshold, 1),
            ("pollution_direct_dof_threshold", self.pollution_direct_dof_threshold, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        if self.geometry_order > 4:
            raise ValueError("geometry_order must not exceed the verified order four")
        if self.angular_cells % 2:
            raise ValueError("angular_cells must be even")


@dataclass(frozen=True, slots=True)
class FrozenFieldIslandResult:
    """Scalar diagnostics returned by one frozen (M4a)--(M4b) solve."""

    epsilon_kappa: float
    island_width: float
    critical_width: float
    diagnostics: dict[str, float | int | str | bool]


def exact_island_width(*, epsilon_1: float, t1: float = 0.38) -> float:
    r"""Derive ``w_island=4 sqrt(epsilon_1/(2 t1))`` from conserved ``K`` roots."""
    if not isfinite(epsilon_1) or epsilon_1 < 0.0:
        raise ValueError("epsilon_1 must be finite and non-negative")
    if not isfinite(t1) or t1 <= 0.0:
        raise ValueError("t1 must be finite and positive")
    return 4.0 * sqrt(epsilon_1 / (2.0 * t1))


def critical_layer_width(
    *, epsilon_kappa: float, resonance_radius: float, t1: float = 0.38, major_radius: float = 1.0
) -> float:
    r"""Derive ``w_c=epsilon_kappa^(1/4)sqrt(R0/(m d-iota/dr))`` for ``m=2``."""
    for name, value in (
        ("epsilon_kappa", epsilon_kappa),
        ("resonance_radius", resonance_radius),
        ("t1", t1),
        ("major_radius", major_radius),
    ):
        if not isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    transform_radial_derivative = 2.0 * t1 * resonance_radius
    return float(epsilon_kappa**0.25 * sqrt(major_radius / (2.0 * transform_radial_derivative)))


class FrozenFieldIslandSolver:
    """Solve only note equations (M4a)--(M4b) on the Section-8.6 frozen field."""

    def solve(
        self,
        config: FrozenFieldIslandConfig,
        pressure_profile: PressureProfile | None = None,
    ) -> FrozenFieldIslandResult:
        """Run one live benchmark row, including the mandatory falsifiability controls."""
        if not isinstance(config, FrozenFieldIslandConfig):
            raise TypeError("config must be a FrozenFieldIslandConfig")
        from remec.fem._frozen_field_island import run_frozen_field_island

        return run_frozen_field_island(config, pressure_profile)
