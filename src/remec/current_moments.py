"""Mollified shell-current moments for note equations ``(M2)``--``(M3b)``."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from math import pi
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from remec.level_set import MollifiedVolumeMap, compact_moment_matched_heaviside

CurrentComponent = Literal["parallel", "diamagnetic", "regularizing", "total"]


@dataclass(frozen=True, slots=True)
class M2ToroidalCurrentSamples:
    r"""Quadrature samples of every note-``(M2)`` contribution dotted with ``grad(phi)``.

    The fields correspond to
    ``J = u B + B cross grad(p)/B_safe**2 - D_u grad_r(utilde)``.
    ``normalized_volume`` must be the volume map's shared s field in exactly the same
    quadrature-point ordering as all three components; the integrator checks it before
    accumulating moments. Keeping the three terms separate lets the independent ``(M3b)`` diagnostic catch
    an omitted diamagnetic or regularizing contribution rather than only comparing a
    pre-summed current assembled by the solver.
    """

    normalized_volume: ArrayLike
    parallel: ArrayLike
    diamagnetic: ArrayLike
    regularizing: ArrayLike


@dataclass(frozen=True, slots=True)
class ShellCurrentMoments:
    r"""Cumulative ``I_tor(s)`` and local-shell moments for all ``(M2)`` terms.

    Cumulative values implement note ``(Itor)``,
    ``I_tor(s)=(2*pi)**-1 int_{Omega_s} J dot grad(phi) dV``. Shellwise
    values are exact differences of adjacent cumulative rows and therefore implement
    the partition used by note ``(shell_constraints)``.
    """

    normalized_volume: NDArray[np.float64]
    shell_edges: NDArray[np.float64]
    cumulative_parallel: NDArray[np.float64]
    cumulative_diamagnetic: NDArray[np.float64]
    cumulative_regularizing: NDArray[np.float64]
    cumulative_total: NDArray[np.float64]
    shellwise_parallel: NDArray[np.float64]
    shellwise_diamagnetic: NDArray[np.float64]
    shellwise_regularizing: NDArray[np.float64]
    shellwise_total: NDArray[np.float64]

    def cumulative(self, component: CurrentComponent) -> NDArray[np.float64]:
        """Return a copy of one component's cumulative ``I_tor(s)`` rows."""
        values = getattr(self, f"cumulative_{component}")
        return np.array(values, dtype=np.float64, copy=True)

    def shellwise(self, component: CurrentComponent) -> NDArray[np.float64]:
        """Return a copy of one component's local annular ``(M3b)`` rows."""
        values = getattr(self, f"shellwise_{component}")
        return np.array(values, dtype=np.float64, copy=True)


def mollified_shell_current_moments(
    volume_map: MollifiedVolumeMap,
    samples: M2ToroidalCurrentSamples,
    shell_edges: ArrayLike,
) -> ShellCurrentMoments:
    r"""Integrate cumulative and shellwise toroidal current with mollified layer sets.

    The exact ``s=0`` and ``s=1`` rows are imposed after using the compact
    moment-matched Heaviside from ``(mollified_V)`` on interior edges. The mollifier
    widths are the original spatial chi widths mapped through the same shared
    ``s=V_chi/V_omega`` field used by ``(M4b)``. No nodal histogram or independent
    radial label is constructed.
    """
    if not isinstance(volume_map, MollifiedVolumeMap):
        raise TypeError("volume_map must be a MollifiedVolumeMap")
    if not isinstance(samples, M2ToroidalCurrentSamples):
        raise TypeError("samples must be M2ToroidalCurrentSamples")
    edges = np.asarray(shell_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)):
        raise ValueError("shell edges must be a finite one-dimensional partition")
    if edges[0] != 0.0 or edges[-1] != 1.0:
        raise ValueError("shell edges must cover exactly [0, 1]")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("shell edges must be strictly increasing")

    normalized_volume = volume_map.quadrature_normalized_volume
    weights = volume_map.quadrature_weights
    widths = volume_map.quadrature_normalized_mollifier_widths
    sample_coordinate = _validated_component(
        samples.normalized_volume, len(weights), "normalized-volume"
    )
    if not np.array_equal(sample_coordinate, normalized_volume):
        raise ValueError(
            "normalized-volume samples must match the volume map's quadrature ordering"
        )
    cell_widths = volume_map.quadrature_normalized_cell_widths
    mollifier_widths = volume_map.quadrature_normalized_mollifier_widths
    for left, right in pairwise(edges):
        in_shell = (normalized_volume >= left) & (normalized_volume <= right)
        if not np.any(in_shell) or (right - left) / float(np.max(cell_widths[in_shell])) < 3.0:
            raise ValueError(
                "shell partition is unresolved: every shell must span at least three "
                "local radial-cell widths"
            )
        if (right - left) / float(np.max(mollifier_widths[in_shell])) < 2.0:
            raise ValueError(
                "shell partition is unresolved: every shell must span at least two "
                "local mapped mollifier widths"
            )

    component_samples = {
        "parallel": _validated_component(samples.parallel, len(weights), "parallel"),
        "diamagnetic": _validated_component(samples.diamagnetic, len(weights), "diamagnetic"),
        "regularizing": _validated_component(samples.regularizing, len(weights), "regularizing"),
    }
    membership = np.empty((len(edges), len(weights)), dtype=np.float64)
    membership[0] = 0.0
    membership[-1] = 1.0
    for index, edge in enumerate(edges[1:-1], start=1):
        argument = (edge - normalized_volume) / widths
        membership[index] = compact_moment_matched_heaviside(argument)

    factor = 1.0 / (2.0 * pi)
    cumulative_parallel = factor * membership @ (weights * component_samples["parallel"])
    cumulative_diamagnetic = factor * membership @ (weights * component_samples["diamagnetic"])
    cumulative_regularizing = factor * membership @ (weights * component_samples["regularizing"])
    cumulative_total = cumulative_parallel + cumulative_diamagnetic + cumulative_regularizing
    return ShellCurrentMoments(
        normalized_volume=np.array(normalized_volume, copy=True),
        shell_edges=np.array(edges, copy=True),
        cumulative_parallel=cumulative_parallel,
        cumulative_diamagnetic=cumulative_diamagnetic,
        cumulative_regularizing=cumulative_regularizing,
        cumulative_total=cumulative_total,
        shellwise_parallel=np.diff(cumulative_parallel),
        shellwise_diamagnetic=np.diff(cumulative_diamagnetic),
        shellwise_regularizing=np.diff(cumulative_regularizing),
        shellwise_total=np.diff(cumulative_total),
    )


def _validated_component(value: ArrayLike, size: int, name: str) -> NDArray[np.float64]:
    samples = np.asarray(value, dtype=float).reshape(-1)
    if len(samples) != size or not np.all(np.isfinite(samples)):
        raise ValueError(f"{name} current samples must be finite and match the volume map")
    return samples
