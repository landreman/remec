"""Norms for separately monitored solver blocks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import hypot, isfinite


@dataclass(frozen=True, slots=True)
class BlockNorms:
    """Raw and physically scaled coefficient-vector norms for named solver blocks."""

    blocks: dict[str, float]
    scaled_blocks: dict[str, float]
    combined: float


def block_l2_norms(
    blocks: Mapping[str, Iterable[float]], *, scales: Mapping[str, float]
) -> BlockNorms:
    """Compute raw and scale-normalized block norms required by DESIGN.md §13.2.

    ``combined`` is a diagnostic summary of the scaled blocks only.  It MUST NOT be
    used alone as a convergence criterion; §13.2 requires each physical residual and
    invariant check independently.
    """
    if set(blocks) != set(scales):
        raise ValueError("scales must provide exactly one entry for every block")
    normalized_scales = {name: float(scale) for name, scale in scales.items()}
    if any(not isfinite(scale) or scale <= 0.0 for scale in normalized_scales.values()):
        raise ValueError("scales must be finite and positive")

    individual: dict[str, float] = {}
    for name, entries in blocks.items():
        values = [float(value) for value in entries]
        if not all(isfinite(value) for value in values):
            raise ValueError(f"block {name!r} contains a non-finite value")
        individual[name] = hypot(*values)
    scaled = {name: value / normalized_scales[name] for name, value in individual.items()}
    return BlockNorms(
        blocks=individual,
        scaled_blocks=scaled,
        combined=hypot(*scaled.values()),
    )
