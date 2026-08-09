"""Norms for separately monitored solver blocks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import fsum, sqrt


@dataclass(frozen=True, slots=True)
class BlockNorms:
    """Euclidean norms for named blocks and their joint Euclidean norm."""

    blocks: dict[str, float]
    combined: float


def block_l2_norms(blocks: Mapping[str, Iterable[float]]) -> BlockNorms:
    """Compute ``||x_i||₂`` and ``sqrt(sum_i ||x_i||₂²)`` for named blocks."""
    individual = {
        name: sqrt(fsum(float(value) ** 2 for value in entries)) for name, entries in blocks.items()
    }
    return BlockNorms(
        blocks=individual, combined=sqrt(fsum(value**2 for value in individual.values()))
    )
