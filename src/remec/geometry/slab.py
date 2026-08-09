"""Two-dimensional slab geometry."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class Slab2D:
    """A rectangular two-dimensional domain with a target element size."""

    maxh: float

    def __post_init__(self) -> None:
        if not isfinite(self.maxh) or self.maxh <= 0.0:
            raise ValueError("maxh must be finite and positive")

    @classmethod
    def unit_square(cls, *, maxh: float) -> Slab2D:
        """Return the unit-square slab used by early kernel verification."""
        return cls(maxh=maxh)
