"""Public option records shared by future solver entry points."""

from dataclasses import dataclass
from typing import Literal

RegularizationGradient = Literal["perpendicular", "full"]
CurrentContinuityStabilization = Literal["none", "supg"]


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    """Execution and runtime-selected physics defaults from DESIGN.md §5, §9.4, and §21.

    Solver implementations will validate and apply ``threads`` before entering an
    NGSolve ``TaskManager``.  ``min_layer_cells`` implements the documented default
    resolution threshold for the w_c and δ diagnostics. ``regularization_gradient``
    records the note-(M3) current-viscosity variant in configuration and checkpoint
    metadata; the derived perpendicular closure remains the default.
    """

    threads: int = 1
    min_layer_cells: int = 6
    regularization_gradient: RegularizationGradient = "perpendicular"

    def __post_init__(self) -> None:
        if self.threads < 1:
            raise ValueError("threads must be at least one")
        if self.min_layer_cells < 1:
            raise ValueError("min_layer_cells must be at least one")
        if self.regularization_gradient not in ("perpendicular", "full"):
            raise ValueError("regularization_gradient must be 'perpendicular' or 'full'")
