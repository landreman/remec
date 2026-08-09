"""Public option records shared by future solver entry points."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    """Execution defaults from DESIGN.md §5 and §21.

    Solver implementations will validate and apply ``threads`` before entering an
    NGSolve ``TaskManager``.  ``min_layer_cells`` implements the documented default
    resolution threshold for the w_c and δ diagnostics.
    """

    threads: int = 1
    min_layer_cells: int = 6

    def __post_init__(self) -> None:
        if self.threads < 1:
            raise ValueError("threads must be at least one")
        if self.min_layer_cells < 1:
            raise ValueError("min_layer_cells must be at least one")
