"""Public solver-field adapters for diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray


def make_hdiv_field_evaluator(
    mesh: Any,
    field: Any,
    *,
    periodic_z_length: float | None = None,
) -> Callable[[float, float, float], NDArray[np.float64]]:
    """Return an ordinary callable for a solver's note-(M1) H(div) field.

    This is the public diagnostics wrapper; NGSolve validation and evaluation remain
    confined to the internal FEM adapter. ``periodic_z_length`` wraps repeated
    Poincare sections into the solver mesh's fundamental period.
    """
    from remec.fem._field_evaluation import make_hdiv_field_evaluator as _internal_adapter

    return _internal_adapter(mesh, field, periodic_z_length=periodic_z_length)
