"""Internal NGSolve-field adapters for diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any

import numpy as np
from numpy.typing import NDArray


def make_hdiv_field_evaluator(
    mesh: Any,
    field: Any,
    *,
    periodic_z_length: float | None = None,
) -> Callable[[float, float, float], NDArray[np.float64]]:
    """Adapt an H(div) magnetic field satisfying (M1) to a point evaluator.

    NGSolve remains behind this internal adapter; the public Poincare tracer consumes
    the returned ordinary Python callable. ``periodic_z_length`` maps repeated section
    planes back into the finite-element mesh's fundamental period.
    """
    try:
        if mesh.dim != 3 or field.space.mesh is not mesh:
            raise ValueError("field must belong to the supplied three-dimensional mesh")
    except AttributeError as error:
        raise TypeError("mesh and field must be NGSolve objects") from error
    if periodic_z_length is not None and (
        not isfinite(periodic_z_length) or periodic_z_length <= 0.0
    ):
        raise ValueError("periodic_z_length must be positive and finite")

    def evaluate(x: float, y: float, z: float) -> NDArray[np.float64]:
        evaluation_z = z
        if periodic_z_length is not None:
            evaluation_z = z % periodic_z_length
        try:
            value = np.asarray(field(mesh(x, y, evaluation_z)), dtype=float)
        except Exception as error:
            raise ValueError("field-line point lies outside the H(div) mesh") from error
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError("H(div) evaluator must produce three finite components")
        return value

    return evaluate
