"""Compatible-current h-refinement leg for milestone 5.5 and ADR 0006."""

from __future__ import annotations

import csv
from itertools import pairwise
from math import sqrt
from pathlib import Path

import numpy as np

_TABLE = Path(__file__).with_name("axisymmetric_nonideal_refinement.csv")


def _records(study: str) -> list[dict[str, float]]:
    with _TABLE.open(newline="") as table_file:
        return [
            {key: float(value) for key, value in row.items() if key != "study"}
            for row in csv.DictReader(table_file)
            if row["study"] == study
        ]


def test_compatible_current_correction_meets_adr_0006_escalation_gate() -> None:
    """The three finest meshes meet ADR 0006's effective-h least-squares gate."""
    rows = _records("projection_refinement")
    fine = rows[-3:]
    log_h = [np.log(1.0 / sqrt(row["elements"])) for row in fine]
    log_correction = [np.log(row["projection_correction_relative_norm"]) for row in fine]
    least_squares_rate = float(np.polyfit(log_h, log_correction, 1)[0])

    assert least_squares_rate >= 1.0
    assert rows[-1]["projection_correction_relative_norm"] < 0.10
    assert all(row["toroidal_flux_relative_error"] < 1.0e-10 for row in rows)


def test_fixed_pressure_record_meets_adr_0006_field_error_gate() -> None:
    """The six-stage measured ladder clears the field-floor and monotonicity criteria."""
    rows = _records("fixed_pressure_regularization")
    errors = [row["nonideal_to_analytic_relative_l2_error"] for row in rows]

    assert len(rows) >= 5
    assert rows[-1]["current_diffusivity"] <= 0.00375
    assert rows[-1]["perpendicular_ratio"] <= 0.0075
    assert errors[-1] <= 0.80 * errors[0]
    assert all(fine <= 1.02 * min(errors[:index]) for index, fine in enumerate(errors[1:], 1))
    assert all(fine < coarse for coarse, fine in pairwise(errors))
    assert all(row["toroidal_flux_relative_error"] < 1.0e-10 for row in rows)
