"""Independent regularization and compatible-current refinement legs for milestone 5.5."""

from __future__ import annotations

import csv
from itertools import pairwise
from math import log, sqrt
from pathlib import Path

import pytest

from remec.solvers.axisymmetric_nonideal import run_zheng_nonideal_continuation
from remec.solvers.continuation import ContinuationStage

_TABLE = Path(__file__).with_name("axisymmetric_nonideal_refinement.csv")
_FIXED_PRESSURE_STAGES = (
    ContinuationStage(1.0, 0.060, 0.120),
    ContinuationStage(1.0, 0.030, 0.060),
    ContinuationStage(1.0, 0.015, 0.030),
)


def _records(study: str) -> list[dict[str, float]]:
    with _TABLE.open(newline="") as table_file:
        return [
            {key: float(value) for key, value in row.items() if key != "study"}
            for row in csv.DictReader(table_file)
            if row["study"] == study
        ]


@pytest.mark.slow
def test_regularization_bias_decreases_with_pressure_held_fixed() -> None:
    """The non-ideal trend is attributable to D_u and epsilon_kappa, not a pressure ramp."""
    rows = run_zheng_nonideal_continuation(
        plasma_current=0.8e6,
        stages=_FIXED_PRESSURE_STAGES,
        maxh=0.32,
        polynomial_order=2,
        require_decreasing_projection_correction=False,
    ).stages
    expected = _records("fixed_pressure_regularization")

    assert all(
        fine.nonideal_to_analytic_relative_l2_error < coarse.nonideal_to_analytic_relative_l2_error
        for coarse, fine in pairwise(rows)
    )
    assert all(row.pressure_profile_error < 1.0e-10 for row in rows)
    for row, record in zip(rows, expected, strict=True):
        assert row.nonideal_to_analytic_relative_l2_error == pytest.approx(
            record["nonideal_to_analytic_relative_l2_error"], rel=0.08
        )
        assert row.projection_correction_relative_norm == pytest.approx(
            record["projection_correction_relative_norm"], rel=0.08
        )


@pytest.mark.slow
@pytest.mark.parametrize("maxh", [0.24, 0.14, 0.10])
def test_compatible_current_correction_matches_refinement_record(maxh: float) -> None:
    """Fresh single-stage solves pin the compatible-current h-refinement table."""
    row = run_zheng_nonideal_continuation(
        plasma_current=0.8e6,
        stages=(_FIXED_PRESSURE_STAGES[-1],),
        maxh=maxh,
        polynomial_order=2,
    ).stages[0]
    expected = next(
        record for record in _records("projection_refinement") if record["maxh"] == maxh
    )

    assert row.projection_correction_relative_norm == pytest.approx(
        expected["projection_correction_relative_norm"], rel=0.08
    )
    assert row.current_profile_error < 1.0e-10
    assert row.projected_current_profile_error < 1.0e-10


def test_compatible_current_correction_has_positive_fine_mesh_rates() -> None:
    """The monitored §27.4 correction converges despite one pre-asymptotic mesh reversal."""
    rows = _records("projection_refinement")
    rates = []
    for coarse, fine in pairwise(rows):
        coarse_h = 1.0 / sqrt(coarse["elements"])
        fine_h = 1.0 / sqrt(fine["elements"])
        rates.append(
            log(
                coarse["projection_correction_relative_norm"]
                / fine["projection_correction_relative_norm"]
            )
            / log(coarse_h / fine_h)
        )

    assert rates[-2] > 1.5
    assert rates[-1] > 1.5
    assert rows[-1]["projection_correction_relative_norm"] < 0.10
