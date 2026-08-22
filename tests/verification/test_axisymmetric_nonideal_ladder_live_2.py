"""Live rows 3--4 of the fixed-pressure ADR-0006 ladder."""

import pytest
from axisymmetric_nonideal_live_helpers import check_fixed_pressure_segment


@pytest.mark.slow
def test_fixed_pressure_ladder_rows_3_and_4_are_reproduced_live() -> None:
    """The first overlap reaches the fourth checked-in row."""
    check_fixed_pressure_segment(2, 4)
