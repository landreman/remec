"""Live rows 1--3 of the fixed-pressure ADR-0006 ladder."""

import pytest
from axisymmetric_nonideal_live_helpers import check_fixed_pressure_segment


@pytest.mark.slow
def test_fixed_pressure_ladder_rows_1_through_3_are_reproduced_live() -> None:
    """The coarse half of the ladder matches its checked-in rows."""
    check_fixed_pressure_segment(0, 3)
