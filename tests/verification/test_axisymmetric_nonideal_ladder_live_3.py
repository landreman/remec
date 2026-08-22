"""Live row 5 of the fixed-pressure ADR-0006 ladder."""

import pytest
from axisymmetric_nonideal_live_helpers import check_fixed_pressure_restart


@pytest.mark.slow
def test_fixed_pressure_ladder_row_5_is_reproduced_live() -> None:
    """The row-4 checkpoint warm-starts a complete live row-5 solve."""
    check_fixed_pressure_restart(4, 1)
