"""Live row 4 of the fixed-pressure ADR-0006 ladder."""

import pytest
from axisymmetric_nonideal_live_helpers import check_fixed_pressure_restart


@pytest.mark.slow
def test_fixed_pressure_ladder_row_4_is_reproduced_live() -> None:
    """The row-3 checkpoint warm-starts a complete live row-4 solve."""
    check_fixed_pressure_restart(3, 1)
