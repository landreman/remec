"""Live row 6 of the fixed-pressure ADR-0006 ladder."""

import pytest
from axisymmetric_nonideal_live_helpers import check_fixed_pressure_restart


@pytest.mark.slow
def test_fixed_pressure_ladder_row_6_is_reproduced_live() -> None:
    """The accepted endpoint checkpoint reruns the complete live endpoint map."""
    check_fixed_pressure_restart(5, 4)
