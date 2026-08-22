"""Live milestone-5.5 acceptance rows for the 1.0-MA current family."""

import pytest
from axisymmetric_nonideal_live_helpers import (
    check_acceptance_cold_start,
    check_acceptance_restart,
)


@pytest.mark.slow
def test_current_10_acceptance_row_1_is_reproduced_live() -> None:
    """The first 1.0-MA row converges from the analytic initial field."""
    check_acceptance_cold_start(1, 1.0e6)


def test_current_10_acceptance_row_2_is_reproduced_live() -> None:
    """The accepted checkpoint reruns the complete live row-2 map."""
    check_acceptance_restart(1, 1.0e6, 1)


def test_current_10_acceptance_row_3_is_reproduced_live() -> None:
    """The accepted checkpoint reruns the complete live row-3 map."""
    check_acceptance_restart(1, 1.0e6, 2)
