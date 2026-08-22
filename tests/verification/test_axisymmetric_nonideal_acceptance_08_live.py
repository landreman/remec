"""Live milestone-5.5 acceptance rows for the 0.8-MA current family."""

import pytest
from axisymmetric_nonideal_live_helpers import (
    check_acceptance_cold_start,
    check_acceptance_restart,
)


@pytest.mark.slow
def test_current_08_acceptance_row_1_is_reproduced_live() -> None:
    """The first 0.8-MA row converges from the analytic initial field."""
    check_acceptance_cold_start(0, 0.8e6)


def test_current_08_acceptance_row_2_is_reproduced_live() -> None:
    """The accepted checkpoint reruns the complete live row-2 map."""
    check_acceptance_restart(0, 0.8e6, 1)


def test_current_08_acceptance_row_3_is_reproduced_live() -> None:
    """The accepted checkpoint reruns the complete live row-3 map."""
    check_acceptance_restart(0, 0.8e6, 2)
