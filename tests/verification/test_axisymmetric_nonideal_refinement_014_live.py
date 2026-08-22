"""Live `maxh=0.14` ADR-0006 projection row."""

from axisymmetric_nonideal_live_helpers import check_refinement_restart


def test_projection_row_014_is_reproduced_live() -> None:
    """The accepted checkpoint reruns the complete second-finest solve."""
    check_refinement_restart(0.14, "axisymmetric_nonideal_refinement_014_restart_state.csv")
