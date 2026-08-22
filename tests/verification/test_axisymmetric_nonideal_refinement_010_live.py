"""Live `maxh=0.10` ADR-0006 projection row."""

from axisymmetric_nonideal_live_helpers import check_refinement_coordinate_restart


def test_projection_row_010_is_reproduced_live() -> None:
    """A coordinate-keyed checkpoint reruns the complete finest solve portably."""
    check_refinement_coordinate_restart(
        0.10,
        "axisymmetric_nonideal_refinement_010_coordinate_restart.csv",
    )
