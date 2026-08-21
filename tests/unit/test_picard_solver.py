"""Public contracts for the damped note-(M1)--(M4b) Picard driver."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("damping", 0.0, "damping"),
        ("damping", 1.01, "damping"),
        ("max_iterations", 0, "max_iterations"),
        ("residual_tolerance", 0.0, "residual_tolerance"),
        ("state_update_tolerance", float("nan"), "state_update_tolerance"),
        ("pressure_profile_tolerance", -1.0, "pressure_profile_tolerance"),
        ("current_profile_tolerance", float("inf"), "current_profile_tolerance"),
        ("invariant_tolerance", 0.0, "invariant_tolerance"),
        ("magnetic_scale", -1.0, "magnetic_scale"),
        ("floor_sensitivity_tolerance", 0.0, "floor_sensitivity_tolerance"),
        ("minimum_layer_cells", 0.0, "minimum_layer_cells"),
        ("anderson_depth", -1, "anderson_depth"),
        ("anderson_depth", True, "anderson_depth"),
        ("anderson_depth", 1.5, "anderson_depth"),
        ("anderson_regularization", 0.0, "anderson_regularization"),
        ("anderson_condition_limit", 1.0, "anderson_condition_limit"),
    ],
)
def test_picard_options_reject_invalid_or_unimplemented_controls(
    keyword: str,
    value: float,
    message: str,
) -> None:
    """Invalid damping, tolerance, and Anderson controls fail at configuration time."""
    from remec.solvers.picard import PicardOptions

    arguments = {"magnetic_scale": 1.0, keyword: value}
    with pytest.raises(ValueError, match=message):
        PicardOptions(**arguments)


def test_picard_options_are_public_and_deterministically_serializable() -> None:
    """The nonlinear controls can enter configuration digests and public problem APIs."""
    from remec import PicardOptions
    from remec.common.serialization import canonical_json
    from remec.solvers import DampedPicardSolver

    options = PicardOptions(
        magnetic_scale=2.0,
        damping=0.25,
        max_iterations=17,
        anderson_depth=5,
    )

    assert DampedPicardSolver.__name__ == "DampedPicardSolver"
    assert '"damping":0.25' in canonical_json(options)
    assert '"max_iterations":17' in canonical_json(options)
    assert '"magnetic_scale":2.0' in canonical_json(options)
    assert '"anderson_depth":5' in canonical_json(options)

    with pytest.raises(TypeError, match="magnetic_scale"):
        PicardOptions()  # type: ignore[call-arg]
