"""Backend-independent DESIGN §13.3 Anderson history contracts."""

from __future__ import annotations

import numpy as np
import pytest

from remec.solvers._anderson import AndersonAccelerator


def _accelerator(*, depth: int = 2) -> AndersonAccelerator:
    return AndersonAccelerator(
        depth=depth,
        damping=0.3,
        regularization=1.0e-12,
        condition_limit=1.0e10,
    )


def test_anderson_accelerates_a_coupled_linear_fixed_point_with_bounded_history() -> None:
    """The flattened-vector kernel solves a coupled affine map without backend state."""
    accelerator = _accelerator(depth=2)
    matrix = np.asarray(((0.6, 0.2), (-0.1, -0.5)), dtype=float)
    source = np.asarray((0.4, -0.2), dtype=float)
    exact = np.linalg.solve(np.eye(2) - matrix, source)
    state = np.asarray((1.0, 1.0), dtype=float)
    methods: list[str] = []

    for _ in range(8):
        update = accelerator.update(state, source + matrix @ state)
        methods.append(update.method)
        state = update.state

    assert np.linalg.norm(state - exact) < 1.0e-9
    assert "anderson" in methods
    assert accelerator.history_size <= 3


def test_rank_deficient_history_restarts_and_returns_exact_damped_fallback() -> None:
    """Dependent residual differences cannot enter the regularized least-squares solve."""
    accelerator = _accelerator(depth=5)
    accelerator.update(np.asarray((0.0, 0.0)), np.asarray((1.0, 0.0)))
    accelerator.update(np.asarray((1.0, 1.0)), np.asarray((3.0, 1.0)))
    current = np.asarray((2.0, 2.0))
    image = np.asarray((5.0, 2.0))

    update = accelerator.update(current, image)

    assert update.method == "damped_fallback"
    assert update.restarted
    assert update.rejection_reason == "ill_conditioned_history"
    assert update.condition_number is None
    assert update.history_size == 3
    assert accelerator.history_size == 1
    assert update.state == pytest.approx(current + 0.3 * (image - current))


def test_anderson_copies_free_vectors_and_never_mutates_adapter_owned_inputs() -> None:
    """Fixed harmonic coefficients/traces can remain outside the free-vector history."""
    accelerator = _accelerator(depth=1)
    free_state = np.asarray((0.2, -0.4), dtype=float)
    image = np.asarray((0.5, 0.1), dtype=float)
    original_state = free_state.copy()
    original_image = image.copy()

    update = accelerator.update(free_state, image)
    free_state[:] = 99.0
    image[:] = -99.0

    assert update.state == pytest.approx(original_state + 0.3 * (original_image - original_state))
    assert accelerator.history_size == 1
