"""Backend-independent DESIGN §13.3 Anderson history contracts."""

from __future__ import annotations

import numpy as np
import pytest

from remec.solvers._anderson import AndersonAccelerator


def _accelerator(*, depth: int = 2, condition_limit: float = 1.0e5) -> AndersonAccelerator:
    return AndersonAccelerator(
        depth=depth,
        damping=0.3,
        regularization=1.0e-12,
        condition_limit=condition_limit,
    )


@pytest.mark.parametrize("scale", [1.0e-6, 1.0, 1.0e6])
def test_anderson_is_scale_equivariant_on_a_coupled_linear_fixed_point(scale: float) -> None:
    """Relative convergence and update decisions do not depend on the state's units."""
    accelerator = _accelerator(depth=2)
    matrix = np.asarray(((0.6, 0.2), (-0.1, -0.5)), dtype=float)
    source = scale * np.asarray((0.4, -0.2), dtype=float)
    exact = np.linalg.solve(np.eye(2) - matrix, source)
    state = scale * np.asarray((1.0, 1.0), dtype=float)
    methods: list[str] = []

    for _ in range(8):
        update = accelerator.update(state, source + matrix @ state)
        methods.append(update.method)
        state = update.state

    assert np.linalg.norm(state - exact) / np.linalg.norm(exact) < 1.0e-9
    assert methods == [
        "damped",
        "anderson",
        "anderson",
        "anderson",
        "damped_fallback",
        "damped_fallback",
        "damped_fallback",
        "damped_fallback",
    ]
    assert accelerator.history_size <= 3


def test_depth_five_history_reaches_full_rank_and_rolls_over() -> None:
    """A six-component map exercises five independent secant columns before rollover."""
    accelerator = _accelerator(depth=5)
    matrix = np.diag(np.asarray((0.72, 0.55, 0.31, 0.08, -0.23, -0.47)))
    source = np.asarray((0.4, -0.3, 0.2, -0.1, 0.15, -0.25))
    state = np.asarray((0.1, 0.2, -0.3, 0.4, -0.5, 0.6))
    updates = []

    for _ in range(7):
        update = accelerator.update(state, source + matrix @ state)
        updates.append(update)
        state = update.state

    assert updates[5].history_size == 6
    assert updates[5].method == "anderson"
    assert updates[5].condition_number is not None
    assert updates[5].condition_number < accelerator.condition_limit
    assert updates[6].history_size == 6
    assert accelerator.history_size == 6


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
    assert update.rejection_reason == "rank_deficient_history"
    assert update.condition_number is None
    assert update.history_size == 3
    assert accelerator.history_size == 1
    assert update.state == pytest.approx(current + 0.3 * (image - current))


def test_condition_limit_restarts_before_the_regularized_filter_loses_accuracy() -> None:
    """A full-rank history beyond the documented effective cutoff is rejected."""
    accelerator = _accelerator(depth=2, condition_limit=1.0e5)
    accelerator.update(np.asarray((0.0, 0.0)), np.asarray((0.0, 0.0)))
    accelerator.update(np.asarray((1.0, 0.0)), np.asarray((2.0, 0.0)))
    current = np.asarray((2.0, 1.0))
    image = current + np.asarray((1.0, 1.0e-6))

    update = accelerator.update(current, image)

    assert update.method == "damped_fallback"
    assert update.rejection_reason == "ill_conditioned_history"
    assert update.condition_number == pytest.approx(1.0e6)


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
