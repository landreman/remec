"""Backend-independent Anderson acceleration for flattened free-state vectors."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AndersonUpdate:
    """One DESIGN §13.3 accepted update or damped fallback decision."""

    state: FloatArray
    method: str
    history_size: int
    condition_number: float | None
    restarted: bool
    rejection_reason: str | None


class AndersonAccelerator:
    r"""Accelerate flattened free magnetic coefficients after the ``(M1)`` solve.

    For fixed-point image ``g_k`` and residual ``f_k=g_k-x_k``, the type-II update is

    ``x[k+1] = x[k] + beta f[k] - (Delta X + beta Delta F) gamma``,

    where ``gamma`` minimizes ``||f[k]-Delta F gamma||² + lambda max(1,
    ||Delta F||_F²) ||gamma||²``.  Only the
    backend adapter's free vector enters this history; fixed harmonic-flux coefficients
    and essential traces remain outside it.  Rank-deficient or overly conditioned
    histories are cleared and return the scalar damped Picard update.
    """

    def __init__(
        self,
        *,
        depth: int,
        damping: float,
        regularization: float,
        condition_limit: float,
    ) -> None:
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("depth must be a positive integer")
        if not isfinite(damping) or not 0.0 < damping <= 1.0:
            raise ValueError("damping must be finite and lie in (0, 1]")
        if not isfinite(regularization) or regularization <= 0.0:
            raise ValueError("regularization must be finite and positive")
        if not isfinite(condition_limit) or condition_limit <= 1.0:
            raise ValueError("condition_limit must be finite and exceed one")
        self.depth = depth
        self.damping = damping
        self.regularization = regularization
        self.condition_limit = condition_limit
        self._states: list[FloatArray] = []
        self._residuals: list[FloatArray] = []

    @property
    def history_size(self) -> int:
        """Return the number of retained fixed-point evaluations."""
        return len(self._states)

    def _restart_with(self, state: FloatArray, residual: FloatArray) -> None:
        """Clear an unusable least-squares history but retain the current pair."""
        self._states = [state.copy()]
        self._residuals = [residual.copy()]

    def _fallback(
        self,
        state: FloatArray,
        residual: FloatArray,
        *,
        history_size: int,
        condition_number: float | None,
        reason: str,
    ) -> AndersonUpdate:
        """Restart and return ``x + beta (g-x)`` after a rejected attempt."""
        self._restart_with(state, residual)
        return AndersonUpdate(
            state=state + self.damping * residual,
            method="damped_fallback",
            history_size=history_size,
            condition_number=condition_number,
            restarted=True,
            rejection_reason=reason,
        )

    def update(self, state: FloatArray, fixed_point_image: FloatArray) -> AndersonUpdate:
        """Append one map evaluation and return an accelerated or damped free state."""
        state_array = np.asarray(state, dtype=float)
        image_array = np.asarray(fixed_point_image, dtype=float)
        if state_array.size == 0 or state_array.shape != image_array.shape:
            raise ValueError("state and fixed_point_image must have the same non-empty shape")
        if not np.all(np.isfinite(state_array)) or not np.all(np.isfinite(image_array)):
            raise ValueError("state and fixed_point_image must be finite")
        original_shape = state_array.shape
        current = np.array(state_array.reshape(-1), copy=True)
        residual = np.array((image_array - state_array).reshape(-1), copy=True)
        self._states.append(current)
        self._residuals.append(residual)
        maximum_pairs = self.depth + 1
        if len(self._states) > maximum_pairs:
            self._states = self._states[-maximum_pairs:]
            self._residuals = self._residuals[-maximum_pairs:]

        history_size = len(self._states)
        if history_size == 1:
            return AndersonUpdate(
                state=(current + self.damping * residual).reshape(original_shape),
                method="damped",
                history_size=history_size,
                condition_number=None,
                restarted=False,
                rejection_reason=None,
            )

        delta_states = np.column_stack(
            [right - left for left, right in zip(self._states[:-1], self._states[1:])]
        )
        delta_residuals = np.column_stack(
            [right - left for left, right in zip(self._residuals[:-1], self._residuals[1:])]
        )
        singular_values = np.linalg.svd(delta_residuals, compute_uv=False)
        largest = float(singular_values[0]) if singular_values.size else 0.0
        rank_tolerance = (
            np.finfo(float).eps * max(delta_residuals.shape) * largest if largest > 0.0 else 0.0
        )
        rank = int(np.count_nonzero(singular_values > rank_tolerance))
        if rank < delta_residuals.shape[1]:
            fallback = self._fallback(
                current,
                residual,
                history_size=history_size,
                condition_number=None,
                reason="ill_conditioned_history",
            )
            return AndersonUpdate(
                state=fallback.state.reshape(original_shape),
                method=fallback.method,
                history_size=fallback.history_size,
                condition_number=fallback.condition_number,
                restarted=fallback.restarted,
                rejection_reason=fallback.rejection_reason,
            )
        condition_number = largest / float(singular_values[-1])
        if not isfinite(condition_number) or condition_number > self.condition_limit:
            fallback = self._fallback(
                current,
                residual,
                history_size=history_size,
                condition_number=condition_number,
                reason="ill_conditioned_history",
            )
            return AndersonUpdate(
                state=fallback.state.reshape(original_shape),
                method=fallback.method,
                history_size=fallback.history_size,
                condition_number=fallback.condition_number,
                restarted=fallback.restarted,
                rejection_reason=fallback.rejection_reason,
            )

        gram = delta_residuals.T @ delta_residuals
        regularization_scale = max(1.0, float(np.sum(delta_residuals * delta_residuals)))
        regularized_gram = gram + (
            self.regularization * regularization_scale * np.eye(gram.shape[0])
        )
        try:
            coefficients = np.linalg.solve(
                regularized_gram,
                delta_residuals.T @ residual,
            )
        except np.linalg.LinAlgError:
            fallback = self._fallback(
                current,
                residual,
                history_size=history_size,
                condition_number=condition_number,
                reason="least_squares_failure",
            )
            return AndersonUpdate(
                state=fallback.state.reshape(original_shape),
                method=fallback.method,
                history_size=fallback.history_size,
                condition_number=fallback.condition_number,
                restarted=fallback.restarted,
                rejection_reason=fallback.rejection_reason,
            )
        accelerated = (
            current
            + self.damping * residual
            - (delta_states + self.damping * delta_residuals) @ coefficients
        )
        if not np.all(np.isfinite(accelerated)):
            fallback = self._fallback(
                current,
                residual,
                history_size=history_size,
                condition_number=condition_number,
                reason="nonfinite_accelerated_state",
            )
            return AndersonUpdate(
                state=fallback.state.reshape(original_shape),
                method=fallback.method,
                history_size=fallback.history_size,
                condition_number=fallback.condition_number,
                restarted=fallback.restarted,
                rejection_reason=fallback.rejection_reason,
            )
        return AndersonUpdate(
            state=np.asarray(accelerated.reshape(original_shape), dtype=float),
            method="anderson",
            history_size=history_size,
            condition_number=condition_number,
            restarted=False,
            rejection_reason=None,
        )
