"""Unit contracts for the Section-8.6 frozen-field island benchmark."""

from __future__ import annotations

import pytest

from remec.solvers.frozen_field_island import critical_layer_width, exact_island_width


def test_reiman_greenside_widths_are_derived_from_the_hamiltonian_balance() -> None:
    """The exact island width and Fitzpatrick width reproduce §8.6 reference numbers."""
    resonance_radius = ((0.5 - 0.29) / 0.38) ** 0.5

    assert exact_island_width(epsilon_1=1.0e-3) == pytest.approx(0.14509525, rel=5.0e-8)
    expected = {1.0e-2: 0.29749, 1.0e-3: 0.16730, 1.0e-4: 0.094074, 1.0e-6: 0.029749}
    for epsilon_kappa, width in expected.items():
        assert critical_layer_width(
            epsilon_kappa=epsilon_kappa,
            resonance_radius=resonance_radius,
        ) == pytest.approx(width, rel=2.0e-4)


def test_width_formulas_reject_nonphysical_inputs() -> None:
    """A benchmark cannot silently manufacture widths from inadmissible parameters."""
    with pytest.raises(ValueError):
        exact_island_width(epsilon_1=-1.0)
    with pytest.raises(ValueError):
        critical_layer_width(epsilon_kappa=0.0, resonance_radius=0.7)
