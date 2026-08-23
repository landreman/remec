"""Independent analytic contracts for the Section-8.6 magnetic field."""

from __future__ import annotations

from math import pi, sqrt

import numpy as np
import pytest

from remec.reiman_greenside import ReimanGreensideField


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"t0": float("inf")}, "t0"),
        ({"t1": 0.0}, "t1"),
        ({"epsilon_1": float("nan")}, "epsilon_1"),
        ({"epsilon_2": float("inf")}, "epsilon_2"),
        ({"major_radius": 0.0}, "major_radius"),
    ],
)
def test_reiman_greenside_rejects_invalid_parameters(
    arguments: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ReimanGreensideField(**arguments)


def test_integrable_transform_and_reference_resonances_are_derived() -> None:
    field = ReimanGreensideField()
    assert field.rotational_transform(0.0) == pytest.approx(0.29)
    assert field.rotational_transform(1.0) == pytest.approx(0.67)
    assert field.resonance_radius(0.5) == pytest.approx(0.7433919416750281)
    assert field.resonance_radius(1.0 / 3.0) == pytest.approx(0.33769081675298523)
    with pytest.raises(ValueError, match="outside"):
        field.resonance_radius(0.2)


def test_cartesian_field_matches_independent_polar_transcription() -> None:
    field = ReimanGreensideField(epsilon_1=1.0e-3, epsilon_2=2.0e-4, major_radius=1.7)
    radius = 0.63
    theta = -0.41
    phi = 0.72
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = field.major_radius * phi
    zeta_2 = 2.0 * theta - phi
    zeta_3 = 3.0 * theta - phi
    radial = (
        -2.0 * field.epsilon_1 * radius * np.sin(zeta_2)
        - 3.0 * field.epsilon_2 * radius**2 * np.sin(zeta_3)
    ) / field.major_radius
    poloidal = (
        radius * (field.t0 + field.t1 * radius**2)
        - 2.0 * field.epsilon_1 * radius * np.cos(zeta_2)
        - 3.0 * field.epsilon_2 * radius**2 * np.cos(zeta_3)
    ) / field.major_radius
    expected = np.array(
        [
            radial * np.cos(theta) - poloidal * np.sin(theta),
            radial * np.sin(theta) + poloidal * np.cos(theta),
            1.0,
        ]
    )
    np.testing.assert_allclose(field(x, y, z), expected, atol=2.0e-15)


def test_closed_form_vector_potential_has_numerical_curl_equal_to_b() -> None:
    field = ReimanGreensideField(epsilon_1=0.017, epsilon_2=-0.006, major_radius=1.3)
    point = np.array([0.37, -0.29, 0.81])
    step = 2.0e-6
    jacobian = np.empty((3, 3))
    for coordinate in range(3):
        offset = np.zeros(3)
        offset[coordinate] = step
        jacobian[:, coordinate] = (
            field.vector_potential(*(point + offset)) - field.vector_potential(*(point - offset))
        ) / (2.0 * step)
    numerical_curl = np.array(
        [
            jacobian[2, 1] - jacobian[1, 2],
            jacobian[0, 2] - jacobian[2, 0],
            jacobian[1, 0] - jacobian[0, 1],
        ]
    )
    np.testing.assert_allclose(numerical_curl, field(*point), atol=3.0e-10)
    assert field(*point)[2] == 1.0
    assert np.linalg.norm(field(*point)) >= 1.0


def test_axis_is_smooth_and_periodic_in_phi() -> None:
    field = ReimanGreensideField(epsilon_1=0.02, epsilon_2=0.01, major_radius=2.5)
    np.testing.assert_allclose(field(0.0, 0.0, 0.3), np.array([0.0, 0.0, 1.0]))
    point = (0.21, -0.18, 0.43)
    np.testing.assert_allclose(
        field(*point),
        field(point[0], point[1], point[2] + 2.0 * pi * field.major_radius),
        atol=2.0e-15,
    )
    assert field.resonance_radius(0.5) == pytest.approx(sqrt((0.5 - 0.29) / 0.38))
