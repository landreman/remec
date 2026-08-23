"""Reiman--Greenside analytic magnetic field from DESIGN section 8.6."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, sin, sqrt

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class ReimanGreensideField:
    r"""Section-8.6 field ``B = grad(Psi_t)xgrad(Theta) + grad(Phi)xgrad(Psi_p)``."""

    t0: float = 0.29
    t1: float = 0.38
    epsilon_1: float = 0.0
    epsilon_2: float = 0.0
    major_radius: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("t0", self.t0),
            ("t1", self.t1),
            ("epsilon_1", self.epsilon_1),
            ("epsilon_2", self.epsilon_2),
            ("major_radius", self.major_radius),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.t1 <= 0.0:
            raise ValueError("t1 must be positive")
        if self.major_radius <= 0.0:
            raise ValueError("major_radius must be positive")

    def psi_p(self, x: float, y: float, z: float) -> float:
        r"""Return the field-line Hamiltonian ``Psi_p``.

        This is the Cartesian polynomial transcription of DESIGN section 8.6,

        ``Psi_p = t0*Psi_t + t1*Psi_t^2
        - epsilon_1*r^2*cos(2*Theta-Phi)
        - epsilon_2*r^3*cos(3*Theta-Phi)``, ``Psi_t=r^2/2``.
        """
        phi = z / self.major_radius
        cosine = cos(phi)
        sine = sin(phi)
        radius_squared = x * x + y * y
        harmonic_2 = (x * x - y * y) * cosine + 2.0 * x * y * sine
        harmonic_3 = (x**3 - 3.0 * x * y * y) * cosine + (3.0 * x * x * y - y**3) * sine
        return float(
            0.5 * self.t0 * radius_squared
            + 0.25 * self.t1 * radius_squared**2
            - self.epsilon_1 * harmonic_2
            - self.epsilon_2 * harmonic_3
        )

    def vector_potential(self, x: float, y: float, z: float) -> npt.NDArray[np.float64]:
        r"""Return ``A = Psi_t grad(Theta) - Psi_p grad(Phi)`` in Cartesian form."""
        psi_p = self.psi_p(x, y, z)
        return np.asarray((-0.5 * y, 0.5 * x, -psi_p / self.major_radius), dtype=float)

    def magnetic_field(self, x: float, y: float, z: float) -> npt.NDArray[np.float64]:
        r"""Return the exact Cartesian ``B = curl(A)`` satisfying note equation (M1)."""
        phi = z / self.major_radius
        cosine = cos(phi)
        sine = sin(phi)
        radius_squared = x * x + y * y
        psi_x = (
            self.t0 * x
            + self.t1 * radius_squared * x
            - self.epsilon_1 * (2.0 * x * cosine + 2.0 * y * sine)
            - self.epsilon_2 * (3.0 * (x * x - y * y) * cosine + 6.0 * x * y * sine)
        )
        psi_y = (
            self.t0 * y
            + self.t1 * radius_squared * y
            - self.epsilon_1 * (-2.0 * y * cosine + 2.0 * x * sine)
            - self.epsilon_2 * (-6.0 * x * y * cosine + 3.0 * (x * x - y * y) * sine)
        )
        return np.asarray((-psi_y / self.major_radius, psi_x / self.major_radius, 1.0))

    def __call__(self, x: float, y: float, z: float) -> npt.NDArray[np.float64]:
        """Evaluate the field for the public Poincare-tracer protocol."""
        return self.magnetic_field(x, y, z)

    def rotational_transform(self, radius: float) -> float:
        r"""Return the integrable transform ``iota(r) = t0 + t1*r^2``."""
        if not isfinite(radius) or radius < 0.0:
            raise ValueError("radius must be finite and non-negative")
        return self.t0 + self.t1 * radius**2

    def resonance_radius(self, transform: float) -> float:
        """Return the radius at which the integrable transform reaches ``transform``."""
        if not isfinite(transform):
            raise ValueError("transform must be finite")
        radius_squared = (transform - self.t0) / self.t1
        if radius_squared < 0.0:
            raise ValueError("transform lies outside the integrable profile above the axis")
        return sqrt(radius_squared)
