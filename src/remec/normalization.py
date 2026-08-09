"""Nondimensional reference scales used by the FEM core."""

from dataclasses import dataclass
from math import pi


@dataclass(frozen=True, slots=True)
class Normalization:
    """Record the reference scales required by DESIGN.md §6.

    The derived scales are p₀ = B₀²/μ₀, J₀ = B₀/(μ₀L₀), A₀ = B₀L₀,
    u₀ = 1/(μ₀L₀), and Dᵤ₀ = B₀L₀.  The χ transport and source scales are
    explicit so checkpoint metadata can preserve a nondimensional problem.
    """

    reference_length: float
    reference_field: float
    chi_transport_scale: float = 1.0
    chi_source_scale: float = 1.0
    mu0: float = 4.0e-7 * pi

    def __post_init__(self) -> None:
        for name, value in (
            ("reference_length", self.reference_length),
            ("reference_field", self.reference_field),
            ("chi_transport_scale", self.chi_transport_scale),
            ("chi_source_scale", self.chi_source_scale),
            ("mu0", self.mu0),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")

    @property
    def pressure_scale(self) -> float:
        """Return p₀ = B₀²/μ₀."""
        return self.reference_field**2 / self.mu0

    @property
    def current_density_scale(self) -> float:
        """Return J₀ = B₀/(μ₀L₀)."""
        return self.reference_field / (self.mu0 * self.reference_length)

    @property
    def vector_potential_scale(self) -> float:
        """Return A₀ = B₀L₀."""
        return self.reference_field * self.reference_length

    @property
    def u_scale(self) -> float:
        """Return u₀ = 1/(μ₀L₀)."""
        return 1.0 / (self.mu0 * self.reference_length)

    @property
    def du_scale(self) -> float:
        """Return Dᵤ₀ = B₀L₀."""
        return self.vector_potential_scale
