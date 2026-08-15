"""Regularized MHD Equilibrium Code."""

from importlib.metadata import version
from typing import Final

from remec.current_moments import (
    M2ToroidalCurrentSamples,
    ShellCurrentMoments,
    mollified_shell_current_moments,
)
from remec.normalization import Normalization
from remec.options import RuntimeOptions
from remec.profiles import (
    AnalyticPressureProfile,
    AnalyticToroidalCurrentProfile,
    InvalidProfileError,
    PressureProfile,
    TabulatedPressureProfile,
    TabulatedToroidalCurrentProfile,
    ToroidalCurrentProfile,
    TransplantedProfile,
)

__version__: Final = version("remec")

__all__ = [
    "AnalyticPressureProfile",
    "AnalyticToroidalCurrentProfile",
    "InvalidProfileError",
    "M2ToroidalCurrentSamples",
    "Normalization",
    "PressureProfile",
    "RuntimeOptions",
    "ShellCurrentMoments",
    "TabulatedPressureProfile",
    "TabulatedToroidalCurrentProfile",
    "ToroidalCurrentProfile",
    "TransplantedProfile",
    "__version__",
    "mollified_shell_current_moments",
]
