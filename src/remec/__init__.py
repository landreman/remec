"""Regularized MHD Equilibrium Code."""

from importlib.metadata import version
from typing import Final

from remec.normalization import Normalization
from remec.options import RuntimeOptions
from remec.profiles import (
    AnalyticVolumeProfile,
    InvalidProfileError,
    TabulatedVolumeProfile,
    TransplantedProfile,
)

__version__: Final = version("remec")

__all__ = [
    "AnalyticVolumeProfile",
    "InvalidProfileError",
    "Normalization",
    "RuntimeOptions",
    "TabulatedVolumeProfile",
    "TransplantedProfile",
    "__version__",
]
