"""Regularized MHD Equilibrium Code."""

from importlib.metadata import version
from typing import Final

from remec.normalization import Normalization
from remec.options import RuntimeOptions

__version__: Final = version("remec")

__all__ = ["Normalization", "RuntimeOptions", "__version__"]
