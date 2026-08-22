"""Physical-domain definitions used by the FEM kernels."""

from remec.geometry.axisymmetric import AxisymmetricFluxContourDomain, AxisymmetricRZDomain
from remec.geometry.slab import Slab2D
from remec.geometry.solid_torus import AnalyticSolidTorus

__all__ = [
    "AnalyticSolidTorus",
    "AxisymmetricFluxContourDomain",
    "AxisymmetricRZDomain",
    "Slab2D",
]
