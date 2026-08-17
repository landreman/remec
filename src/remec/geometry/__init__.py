"""Physical-domain definitions used by the FEM kernels."""

from remec.geometry.axisymmetric import AxisymmetricRZDomain
from remec.geometry.slab import Slab2D
from remec.geometry.solid_torus import AnalyticSolidTorus

__all__ = ["AnalyticSolidTorus", "AxisymmetricRZDomain", "Slab2D"]
