"""Linear and nonlinear solver kernels."""

from remec.options import CurrentContinuityStabilization
from remec.solvers.anisotropic_diffusion import (
    AnisotropicDiffusionResult,
    AnisotropicDiffusionSolver,
    AnisotropyPollutionError,
    AnisotropyPollutionWarning,
    EnergyDiagnostics,
    FloorSensitivityDiagnostic,
    FloorSensitivityError,
    FloorSensitivityWarning,
    PollutionSafetyDiagnostic,
    SpatialAnisotropicConductivity,
)
from remec.solvers.current_continuity import (
    ConstrainedCurrentContinuityResult,
    ConstrainedCurrentContinuitySolver,
    CurrentContinuityFormulation,
    CurrentContinuityResult,
    CurrentContinuitySolver,
    CurrentLayerResolutionDiagnostic,
    FrozenCurrentConstraintGeometry,
    FrozenCurrentContinuityCoefficients,
    UnresolvedCurrentLayerError,
    UnresolvedCurrentLayerWarning,
)

__all__ = [
    "AnisotropicDiffusionResult",
    "AnisotropicDiffusionSolver",
    "AnisotropyPollutionError",
    "AnisotropyPollutionWarning",
    "ConstrainedCurrentContinuityResult",
    "ConstrainedCurrentContinuitySolver",
    "CurrentContinuityFormulation",
    "CurrentContinuityResult",
    "CurrentContinuitySolver",
    "CurrentContinuityStabilization",
    "CurrentLayerResolutionDiagnostic",
    "EnergyDiagnostics",
    "FloorSensitivityDiagnostic",
    "FloorSensitivityError",
    "FloorSensitivityWarning",
    "FrozenCurrentConstraintGeometry",
    "FrozenCurrentContinuityCoefficients",
    "PollutionSafetyDiagnostic",
    "SpatialAnisotropicConductivity",
    "UnresolvedCurrentLayerError",
    "UnresolvedCurrentLayerWarning",
]
