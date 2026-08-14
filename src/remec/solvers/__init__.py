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
    CurrentContinuityFormulation,
    CurrentContinuityResult,
    CurrentContinuitySolver,
    CurrentLayerResolutionDiagnostic,
    FrozenCurrentContinuityCoefficients,
    PrescribedCurrentProfile,
    UnresolvedCurrentLayerError,
    UnresolvedCurrentLayerWarning,
)

__all__ = [
    "AnisotropicDiffusionResult",
    "AnisotropicDiffusionSolver",
    "AnisotropyPollutionError",
    "AnisotropyPollutionWarning",
    "CurrentContinuityFormulation",
    "CurrentContinuityResult",
    "CurrentContinuitySolver",
    "CurrentContinuityStabilization",
    "CurrentLayerResolutionDiagnostic",
    "EnergyDiagnostics",
    "FloorSensitivityDiagnostic",
    "FloorSensitivityError",
    "FloorSensitivityWarning",
    "FrozenCurrentContinuityCoefficients",
    "PollutionSafetyDiagnostic",
    "PrescribedCurrentProfile",
    "SpatialAnisotropicConductivity",
    "UnresolvedCurrentLayerError",
    "UnresolvedCurrentLayerWarning",
]
