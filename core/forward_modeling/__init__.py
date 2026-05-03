from .base import BaseForwardSolver
from .gravity import GravityForwardSolver
from .magnetic import MagneticForwardSolver
from .electrical import ElectricalForwardSolver
from .seismic import SeismicForwardSolver

__all__ = [
    "BaseForwardSolver",
    "GravityForwardSolver",
    "MagneticForwardSolver",
    "ElectricalForwardSolver",
    "SeismicForwardSolver"
]
