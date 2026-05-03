import torch
from .base import BaseForwardSolver
from deepwave import scalar
from loguru import logger
import time

class SeismicForwardSolver(BaseForwardSolver):
    """
    Seismic Forward Modeling Solver (Acoustic scalar wave equation using deepwave).
    """
    def __init__(self, dx, dt, source_amplitudes, source_locations, receiver_locations, pml_freq=25.0, accuracy=4, pml_width=None):
        super().__init__()
        self.dx = dx
        self.dt = dt
        self.source_amplitudes = source_amplitudes
        self.source_locations = source_locations
        self.receiver_locations = receiver_locations
        self.pml_freq = pml_freq
        self.accuracy = accuracy
        self.pml_width = pml_width

    def forward(self, v: torch.Tensor):
        """
        Forward seismic wave propagation.
        Args:
            v: 2D or 3D tensor representing the velocity model.
        Returns:
            out: Deepwave output dictionary containing 'receiver_amplitudes' and other fields.
        """
        scalar_kwargs = {
            "source_amplitudes": self.source_amplitudes,
            "source_locations": self.source_locations,
            "receiver_locations": self.receiver_locations,
            "accuracy": self.accuracy,
        }
        if self.pml_freq is not None:
            scalar_kwargs["pml_freq"] = self.pml_freq
        # Deepwave expects pml_width to be an int or a sequence of ints; passing
        # None explicitly raises, while omitting it uses the library default.
        if self.pml_width is not None:
            scalar_kwargs["pml_width"] = self.pml_width

        out = scalar(
            v,
            self.dx,
            self.dt,
            **scalar_kwargs,
        )
        return out
