import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseForwardSolver(nn.Module, ABC):
    """
    Base class for all forward modeling solvers (Gravity, Magnetic, Electrical, Seismic).
    """
    def __init__(self):
        super(BaseForwardSolver, self).__init__()
        
    @abstractmethod
    def forward(self, model: torch.Tensor, **kwargs):
        """
        Run forward modeling on the input physical model.
        
        Args:
            model (torch.Tensor): 3D tensor of the physical property 
                                 (e.g., density, susceptibility, resistivity, velocity)
                                 Shape: (nx, ny, nz)
                                 
        Returns:
            torch.Tensor: The forward modeled data.
        """
        pass
