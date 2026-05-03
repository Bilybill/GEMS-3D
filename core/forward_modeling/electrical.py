import torch
from .base import BaseForwardSolver
from Electrical.forward_modeling.mt_forward import MTForward3D

class ElectricalForwardSolver(BaseForwardSolver):
    """
    Electrical Forward Modeling Solver (MT 3D).
    """
    def __init__(self, freqs=None, dx=1.0, dy=1.0, dz=1.0):
        super().__init__()
        self.solver = MTForward3D(freqs, dx, dy, dz)

    def forward(self, rho_tensor: torch.Tensor):
        """
        Forward MT 3D.
        Args:
            rho_tensor: 3D contiguous tensor of shape (NX, NY, NZ) with resistivity values (Ohm-m).
        Returns:
            app_res: Apparent resistivity tensor of shape (n_freqs, NX, NY, 2),
                where the last dimension is (Zxy, Zyx). Returned on the same
                device as `rho_tensor`.
            phase: Phase tensor of shape (n_freqs, NX, NY, 2),
                where the last dimension is (Zxy, Zyx). Returned on the same
                device as `rho_tensor`.
        """
        return self.solver(rho_tensor)

    @property
    def last_freqs(self):
        return self.solver.last_freqs
