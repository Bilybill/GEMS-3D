import torch
from .base import BaseForwardSolver
from Electrical.forward_modeling.mt_forward import MTForward3D

class ElectricalForwardSolver(BaseForwardSolver):
    """
    Electrical Forward Modeling Solver (MT 3D).
    """
    def __init__(
        self,
        freqs=None,
        dx=1.0,
        dy=1.0,
        dz=1.0,
        *,
        boundary_mode="legacy",
        air_layers=8,
        air_resistivity=1.0e8,
        solver_tolerance=None,
        solver_max_iterations=50000,
        frequency_continuation=None,
        padding_layers=10,
        padding_growth=1.4,
    ):
        super().__init__()
        self.solver = MTForward3D(
            freqs=freqs,
            dx=dx,
            dy=dy,
            dz=dz,
            boundary_mode=boundary_mode,
            air_layers=air_layers,
            air_resistivity=air_resistivity,
            solver_tolerance=solver_tolerance,
            solver_max_iterations=solver_max_iterations,
            frequency_continuation=frequency_continuation,
            padding_layers=padding_layers,
            padding_growth=padding_growth,
        )

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

    @property
    def settings(self):
        """Return the effective MT configuration used by the CUDA solver."""
        return {
            "boundary_mode": self.solver.boundary_mode,
            "air_layers": (
                self.solver.air_layers
                if self.solver.boundary_mode == "explicit_air"
                else 0
            ),
            "air_resistivity": self.solver.air_resistivity,
            "solver_tolerance": self.solver.solver_tolerance,
            "solver_max_iterations": self.solver.solver_max_iterations,
            "frequency_continuation": self.solver.frequency_continuation,
            "padding_layers": self.solver.padding_layers,
            "padding_growth": self.solver.padding_growth,
        }
