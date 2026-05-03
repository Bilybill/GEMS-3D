import torch
from .base import BaseForwardSolver

class MagneticForwardSolver(BaseForwardSolver):
    """
    Magnetic Forward Modeling Solver (Total Magnetic Intensity).
    """
    def __init__(self, dx, dy, dz, heights_m, obs_conf, inc, dec, inc0=None, dec0=None, B0=50000.0, output_unit="nt", pad_factor=2, input_type="susceptibility", algorithm="standard_B"):
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.heights_m = heights_m
        self.obs_conf = obs_conf
        self.inc = inc
        self.dec = dec
        self.inc0 = inc0 if inc0 is not None else inc
        self.dec0 = dec0 if dec0 is not None else dec
        self.B0 = B0
        self.output_unit = output_unit
        self.pad_factor = pad_factor
        self.input_type = input_type
        self.algorithm = algorithm

    def forward(self, model: torch.Tensor):
        """
        Forward magnetic TMI.
        Args:
            model: 3D tensor of shape (nx, ny, nz) representing susceptibility or magnetization.
        """
        from Magnetic.forward_modeling.mag_forward import forward_mag_tmi
        data, meta = forward_mag_tmi(
            model,
            self.dx, self.dy, self.dz,
            self.heights_m,
            self.obs_conf,
            input_type=self.input_type,
            B0=self.B0,
            I_deg=self.inc,
            A_deg=self.dec,
            M_I_deg=self.inc0,
            M_A_deg=self.dec0,
            output_unit=self.output_unit,
            pad_factor=self.pad_factor,
            mode=self.algorithm
        )
        return data, meta
