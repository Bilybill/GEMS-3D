import torch
import torch.nn as nn

try:
    from . import mt_forward_cuda
except ImportError:
    try:
        import mt_forward_cuda  # type: ignore[no-redef]
    except ImportError:
        mt_forward_cuda = None


PHOENIX_COEFFS = [8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0]


def _estimate_boundary_background_rho(rho_tensor: torch.Tensor) -> float:
    nx, ny, nz = rho_tensor.shape
    boundary_mask = torch.zeros((nx, ny, nz), dtype=torch.bool, device=rho_tensor.device)
    boundary_mask[0, :, :] = True
    boundary_mask[-1, :, :] = True
    boundary_mask[:, 0, :] = True
    boundary_mask[:, -1, :] = True
    boundary_mask[:, :, 0] = True
    boundary_mask[:, :, -1] = True
    return float(rho_tensor[boundary_mask].mean().item())


def _generate_phoenix_frequencies(f_min: float, f_max: float) -> list[float]:
    if f_min > f_max:
        f_min, f_max = f_max, f_min

    max_power = int(torch.ceil(torch.log10(torch.tensor(f_max, dtype=torch.float64))).item())
    min_power = int(torch.floor(torch.log10(torch.tensor(f_min, dtype=torch.float64))).item())

    freqs = []
    for p in range(max_power, min_power - 1, -1):
        power_of_10 = 10.0 ** p
        for coeff in PHOENIX_COEFFS:
            current_f = coeff * power_of_10
            if current_f <= f_max and current_f >= f_min:
                freqs.append(float(current_f))

    if not freqs:
        freqs = [float(f_max), float(f_min)]
    return freqs


def generate_mt_frequencies(f_min: float, f_max: float) -> list[float]:
    """
    Public helper for building the Phoenix-style MT frequency list used by
    the project from a requested frequency range.
    """
    return _generate_phoenix_frequencies(float(f_min), float(f_max))

def resolve_auto_mt_frequencies(rho_tensor: torch.Tensor, dz: float):
    bg_rho = _estimate_boundary_background_rho(rho_tensor)
    skin_depth_min = dz * 2.0
    total_depth = rho_tensor.shape[2] * dz
    skin_depth_max = total_depth / 1

    f_max = bg_rho * (503.0 / skin_depth_min) ** 2
    f_min = bg_rho * (503.0 / skin_depth_max) ** 2
    freqs = _generate_phoenix_frequencies(f_min, f_max)
    return freqs, bg_rho, f_min, f_max

class MTForward3D(nn.Module):
    def __init__(self, freqs=None, dx=1.0, dy=1.0, dz=1.0):
        """
        PyTorch wrapper for CUDA MT 3D Forward Modeling.
        Args:
            freqs: Optional list of frequencies (Hz). If omitted or empty,
                the frequency range is auto-generated from the model extent and
                boundary background resistivity following MTForward3D.
            dx: Cell width in x-direction (m)
            dy: Cell width in y-direction (m)
            dz: Cell width in z-direction (m)
        """
        super(MTForward3D, self).__init__()
        self.freqs = None if freqs is None else list(freqs)
        self.last_freqs = None
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def forward(self, rho_tensor):
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
        if mt_forward_cuda is None:
            raise ImportError(
                "MT forward modeling requires the compiled mt_forward_cuda extension. "
                "Build Electrical/forward_modeling before running the MT solver, or pass --skip_mt "
                "when using the multiphysics pipeline without MT."
            )
        input_device = rho_tensor.device
        rho_prepared = rho_tensor.detach().to(torch.float64)
        # The original MTForward3D entry path is stable when the extension receives
        # a CPU tensor and performs its own host/device staging internally. The
        # newer CUDA-input branch is still less stable in large pipeline calls, so
        # we keep the public API device-agnostic but normalize the extension input
        # back to the known-good CPU entry path here.
        rho_solver_input = rho_prepared.cpu().contiguous()
        if self.freqs:
            freqs = list(self.freqs)
        else:
            freqs, _, _, _ = resolve_auto_mt_frequencies(rho_solver_input, self.dz)

        self.last_freqs = tuple(freqs)
        app_res, phase = mt_forward_cuda.compute_mt_3d(rho_solver_input, self.dx, self.dy, self.dz, freqs)
        if input_device.type != "cpu":
            app_res = app_res.to(input_device)
            phase = phase.to(input_device)
        # The custom CUDA extension does not surface all launch errors immediately.
        # Synchronize here so failures are attributed to MT instead of a later torch call.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return app_res, phase
